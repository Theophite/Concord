"""Iteration speed of the wired trainer: per-step wall time + breakdown
(TE encode, UNet fwd, bwd = UNet-winner + token self-step, rebalance, aux)."""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from concord_winner import ConcordConfig, configure_optimizer, winner_step, GatedRebalance
from control_plane import TokenSpec, apply_token_spec
from sdxl_train import encode_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--steps", type=int, default=8)
args = ap.parse_args()
dev, dt = torch.device("cuda"), torch.bfloat16

pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)
sched = DDPMScheduler.from_config(pipe.scheduler.config)

config = ConcordConfig(lr=5e-5)
specs = [TokenSpec("<cncd>", "train", init="dog"), TokenSpec("tok", "sanitize")]
layers, aux, cfg = configure_optimizer(pipe.unet, dev, config)
cps = {}
for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                     ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
    apply_token_spec(te, tok, specs, lr=5e-2)
    cps[tag] = (te, tok)
import gc; gc.collect(); torch.cuda.empty_cache()

gate = GatedRebalance(layers)
lat = torch.randn(1, 4, args.res // 8, args.res // 8, device=dev, dtype=dt)
tids = torch.tensor([[args.res, args.res, 0, 0, args.res, args.res]], device=dev, dtype=dt)


def step(it):
    s = torch.cuda.synchronize; s(); t0 = time.time()
    winner_step(it, args.steps, layers, config=cfg)
    aux.zero_grad(set_to_none=True)
    noise = torch.randn_like(lat)
    t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
    noisy = sched.add_noise(lat, noise, t)
    s(); t1 = time.time()
    ehs, pooled = encode_prompt("a photo of <cncd> by the sea", cps)
    s(); t2 = time.time()
    out = pipe.unet(noisy, t, encoder_hidden_states=ehs.to(dt),
                    added_cond_kwargs={"text_embeds": pooled.to(dt), "time_ids": tids}).sample
    s(); t3 = time.time()
    torch.nn.functional.mse_loss(out.float(), noise.float()).backward()
    s(); t4 = time.time()
    gate()                                       # gated rebalance (skips no-op launches)
    s(); t5 = time.time()
    aux.step()
    s(); t6 = time.time()
    return dict(total=t6 - t0, te=t2 - t1, fwd=t3 - t2, bwd=t4 - t3, reb=t5 - t4, aux=t6 - t5)


print(f"=== trainer iteration speed @ {args.res} (UNet winner + 2 TEs + token + sanitize) ===")
rows = [step(it) for it in range(args.steps)]
med = {k: statistics.median(r[k] for r in rows[2:]) for k in rows[0]}        # skip warmup
print(f"  TE encode {med['te']*1e3:6.0f} ms | UNet fwd {med['fwd']*1e3:6.0f} | "
      f"bwd {med['bwd']*1e3:6.0f} | rebalance {med['reb']*1e3:6.0f} | aux {med['aux']*1e3:5.0f}")
print(f"[RESULT] {med['total']*1e3:.0f} ms/step  =  {1/med['total']:.2f} steps/s  "
      f"(peak {torch.cuda.max_memory_reserved()/1024**3:.1f} GB)")
