"""How often does rebalance actually FIRE (change an exponent)? If it's rare, the
840 ms of unconditional rebalance launches is almost all wasted -- a flag set in the
apply kernel + conditional dispatch would reclaim it. Measures the per-step fraction
of the 794 UNet layers whose row/col exponent changed during rebalance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from concord_winner import ConcordConfig, configure_optimizer, winner_step
from control_plane import TokenSpec, apply_token_spec
from sdxl_train import encode_prompt

dev, dt = torch.device("cuda"), torch.bfloat16
pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)
sched = DDPMScheduler.from_config(pipe.scheduler.config)

config = ConcordConfig(lr=5e-5)
layers, aux, cfg = configure_optimizer(pipe.unet, dev, config)
cps = {}
for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                     ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
    apply_token_spec(te, tok, [TokenSpec("<cncd>", "train", init="dog")], lr=5e-2)
    cps[tag] = (te, tok)
import gc; gc.collect(); torch.cuda.empty_cache()

RES, N = 512, 40
lat = torch.randn(1, 4, RES // 8, RES // 8, device=dev, dtype=dt)
tids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)

print(f"=== rebalance fire-rate over {N} steps (lr={config.lr}, {len(layers)} layers) ===")
total_fired = 0
for it in range(N):
    winner_step(it, N, layers, config=cfg)
    aux.zero_grad(set_to_none=True)
    noise = torch.randn_like(lat)
    t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
    ehs, pooled = encode_prompt("a photo of <cncd>", cps)
    out = pipe.unet(sched.add_noise(lat, noise, t), t, encoder_hidden_states=ehs.to(dt),
                    added_cond_kwargs={"text_embeds": pooled.to(dt), "time_ids": tids}).sample
    F.mse_loss(out.float(), noise.float()).backward()
    before = [(m.row_exp.clone(), m.col_exp.clone()) for m in layers]
    for m in layers:
        m.rebalance()
    fired = sum(int((m.row_exp != re).any() or (m.col_exp != ce).any())
                for m, (re, ce) in zip(layers, before))
    total_fired += fired
    if it % 5 == 0 or fired:
        print(f"  step {it:2d}: {fired:3d}/{len(layers)} layers rebalanced")

avg = total_fired / N
print(f"\n[RESULT] avg {avg:.1f}/{len(layers)} layers fire per step ({100*avg/len(layers):.1f}%). "
      f"Unconditional rebalance launches all {len(layers)} every step (~840 ms); a flag + "
      f"conditional dispatch would launch ~{avg:.0f} -> reclaim ~{100*(1-avg/len(layers)):.0f}% of it.")
