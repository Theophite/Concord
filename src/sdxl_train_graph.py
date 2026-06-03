"""Graphed SDXL trainer: the whole GPU step -- both TE forwards (with the trainable
token), the diffusion noising, the UNet, the loss, and the fused-in-backward Concord
step + rebalance -- captured ONCE in a CUDA graph and replayed each iteration. This
erases the bsz=1 Triton launch overhead on the backward (the floor after gating
rebalance).

The pieces that make it legal + correct (all proven separately first):
  - CHECKPOINTING caps activations so the capture pool fits at 1024 (ckpt@1024 used
    LESS than no-ckpt@512). We force non-reentrant + preserve_rng_state=False so there
    is NO get/set_rng_state host sync to break capture -- correct because the UNet is
    dropout-free, so the recompute is bit-identical regardless.
  - NOISE rides the default CUDA generator; torch.cuda.graph() makes it capturable, so
    the winner's fluctuation draw (and the diffusion noise/timestep) is FRESH per
    replay (verified: 3/3 distinct draws). No divergence.
  - The SCHEDULE (lr / sigma / ratio floors) lives in DEVICE tensors that winner_step
    updates OUTSIDE the graph -> replays see the moving schedule without recapture.
  - Host work (tokenize, aux SGD) stays outside; the prompt is fixed -> static ids.

Single-image overfit here (static latent). Multi-image is a copy_ into static_lat /
static_ids before each replay -- the graph is identical.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

# capture-legal checkpoint: non-reentrant, no RNG-state sync (UNet is dropout-free)
import torch.utils.checkpoint as _ckpt
_orig_ckpt = _ckpt.checkpoint


def _capturable_checkpoint(function, *a, use_reentrant=None, preserve_rng_state=True, **kw):
    return _orig_ckpt(function, *a, use_reentrant=False, preserve_rng_state=False, **kw)


_ckpt.checkpoint = _capturable_checkpoint

import torch
import torch.nn.functional as F

from concord_winner import configure_optimizer, winner_step
from control_plane import apply_token_spec
from sdxl_train import tokenize_prompt, encode_from_ids, dev, dt


def train_graphed(pipe, opt_config, token_specs, lat, prompt, time_ids, steps, sched,
                  warmup=3, log_every=30):
    pipe.unet.enable_gradient_checkpointing()                  # cap activations -> graph fits
    layers, aux_opt, cfg = configure_optimizer(pipe.unet, dev, opt_config)
    cps = {}
    for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                         ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
        apply_token_spec(te, tok, token_specs, lr=5e-2)
        cps[tag] = (te, tok)               # SDXL CLIP dropout=0 -> train mode is capture-safe,
                                           # and avoids any train-gated token self-step surprise
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    sched.alphas_cumprod = sched.alphas_cumprod.to(dev)        # on-device: add_noise won't CPU->CUDA sync
    static_lat = lat.detach().clone()                          # fixed (multi-image: copy_ in)
    static_ids = tokenize_prompt(prompt, cps)                  # fixed prompt -> static ids
    aux_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    w0 = layers[0].weight.detach().float().clone()

    def step():
        for p in aux_params:
            if p.grad is not None:
                p.grad.zero_()             # in-place: keep .grad addresses static for replay
        noise = torch.randn_like(static_lat)                   # graph RNG -> fresh per replay
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
        noisy = sched.add_noise(static_lat, noise, t)
        ehs, pooled = encode_from_ids(static_ids, cps)         # TE fwd (+ token) in-graph
        pred = pipe.unet(noisy, t, encoder_hidden_states=ehs.to(dt),
                         added_cond_kwargs={"text_embeds": pooled.to(dt),
                                            "time_ids": time_ids}).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()                                        # UNet + token self-step
        for m in layers:
            m.rebalance()                                      # in-graph: no launch overhead
        return loss

    winner_step(0, steps, layers, config=cfg)                  # seed the device tensors pre-capture
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                                 # side-stream warmup
        for _ in range(warmup):
            step(); aux_opt.step()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap_loss = step()
    print(f"[graph] captured whole step (TE+UNet+loss+bwd+rebalance), ckpt on")

    for it in range(steps):
        winner_step(it, steps, layers, config=cfg)             # host: move the schedule
        g.replay()                                             # GPU: the whole step
        aux_opt.step()                                         # host: aux params (read static .grad)
        if it % log_every == 0 or it == steps - 1:
            dw = ((layers[0].weight.detach().float() - w0).norm() / w0.norm()).item()
            print(f"  [{it:3d}] loss {cap_loss.item():.4f} | UNet w moved {dw:.2e}")
    return layers, cps


if __name__ == "__main__":
    import numpy as np
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from concord_winner import ConcordConfig
    from control_plane import TokenSpec

    pipe = StableDiffusionXLPipeline.from_single_file(
        r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=dt).to(dev)
    pipe.set_progress_bar_config(disable=True)
    for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    ref = pipe("a photograph of a corgi puppy in grass", num_inference_steps=25,
               height=1024, width=1024, generator=torch.Generator(dev).manual_seed(1)).images[0]

    opt_config = ConcordConfig(lr=5e-5, noise=True, aux="sgd")
    token_specs = [TokenSpec("<cncd>", "train", init="dog"), TokenSpec("tok", "sanitize")]

    RES = 512
    arr = np.array(ref.resize((RES, RES))).astype("float32") / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev, dt) * 2 - 1
    with torch.no_grad():
        lat = pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor
    tids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)

    import time
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    layers, cps = train_graphed(pipe, opt_config, token_specs, lat, "a photo of <cncd>",
                                tids, 120, sched)
    torch.cuda.synchronize(); wall = time.time() - t0

    teL, tokL = cps["L"]
    cp = teL.get_input_embeddings()
    z = lambda w: cp(torch.tensor([[tokL.convert_tokens_to_ids(w)]], device=dev)).norm().item()
    print(f"\n[RESULT] peak {torch.cuda.max_memory_reserved()/1024**3:.1f} GB | "
          f"120 steps in {wall:.1f}s ({wall/120*1e3:.0f} ms/step incl. capture) | "
          f"'<cncd>' norm {z('<cncd>'):.3f}, 'tok' norm {z('tok'):.3f} | graphed backward")
