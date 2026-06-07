"""Localize the slowness: vanilla SDXL fwd+bwd vs Concord-swapped, per-step and
per-phase. Random init (timing is weight-value-independent), so no checkpoint
load. Step-by-step timing separates one-time Triton compile (step 0) from
steady-state per-step cost.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

from sdxl_fit_smoketest import SDXL_UNET_CONFIG, TINY_UNET_CONFIG, _gb
from concord_winner import swap_unet_to_winner, winner_step

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["tiny", "sdxl"], default="sdxl")
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--ckpt", action="store_true")
ap.add_argument("--steps", type=int, default=8)
args = ap.parse_args()

dev, dt = torch.device("cuda"), torch.bfloat16
cfg = SDXL_UNET_CONFIG if args.size == "sdxl" else TINY_UNET_CONFIG
unet = UNet2DConditionModel(**cfg).to(dev, dt)
unet.train()
if args.ckpt:
    unet.enable_gradient_checkpointing()

B, lat = args.batch, args.res // 8
g = torch.Generator(device=dev).manual_seed(0)
rnd = lambda *s: torch.randn(*s, device=dev, dtype=dt, generator=g)
sample0 = rnd(B, 4, lat, lat)
ts = torch.tensor([500] * B, device=dev)
ehs = rnd(B, 77, 2048)
add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}
target = rnd(B, 4, lat, lat)
fwd = lambda s: unet(s, ts, encoder_hidden_states=ehs, added_cond_kwargs=add_cond).sample


def step(layers=None, it=0, n=8):
    torch.cuda.synchronize(); t0 = time.time()
    if layers is not None:
        winner_step(it, n, layers, 5e-4, warmup=1)
    s = sample0.clone().requires_grad_(True)
    torch.cuda.synchronize(); t_sched = time.time()
    out = fwd(s)
    torch.cuda.synchronize(); t_fwd = time.time()
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    torch.cuda.synchronize(); t_bwd = time.time()
    if layers is not None:
        for m in layers:
            m.rebalance()
        torch.cuda.synchronize()
    t_reb = time.time()
    return dict(total=t_reb - t0, sched=t_sched - t0, fwd=t_fwd - t_sched,
                bwd=t_bwd - t_fwd, reb=t_reb - t_bwd)


print(f"=== {args.size} res={args.res} bsz={B} ckpt={args.ckpt} ===")
print("--- VANILLA (nn.Linear/Conv2d, cuBLAS/cuDNN) ---")
for it in range(4):
    d = step(None)
    print(f"  step {it}: total={d['total']*1e3:7.1f}ms  fwd={d['fwd']*1e3:6.1f}  bwd={d['bwd']*1e3:6.1f}")
van = d['total']

van_grad_free = None
unet.zero_grad(set_to_none=True)            # drop the ~5GB of vanilla-baseline grads
import gc
layers = swap_unet_to_winner(unet, dev, 5e-4, verbose=False)
gc.collect(); torch.cuda.empty_cache()      # release the GC'd vanilla modules' storage
torch.cuda.reset_peak_memory_stats()        # measure CONCORD peak only (no baseline inflation)
print(f"--- CONCORD winner ({len(layers)} layers) ---")
for it in range(args.steps):
    d = step(layers, it, args.steps)
    note = "  <- step0 = Triton compile" if it == 0 else ""
    print(f"  step {it}: total={d['total']*1e3:7.1f}ms  sched={d['sched']*1e3:5.1f}  "
          f"fwd={d['fwd']*1e3:6.1f}  bwd={d['bwd']*1e3:6.1f}  reb={d['reb']*1e3:6.1f}{note}")
con = d['total']
print(f"\n[ratio] steady concord/vanilla = {con/van:.1f}x  "
      f"(vanilla {van*1e3:.0f}ms -> concord {con*1e3:.0f}ms/step)")
print(f"[peak] {_gb(torch.cuda.max_memory_reserved()):.2f} GB")
