"""Smoketest of the WINNING Concord config (concord_winner.py) on SDXL.

Not a quality benchmark (that needs real data + multi-seed). This proves the
winner is correctly wired and MECHANICALLY sound on the SDXL UNet:

  1. swap to the winner lineage (src/prototype_packed_b.py, both halves) works;
  2. every winner knob is LIVE (ratio_coh, fixed_coh, isotropic noise, the
     decaying floors) -- printed mid-run so it can't be silently off;
  3. the fused-in-backward step DESCENDS -- single fixed-batch overfit, loss
     must drop (a broken optimizer can't overfit one batch);
  4. (real-init) the swap is non-destructive -- step-0 forward matches original;
  5. peak VRAM < 24 GB with the full winner active;
  6. consolidated_weight() materializes a deploy weight (drop s_fast).

Run:
  python src/sdxl_winner_smoketest.py --size tiny
  python src/sdxl_winner_smoketest.py --checkpoint C:\\Concord\\albedobaseXL_v21.safetensors
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

from sdxl_fit_smoketest import SDXL_UNET_CONFIG, TINY_UNET_CONFIG, _gb
from prototype_packed_b import ConcordLinearPackedB, ConcordConv2dPackedB
from concord_winner import (swap_unet_to_winner, winner_step, active_config,
                            consolidated_state_dict, make_aux_optimizer, WINNER)


def build_or_load(size, checkpoint, dev, dt):
    if checkpoint:
        from sdxl_real_checkpoint import load_unet_single_file
        return load_unet_single_file(checkpoint, dt).to(dev)
    cfg = SDXL_UNET_CONFIG if size == "sdxl" else TINY_UNET_CONFIG
    return UNet2DConditionModel(**cfg).to(dev, dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["tiny", "sdxl"], default="tiny")
    ap.add_argument("--checkpoint", default=None, help="real single-file SDXL (implies sdxl size)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4, help="smoketest LR (NOT a validated SDXL LR)")
    ap.add_argument("--ckpt", action="store_true", help="grad checkpointing (for the VRAM check)")
    ap.add_argument("--no_noise", action="store_true", help="ablate the fluctuation half")
    ap.add_argument("--target", choices=["random", "input"], default="random",
                    help="overfit target: random (predict-zero is an attractor) or "
                         "input (fittable: predict the input, no trivial escape)")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev, dt = torch.device("cuda"), torch.bfloat16
    free0, total0 = torch.cuda.mem_get_info()
    print(f"[gpu] free {_gb(free0):.2f}/{_gb(total0):.2f} GB | size={args.size} "
          f"ckpt={'real' if args.checkpoint else 'random'} noise={not args.no_noise}")
    torch.cuda.reset_peak_memory_stats()

    unet = build_or_load(args.size, args.checkpoint, dev, dt)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"[build] UNet {n_params/1e9:.4f} B params")

    # one fixed batch (seeded) -> overfit target
    g = torch.Generator(device=dev).manual_seed(0)
    B, lat = 1, args.res // 8
    rnd = lambda *s: torch.randn(*s, device=dev, dtype=dt, generator=g)
    sample0 = rnd(B, 4, lat, lat)
    ts = torch.tensor([500], device=dev)
    ehs = rnd(B, 77, 2048)
    add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}
    target = sample0.clone() if args.target == "input" else rnd(B, 4, lat, lat)

    fwd = lambda s: unet(s, ts, encoder_hidden_states=ehs,
                         added_cond_kwargs=add_cond).sample

    # (4) non-destructive swap: reference forward BEFORE swap
    do_faith = args.checkpoint is not None
    if do_faith:
        unet.eval()
        with torch.no_grad():
            ref = fwd(sample0).float()

    layers = swap_unet_to_winner(unet, dev, args.lr)

    if do_faith:
        with torch.no_grad():
            out0 = fwd(sample0).float()
        rel = (out0 - ref).norm() / ref.norm()
        cos = F.cosine_similarity(out0.flatten(), ref.flatten(), dim=0)
        print(f"[faithful] step-0 forward vs original: rel_L2={rel.item():.4%} "
              f"cos={cos.item():.6f}  {'OK' if cos > 0.999 else 'BAD'}")

    unet.train()
    if args.ckpt:
        unet.enable_gradient_checkpointing()

    aux = [p for p in unet.parameters() if p.requires_grad]
    aux_opt = make_aux_optimizer(aux, args.lr) if aux else None   # SGD, not AdamW
    print(f"[setup] concord layers={len(layers)}  aux tensors={len(aux)} "
          f"({sum(p.numel() for p in aux)/1e6:.2f}M) -> SGD  peak_lr={args.lr}")

    # (3) single-batch overfit -- loss must drop
    warmup = max(1, args.steps // 10)
    losses = []
    log_every = max(1, args.steps // 6)
    for it in range(args.steps):
        lr = winner_step(it, args.steps, layers, peak_lr=args.lr, warmup=warmup,
                         noise=not args.no_noise)
        if aux_opt:
            aux_opt.zero_grad(set_to_none=True)
        s = sample0.clone().requires_grad_(True)
        loss = F.mse_loss(fwd(s).float(), target.float())
        loss.backward()
        if aux_opt:
            aux_opt.step()
        for m in layers:
            m.rebalance()
        torch.cuda.synchronize()
        losses.append(loss.item())
        if it % log_every == 0 or it == args.steps - 1:
            ac = active_config()
            print(f"[step {it:3d}] loss={loss.item():.5f} lr={lr:.2e} "
                  f"sigma={ac['sigma_now']} chase_floor={ac['chase_floor_now']} "
                  f"leak_floor={ac['leak_floor_now']}")

    # (2) prove the winner config was engaged
    print(f"[active] {active_config()}")

    # (6) deploy path: consolidated_weight() on a few layers
    named = [(n, m) for n, m in unet.named_modules()
             if isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB))]
    sd = consolidated_state_dict([m for _, m in named[:4]], [n for n, _ in named[:4]])
    ok = all(torch.isfinite(v).all() for v in sd.values())
    print(f"[deploy] consolidated_weight() on {len(named[:4])} layers -> "
          f"{len(sd)} tensors, finite={ok}, e.g. "
          f"{named[0][0]}.weight {tuple(sd[named[0][0]+'.weight'].shape)}")

    drop = 100 * (losses[0] - losses[-1]) / losses[0]
    peak = _gb(torch.cuda.max_memory_reserved())
    print("=" * 64)
    print(f"[RESULT] loss {losses[0]:.5f} -> {losses[-1]:.5f} ({drop:+.1f}% over "
          f"{args.steps} steps)  {'DESCENDS' if losses[-1] < losses[0] else 'NO DESCENT'}")
    print(f"[RESULT] peak reserved {peak:.2f} GB  {'FITS 24GB' if peak < 24 else 'OVER'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
