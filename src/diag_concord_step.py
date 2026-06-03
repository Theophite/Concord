"""Diagnostic: do the Concord packed-B weights ACTUALLY update, or is the loss
drop just the aux AdamW zeroing the output?

Uses the BARE recipe (defaults; ratio_coh / noise / gf_consol all OFF) so nothing
masks the core fused step, and DISABLES the aux optimizer by default so only the
Concord step can move anything. Measures, per Concord layer:
  - ||Δweight|| / ||weight||           (did the live weight move?)
  - fraction of packed_w int32 words that changed   (did the fused step WRITE?)
on tiny (control) vs real albedobaseXL.

Run:
  python src/diag_concord_step.py --size tiny
  python src/diag_concord_step.py --checkpoint C:\\Concord\\albedobaseXL_v21.safetensors
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

from sdxl_fit_smoketest import SDXL_UNET_CONFIG, TINY_UNET_CONFIG
import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB, ConcordConv2dPackedB


def _s(v):
    return v[0] if isinstance(v, (tuple, list)) else v


def bare_swap(unet, dev, lr):
    layers = []
    for parent in list(unet.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                c = ConcordLinearPackedB(child.in_features, child.out_features,
                                         bias=child.bias is not None, device=dev,
                                         alpha=0.1, lr=lr)
                W2d = child.weight.data
            elif isinstance(child, nn.Conv2d):
                k = _s(child.kernel_size)
                c = ConcordConv2dPackedB(child.in_channels, child.out_channels, k,
                                         stride=_s(child.stride), padding=_s(child.padding),
                                         bias=child.bias is not None, device=dev,
                                         alpha=0.1, lr=lr)
                W2d = child.weight.data.reshape(child.out_channels, -1)
            else:
                continue
            with torch.no_grad():
                c.load_weights(W2d.float())
                if child.bias is not None:
                    c.bias.data.copy_(child.bias.data.to(c.bias.dtype))
            setattr(parent, name, c)
            layers.append(c)
    # BARE: explicitly disable the experimental halves so we test the core step.
    ppb.set_ratio_coh(False)
    ppb.set_sigmag_noise(False, isotropic=False)
    ppb.set_fixed_coh(True)   # the validated fixed coherence gate (baked default)
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["tiny", "sdxl"], default="tiny")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--aux", action="store_true", help="also enable aux AdamW (default OFF)")
    args = ap.parse_args()

    dev, dt = torch.device("cuda"), torch.bfloat16
    if args.checkpoint:
        from sdxl_real_checkpoint import load_unet_single_file
        unet = load_unet_single_file(args.checkpoint, dt).to(dev)
        tag = "real"
    else:
        cfg = SDXL_UNET_CONFIG if args.size == "sdxl" else TINY_UNET_CONFIG
        unet = UNet2DConditionModel(**cfg).to(dev, dt)
        tag = args.size
    unet.train()

    layers = bare_swap(unet, dev, args.lr)
    print(f"[{tag}] bare-swapped {len(layers)} Concord layers | aux={'ON' if args.aux else 'OFF'} "
          f"| lr={args.lr} const | ratio_coh=OFF noise=OFF")

    # snapshots BEFORE any step
    w0 = [m.weight.detach().float().clone() for m in layers]
    packed0 = [m.packed_w.detach().clone() for m in layers]

    aux = [p for p in unet.parameters() if p.requires_grad]
    aux_opt = torch.optim.SGD(aux, lr=args.lr, momentum=0.9) if args.aux else None

    g = torch.Generator(device=dev).manual_seed(0)
    B, lat = 1, args.res // 8
    rnd = lambda *s: torch.randn(*s, device=dev, dtype=dt, generator=g)
    sample0 = rnd(B, 4, lat, lat)
    ts = torch.tensor([500], device=dev)
    ehs = rnd(B, 77, 2048)
    add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}
    target = rnd(B, 4, lat, lat)
    fwd = lambda s: unet(s, ts, encoder_hidden_states=ehs, added_cond_kwargs=add_cond).sample

    losses = []
    for it in range(args.steps):
        for m in layers:
            m.lr = args.lr
        if aux_opt:
            aux_opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(fwd(sample0.clone().requires_grad_(True)).float(), target.float())
        loss.backward()
        if aux_opt:
            aux_opt.step()
        for m in layers:
            m.rebalance()
        torch.cuda.synchronize()
        losses.append(loss.item())
        if it % max(1, args.steps // 5) == 0 or it == args.steps - 1:
            print(f"  [step {it:3d}] loss={loss.item():.5f}")

    # did the live weights move, and did the packed int words change?
    dw = torch.tensor([((m.weight.detach().float() - w0i).norm()
                        / (w0i.norm() + 1e-12)).item() for m, w0i in zip(layers, w0)])
    pc = torch.tensor([(m.packed_w != p0).float().mean().item()
                       for m, p0 in zip(layers, packed0)])
    moved = (dw > 1e-4).float().mean().item()
    print("=" * 64)
    print(f"[{tag}] loss {losses[0]:.5f} -> {losses[-1]:.5f}")
    print(f"[{tag}] ||Δw||/||w||  median={dw.median():.2e}  max={dw.max():.2e}  "
          f"frac layers moved(>1e-4)={moved:.2%}")
    print(f"[{tag}] packed_w words changed: median={pc.median():.2%}  max={pc.max():.2%}")
    if moved < 0.5:
        print(f"[{tag}] *** CONCORD STEP IS (mostly) A NO-OP -- weights not moving ***")
    else:
        print(f"[{tag}] concord weights ARE moving.")
    print("=" * 64)


if __name__ == "__main__":
    main()
