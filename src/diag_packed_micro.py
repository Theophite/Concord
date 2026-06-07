"""Surgical test: does the BARE packed-B fused step move a SINGLE Linear's
weights -- across magnitude scales and on REAL SDXL layers?

No UNet, no diffusers model build: just one ConcordLinearPackedB doing a linear
regression W -> W_target. If the live weight moves (dw>0) and packed_w words
change, the core step works at that scale/distribution. Isolates "is the fused
step a no-op on real-scale weights" from all UNet/aux confounds. Runs in seconds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from safetensors import safe_open

import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB

dev = torch.device("cuda")
ppb.set_ratio_coh(False)
ppb.set_sigmag_noise(False, isotropic=False)
ppb.set_fixed_coh(True)
CKPT = r"C:\Concord\albedobaseXL_v21.safetensors"


def test(name, W, lr=5e-4, steps=40):
    W = W.float().to(dev)
    o, i = W.shape
    torch.manual_seed(0)
    m = ConcordLinearPackedB(i, o, bias=False, device=dev, alpha=0.1, lr=lr)
    m.load_weights(W)
    p0 = m.packed_w.clone()
    w0 = m.weight.detach().float().clone()
    recon0 = (w0 - W).norm() / W.norm()              # load_weights fidelity
    Wt = W + torch.randn_like(W) * (W.std() + 1e-9) * 0.5   # perturbed target
    x = torch.randn(128, i, device=dev)
    y = x @ Wt.T
    l0 = None
    for _ in range(steps):
        m.lr = lr
        out = m(x.detach().requires_grad_(True))
        loss = ((out - y) ** 2).mean()
        loss.backward()
        m.rebalance()
        if l0 is None:
            l0 = loss.item()
    torch.cuda.synchronize()
    dw = ((m.weight.detach().float() - w0).norm() / (w0.norm() + 1e-12)).item()
    pc = (m.packed_w != p0).float().mean().item()
    print(f"  {name[:46]:46} shape={str(tuple(W.shape)):12} Wstd={W.std():.4f} "
          f"recon={recon0:.1e}  dw={dw:.2e} packed_chg={pc:5.1%}  "
          f"loss {l0:.3e}->{loss.item():.3e}")


print("=== synthetic scales (init-like vs real-like magnitudes) ===")
for s in (1.0, 0.1, 0.05, 0.02, 0.01):
    torch.manual_seed(1)
    test(f"synthetic_std~{s}", torch.randn(512, 512) * s)

print("=== real albedobaseXL Linear weights (square, mid-size) ===")
picked = []
with safe_open(CKPT, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith("model.diffusion_model") and k.endswith(".weight"):
            sl = f.get_slice(k)
            sh = sl.get_shape()
            if len(sh) == 2 and sh[0] == sh[1] and 320 <= sh[0] <= 1280:
                picked.append((k, f.get_tensor(k)))
                if len(picked) >= 4:
                    break
for k, W in picked:
    test(k.replace("model.diffusion_model.", ""), W)
