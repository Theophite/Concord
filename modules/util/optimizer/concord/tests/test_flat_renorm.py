"""Does renormalize_flat coarsen the per-row block-float scale when the learned offset wants
to be MUCH larger than the (too-fine) init scale?

Hypothesis (the flat@0.1 dead-net): a small substrate -> tiny init scale -> the offset
overshoots the int8 mantissa every step -> the clamp eats the excess, and renorm (only +1
octave/step) can't catch up -> the weight stalls and never reaches the target.

Test: fit ONE fixed LARGE target W_true from several substrate scales (which set the init
scale). If renorm works, all converge (the fine ones' row_exp climbs to track). If renorm is
insufficient, the fine-scale runs stall (MSE stays high, max|w| never reaches the target).

CPU-only:  CUDA_VISIBLE_DEVICES="" python tests/test_flat_renorm.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dual_dissipation_ref as ref          # noqa: E402
import dual_dissipation_flat_ref as flat     # noqa: E402

torch.manual_seed(0)
N, K = 6, 16
W_true = torch.randn(N, K) * 2.0             # LARGE target (|W| up to ~6)
X = torch.randn(3000, K)
Y = X @ W_true.T
TGT_MAX = float(W_true.abs().max())


def mse(W):
    return float(((X @ W.T - Y) ** 2).mean())


def grad_on(Wlive, xb, yb):
    Wp = Wlive.detach().clone().requires_grad_(True)
    ((xb @ Wp.T - yb) ** 2).mean().backward()
    return Wp.grad.detach()


CFG = dict(alpha=0.1, drift_cancel_C=0.02, alpha_v_fast=0.001, coh_kappa=1.0, v_scale=1.0,
           precond_p=0.5, eps=1.0, step_cap=10.0, min_leak=0.1, evap_build_min=128.0,
           beta1=0.0, beta2=0.99, use_coh_vhat=True, mass_preserve=True, chase_floor=0.1,
           leak_floor=0.05, consf=1.0)


def run(sub_scale):
    layer = flat.DualDissipationFlatLayer(N, K, enabled=True, grad_accum_M=8, bracket_d=0.5,
                                          seed=1, substrate_mode=ref.SUBSTRATE_KAIMING)
    base = sub_scale * ref.make_substrate(N, K, layer.substrate_seed, layer.substrate_mode)
    layer.load_weights(torch.zeros(N, K), base=base)
    re0 = float(layer.row_exp.float().mean())
    g = torch.Generator().manual_seed(7)
    for step in range(2000):
        idx = torch.randperm(3000, generator=g)[:256]
        xb, yb = X[idx], Y[idx]
        Wlive = layer.live_weight()
        gL = grad_on(Wlive, xb[:128], yb[:128])
        gH = grad_on(Wlive, xb[128:], yb[128:])
        layer.step(gL, 0.05, grad_W_H=gH, gf_consol=0.5, **CFG)
    w = layer.live_weight()
    return mse(w), re0, float(layer.row_exp.float().mean()), float(w.abs().max())


print(f"target: max|W|={TGT_MAX:.2f}, init MSE ~ {mse(torch.zeros(N, K)):.2f}\n")
print(f"{'sub_scale':>9s} {'init_rowexp':>11s} {'final_rowexp':>12s} {'max|w|':>7s} {'finalMSE':>9s}  verdict")
ok_all = True
for s in (1.0, 0.3, 0.1, 0.03, 0.01):
    m, re0, re1, mx = run(s)
    reached = mx > 0.6 * TGT_MAX and m < 0.2
    ok_all = ok_all and reached
    print(f"{s:>9.3f} {re0:>11.1f} {re1:>12.1f} {mx:>7.2f} {m:>9.4f}  {'ok' if reached else 'STALLED'}")

print()
print("renormalization WORKS across scales" if ok_all
      else "renormalization INSUFFICIENT at fine init scales -> the offset overshoots + clamps "
           "faster than row_exp can climb (+1 octave/step). Fix: multi-octave bump.")
sys.exit(0 if ok_all else 1)
