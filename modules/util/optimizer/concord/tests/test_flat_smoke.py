"""Functional smoke for the same-scale (flat) dual-dissipation reference.

CPU-only (safe while a live run holds the GPU):
    CUDA_VISIBLE_DEVICES="" python tests/test_flat_smoke.py

Checks, on a tiny linear regression learned FROM SCRATCH (random substrate + zeroed offset):
  1. it LEARNS         -- live MSE drops far below the substrate-only baseline;
  2. deploy TRACKS live -- deploy MSE ~ live MSE (the exact chase consolidates the fast into
                           s_slow, so dropping the fast at deploy loses ~nothing: the LSB
                           pathology -- held-but-not-admitted fast mass -- is gone);
  3. dissipation       -- under ZERO gradient the fast register (e_L+e_H) drains toward 0 but
                           the consolidated weight (deploy) PERSISTS.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dual_dissipation_ref as ref          # noqa: E402
import dual_dissipation_flat_ref as flat     # noqa: E402

torch.manual_seed(0)
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name:26s} {detail}")


N, K, B, NDATA = 16, 32, 256, 4000
W_true = torch.randn(N, K) * 0.3
X = torch.randn(NDATA, K)
Y = X @ W_true.T + 0.01 * torch.randn(NDATA, N)


def mse_of(Wmat):
    return float(((X @ Wmat.T - Y) ** 2).mean())


def grad_on(Wlive, xb, yb):
    Wp = Wlive.detach().clone().requires_grad_(True)
    loss = ((xb @ Wp.T - yb) ** 2).mean()
    loss.backward()
    return Wp.grad.detach(), float(loss)


def fine_mag(layer):
    e_H, e_L, _, _ = ref.unpack_dual(layer.packed_w)
    return float((e_H.abs() + e_L.abs()).to(torch.float32).mean())


layer = flat.DualDissipationFlatLayer(N, K, enabled=True, grad_accum_M=8, bracket_d=0.5,
                                      seed=1, substrate_mode=ref.SUBSTRATE_KAIMING)
base = 0.2 * ref.make_substrate(N, K, layer.substrate_seed, layer.substrate_mode)
layer.load_weights(torch.zeros(N, K), base=base)

baseline_mse = mse_of(layer.live_weight())          # substrate only (offset 0)
cfg = dict(alpha=0.1, drift_cancel_C=0.02, alpha_v_fast=0.001, coh_kappa=1.0, v_scale=1.0,
           precond_p=0.5, eps=1.0, step_cap=10.0, min_leak=0.1, evap_build_min=128.0,
           beta1=0.0, beta2=0.99, use_coh_vhat=True, mass_preserve=True, chase_floor=0.1,
           leak_floor=0.05, consf=1.0)
gf, lr = 0.5, 0.05
g = torch.Generator().manual_seed(123)

print(f"baseline (substrate-only) MSE = {baseline_mse:.4f}")
for step in range(1000):
    idx = torch.randperm(NDATA, generator=g)[:B]
    xb, yb = X[idx], Y[idx]
    h = B // 2
    Wlive = layer.live_weight()
    gL, _ = grad_on(Wlive, xb[:h], yb[:h])
    gH, _ = grad_on(Wlive, xb[h:], yb[h:])
    layer.step(gL, lr, grad_W_H=gH, gf_consol=gf, **cfg)
    if step % 200 == 0:
        print(f"  step {step:4d}: live={mse_of(layer.live_weight()):.4f} "
              f"deploy={mse_of(layer.deploy_weight()):.4f} fine|e|={fine_mag(layer):.1f}")

live_mse = mse_of(layer.live_weight())
deploy_mse = mse_of(layer.deploy_weight())
print(f"trained: live={live_mse:.4f} deploy={deploy_mse:.4f} (baseline {baseline_mse:.4f})")

check("learns (live << baseline)", live_mse < 0.25 * baseline_mse,
      f"live={live_mse:.4f} vs baseline={baseline_mse:.4f}")
check("deploy tracks live", abs(deploy_mse - live_mse) < 0.15 * baseline_mse + 0.02,
      f"|deploy-live|={abs(deploy_mse - live_mse):.4f}")

# ---- zero-gradient: fast drains, consolidated weight persists ----
fine_before = fine_mag(layer)
deploy_before = mse_of(layer.deploy_weight())
zero = torch.zeros(N, K)
for _ in range(300):
    layer.step(zero, lr, grad_W_H=zero, gf_consol=gf, **cfg)
fine_after = fine_mag(layer)
deploy_after = mse_of(layer.deploy_weight())
print(f"zero-grad: fine|e| {fine_before:.1f}->{fine_after:.1f}  "
      f"deploy MSE {deploy_before:.4f}->{deploy_after:.4f}")
check("fast drains under zero grad", fine_after < 0.5 * fine_before + 0.5,
      f"{fine_before:.1f}->{fine_after:.1f}")
check("consolidated weight persists", deploy_after < baseline_mse * 0.5,
      f"deploy {deploy_before:.4f}->{deploy_after:.4f} (baseline {baseline_mse:.4f})")

print()
nf = sum(1 for ok in _results if not ok)
print(f"{'ALL PASS' if nf == 0 else str(nf) + ' FAILED'}  ({len(_results)} checks)")
sys.exit(1 if nf else 0)
