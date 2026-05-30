"""Probe: are v_proxy (drift-cancel noise²) and the gf-trust floor (δ²·v̂)
actually non-negligible vs eps in the AdamW int8 denominator
sqrt(v_proxy + δ²·v̂ + eps)?

Builds an fc1-sized ConcordLinearFusedInt8, drives it with realistic
gradients for a while, then recomputes the three denominator terms
element-wise from the live buffers and compares their medians to eps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import torch
import torch.nn.functional as F
from concord_linear_fused import ConcordLinearFusedInt8
from prototype_packed_b import compute_drift_cancel_C

torch.manual_seed(0); torch.cuda.manual_seed_all(0)
dev = 'cuda'
IN, OUT = 4096, 512          # fc1 dims
alpha, alpha_v_fast = 0.1, 0.001
eps = 1.0
drift_C = compute_drift_cancel_C(alpha, alpha_v_fast)
gf_delta_sq = 1.0 / (10.0 ** 2)   # = 0.01

lin = ConcordLinearFusedInt8(IN, OUT, bias=False, device=dev)
# Init weights at a realistic trained magnitude.
W0 = torch.randn(OUT, IN, device=dev) * 0.02
lin.load_weights_finetune(W0)
lin.set_optimizer_kind('adamw', weight_decay=0.01, eps=eps)
lin.optimizer_v_kind = 'three_accum'
lin.alpha, lin.alpha_v_fast = alpha, alpha_v_fast
lin.drift_cancel_C = drift_C
lin.wd_sv = lin.wd_sf = 1e-5
lin.enable_gf_trust(delta_sq=gf_delta_sq)
lin.lr = 0.02                 # ~ lr * v_lr_scale in the real run

# Drive with a fixed random regression task so grads are realistic.
# Xb requires grad so the autograd graph is built (in real training the
# upstream activations carry requires_grad); the int8 apply kernel + gf
# EMA update run as a side effect of backward.
Xb = torch.randn(64, IN, device=dev, dtype=torch.bfloat16).requires_grad_(True)
Yt = torch.randn(64, OUT, device=dev, dtype=torch.bfloat16)
for step in range(60):
    y = lin(Xb)
    loss = F.mse_loss(y.float(), Yt.float())
    loss.backward()

# Recompute the grad_W actually used on a fresh step (matches kernel input).
y = lin(Xb)
loss = F.mse_loss(y.float(), Yt.float())
gy = torch.autograd.grad(loss, y, retain_graph=False)[0].to(torch.bfloat16)
grad_W = torch.matmul(gy.transpose(0, 1), Xb).float()   # [OUT, IN]

# ---- recompute denominator terms element-wise from live buffers ----
s_slow = lin.s_slow.float()
s_fast = lin.s_fast.float()
v_slow_full = lin.v_slow_i8.float() * lin.v_slow_factor
row_e = lin.row_exp.float()[:, None]
col_e = lin.col_exp.float()[None, :]
scale_fwd = torch.exp2(row_e + col_e - lin.MANTISSA_BIAS)

noise = s_fast - drift_C * (s_slow - v_slow_full)     # slow_scale=1
noise_in_w = noise * scale_fwd
v_proxy = noise_in_w * noise_in_w * 1.0               # v_scale=1

v_hat = lin.v_row[:, None] * lin.v_col[None, :] * lin._sum_v_inv
gf_floor = gf_delta_sq * v_hat

def stats(name, t):
    t = t.flatten()
    q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99], device=t.device))
    print(f"{name:18s} median={q[0].item():.3e}  p90={q[1].item():.3e}  "
          f"p99={q[2].item():.3e}  max={t.max().item():.3e}")

print(f"drift_C = {drift_C:.5f}   eps = {eps}   gf_delta_sq = {gf_delta_sq}")
print(f"scale_fwd median = {scale_fwd.median().item():.3e}")
stats("v_proxy (noise2)", v_proxy)
stats("gf_floor (d2*vhat)", gf_floor)
stats("grad_W^2", grad_W ** 2)
print(f"eps = {eps:.3e}  (constant)")
print()

# Effect on the actual step: denom with full term vs eps-only.
denom_full = torch.sqrt(v_proxy + gf_floor + eps)
denom_eps_only = torch.sqrt(torch.full_like(v_proxy, eps))
ratio = (denom_full / denom_eps_only)
print("denom_full / denom_eps_only  (1.0 => preconditioner inert):")
stats("  denom_ratio", ratio)
# How much would the step change if eps were tiny (preconditioner live)?
denom_tiny_eps = torch.sqrt(v_proxy + gf_floor + 1e-12)
print("denom(eps=1.0) / denom(eps=1e-12)  (>>1 => eps is what sets the step):")
stats("  eps_vs_tiny", denom_full / denom_tiny_eps)
