"""Regression: a BARE ConcordLinear/Conv2dPackedB must now BE the validated
production optimizer (rank-1 v-hat AdamW + fixed coherence gate), with no knob
setting. Asserts the baked config + that it trains + that disable_cohpre works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import torch
import torch.nn.functional as F
import prototype_packed_b as ppb
from prototype_packed_b import ConcordLinearPackedB, ConcordConv2dPackedB

dev = "cuda"
torch.manual_seed(0); torch.cuda.manual_seed_all(0)


def check_cfg(m, name):
    assert m.optimizer_kind == 'adamw', (name, m.optimizer_kind)
    assert abs(m._eps_value - 1e-10) < 1e-20, (name, m._eps_value)
    assert m.v_scale == 0.0, (name, m.v_scale)
    assert m.gf_trust_delta_sq == 1.0, (name, m.gf_trust_delta_sq)
    assert m.precond_p == 0.5, (name, m.precond_p)
    assert m._coh_pre is not None and m._coh_pre.shape == m.packed_w.shape, name
    print(f"  [{name:7}] adamw eps=1e-10 v_scale=0 gf_trust=1 precond=0.5 gate=ON  OK")


assert ppb._USE_FIXED_COH is True, "fixed coherence gate not the default"
print("global _USE_FIXED_COH = True  OK")
check_cfg(ConcordLinearPackedB(64, 128, bias=False, device=dev), "Linear")
check_cfg(ConcordConv2dPackedB(3, 16, 3, padding=1, bias=False, device=dev), "Conv2d")

# Bare default must train a toy regression (loss drops).
torch.manual_seed(1)
W = ConcordLinearPackedB(32, 32, bias=False, device=dev); W.lr = 0.02
tgt = torch.randn(32, 32, device=dev) * 0.3
x = torch.randn(256, 32, device=dev)
y = x @ tgt.T
losses = []
for _ in range(80):
    xb = x.detach().requires_grad_(True)
    loss = F.mse_loss(W(xb), y)
    loss.backward()
    W.rebalance()
    losses.append(loss.item())
print(f"  bare-default fit: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
assert losses[-1] < 0.9 * losses[0], "bare default not training"

# Ablation path: disable_cohpre() turns the gate off and still steps.
W.disable_cohpre(); assert W._coh_pre is None
loss = F.mse_loss(W(x.detach().requires_grad_(True)), y); loss.backward(); W.rebalance()
print("  disable_cohpre() + step OK")
print("ALL BAKED-DEFAULT CHECKS PASSED")
