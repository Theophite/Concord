"""make_graphed_callables + rebalance, clean (both eager and graphed run rebalance).
probe2 proved fwd+bwd capture is correct (graphed==eager, no rebalance). This adds
rebalance back to find the right integration. rebalance() reads _row_max_buf/_col_max_buf
populated by the apply kernel INSIDE the captured backward, then launches kernels that
mutate packed_w + row_exp/col_exp.
  - If graphed+rebalance == eager+rebalance: eager rebalance after the graphed call is
    fine (buffers are the real ones) -> trivial harness integration.
  - If diverge: make_graphed_callables snapshots _row_max_buf -> rebalance reads stale;
    need to read the buffer the captured bwd actually writes (or rebalance differently).
Run: python tools/probe_graph3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch, torch.nn as nn, torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB

dev = "cuda"

def build():
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    m = nn.Sequential(
        ConcordLinearPackedB(64, 128, bias=False, device=dev),
        nn.GELU(),
        ConcordLinearPackedB(128, 64, bias=False, device=dev),
    )
    for layer in m:
        if hasattr(layer, "lr"): layer.lr = 0.02
    return m

def concords(m): return [l for l in m if hasattr(l, "rebalance")]

torch.manual_seed(1)
X = torch.randn(256, 64, device=dev)
Y = torch.randn(256, 64, device=dev) * 0.3
N = 120   # long enough that rebalance likely fires

# ---- EAGER + rebalance ----
m1 = build()
eager = []
for i in range(N):
    x = X.detach().requires_grad_(True)
    loss = F.mse_loss(m1(x), Y); loss.backward()
    for l in concords(m1): l.rebalance()
    eager.append(loss.item())
print(f"[eager  +reb] {eager[0]:.5f} -> {eager[-1]:.5f}")

# ---- GRAPHED + eager rebalance after replay ----
m2 = build()
sample = torch.randn(256, 64, device=dev, requires_grad=True)
# record buffer ptrs before/after capture to detect snapshotting
pre = [l._row_max_buf.data_ptr() for l in concords(m2)]
graphed = torch.cuda.make_graphed_callables(m2, (sample,))
post = [l._row_max_buf.data_ptr() for l in concords(m2)]
print(f"[buf] _row_max_buf ptr unchanged by capture: {pre == post}")
g = []
for i in range(N):
    x = X.detach().requires_grad_(True)
    loss = F.mse_loss(graphed(x), Y); loss.backward()
    for l in concords(m2): l.rebalance()
    g.append(loss.item())
print(f"[graphed+reb] {g[0]:.5f} -> {g[-1]:.5f}")
md = max(abs(a-b) for a, b in zip(eager, g))
print(f"[compare] max |eager-graphed| over {N} = {md:.6f}")
print("[VERDICT] " + ("MATCH -> eager rebalance after graphed call WORKS. Port to harness."
                      if md < 1e-2 else
                      "DIVERGE -> rebalance reads a snapshotted buffer; needs fix."))
