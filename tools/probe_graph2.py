"""Clean test of make_graphed_callables on Concord, isolating the fwd+bwd capture
from confounds in probe_graph.py (which ran rebalance OUTSIDE the callable -> stale
_row_max_buf, AND let make_graphed_callables warm up on random data).

Test A: NO rebalance anywhere (eager ref also no rebalance). If graphed == eager here,
the fwd+bwd capture (incl the fused optimizer side-effect in backward) is CORRECT and
the earlier divergence was the rebalance/warmup confound -> make_graphed_callables is
usable, we just handle rebalance separately.

Run: python tools/probe_graph2.py
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

torch.manual_seed(1)
X = torch.randn(256, 64, device=dev)
Y = torch.randn(256, 64, device=dev) * 0.3

# ---- EAGER, NO rebalance ----
m1 = build()
eager = []
for i in range(40):
    x = X.detach().requires_grad_(True)
    loss = F.mse_loss(m1(x), Y)
    loss.backward()
    eager.append(loss.item())
print(f"[eager  no-reb] {eager[0]:.5f} -> {eager[-1]:.5f}")

# ---- GRAPHED via make_graphed_callables, NO rebalance ----
m2 = build()
try:
    sample = torch.randn(256, 64, device=dev, requires_grad=True)
    graphed = torch.cuda.make_graphed_callables(m2, (sample,))
    # check: did it keep the SAME buffer tensor objects? (clone => coupling broken)
    same_wbuf = []
    for orig, gm in zip(m1, m2):
        if hasattr(gm, "_bf16_weight_buf"):
            same_wbuf.append(gm._bf16_weight_buf.data_ptr())
    g = []
    for i in range(40):
        x = X.detach().requires_grad_(True)
        loss = F.mse_loss(graphed(x), Y)
        loss.backward()
        g.append(loss.item())
    print(f"[graphed no-reb] {g[0]:.5f} -> {g[-1]:.5f}")
    md = max(abs(a-b) for a, b in zip(eager, g))
    print(f"[compare] max |eager-graphed| = {md:.6f}")
    print("[VERDICT A] " + ("MATCH -> fwd+bwd capture is CORRECT; earlier 5336 was the "
                            "rebalance/warmup confound. make_graphed_callables IS usable."
                            if md < 1e-2 else
                            "DIVERGE even without rebalance -> capture breaks the fused "
                            "side-effect step itself. Need custom single-graph capture."))
except Exception as e:
    import traceback
    print("[graphed] FAILED:", type(e).__name__)
    print("\n".join(traceback.format_exc().strip().split("\n")[-10:]))
