"""Probe: can torch.cuda.make_graphed_callables graph a ConcordLinearPackedB whose
optimizer step is FUSED into the backward (state in buffers, mutated by side effect,
zero nn.Parameters)? Three questions:
  1. Does make_graphed_callables accept a param-less module (buffers only)?
  2. Do the side-effect kernel writes (packed_w update) REPLAY from the captured bwd?
  3. Is graphed numerically == eager? (deterministic SR via step_counter)
Tiny + fast -> survives the flaky box. Run: python tools/probe_graph.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB

dev = "cuda"
torch.manual_seed(0)

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

def step_eager(m, x, y):
    for layer in m:
        if hasattr(layer, "rebalance"): pass
    out = m(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    for layer in m:
        if hasattr(layer, "rebalance"): layer.rebalance()
    return loss.item()

# fixed data
torch.manual_seed(1)
X = torch.randn(256, 64, device=dev)
Y = torch.randn(256, 64, device=dev) * 0.3

# ---- EAGER reference ----
m1 = build()
eager_losses = []
for i in range(40):
    x = X.detach().requires_grad_(True)
    eager_losses.append(step_eager(m1, x, Y))
print(f"[eager]   loss {eager_losses[0]:.5f} -> {eager_losses[-1]:.5f}")

# ---- GRAPHED via make_graphed_callables ----
print("\n[graphed] attempting make_graphed_callables on param-less Concord module...")
m2 = build()
try:
    sample = torch.randn(256, 64, device=dev, requires_grad=True)
    # make_graphed_callables graphs fwd+bwd; the fused apply kernel (packed_w side
    # effect) lives in the autograd Function backward -> should be captured + replay.
    graphed = torch.cuda.make_graphed_callables(m2, (sample,))
    print("[graphed] make_graphed_callables RETURNED ok")
    g_losses = []
    for i in range(40):
        x = X.detach().requires_grad_(True)
        out = graphed(x)
        loss = F.mse_loss(out, Y)
        loss.backward()
        for layer in m2:
            if hasattr(layer, "rebalance"): layer.rebalance()
        g_losses.append(loss.item())
    print(f"[graphed] loss {g_losses[0]:.5f} -> {g_losses[-1]:.5f}")
    # correctness: does graphed track eager?
    import statistics
    md = max(abs(a-b) for a,b in zip(eager_losses, g_losses))
    print(f"[compare] max |eager-graphed| over 40 steps = {md:.6f}")
    print("[VERDICT] " + ("MATCH (graphs the fused step correctly)" if md < 1e-3
                          else "DIVERGE (side-effect step not replaying right)"))
except Exception as e:
    import traceback
    print("[graphed] FAILED:", type(e).__name__)
    tb = traceback.format_exc()
    # print last 12 lines (the real cause)
    print("\n".join(tb.strip().split("\n")[-12:]))
