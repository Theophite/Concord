"""Definitive, confound-free test of the one thing the winner's fluctuation noise
relies on under capture: does torch.randn on the DEFAULT generator advance per
CUDA-graph replay (fresh draw each time), or is it frozen at the captured draw?

If it advances, the winner's backward noise (prototype_packed_b L1424, default
generator) is fresh every replay -> no divergence from eager. Testing this through
the trainer is confounded (the Concord step self-modifies the weights, so the loss
changes across replays even if the noise were frozen). So test the RNG in isolation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import torch

dev = torch.device("cuda")


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


# --- 1. raw torch.randn under replay (the exact primitive the noise uses) ---
x = torch.zeros(8, device=dev)
g = capture(lambda: x.copy_(torch.randn(8, device=dev)))
g.replay(); a = x.clone()
g.replay(); b = x.clone()
g.replay(); c = x.clone()
draws = [a, b, c]
distinct = len({tuple(round(v, 5) for v in d.tolist()) for d in draws})
print("=== torch.randn (default generator) across 3 graph replays ===")
for i, d in enumerate(draws):
    print(f"  replay {i}: {[round(v, 4) for v in d.tolist()[:5]]} ...")
print(f"[RESULT] {distinct}/3 distinct -> "
      f"{'ADVANCES per replay (noise is fresh, no divergence)' if distinct == 3 else 'FROZEN (would diverge!)'}")

# --- 2. randn_like (L1424 isotropic path) + a reduction, same question ---
y = torch.zeros((), device=dev)
buf = torch.randn(256, 256, device=dev)
g2 = capture(lambda: y.copy_(torch.randn_like(buf).norm()))
norms = []
for _ in range(3):
    g2.replay(); norms.append(round(y.item(), 4))
print("\n=== torch.randn_like(...).norm() across 3 replays (the L1424 shape) ===")
print(f"  norms: {norms}")
print(f"[RESULT] {'ADVANCES' if len(set(norms)) == 3 else 'FROZEN'} "
      f"(distinct norms => fresh draw each replay)")
