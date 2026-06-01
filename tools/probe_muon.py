"""Muon-chase prototype probe (standalone, seconds, survives flaky box).

Muon = NS5-orthogonalize the momentum, then step. Concord's s_fast IS the momentum
buffer, and the chase (alpha*s_fast, SR-rounded to int8, into s_slow) is where it would
hook. Two questions BEFORE any kernel surgery:

  Q1 (correctness): does a Newton-Schulz5 iteration actually orthogonalize (singular
     values -> ~1) a representative s_fast matrix? Implement + verify vs torch SVD.

  Q2 (THE novel question): orthogonality is a property of the FULL update, but the chase
     only moves alpha=10% of s_fast and SR-ROUNDS it to int8 (128-mantissa quanta). Does
     the orthogonal structure SURVIVE that quantized partial transfer into s_slow, or does
     int8 SR rounding destroy the singular-value flatness NS5 imposed? Measure the singular
     spectrum of what actually accumulates in s_slow over many chases, orthogonalized vs not.

No kernel edits. Pure torch. If Q2 says "survives", the kernel hook is worth building;
if "destroyed by SR", Muon-chase is incompatible with the int cascade and we learned that
for free.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torch

dev = "cuda"
torch.manual_seed(0)


def ns5(G, steps=5):
    """Newton-Schulz5 orthogonalization (Muon's quintic, bf16). Returns U V^T ~ for G=U S V^T
    -> singular values driven to ~1. Coeffs from the Muon reference (a,b,c)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)          # spectral-norm-ish normalize so SVs enter [0,1]
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.float()


def sv_stats(M):
    s = torch.linalg.svdvals(M.float())
    s = s / (s.max() + 1e-12)          # normalize to top SV
    return s.mean().item(), s.min().item(), (s > 0.1).float().mean().item()


print("=== Q1: does NS5 orthogonalize? (SV ratio min/mean -> ~1 means flat spectrum) ===")
for (n, k) in [(128, 64), (384, 384), (1536, 384)]:
    G = torch.randn(n, k, device=dev)
    s_mean0, s_min0, frac0 = sv_stats(G)
    O = ns5(G)
    s_mean1, s_min1, frac1 = sv_stats(O)
    print(f"  [{n}x{k}] raw: mean_sv/max={s_mean0:.3f} min/max={s_min0:.3f} frac>0.1={frac0:.2f}"
          f"  | NS5: mean/max={s_mean1:.3f} min/max={s_min1:.3f} frac>0.1={frac1:.2f}")
print("  (NS5 working => mean/max and min/max both jump toward ~1: flat = orthogonal)")

print("\n=== Q2: does orthogonality SURVIVE the SR int8 chase into s_slow? ===")
# Model the cascade on ONE matrix: repeatedly (a) get a 'gradient' tick into s_fast,
# (b) optionally NS5 s_fast, (c) chase alpha*s_fast SR-rounded to int8*128 into s_slow.
# Then look at s_slow's singular spectrum: orthogonalized-chase vs plain-chase.
N, K = 384, 384
alpha = 0.1
SCALE = 128.0  # int8 mantissa quantum (s_slow stored as int8, *128 in m_eff units)

def run_cascade(orthogonalize, steps=200):
    torch.manual_seed(1)
    s_fast = torch.zeros(N, K, device=dev)
    s_slow_i8 = torch.zeros(N, K, device=dev)   # integer accumulator (we keep float, round on tick)
    g = torch.Generator(device=dev); g.manual_seed(7)
    for t in range(steps):
        # a structured 'gradient' (low-rank signal + noise), into s_fast (mantissa units)
        sig = (torch.randn(N, 8, device=dev, generator=g) @ torch.randn(8, K, device=dev, generator=g)) * 30
        noise = torch.randn(N, K, device=dev, generator=g) * 30
        s_fast = s_fast + sig + noise
        src = ns5(s_fast) * s_fast.norm() / (N**0.5) if orthogonalize else s_fast
        # chase: alpha*src, SR-round to int8 units (quantum=128 mantissa)
        chase = alpha * src / SCALE
        floor = torch.floor(chase)
        frac = chase - floor
        tick = floor + (torch.rand(N, K, device=dev, generator=g) < frac).float()
        s_slow_i8 = s_slow_i8 + tick
        s_fast = s_fast - tick * SCALE
    return s_slow_i8 * SCALE   # s_slow in mantissa units

for orth in (False, True):
    M = run_cascade(orth)
    mean, mn, frac = sv_stats(M)
    print(f"  orthogonalize_chase={orth!s:5}: s_slow spectrum  mean_sv/max={mean:.3f} "
          f"min/max={mn:.4f} frac>0.1={frac:.2f}")
print("  (if orth=True keeps a FLATTER s_slow spectrum than False, orthogonality SURVIVES")
print("   the SR int8 cascade -> kernel hook worth building. If they're equal, SR destroyed it.)")
