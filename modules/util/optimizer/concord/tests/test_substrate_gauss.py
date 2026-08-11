"""CPU statistical validation of the seed-hash Gaussian substrate (substrate_gauss.py).

Run CPU-only (safe while a live run holds the GPU):
    CUDA_VISIBLE_DEVICES="" python tests/test_substrate_gauss.py

Validates: determinism, the four moments (mean/std/skew/kurtosis), normality (KS vs the
standard-normal CDF), independence across positions and across seeds, finiteness + the
bounded tail (log(0) guard), Xavier/Kaiming std scaling, and that make_substrate ==
per-element gauss_unit*std (the pos convention the kernel relies on).
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import substrate_gauss as sg  # noqa: E402


def _mix32_numpy(seed, pos, salt):
    """Ground-truth murmur fmix32 in native numpy uint32 (true wrap + logical >>).
    What the Triton uint32 path computes; the torch int64-masked mirror must match."""
    s = np.uint32(seed)
    p = pos.astype(np.uint32)
    sa = np.uint32(salt)
    with np.errstate(over="ignore"):
        h = (s * np.uint32(sg._C_SEED)) ^ (p * np.uint32(sg._C_POS)) ^ sa
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(sg._F1)
        h = h ^ (h >> np.uint32(13))
        h = h * np.uint32(sg._F2)
        h = h ^ (h >> np.uint32(16))
    return h.astype(np.uint64)

torch.manual_seed(0)
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name:24s} {detail}")


def moments(x):
    x = x.double().flatten()
    m = x.mean()
    c = x - m
    var = (c * c).mean()
    sd = var.sqrt()
    skew = (c.pow(3)).mean() / sd.pow(3)
    kurt = (c.pow(4)).mean() / var.pow(2)          # raw kurtosis; normal == 3
    return m.item(), sd.item(), skew.item(), kurt.item()


def corr(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean().item() / (a.std(unbiased=False).item() * b.std(unbiased=False).item())


def ks_vs_normal(x):
    s, _ = torch.sort(x.double().flatten())
    n = s.numel()
    ecdf = torch.arange(1, n + 1, dtype=torch.float64) / n
    cdf = 0.5 * (1.0 + torch.erf(s / math.sqrt(2.0)))
    return torch.max(torch.abs(ecdf - cdf)).item()


# 0. Hash matches native numpy uint32 (proves the torch int64-masked mirror == true
#    unsigned semantics == what the Triton uint32 path runs).
_np_pos = np.arange(50_000, dtype=np.int64)
_t_pos = torch.from_numpy(_np_pos)
_ok_hash = True
for _salt in (sg.SUBSTRATE_SALT_A, sg.SUBSTRATE_SALT_B):
    _t = sg._mix32(31337, _t_pos, _salt).numpy().astype(np.uint64)
    _n = _mix32_numpy(31337, _np_pos, _salt)
    _ok_hash = _ok_hash and bool((_t == _n).all())
check("hash==numpy uint32", _ok_hash)

# 1. Determinism — same (seed, pos) -> identical, every call.
pos = torch.arange(10_000, dtype=torch.int64)
check("determinism", torch.equal(sg.gauss_unit(12345, pos), sg.gauss_unit(12345, pos)))

# 2. Moments of N(0,1) on a large sample.
z = sg.gauss_unit(777, torch.arange(2_000_000, dtype=torch.int64))
m, sd, sk, ku = moments(z)
check("mean ~ 0", abs(m) < 0.01, f"mean={m:+.4f}")
check("std ~ 1", abs(sd - 1.0) < 0.01, f"std={sd:.4f}")
check("skew ~ 0", abs(sk) < 0.03, f"skew={sk:+.4f}")
check("kurtosis ~ 3", abs(ku - 3.0) < 0.05, f"kurt={ku:.4f}")

# 3. Normality — KS distance to the standard-normal CDF below the ~alpha=0.01 critical band.
sub = z[:100_000]
ks = ks_vs_normal(sub)
crit = 2.0 / math.sqrt(sub.numel())
check("KS normal", ks < crit, f"KS={ks:.5f} < crit={crit:.5f}")

# 4. Independence — adjacent positions, and same positions under different seeds.
big = torch.arange(400_000, dtype=torch.int64)
c_adj = corr(sg.gauss_unit(1, big[0::2]), sg.gauss_unit(1, big[1::2]))
check("indep adjacent pos", abs(c_adj) < 0.01, f"corr={c_adj:+.4f}")
c_seed = corr(sg.gauss_unit(1, big), sg.gauss_unit(2, big))
check("indep across seed", abs(c_seed) < 0.01, f"corr={c_seed:+.4f}")

# 5. Finite + bounded tail (the log(0) guard => no inf; |z| <= ~5.77 at u1 = 2**-24).
check("all finite", torch.isfinite(z).all().item())
check("tail bounded", z.abs().max().item() < 6.0, f"max|z|={z.abs().max().item():.3f}")

# 6. Std scaling — Xavier and Kaiming targets on a representative SDXL-ish shape.
N, K = 320, 1280
subx = sg.make_substrate(N, K, seed=42, mode=sg.SUBSTRATE_XAVIER)
tx = sg.xavier_std(N, K)
ex = subx.double().std(unbiased=False).item()
check("xavier std", abs(ex - tx) / tx < 0.03, f"emp={ex:.5f} target={tx:.5f}")
subk = sg.make_substrate(N, K, seed=42, mode=sg.SUBSTRATE_KAIMING)
tk = sg.kaiming_std(K)
ek = subk.double().std(unbiased=False).item()
check("kaiming std", abs(ek - tk) / tk < 0.03, f"emp={ek:.5f} target={tk:.5f}")

# 7. make_substrate == per-element gauss_unit * std  (the pos = i*K+j convention).
ref = sg.gauss_unit(42, torch.arange(N * K, dtype=torch.int64)).reshape(N, K) * sg.xavier_std(N, K)
check("materialize==perelem", torch.equal(subx, ref))

print()
n_fail = sum(1 for ok in _results if not ok)
print(f"{'ALL PASS' if n_fail == 0 else str(n_fail) + ' FAILED'}  ({len(_results)} checks)")
sys.exit(1 if n_fail else 0)
