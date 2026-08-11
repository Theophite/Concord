"""CPU unit tests for the ERROR-FEEDBACK / NO-DC-BIAS property of the dither-accum
redesign reference (no GPU, no Triton). Scope: ONLY error feedback and the absence
of a systematic (DC) bias in the stochastic-rounding deploy render.

Reference under test:
  modules/util/optimizer/concord/dither_accum_ref.py  (pure-torch, CPU)
Design doc:
  modules/util/optimizer/concord/DITHER_ACCUM_DESIGN.md

WHAT "ERROR FEEDBACK / NO DC BIAS" MEANS HERE
---------------------------------------------
The deploy render quantizes to the coarse (128-mantissa) grid with a SEPARATE
sub-LSB denormal channel rendered by stochastic rounding (SR). The contract is
that constant, ramp, and random sub-LSB updates are NOT silently dropped: the
decoded DEPLOY weight tracks the TRUE accumulated value within dither noise, and
the MEAN error -> 0 over N draws/steps (no systematic drift). This rests on:

  1. SR PRIMITIVE is DC-unbiased: E[_sr_round(x)] = x for any fractional x
     (constant / ramp / random), the floor+Bernoulli identity
     (dither_accum_ref.py:321-326). This is the root of every no-DC-bias claim.

  2. DENORMAL CHANNEL: a sub-LSB weight (|mantissa| < 1) lives in den_frac
     (dither_accum_ref.py:431-442). The LIVE weight reconstructs it exactly, and
     the deploy mean over SR salts tracks the TRUE value within dither noise
     (E[deploy] = coarse + den_frac = true value), mean mantissa error -> 0, for
     constant / ramp / random sub-LSB values. (mirror of self-test 2,
     dither_accum_ref.py:760-775, and design doc s5.)

  3. DYNAMIC SUB-LSB INFLOW: sustained sub-LSB updates sigma-delta accumulate via
     err_s (the bounded one-LSB remainder, dither_accum_ref.py:531-534) and
     RATCHET the deploy register in proportion to their DC (net) component, while
     the resident un-chased fine velocity |s_fast| stays bounded. On an EXACTLY
     net-zero stream the true accumulated weight change is zero, so the decoded
     deploy must not acquire a systematic offset: the SIGNED grand-mean deploy
     error must -> 0 with a vanishing PER-STEP RATE (no linear integration).
     (mirror of self-test 4, dither_accum_ref.py:815-826.)

     *** This sub-section is where the reference FAILS (see the run summary). ***
     With the den_frac channel ENABLED, the net-zero deploy DC error integrates
     LINEARLY (~ -2e-3 mantissa/coord/step; -16 @8k -> -32 @16k), whereas with
     den_frac DISABLED (legacy) it stays bounded/oscillating (~0 +/- 8). The gap
     is the den_frac whole-unit promotion (dither_accum_ref.py:571-577):
     `whole = trunc(den_frac); v_slow += whole; den_frac -= whole` pushes coarse
     LSBs into v_slow WITHOUT the mass-preserving `s_slow -= ...` counter-move the
     leak uses at line 563-564, so on a net-zero stream the consistently-signed
     leak residual fed into den_frac accumulates one-directionally into the deploy
     word. This is a real, small, but linearly-growing DC bias, NOT pre-existing
     legacy behavior. Reported as a design gap; the threshold is NOT relaxed.

Monte-Carlo note: an SR mean over M independent salts has per-coord standard
error <= 0.5/sqrt(M) mantissa units (Bernoulli variance <= 1/4). With M=8192 that
is ~0.0055; the 0.05-mantissa thresholds below are ~9 sigma of headroom, so a
PASS reflects real DC-unbiasedness rather than a loose bound.

Run with the OneTrainer venv python (CPU only):
  CUDA_VISIBLE_DEVICES="" venv/Scripts/python.exe \
      modules/util/optimizer/concord/tests/test_dither_tracking_cpu.py
"""
import sys
from pathlib import Path

import torch

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

from dither_accum_ref import (  # noqa: E402
    DitherAccumRef,
    _scale_fwd,
    _sr_round,
    _pos_hash,
    is_denormal,
    unpack_word,
)

torch.manual_seed(0)

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ============================================================================
# (1) SR PRIMITIVE is DC-unbiased: E[_sr_round(x)] = x, for constant/ramp/random
#     fractional inputs. This is the floor+Bernoulli identity the whole
#     error-feedback story rests on (dither_accum_ref.py:321-326).
# ============================================================================
print("== (1) SR primitive DC-unbiasedness  E[_sr_round(x)] = x ==")

def sr_mean(x, M):
    """Mean of M independent SR rounds of x, salt-swept like the deploy step_salt."""
    N, K = x.shape
    pos = _pos_hash(N, K)
    hx = torch.zeros(N, K, dtype=torch.int32)   # fixed hash arg; salt provides entropy
    acc = torch.zeros(N, K, dtype=torch.float32)
    for t in range(M):
        salt = (t * 0x9E3779B1) & 0x7FFFFFFF
        acc += _sr_round(x, hx, pos, salt).to(torch.float32)
    return acc / M

N, K = 8, 8
M = 8192
# standard error of the SR mean per coord <= 0.5/sqrt(M); a 0.05 threshold is ~9 sigma.
SR_TOL = 0.05

_x_const = torch.full((N, K), 0.37)                         # constant fraction
_x_ramp = torch.linspace(-2.4, 2.4, N * K).reshape(N, K)    # ramp across the grid
torch.manual_seed(4)
_x_rand = torch.rand(N, K) * 8.0 - 4.0                       # random in [-4, 4)

for name, x in [("constant", _x_const), ("ramp", _x_ramp), ("random", _x_rand)]:
    err = sr_mean(x, M) - x
    mean_err = float(err.mean())
    max_err = float(err.abs().max())
    check(f"SR mean tracks x ({name}): max|coord err| < {SR_TOL}", max_err < SR_TOL,
          f"max|coord|={max_err:.5f} mantissa")
    check(f"SR has NO net DC bias ({name}): |grand mean err| < {SR_TOL / 5}",
          abs(mean_err) < SR_TOL / 5, f"mean={mean_err:+.6f} mantissa")
    # integer round-trip sanity: SR of an exact integer is that integer (no dither)
    xi = torch.round(x)
    check(f"SR of integer is exact ({name}) -> no spurious dither",
          torch.equal(sr_mean(xi, 16), xi), "")


# ============================================================================
# (2) DENORMAL CHANNEL: sub-LSB constant/ramp/random VALUES -> live reconstructs
#     exactly, decoded DEPLOY mean tracks the true value within dither noise,
#     MEAN mantissa error -> 0. (mirror of self-test 2, dither_accum_ref.py:760.)
# ============================================================================
print("== (2) denormal channel: decoded deploy tracks the true sub-LSB value ==")

def build_denormal(kind, n=8, k=8, seed=2):
    """A weight matrix with a large row max (-> high row_exp) and a band of
    genuinely sub-LSB (|mantissa| < 1) coords carrying constant / ramp / random
    values. Returns (W, coords)."""
    torch.manual_seed(seed)
    W = torch.zeros(n, k)
    W[0, 0] = 2.0                       # row-0 max -> row_exp high -> coords below are sub-LSB
    coords = [(0, j) for j in range(1, 7)]
    for idx, c in enumerate(coords):
        if kind == "constant":
            W[c] = 3e-6
        elif kind == "ramp":
            W[c] = 1e-6 * (idx + 1)
        else:  # random sub-LSB
            W[c] = float(torch.randn(1)) * 3e-6
    return W, coords

DEP_TOL = 0.05      # per-coord deploy-mean mantissa error threshold (>> MC noise at M)
M2 = 12288

for kind in ["constant", "ramp", "random"]:
    W, coords = build_denormal(kind)
    ref = DitherAccumRef(N, K, dither_enabled=True, seed=2)
    ref.load_weights(W)
    dn = is_denormal(ref.packed_w)
    scale = _scale_fwd(ref.row_exp, ref.col_exp)
    live = ref.live_weight()

    all_denorm = all(bool(dn[c]) for c in coords)
    check(f"sub-LSB coords detected denormal ({kind})", all_denorm,
          f"{sum(int(dn[c]) for c in coords)}/{len(coords)} flagged")

    # live reconstructs the true sub-LSB value exactly (it is held in den_frac)
    live_ok = all(abs(live[c].item() - W[c].item()) <= 1e-12 + 1e-6 * abs(W[c].item())
                  for c in coords)
    check(f"LIVE weight reconstructs true sub-LSB value ({kind})", live_ok,
          f"e.g. {coords[0]} live={live[coords[0]].item():.3e} true={W[coords[0]].item():.3e}")

    # decoded deploy MEAN over salts tracks the true value within dither noise
    dep_mean = torch.stack([ref.deploy_weight(step_salt=t) for t in range(M2)], 0).mean(0)
    coord_err = torch.tensor([((dep_mean[c] - W[c]) / scale[c]).item() for c in coords])
    max_err = float(coord_err.abs().max())
    mean_err = float(coord_err.mean())
    check(f"deploy mean tracks true sub-LSB value within dither ({kind}): "
          f"max|coord err| < {DEP_TOL}", max_err < DEP_TOL,
          f"max|coord|={max_err:.4f} mantissa")
    check(f"deploy has NO systematic (DC) error ({kind}): |mean err| < {DEP_TOL / 2}",
          abs(mean_err) < DEP_TOL / 2, f"mean={mean_err:+.5f} mantissa")
    # sign-preservation: each denormal coord's deploy mean has the true sign (or ~0)
    sign_ok = all((dep_mean[c].item() == 0.0) or
                  (dep_mean[c].item() * W[c].item() > 0) for c in coords)
    check(f"deploy mean preserves the sub-LSB sign ({kind})", sign_ok, "")


# ============================================================================
# (3) DYNAMIC SUB-LSB INFLOW: sustained constant/ramp/random sub-LSB updates
#     sigma-delta accumulate and RATCHET the deploy in proportion to their DC
#     component, with a BOUNDED resident fine gap (no growing DC drift). A
#     zero-mean random stream produces NO systematic deploy drift.
#     (mirror of self-test 4, dither_accum_ref.py:815-826.)
# ============================================================================
print("== (3) dynamic sub-LSB inflow: error-feedback ratchet, no DC drift ==")

def drive(grads, lr=5e-4, seed=8, dither_enabled=True):
    """Drive the accumulator with a pre-built [steps,n,k] sub-LSB inflow stream
    (large init -> tiny per-step mantissa motion). gf_consol=0 / drift_cancel_C=0
    isolates the inflow path. Returns a dict with the SIGNED grand-mean deploy
    error (the DC-bias estimator), the abs sum, the worst coord, and the resident
    fine-velocity high-water |s_fast|. All deploy quantities in mantissa units."""
    torch.manual_seed(seed)
    n, kk = grads.shape[1], grads.shape[2]
    ref = DitherAccumRef(n, kk, dither_enabled=dither_enabled, seed=seed)
    W = torch.full((n, kk), 1000.0)     # big -> sub-LSB per-step mantissa motion
    ref.load_weights(W)
    scale = _scale_fwd(ref.row_exp, ref.col_exp)
    dep0 = ref.deploy_weight(deterministic=True).clone()
    max_sfast = 0
    for t in range(grads.shape[0]):
        ref.step(grads[t], lr=lr, gf_consol=0.0, drift_cancel_C=0.0, chase_floor=0.1)
        sf, _, _ = unpack_word(ref.packed_w)
        max_sfast = max(max_sfast, int(sf.abs().max()))
    dm = (ref.deploy_weight(deterministic=True) - dep0) / scale
    return {
        "signed_mean": float(dm.sum()) / dm.numel(),   # DC-bias estimator (mantissa/coord)
        "abs_sum": float(dm.abs().sum()),
        "worst": float(dm.abs().max()),
        "max_sfast": max_sfast,
    }

# nonzero-DC streams (constant, ramp) -> deploy ratchets (error feedback: not dropped)
STEPS = 1500
n0, k0 = 4, 4
for kind in ["constant", "ramp"]:
    if kind == "constant":
        G = torch.ones(STEPS, n0, k0)
    else:
        G = (0.2 + 0.0005 * torch.arange(STEPS).float())[:, None, None].expand(STEPS, n0, k0)
    r = drive(G.clone())
    check(f"sustained sub-LSB inflow ratchets deploy ({kind}): moved > 0", r["abs_sum"] > 0,
          f"moved {r['abs_sum']:.2f} mantissa")
    # resident fine velocity stays bounded (< one coarse LSB region) -> no integrating drift
    check(f"resident fine gap bounded under inflow ({kind}): max|s_fast| < 256",
          r["max_sfast"] < 256, f"max|s_fast|={r['max_sfast']}")

# EXACTLY-zero-DC random stream: per-coord time mean subtracted so the FULL stream's
# net inflow == 0 per coord. The true accumulated weight change is then ZERO, so the
# decoded deploy must NOT acquire a systematic (DC) offset: the SIGNED grand-mean
# deploy error must -> 0 and, crucially, its PER-STEP RATE must -> 0 (no linear
# integration). We measure the signed DC at two horizons (8k and 16k steps) and
# require the per-step DC RATE not to persist as the horizon doubles.
def zeromean_stream(steps, seed):
    torch.manual_seed(seed)
    G = torch.randn(steps, n0, k0)
    return G - G.mean(dim=0, keepdim=True)     # per-coord net inflow exactly 0

S1, S2 = 8000, 16000
G1 = zeromean_stream(S1, seed=21)
G2 = zeromean_stream(S2, seed=21)
net1 = float(G1.sum(0).abs().max())
r1 = drive(G1, seed=8)
r2 = drive(G2, seed=8)
rate1 = abs(r1["signed_mean"]) / S1                # |DC| per coord per step
rate2 = abs(r2["signed_mean"]) / S2
# companion localizer: the SAME net-zero stream with the den_frac channel DISABLED
# (legacy mode) -- isolates whether any DC drift is from the den_frac promotion path.
r1_leg = drive(G1, seed=8, dither_enabled=False)
r2_leg = drive(G2, seed=8, dither_enabled=False)

check("zero-DC random stream is genuinely net-zero per coord", net1 < 1e-3,
      f"max|net inflow|={net1:.2e}")
# THE no-DC-bias assertion: the signed DC deploy error does NOT integrate linearly
# with step count (its per-step rate must vanish, ~ < 1e-4 mantissa/coord/step).
DC_RATE_TOL = 1e-4
check("zero-DC random inflow: signed deploy DC RATE -> 0 (no integrating DC bias)",
      rate2 < DC_RATE_TOL,
      f"signed DC: {r1['signed_mean']:+.2f}@{S1}, {r2['signed_mean']:+.2f}@{S2} "
      f"mantissa/coord -> rate {rate1:.1e}, {rate2:.1e} /step")
# the legacy (den_frac OFF) path on the identical stream, for localization
check("LOCALIZER: den_frac-OFF (legacy) deploy DC stays bounded on the same stream",
      abs(r2_leg["signed_mean"]) <= abs(r1_leg["signed_mean"]) + 8.0 + 1e-6,
      f"legacy signed DC: {r1_leg['signed_mean']:+.2f}@{S1}, "
      f"{r2_leg['signed_mean']:+.2f}@{S2} mantissa/coord (bounded, oscillating)")
check("zero-DC random inflow leaves resident fine gap bounded",
      r2["max_sfast"] < 256, f"max|s_fast|={r2['max_sfast']}")


n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} dither error-feedback / no-DC-bias CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
