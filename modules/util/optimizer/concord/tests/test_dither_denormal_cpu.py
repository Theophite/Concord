"""CPU unit tests for the DENORMAL / OUT-OF-RANGE-small architect key point of the
dither-accum redesign (pure-torch, CPU, no Triton, no pytest).

Reference under test:  modules/util/optimizer/concord/dither_accum_ref.py
Design doc:            modules/util/optimizer/concord/DITHER_ACCUM_DESIGN.md (sec 5, 6)

SCOPE -- this file tests ONLY the architect's denormal claim, nothing else:

  A weight FAR below its row max-abs is sub-LSB relative to the SHARED block-float
  row exponent (scale_ij = 2^(row_exp_i + col_exp_j - bias), set from the row max;
  load_weights:2666-2669 / dither_accum_ref.load_weights). The deploy x128
  quantization (deploy LSB = 128*scale) would floor it to 0. The redesign keeps
  such an "out-of-range-small" weight ALIVE via the SEPARATE fractional denormal
  channel den_frac (the fast-accumulator denormal extension), and -- crucially --
  borrowing that den_frac/int8 bit budget does NOT corrupt the dither-carry (err_s)
  or the velocity (the bounded int16 s_fast) role, because den_frac is its OWN
  fp companion buffer, NOT a bit-split of the e_v byte (DITHER_ACCUM_DESIGN.md sec 5,
  dither_accum_ref.py:54-64, 148-152, 566-577).

We assert, on a row whose weights span a WIDE range so the small ones are sub-LSB:
  1. those out-of-range-small weights are REPRESENTED (is_denormal True; live weight
     reconstructs the true sub-LSB value, NOT floored to 0), while the row's normal
     coords are NOT flagged denormal;
  2. the value round-trips through encode/decode -- live exact, deploy non-zero in
     expectation over the dither (DC-bias nulled), deterministic export reproducible;
  3. CONTRAST: with the denormal extension OFF (dither_enabled=False / den_frac_bits=0)
     the SAME sub-LSB weights ARE floored to 0 -- isolating den_frac as the mechanism;
  4. borrowing the den_frac bit budget does NOT corrupt the dither-carry/velocity role:
     - a NORMAL coord's sub-LSB residual lands in err_s (the dither carry), NOT den_frac;
     - at a denormal coord nothing spills into s_fast (velocity) or err_s (dither carry);
     - den_frac is orthogonal to s_fast/err_s in the packed word (two states differing
       only in a denormal value have byte-identical velocity + dither carry elsewhere);
     - under training with denormals present, err_s and den_frac each stay BOUNDED (<1),
       |s_fast| stays in int16 range, and the live state is recoverable from the packed
       word ALONE (sidecars dropped) to <= 1.5 mantissa units.

Run:  CUDA_VISIBLE_DEVICES="" venv/Scripts/python.exe \
        modules/util/optimizer/concord/tests/test_dither_denormal_cpu.py
"""
import sys
from pathlib import Path

import torch

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import dither_accum_ref as ref
from dither_accum_ref import (
    DitherAccumRef,
    is_denormal,
    unpack_word,
    _scale_fwd,
    assert_carries_bounded,
    assert_recoverable_without_sidecar,
)

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ------------------------------------------------------------------ #
# A WIDE-RANGE row: max-abs 4.0 sets a HIGH shared row exponent, so several small
# coords fall below one mantissa unit (sub-LSB) relative to that exponent. With
# row_exp = ceil(log2(4)+1) = 3 and bias 15, scale = 2^(3-15) = 2.44e-4, so the
# deploy LSB is 128*scale ~= 0.03125 -- everything below ~scale is sub-LSB.
# ------------------------------------------------------------------ #
torch.manual_seed(0)
N, K = 4, 8
W = torch.zeros(N, K)
W[0, 0] = 4.0          # row max  -> sets row_exp high                     NORMAL
W[0, 1] = 2.0          #                                                   NORMAL
W[0, 2] = 1e-3         # mantissa ~= 4.1  (>= 1)                           NORMAL
W[0, 3] = 5e-5         # mantissa ~= 0.20 (< 1)  out-of-range-small -> DENORMAL
W[0, 4] = 3e-6         # mantissa ~= 0.012        out-of-range-small -> DENORMAL
W[0, 5] = -8e-5        # mantissa ~= -0.33        out-of-range-small -> DENORMAL (signed)
W[0, 6] = 1.0          #                                                   NORMAL
W[1:] = torch.randn(N - 1, K) * 0.1                                       # other rows normal

DENORMAL = [(0, 3), (0, 4), (0, 5)]
NORMAL_IN_ROW0 = [(0, 0), (0, 1), (0, 2), (0, 6)]

print("== wide-range row: out-of-range-small weights are REPRESENTED (not floored) ==")
r = DitherAccumRef(N, K, dither_enabled=True, seed=1)
r.load_weights(W)
scale0 = float(_scale_fwd(r.row_exp, r.col_exp)[0, 0])
mant = (W / _scale_fwd(r.row_exp, r.col_exp))
dn = is_denormal(r.packed_w)
check("row span sets a high shared exponent so the small coords ARE sub-LSB",
      all(abs(float(mant[c])) < 1.0 for c in DENORMAL)
      and all(abs(float(mant[c])) >= 1.0 for c in NORMAL_IN_ROW0),
      f"scale={scale0:.3e}, mant_denorm={[round(float(mant[c]),3) for c in DENORMAL]}")
check("the SINGLE is_denormal predicate flags exactly the out-of-range-small coords",
      all(bool(dn[c]) for c in DENORMAL) and not any(bool(dn[c]) for c in NORMAL_IN_ROW0),
      f"flagged={int(dn.sum())}, denorm_flags={[bool(dn[c]) for c in DENORMAL]}")

# the integer mantissa of a denormal coord is EXACTLY 0 (value lives only in den_frac)
sf, ss, vs = unpack_word(r.packed_w)
int_mant = ss * 128 + sf + vs * 128
check("denormal coords have integer mantissa == 0 (value is ONLY in den_frac)",
      all(int(int_mant[c]) == 0 for c in DENORMAL)
      and all(abs(float(r.den_frac[c])) > 0 for c in DENORMAL),
      f"int_mant={[int(int_mant[c]) for c in DENORMAL]}, "
      f"den_frac={[round(float(r.den_frac[c]),4) for c in DENORMAL]}")

live = r.live_weight()
max_rel = max((abs(float(live[c]) - float(W[c])) / (abs(float(W[c])) + 1e-30)) for c in DENORMAL)
check("LIVE weight reconstructs the true sub-LSB value (NOT floored to 0)",
      max_rel < 1e-3,
      "  ".join(f"[{i},{j}]={float(live[i,j]):.3e}(true {float(W[i,j]):.3e})" for (i, j) in DENORMAL))

# ------------------------------------------------------------------ #
print("== round-trip through encode/decode: deploy survives in expectation ==")
# The deploy render is an SR (Bernoulli) of the sub-LSB den_frac fraction p<<1, so the
# Monte-Carlo estimate of E|deploy| has relative standard error ~ sqrt((1-p)/(p*T)).
# The TINIEST denormal here has p ~= 0.012, so a few hundred salts is far too noisy to
# resolve the mean to 10% (that is an ESTIMATOR-variance artifact, not a DC bias -- the
# render is verified unbiased to <1% over 4096+ salts). Use enough salts that the
# estimator converges below the tolerance; this measures the SAME "non-zero, DC-bias-
# nulled, tracks-true" property, just with a statistically adequate sample.
T = 16384
for (i, j) in DENORMAL:
    em = sum(abs(float(r.deploy_weight(step_salt=t)[i, j])) for t in range(T)) / T
    check(f"deploy E|W[{i},{j}]| > 0 and tracks true value over {T} dithers (DC-bias nulled)",
          em > 0 and abs(em - abs(float(W[i, j]))) / (abs(float(W[i, j])) + 1e-30) < 0.10,
          f"E|deploy|={em:.3e}  true={abs(float(W[i,j])):.3e}")

da = r.deploy_weight(deterministic=True)
db = r.deploy_weight(deterministic=True)
check("deterministic export of the denormal render is reproducible (shipping checkpoint)",
      torch.equal(da, db))

# ------------------------------------------------------------------ #
print("== CONTRAST: denormal extension OFF (den_frac_bits=0) floors the small weights ==")
rd = DitherAccumRef(N, K, dither_enabled=False, seed=1)
rd.load_weights(W)
livd = rd.live_weight()
depd = rd.deploy_weight()
check("disabled mode holds NO den_frac (the denormal channel is the only difference)",
      float(rd.den_frac.abs().max()) == 0.0,
      f"max|den_frac|={float(rd.den_frac.abs().max())}")
check("WITHOUT the denormal extension the out-of-range-small weights ARE floored to 0",
      all(float(livd[c]) == 0.0 for c in DENORMAL) and all(float(depd[c]) == 0.0 for c in DENORMAL),
      f"disabled live={[float(livd[c]) for c in DENORMAL]}")
# normal coords are unaffected by the extension (their integer mantissa survives either way)
check("normal coords in the row are NOT floored even with the extension off",
      all(abs(float(livd[c])) > 0 for c in NORMAL_IN_ROW0))

# ------------------------------------------------------------------ #
print("== budget separation: den_frac does NOT corrupt the dither-carry / velocity ==")
# (1) A NORMAL coord's sub-LSB FINE residual lands in err_s (the sigma-delta dither
#     carry), NOT in den_frac. W[0,2] has mantissa ~4.1: integer 4 -> s_fast, the
#     ~0.1 fractional residual -> err_s. den_frac stays 0 there.
check("normal coord's fine residual is in err_s (dither carry), den_frac stays 0 there",
      abs(float(r.err_s[0, 2])) > 0 and float(r.den_frac[0, 2]) == 0.0,
      f"err_s[0,2]={float(r.err_s[0,2]):.4f}  den_frac[0,2]={float(r.den_frac[0,2]):.4f}")

# (2) At a denormal coord NOTHING spills into the velocity (s_fast) or dither carry (err_s);
#     the value is held purely by the SEPARATE den_frac channel.
check("denormal coords leave the velocity register s_fast == 0 (no spill into velocity)",
      all(int(sf[c]) == 0 for c in DENORMAL),
      f"s_fast={[int(sf[c]) for c in DENORMAL]}")
check("denormal coords leave the dither carry err_s == 0 (no spill into the carry)",
      all(float(r.err_s[c]) == 0.0 for c in DENORMAL),
      f"err_s={[float(r.err_s[c]) for c in DENORMAL]}")

# (3) Orthogonality in the packed word: two states that differ ONLY in a denormal
#     value have BYTE-IDENTICAL velocity (s_fast) and dither carry (err_s) at every
#     OTHER coord -- proving den_frac is its own buffer, not stolen e_v velocity bits.
torch.manual_seed(0)
base = torch.randn(N, K) * 0.1
base[0, 0] = 4.0
WA = base.clone(); WA[0, 1] = 3e-6      # [0,1] denormal, den_frac != 0
WB = base.clone(); WB[0, 1] = 0.0       # [0,1] also denormal but den_frac == 0
rA = DitherAccumRef(N, K, dither_enabled=True, seed=5); rA.load_weights(WA)
rB = DitherAccumRef(N, K, dither_enabled=True, seed=5); rB.load_weights(WB)
sfA, _, _ = unpack_word(rA.packed_w)
sfB, _, _ = unpack_word(rB.packed_w)
elsewhere = torch.ones(N, K, dtype=torch.bool); elsewhere[0, 1] = False
check("changing ONLY a denormal value leaves s_fast (velocity) identical everywhere else",
      bool(torch.equal(sfA[elsewhere], sfB[elsewhere])))
check("changing ONLY a denormal value leaves err_s (dither carry) identical everywhere else",
      bool(torch.equal(rA.err_s[elsewhere], rB.err_s[elsewhere])))
check("the den_frac channel IS where the differing denormal value lives",
      abs(float(rA.den_frac[0, 1])) > 0 and float(rB.den_frac[0, 1]) == 0.0,
      f"den_frac: A={float(rA.den_frac[0,1]):.4f}  B={float(rB.den_frac[0,1]):.4f}")

# (4) Under training WITH denormals present, the dither carry + velocity stay healthy:
#     err_s and den_frac each BOUNDED (<1), |s_fast| in int16 range, and the live
#     state is recoverable from the packed word ALONE (sidecars dropped) to <=1.5
#     mantissa units -- i.e. the denormal budget did not become a shadow velocity.
g = torch.randn(N, K) * 0.05
carries_ok = True
max_err_s = max_den = 0.0
max_sfast = 0
for t in range(200):
    r.step(g, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
           coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
    okc, me, md = assert_carries_bounded(r.err_s, r.den_frac)
    carries_ok = carries_ok and okc
    max_err_s = max(max_err_s, me); max_den = max(max_den, md)
    max_sfast = max(max_sfast, int(unpack_word(r.packed_w)[0].abs().max()))
check("over 200 steps with denormals present, err_s (dither carry) stays bounded < 1",
      carries_ok and max_err_s < 1.0 + 1e-4, f"max|err_s|={max_err_s:.3f}")
check("over 200 steps the denormal channel den_frac stays bounded < 1 (no shadow velocity)",
      max_den < 1.0 + 1e-4, f"max|den_frac|={max_den:.3f}")
check("velocity register |s_fast| stays in int16 range throughout",
      max_sfast <= ref.INT16_MAX, f"max|s_fast|={max_sfast}")
okr, gap = assert_recoverable_without_sidecar(r.packed_w, r.err_s, r.den_frac, r.row_exp, r.col_exp)
check("live state recoverable from the packed word ALONE (sidecars dropped) to <=1.5 units",
      okr and gap <= 1.5, f"mantissa gap={gap:.3f}")


# ------------------------------------------------------------------ #
n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} dither-denormal CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
