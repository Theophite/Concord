"""CPU unit tests for the DENORMAL (sub-LSB) survival key point of the CORRECTED
dual-dissipation co-equal packed-B fine accumulator — REWORKED for the e_H
EXPONENT-CLAIM (ADD-2). Pure-torch, CPU, no Triton, no pytest.

Reference under test:  modules/util/optimizer/concord/dual_dissipation_ref.py

──────────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE TESTS  (the architect's REWORKED denormal claim, ADD-2)
──────────────────────────────────────────────────────────────────────────────────
A weight FAR below its row max-abs is sub-LSB relative to the SHARED block-float row
exponent (scale_ij = 2^(row_exp_i + col_exp_j - bias)). The deploy x128 quantization
(deploy LSB = 128*scale) would floor it to 0. The REWORKED design keeps such a coord
ALIVE via a per-element EXPONENT-CLAIM on the HIGH-dissipation arm e_H -- NOT a linear
low-bit fraction (the REMOVED pre-rework field), NOT an fp sidecar (WRONG #1), and NOT
e_H packed as the high byte of an int16 (WRONG #2).

THE TRICK (ADD-2): e_H is the high-dissipation arm -> statistically drained near zero
(by BOTH evap and consolidation) -> its high bits are free. For a DENORMAL element
(coarse word == 0) read e_H's magnitude as an IEEE-style block-float sub-scale value:
    oct  = (|e_H| >> MANT_BITS) & OCT_MAX     (downward octave, 0..3)
    mlow = |e_H| & MLOW_MASK                   (significand, leading-1 implied)
    units = sign(e_H) * (1 + mlow/16) * 2^-(oct+1)        (|units| < 1 mantissa unit)
reaching 2^-1 .. 2^-4 of one mantissa unit BELOW the shared block scale -- downward LOG
reach the int16 never had. A |e_H| guard (EH_VELO_CAP = 64) selects exponent-mode vs
magnitude-mode: a LARGE e_H is a real un-consolidated velocity -> DROP the claim, render
PURE COARSE (it bursts toward normal and graduates via the normal chase). Octave is
capped at oct in 0..3 so every exp-mode |e_H| <= 63 < 64 -> exp-mode and the guard are
PROVABLY DISJOINT (spec HOLE-1).

The corrected invariants asserted here (numbered to the GOAL spec, ADD-2 + point 8):

  1. is_denormal is COARSE-WORD-ONLY: is_denormal == (s_slow==0 AND v_slow==0). It does
     NOT read the fine bits, so reading the exponent claim out of e_H cannot self-nullify
     the predicate (the WRONG #2 BREAK A). Toggling e_H/e_L arbitrarily must NOT move it.

  2. A sub-LSB (denormal) coord SURVIVES to the DEPLOY weight via the e_H exponent claim:
     E|deploy| > 0 over many salts and tracks the true sub-scale value (DC-bias nulled),
     while exp-mode holds; dropping the claim (enabled=False) FLOORS it to 0.

  3. The |e_H| FALLBACK (magnitude mode, |e_H| >= EH_VELO_CAP): a denormal whose e_H
     carries a REAL un-consolidated velocity renders PURE COARSE (graceful skip) -- never
     a silent collision of the exponent tag with velocity. PROVABLY DISJOINT from every
     valid exp-mode encoding (which has |e_H| <= 63).

  4. NORMAL coords are UNAFFECTED by the exponent-claim render: their deploy is the pure
     coarse (s_slow+v_slow)*128 EXACTLY (gate is_denormal & is_exp_mode is False for them).

  5. The deploy render is DC-UNBIASED over many salts: the stochastic-rounded render
     averages to the exact sub-scale value (no systematic DC drift), and the deterministic
     render is reproducible (shipping checkpoint).

  6. CONSERVING graduation (denormal -> normal): when a denormal's running linear value
     crosses one mantissa unit, the whole units PROMOTE into e_L (a fine->fine move, booked
     as inflow) -- NEVER a +128 credit into s_slow (the REMOVED leak). The conservation
     ledger (point 8) holds EVERY step, including live denormals; the deploy render injects
     NO unbooked deploy mantissa.

  7. NO fp sidecars (point 9): the Layer carries no den_frac/err_s buffer; the live + deploy
     weight is recoverable from (packed word + exponents + substrate) alone.

Run:  CUDA_VISIBLE_DEVICES="" venv/Scripts/python.exe \
        modules/util/optimizer/concord/tests/test_dualdis_denormal.py
"""
import sys
from pathlib import Path

import torch

# CPU-only: never touch the GPU (the user may be training). Belt-and-suspenders.
torch.set_grad_enabled(False)

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import dual_dissipation_ref as ref
from dual_dissipation_ref import (
    DualDissipationLayer,
    is_denormal,
    is_exp_mode,
    decode_denormal_units,
    unpack_dual,
    pack_dual,
    deploy_mantissa,
    assert_conservation,
    decode_to_live_weight,
    decode_to_deploy_weight,
    _scale_fwd,
    EXP_BITS,
    MANT_BITS,
    OCT_MAX,
    MLOW_MASK,
    EH_VELO_CAP,
    INT8_MIN,
    INT8_MAX,
)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def make_eH(N, K, value_map):
    """Helper: build an e_H int32 tensor with `value_map` {(i,j): int8_value}."""
    t = torch.zeros(N, K, dtype=torch.int32)
    for (i, j), v in value_map.items():
        t[i, j] = v
    return t


N, K = 4, 8
# A fixed layer just to get well-formed row/col exponents for the render math. The
# accumulator is ZEROED on enabled load (offset 0), so denormal coords are constructed
# directly by packing e_H below (matching __main__ test 3); we do NOT rely on load to
# manufacture sub-LSB coords (the rework zeroes the offset, removing init-time denormals).
torch.manual_seed(0)
W = torch.randn(N, K) * 0.1
r = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=1)
r.load_weights(W)
ROW_EXP, COL_EXP = r.row_exp, r.col_exp
SCALE = _scale_fwd(ROW_EXP, COL_EXP)


# ================================================================================== #
print("== ADD-2 sanity: the exponent-claim decode reaches BELOW one mantissa unit ==")
# Build the three canonical denormal exp-mode coords by packing e_H with sub-unit
# octave encodings (oct<<MANT_BITS)|mlow, coarse word == 0.  Their decode_denormal_units
# must equal SGN*(1+mlow/16)*2^-(oct+1) and lie strictly in (0, 1).
# BUG-2: the (oct,mlow) index is stored OFFSET BY ONE (e_H code a = idx + 1) so a == 0 stays
# RESERVED for true-zero. So the canonical denormal coords pack ((oct<<MANT_BITS)|mlow) + 1.
DEN = {
    (0, 3): ((0 << MANT_BITS) | 0) + 1,    # oct0 mlow0  -> 0.5            (code a=1)
    (0, 4): ((3 << MANT_BITS) | 14) + 1,   # oct3 mlow14 -> 1.875*2^-4 = 0.1171875 (deepest, code a=63)
    (0, 5): -(((1 << MANT_BITS) | 8) + 1),  # oct1 mlow8, NEGATIVE -> -(1.5*2^-2) = -0.375 (code a=-25)
}
DENORMAL = list(DEN.keys())
NORMAL = [(1, 0), (2, 3), (3, 7)]   # we will set non-zero coarse on these

e_H0 = make_eH(N, K, DEN)
packed0 = pack_dual(e_H0,
                    torch.zeros(N, K, dtype=torch.int32),       # e_L = 0
                    torch.zeros(N, K, dtype=torch.int32),       # s_slow = 0
                    torch.zeros(N, K, dtype=torch.int32))       # v_slow = 0
units0 = decode_denormal_units(unpack_dual(packed0)[0])
want = {(0, 3): 0.5, (0, 4): 1.875 * 2 ** -4, (0, 5): -0.375}
check("exp-claim decode = SGN*(1+mlow/16)*2^-(oct+1) and |.| < 1 for the denormal coords",
      all(abs(float(units0[c]) - want[c]) < 1e-7 for c in DENORMAL)
      and all(0 < abs(float(units0[c])) < 1.0 for c in DENORMAL),
      f"units={[round(float(units0[c]), 6) for c in DENORMAL]}")
# deepest reach: the smallest-octave code (oct=OCT_MAX, mlow=0) = idx (OCT_MAX<<MANT_BITS),
# stored as code a = idx + 1, decodes to 2^-(OCT_MAX+1).
check("deepest reach is 2^-(OCT_MAX+1) of a mantissa unit (downward LOG range)",
      abs(float(decode_denormal_units(
          torch.tensor([[(OCT_MAX << MANT_BITS) + 1]], dtype=torch.int32))[0, 0])
          - 2.0 ** -(OCT_MAX + 1)) < 1e-7,
      f"OCT_MAX={OCT_MAX}, EXP_BITS={EXP_BITS}, MANT_BITS={MANT_BITS}, "
      f"deepest=2^-{OCT_MAX + 1}")


# ================================================================================== #
print("== CHECK 1: is_denormal is COARSE-ONLY -- the exponent claim cannot self-nullify it ==")
# The heart of the corrected design (WRONG #2 BREAK A): the predicate must read ONLY the
# coarse word. Take coarse (0,0) and write EVERY combination into the fine bytes -- the
# exponent claim lives in e_H -- the predicate must stay TRUE (it never reads the fine).
coarse_zero = pack_dual(torch.zeros(N, K, dtype=torch.int32),
                        torch.zeros(N, K, dtype=torch.int32),
                        torch.zeros(N, K, dtype=torch.int32),
                        torch.zeros(N, K, dtype=torch.int32))
check("coarse word (0,0) is denormal with empty fine bytes", bool(is_denormal(coarse_zero).all()))

inv_ok = True
for eh in (-128, -64, -63, -16, -1, 0, 1, 5, 16, 47, 63, 64, 127):
    for el in (-128, -7, 0, 3, 127):
        p = pack_dual(torch.full((N, K), eh, dtype=torch.int32),
                      torch.full((N, K), el, dtype=torch.int32),
                      torch.zeros(N, K, dtype=torch.int32),
                      torch.zeros(N, K, dtype=torch.int32))
        if not bool(is_denormal(p).all()):
            inv_ok = False
check("is_denormal stays TRUE for (s_slow,v_slow)=(0,0) across ALL e_H (incl. exp- AND "
      "magnitude-mode),e_L combos -- does NOT self-nullify (WRONG #2 BREAK A avoided)", inv_ok)

# and a non-zero coarse word is NEVER denormal, regardless of the e_H exponent claim
nondn_ok = True
for ss, vs in ((1, 0), (0, 1), (3, -3), (-5, 5), (127, -128)):
    p = pack_dual(make_eH(N, K, DEN),                          # exp claim present in e_H
                  torch.full((N, K), 0, dtype=torch.int32),
                  torch.full((N, K), ss, dtype=torch.int32),
                  torch.full((N, K), vs, dtype=torch.int32))
    if bool(is_denormal(p).any()):
        nondn_ok = False
check("is_denormal stays FALSE whenever the coarse word is non-zero (even with e_H claim set)",
      nondn_ok)


# ================================================================================== #
print("== CHECK 2: a sub-LSB denormal SURVIVES to the deploy weight via the e_H exp claim ==")
# The exp-claim render is gated on (is_denormal & is_exp_mode). The stochastic render is a
# DC-unbiased Bernoulli of the sub-unit value into the integer mantissa grid, so the
# Monte-Carlo estimate of E|deploy| has standard error ~ sqrt(p(1-p)/T). The deepest coord
# here has |units| ~= 0.12, so use enough salts that the estimator converges below tol.
T = 20000
true_w = {c: abs(float(units0[c])) * float(SCALE[c]) for c in DENORMAL}   # |value|*scale
for c in DENORMAL:
    i, j = c
    em = sum(abs(float(decode_to_deploy_weight(
        packed0, ROW_EXP, COL_EXP, step_salt=t)[i, j])) for t in range(T)) / T
    tr = true_w[c]
    check(f"deploy E|W[{i},{j}]| > 0 and tracks the true sub-scale value over {T} salts "
          f"(survives via e_H exp claim)",
          em > 0 and abs(em - tr) / (tr + 1e-30) < 0.06,
          f"E|deploy|={em:.3e}  true={tr:.3e}")

# CONTRAST -- dropping the claim (enabled=False render on the SAME packed word) gives pure
# coarse and floors the denormal to 0. Isolates the exp claim as THE survival mechanism.
dep_coarse_only = decode_to_deploy_weight(packed0, ROW_EXP, COL_EXP,
                                          enabled=False, deterministic=True)
check("dropping the e_H exp claim (enabled=False render) FLOORS the denormals to 0 "
      "-- isolating the exp claim as the survival mechanism",
      all(float(dep_coarse_only[c]) == 0.0 for c in DENORMAL),
      f"coarse-only deploy at denormals={[float(dep_coarse_only[c]) for c in DENORMAL]}")

# the deterministic render ships the EXACT sub-scale value (|units|*scale), preserving sign.
dep_det = decode_to_deploy_weight(packed0, ROW_EXP, COL_EXP, deterministic=True)
check("deterministic deploy ships the EXACT signed sub-scale value (units*scale)",
      all(abs(float(dep_det[c]) - float(units0[c]) * float(SCALE[c])) < 1e-9 for c in DENORMAL),
      f"det deploy={[float(dep_det[c]) for c in DENORMAL]}")


# ================================================================================== #
print("== CHECK 3: |e_H| FALLBACK -- e_H holding REAL velocity renders PURE COARSE ==")
# Build a denormal coord whose e_H carries a real un-consolidated velocity, |e_H| >=
# EH_VELO_CAP. is_exp_mode must be FALSE there -> the deploy gate (is_denormal & is_exp_mode)
# is off -> the render falls back to pure coarse on THAT coord (and the coarse word is 0,
# so it deploys exactly 0). PROVABLY disjoint: every valid exp-mode encoding has |e_H| <= 63.
busy_eH = EH_VELO_CAP + 5          # 69 >= 64 -> MAGNITUDE MODE (real velocity)
free_eH = ((1 << MANT_BITS) | 6) + 1  # 23 < 64 -> EXP MODE (BUG-2 a=idx+1), oct1 mlow6 -> 0.34375
e_H_b = make_eH(N, K, {(0, 0): busy_eH, (0, 1): free_eH})
packed_b = pack_dual(e_H_b,
                     torch.zeros(N, K, dtype=torch.int32),
                     torch.zeros(N, K, dtype=torch.int32),   # coarse 0 -> all denormal
                     torch.zeros(N, K, dtype=torch.int32))

check("disjointness (HOLE-1): every valid exp-mode encoding has |e_H| <= 63 < EH_VELO_CAP",
      ((OCT_MAX << MANT_BITS) | MLOW_MASK) == 63 and 63 < EH_VELO_CAP,
      f"max exp |e_H|={(OCT_MAX << MANT_BITS) | MLOW_MASK}, EH_VELO_CAP={EH_VELO_CAP}")
check("the busy coord is_denormal (coarse 0) yet is_exp_mode == False (e_H is velocity)",
      bool(is_denormal(packed_b)[0, 0]) and not bool(is_exp_mode(e_H_b)[0, 0]),
      f"|e_H[0,0]|={busy_eH} >= {EH_VELO_CAP}")
check("the free coord is_denormal AND is_exp_mode == True (claim trustworthy)",
      bool(is_denormal(packed_b)[0, 1]) and bool(is_exp_mode(e_H_b)[0, 1]))

# deterministic deploy: the BUSY (magnitude-mode) coord must equal the pure-coarse render
# (claim skipped); the FREE coord adds the exp claim.
dep_busy = decode_to_deploy_weight(packed_b, ROW_EXP, COL_EXP, enabled=True, deterministic=True)
dep_pure = decode_to_deploy_weight(packed_b, ROW_EXP, COL_EXP, enabled=False, deterministic=True)
check("BUSY (magnitude-mode) denormal renders PURE COARSE (guard fires -> claim skipped)",
      float(dep_busy[0, 0]) == float(dep_pure[0, 0]),
      f"busy deploy={float(dep_busy[0,0]):.3e}  pure-coarse={float(dep_pure[0,0]):.3e}")
# the coarse word at [0,0] is 0, so pure coarse is exactly 0 -> the busy guard yields 0,
# NOT the velocity: no silent collision of velocity magnitude with the exponent tag.
check("BUSY denormal deploy is exactly 0 (no velocity leaks through the exp-claim render)",
      float(dep_busy[0, 0]) == 0.0)
# the free coord DOES render its claim (deterministic ships units*scale, sign-correct).
check("FREE (exp-mode) denormal renders its exact claim (deterministic units*scale)",
      abs(float(dep_busy[0, 1])
          - float(decode_denormal_units(e_H_b)[0, 1]) * float(SCALE[0, 1])) < 1e-9,
      f"free deploy={float(dep_busy[0,1]):.3e}")

# stochastic across many salts: free survives in expectation, busy stays floored EVERY salt.
T2 = 8192
em_free = sum(abs(float(decode_to_deploy_weight(packed_b, ROW_EXP, COL_EXP,
                                                step_salt=t)[0, 1])) for t in range(T2)) / T2
em_busy = sum(abs(float(decode_to_deploy_weight(packed_b, ROW_EXP, COL_EXP,
                                                step_salt=t)[0, 0])) for t in range(T2)) / T2
check("FREE denormal survives in expectation (E|deploy| > 0 via e_H exp claim)", em_free > 0,
      f"E|deploy free|={em_free:.3e}")
check("BUSY denormal stays floored to 0 across ALL salts (graceful skip, never a collision)",
      em_busy == 0.0, f"E|deploy busy|={em_busy:.3e}")


# ================================================================================== #
print("== CHECK 4: NORMAL coords are UNAFFECTED by the exponent-claim render ==")
# Build a word with non-zero coarse on NORMAL coords AND a populated e_H exp field
# everywhere. For a normal coord the gate (is_denormal & is_exp_mode) is False, so NO claim
# term is added: the enabled deploy must equal the pure-coarse deploy EXACTLY.
e_H_n = make_eH(N, K, DEN)                       # exp field everywhere (incl. normals)
ss_n = torch.zeros(N, K, dtype=torch.int32)
vs_n = torch.zeros(N, K, dtype=torch.int32)
for (i, j) in NORMAL:
    ss_n[i, j] = 3                               # non-zero coarse -> NOT denormal
    vs_n[i, j] = -1
packed_n = pack_dual(e_H_n, torch.zeros(N, K, dtype=torch.int32), ss_n, vs_n)
dep_enabled = decode_to_deploy_weight(packed_n, ROW_EXP, COL_EXP, enabled=True, deterministic=True)
dep_pure_all = decode_to_deploy_weight(packed_n, ROW_EXP, COL_EXP, enabled=False, deterministic=True)
normal_mask = ~is_denormal(packed_n)
check("at every NON-denormal coord, enabled deploy == pure-coarse deploy EXACTLY "
      "(the exp-claim render never touches normal coords)",
      bool(torch.equal(dep_enabled[normal_mask], dep_pure_all[normal_mask])),
      f"n_normal={int(normal_mask.sum())}")
# the named normal coords each deploy their coarse (s_slow+v_slow)*128*scale, ALIVE.
for (i, j) in NORMAL:
    want_w = float((ss_n[i, j] + vs_n[i, j]).item()) * 128.0 * float(SCALE[i, j])
    check(f"named normal coord [{i},{j}] deploys pure coarse (s_slow+v_slow)*128*scale",
          abs(float(dep_enabled[i, j]) - want_w) < 1e-9 and float(dep_enabled[i, j]) != 0.0,
          f"deploy={float(dep_enabled[i,j]):.4e}  want={want_w:.4e}")


# ================================================================================== #
print("== CHECK 5: the deploy exp-claim render is DC-UNBIASED over many salts ==")
# The stochastic-rounded render must average to the exact sub-scale value (no systematic
# DC drift). Use a denormal coord whose units are a clean reference: e_H = oct0 mlow6 ->
# (1 + 6/16) * 2^-1 = 0.6875 mantissa units. Verify E[render mantissa] ~= 0.6875.
mlow_dc = 6
a_dc = ((0 << MANT_BITS) | mlow_dc) + 1              # BUG-2: code a = idx + 1
units_dc = (1.0 + mlow_dc / 16.0) * 2.0 ** -1        # 0.6875 mantissa units
e_H_dc = torch.full((N, K), a_dc, dtype=torch.int32)
packed_dc = pack_dual(e_H_dc,
                      torch.zeros(N, K, dtype=torch.int32),
                      torch.zeros(N, K, dtype=torch.int32),
                      torch.zeros(N, K, dtype=torch.int32))
assert bool(is_denormal(packed_dc).all()) and bool(is_exp_mode(e_H_dc).all())
assert abs(float(decode_denormal_units(e_H_dc)[0, 0]) - units_dc) < 1e-7
Tdc = 30000
acc = torch.zeros(N, K, dtype=torch.float64)
for t in range(Tdc):
    dep = decode_to_deploy_weight(packed_dc, ROW_EXP, COL_EXP, step_salt=t)
    acc += (dep.to(torch.float64) / SCALE.to(torch.float64))     # back to mantissa units
mean_mant = acc / Tdc
max_dc_err = float((mean_mant - units_dc).abs().max())
check(f"E[deploy exp-claim render] == {units_dc} mantissa units over {Tdc} salts (DC-unbiased)",
      max_dc_err < 0.02,
      f"max|E[render]-{units_dc}|={max_dc_err:.4f} mantissa units")

# deterministic render is reproducible (shipping checkpoint determinism)
da = decode_to_deploy_weight(packed_dc, ROW_EXP, COL_EXP, deterministic=True)
db = decode_to_deploy_weight(packed_dc, ROW_EXP, COL_EXP, deterministic=True)
check("deterministic deploy render is reproducible (shipping checkpoint)",
      bool(torch.equal(da, db)))


# ================================================================================== #
print("== CHECK 6: CONSERVING graduation -- denormal -> normal credits e_L, NEVER +128 s_slow ==")
# The architect's key conservation claim for ADD-2. Two assertions:
#   (a) the FULL-STEP conservation ledger holds EVERY step (delta_deploy + delta_fine ==
#       inflow, evap a sink), INCLUDING steps with live exp-denormals -- step() banks the
#       H-arm inflow in the log field and books the graduated whole units into inflow_int,
#       so the global invariant stays green. We replay the reference's own validated regime.
#   (b) on a HAND-BUILT graduation (calling _denormal_graduate directly) the promotion lands
#       in e_L (fine->fine) and the DEPLOY mantissa (s_slow+v_slow)*128 stays UNCHANGED --
#       i.e. NO +128 credit into s_slow (the REMOVED leak).
torch.manual_seed(0)
rg = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=2)
Wg = torch.randn(N, K) * 0.05
rg.load_weights(Wg)                 # from-scratch: substrate from seed, accumulator ZERO
# the reference's __main__ test-4 gradient pattern (drives offset off zero -> denormals).
g = -torch.sign(Wg) * 0.02 + 0.02
max_resid = 0
saw_exp_denorm = 0
saw_graduation = 0
for t in range(300):
    info = rg.step(g, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                   coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0,
                   chase_floor=0.1, return_ledger=True)
    saw_exp_denorm = max(saw_exp_denorm, info["n_exp_denormal"])
    pb, pa, inflow = info["_ledger"]
    ok, resid = assert_conservation(pb, pa, inflow)
    max_resid = max(max_resid, resid)
    assert ok, f"CONSERVATION VIOLATED at step {t}: residual={resid}"
    if int(info["n_exp_denormal"]) > 0:
        saw_graduation += 1
check("FULL-STEP conservation ledger holds EVERY step incl. live exp-denormals "
      "(graduation books whole units as inflow into the fine side, never +128 to s_slow)",
      max_resid == 0 and saw_exp_denorm > 0,
      f"max residual over 300 steps = {max_resid}, saw up to {saw_exp_denorm} "
      f"exp-denormals/step on {saw_graduation} steps")

# (b) hand-built graduation: a denormal e_H near +1 unit plus a positive sub-unit inflow
# must promote the WHOLE unit into e_L and re-encode the residual into e_H -- with the
# DEPLOY mantissa (s_slow+v_slow)*128 UNCHANGED (no +128 credit into s_slow). We call the
# layer's graduation primitive directly so the assertion isolates the promotion path.
rh = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=9)
rh.load_weights(torch.randn(N, K) * 0.02)
e_H_h, e_L_h, ss_h, vs_h = unpack_dual(rh.packed_w.clone())
e_H_h = e_H_h.to(torch.int32); e_L_h = e_L_h.to(torch.int32)
ss_h = torch.zeros(N, K, dtype=torch.int32); vs_h = torch.zeros(N, K, dtype=torch.int32)
e_L_h = torch.zeros(N, K, dtype=torch.int32)
gi, gj = 0, 2
# near-full exp claim: oct0 mlow15 -> (1+15/16)*0.5 = 0.96875 mantissa units. (BUG-2 a=idx+1)
e_H_h[:] = 0
e_H_h[gi, gj] = ((0 << MANT_BITS) | 15) + 1
packed_before_h = pack_dual(e_H_h, e_L_h, ss_h, vs_h)
exp_den_h = is_denormal(packed_before_h) & is_exp_mode(unpack_dual(packed_before_h)[0])
assert bool(exp_den_h[gi, gj]), "hand-built coord must be an exp-mode denormal"
assert abs(float(decode_denormal_units(e_H_h)[gi, gj]) - 0.96875) < 1e-6
dep_mant_before = int(deploy_mantissa(packed_before_h)[gi, gj])
eL_before = int(e_L_h[gi, gj])
# a positive sub-unit inflow that, added to ~0.96875, crosses 1.0 -> promote 1 whole unit.
half_inflow = torch.zeros(N, K, dtype=torch.float32)
half_inflow[gi, gj] = 0.20         # 0.96875 + 0.20 = 1.16875 -> trunc -> 1 unit into e_L
pos = ref._pos_hash(N, K)
e_Hg, e_Lg, grad_inflow = rh._denormal_graduate(
    exp_den_h, e_H_h.clone(), e_L_h.clone(), half_inflow, pos, salt=12345)
check("graduation promotes the WHOLE unit into e_L (fine->fine), booked as inflow",
      int(e_Lg[gi, gj]) == eL_before + 1 and int(grad_inflow[gi, gj]) == 1,
      f"e_L: {eL_before} -> {int(e_Lg[gi,gj])}, grad_inflow={int(grad_inflow[gi,gj])}")
# the residual (~0.16875) re-encodes back into e_H's octave field -> still a sub-unit claim.
resid_units = float(decode_denormal_units(e_Hg)[gi, gj])
check("the sub-unit residual re-encodes into e_H's octave field (still |claim| < 1)",
      abs(resid_units) < 1.0 and abs(resid_units) > 0.0,
      f"residual claim after graduation = {resid_units:.5f} (~0.169 expected pre-SR-grain)")
# the promotion went to e_L, NOT to the deploy word: deploy mantissa UNCHANGED, no +128.
packed_after_h = pack_dual(e_Hg.clamp(INT8_MIN, INT8_MAX), e_Lg.clamp(INT8_MIN, INT8_MAX),
                           ss_h, vs_h)
dep_mant_after = int(deploy_mantissa(packed_after_h)[gi, gj])
check("graduation injects NO +128 deploy credit (s_slow/v_slow untouched -> deploy mantissa "
      "unchanged at the coord; the REMOVED +128 leak does NOT recur)",
      dep_mant_after == dep_mant_before
      and int(unpack_dual(packed_after_h)[2][gi, gj]) == 0
      and int(unpack_dual(packed_after_h)[3][gi, gj]) == 0,
      f"deploy mantissa {dep_mant_before} -> {dep_mant_after}, s_slow=0, v_slow=0")


# ================================================================================== #
print("== CHECK 7: NO fp sidecars -- live/deploy recoverable from the packed word ALONE ==")
# The corrected design (point 9) forbids ANY fp companion buffer for the fine/denormal
# value. The ONLY out-of-word fp state allowed is the seed-derived SUBSTRATE (ADD-1), which
# is regenerable from the seed and never a shadow of the fine register.
has_sidecar = any(hasattr(rg, nm) for nm in
                  ("den_frac", "err_s", "s_fast_fp", "frac_fp", "e_H_fraction_buf"))
check("the Layer carries NO fp sidecar (den_frac / err_s / frac_fp / ... absent)",
      not has_sidecar)

# after training with denormals present, a fresh copy of the packed word + the SAME exps +
# substrate must decode to the identical live and deploy weight (value lives in no sidecar).
fresh = rg.packed_w.clone()
w_live_a = decode_to_live_weight(rg.packed_w, rg.row_exp, rg.col_exp, substrate=rg.substrate)
w_live_b = decode_to_live_weight(fresh, rg.row_exp, rg.col_exp, substrate=rg.substrate)
check("after training with denormals present, live weight is a PURE function of "
      "(packed, exps, substrate) -- recoverable from the word ALONE",
      bool(torch.equal(w_live_a, w_live_b)))
w_dep_a = decode_to_deploy_weight(rg.packed_w, rg.row_exp, rg.col_exp,
                                  substrate=rg.substrate, deterministic=True)
w_dep_b = decode_to_deploy_weight(fresh, rg.row_exp, rg.col_exp,
                                  substrate=rg.substrate, deterministic=True)
check("deploy weight (deterministic) is likewise a pure function of the packed word + exps",
      bool(torch.equal(w_dep_a, w_dep_b)))

# the fine bytes stay int8-bounded through training (no shadow-velocity blow-up).
# int8-bounded means within [INT8_MIN, INT8_MAX] = [-128, 127]; the valid floor INT8_MIN
# has abs 128 (> INT8_MAX=127), so a signed range check is the correct "int8-bounded" test
# (the abs() <= INT8_MAX form wrongly rejects the legitimate -128 the reference clamps to).
e_H_f, e_L_f, _, _ = unpack_dual(rg.packed_w)
check("fine bytes e_L, e_H stay int8-bounded through training (no shadow velocity)",
      int(e_L_f.min()) >= INT8_MIN and int(e_L_f.max()) <= INT8_MAX
      and int(e_H_f.min()) >= INT8_MIN and int(e_H_f.max()) <= INT8_MAX,
      f"e_L in [{int(e_L_f.min())},{int(e_L_f.max())}], e_H in [{int(e_H_f.min())},{int(e_H_f.max())}]")


# ================================================================================== #
n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} dual-dissipation denormal (exp-claim) CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
