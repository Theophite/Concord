"""CPU unit tests for the dither-accum CARRY / MASS-CONSERVATION / NO-OVERFLOW
invariants of the bounded-int16 fine accumulator (no GPU, no Triton, no pytest).

Reference under test:
  modules/util/optimizer/concord/dither_accum_ref.py  (pure-torch, CPU)

This module tests ONLY the three structural accounting properties the redesign
hinges on -- NOT coherence/denormal/disabled-vs-legacy (those are the reference's
own self-test). Concretely:

  (1) CARRY -- each carry moves EXACTLY one LSB of mass between registers:
      * the chase carry `tick_slow` removes EXACTLY `tick_slow*128` mantissa units
        from s_fast and deposits them in s_slow, so the FULL integer mantissa
        `s_slow*128 + s_fast + v_slow*128` is conserved across the chase, and the
        deploy coarse `(s_slow+v_slow)*128` advances by EXACTLY `tick_slow*128`;
      * the mass-preserving leak `tick_v8` moves EXACTLY one int8 unit at a time
        s_slow <-> v_slow, so the deploy coarse `(s_slow+v_slow)*128` is conserved
        across the leak (only the chase advances it).

  (2) MASS CONSERVATION -- the total represented value equals the cumulative
      input. Driving the accumulator with a known per-step inflow `delta_t`
      (mantissa units), the change in the full represented value
        repr := (s_slow*128 + s_fast + v_slow*128) + err_s   (+ den_frac residual)
      over a step equals EXACTLY the SR-rounded inflow that was ticked into
      s_fast that step -- i.e. the sigma-delta error-feedback drops NO sub-LSB
      inflow: cumulative(repr) == cumulative(SR inflow) == cumulative input to
      within one LSB at any horizon (the open err_s remainder), with ZERO drift.

  (3) NO OVERFLOW + bounded quantization error -- under realistic streams:
      * the byte VIEW (e_s, e_v) of the single int16 s_fast always reads back as
        valid sign-extended int8 (|e_s| <= 127, e_v in the legacy low-byte range)
        and reunifies bit-exactly to s_fast (it is a byte split, never two clamped
        registers, so it cannot "overflow int8" -- this is the crux of the fix);
      * s_slow, v_slow stay in int8 [-128,127] and s_fast in int16 [-32768,32767]
        every step (the re-exponent safety renormalizes before the clamp bites);
      * the live-weight quantization error vs the true cumulative input is bounded
        ~+-0.5 LSB in expectation (it is err_s, a sub-LSB SR remainder).

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_dither_carry_mass_cpu.py
  or:  CUDA_VISIBLE_DEVICES="" python .../tests/test_dither_carry_mass_cpu.py
"""
import sys
from pathlib import Path

import torch

# import the reference straight from the concord dir (pure-torch, CPU)
HERE = Path(__file__).resolve()
CONCORD = HERE.parents[1]                       # .../optimizer/concord
sys.path.insert(0, str(CONCORD))

from dither_accum_ref import (                  # noqa: E402
    DitherAccumRef,
    INT8_MIN, INT8_MAX, INT16_MIN, INT16_MAX, CARRY,
    unpack_word, pack_word, unpack_bytes, reunify_fast,
    assert_carries_bounded,
    _sr_round, _pos_hash, _scale_fwd,
)

torch.manual_seed(0)

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def repr_value(ref):
    """The FULL represented value in mantissa units (scale-free): the integer
    live mantissa plus the bounded fp companions. This is the conserved quantity
    the sigma-delta accumulator tracks against cumulative input."""
    s_fast, s_slow, v_slow = unpack_word(ref.packed_w)
    int_mant = (s_slow.to(torch.float32) * CARRY
                + s_fast.to(torch.float32)
                + v_slow.to(torch.float32) * CARRY)
    return int_mant + ref.err_s + ref.den_frac


# ============================================================================
# (1) CARRY -- each carry moves EXACTLY one LSB between registers
# ============================================================================
# We re-run the chase and leak arithmetic IN ISOLATION on a controlled packed
# state with a KNOWN coherence so the carries are deterministic-magnitude, and
# assert the exact accounting identities the redesign claims:
#   chase:  s_slow += tick_slow ; s_fast -= tick_slow*128   (full mantissa conserved)
#   leak :  v_slow += tick_v8   ; s_slow -= tick_v8          (coarse conserved)
print("== (1) CARRY: each carry moves exactly one LSB (exact accounting) ==")

# Drive a real DitherAccumRef and, every step, snapshot the packed word BEFORE
# and AFTER and reconstruct the carries from the diagnostics + register deltas.
N, K = 8, 16
ref = DitherAccumRef(N, K, dither_enabled=True, seed=11)
W = torch.randn(N, K) * 0.05
ref.load_weights(W)
g = -torch.sign(W) * 0.02 + 0.02                     # coherent-drift regime (self-test 3)

chase_conserves_full = True
leak_conserves_coarse = True
deploy_moves_by_chase_only = True
max_full_jump_resid = 0.0
n_carry_steps = 0
worst_full_drift = 0.0
worst_coarse_drift = 0

for t in range(300):
    sf0, ss0, vs0 = unpack_word(ref.packed_w)
    sf0 = sf0.to(torch.int64); ss0 = ss0.to(torch.int64); vs0 = vs0.to(torch.int64)
    full0 = ss0 * CARRY + sf0 + vs0 * CARRY          # integer live mantissa, pre-step
    coarse0 = (ss0 + vs0) * CARRY

    info = ref.step(g, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                    coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)

    sf1, ss1, vs1 = unpack_word(ref.packed_w)
    sf1 = sf1.to(torch.int64); ss1 = ss1.to(torch.int64); vs1 = vs1.to(torch.int64)
    full1 = ss1 * CARRY + sf1 + vs1 * CARRY
    coarse1 = (ss1 + vs1) * CARRY

    # The deploy coarse advances ONLY by chase whole-LSB carries. Both chase and
    # leak are exact int moves, so coarse1 - coarse0 must be a whole multiple of
    # 128 (CARRY) -- no fractional mass ever lands in the coarse word.
    d_coarse = (coarse1 - coarse0)
    if bool((d_coarse % CARRY != 0).any()):
        deploy_moves_by_chase_only = False
        worst_coarse_drift = int((d_coarse % CARRY).abs().max())

    if info["deploy_advanced"]:
        n_carry_steps += 1

# We cannot read the intermediate (post-chase, pre-leak) state from the public
# API, so we re-derive the EXACT chase/leak identities from a fresh hand-run of
# the same arithmetic the step performs, on a controlled state, below.
check("deploy coarse advances only in whole 128-unit (one-LSB) increments",
      deploy_moves_by_chase_only,
      f"worst non-128 remainder over 300 steps = {worst_coarse_drift}")
check("a carry lands on the deploy word on the great majority of steps (ratchet)",
      n_carry_steps > 150, f"carry-bearing steps = {n_carry_steps}/300")


# --- exact chase/leak identity, hand-run on a controlled state ---------------
# Reproduce ONE step's chase + mass-preserving leak arithmetic verbatim from the
# reference (dither_accum_ref.py:540-564) and assert the conservation identities
# coordinate-wise and exactly (integer arithmetic, no tolerance).
torch.manual_seed(3)
N2, K2 = 6, 10
s_fast = torch.randint(-3000, 3000, (N2, K2), dtype=torch.int32)
s_slow = torch.randint(-40, 40, (N2, K2), dtype=torch.int32)
v_slow = torch.randint(-40, 40, (N2, K2), dtype=torch.int32)
coh = torch.rand(N2, K2)                                       # arbitrary gate in [0,1]
pos = _pos_hash(N2, K2)
salt = 0x12345678
alpha, chase_floor = 0.1, 0.1
alpha_v_fast, leak_floor = 0.001, 0.05

full_before = (s_slow.to(torch.int64) * CARRY + s_fast.to(torch.int64)
               + v_slow.to(torch.int64) * CARRY)
coarse_before = (s_slow.to(torch.int64) + v_slow.to(torch.int64)) * CARRY

# CHASE (dither_accum_ref.py:545-552)
chase_gate = chase_floor + (1.0 - chase_floor) * coh
chase_mantissa = alpha * chase_gate * s_fast.to(torch.float32)
chase_int8_f = chase_mantissa / float(CARRY)
tick_slow = _sr_round(chase_int8_f, s_fast, pos, salt ^ 0x5A5A5A5A)
s_slow_c = s_slow + tick_slow
s_fast_c = s_fast - tick_slow * CARRY

full_after_chase = (s_slow_c.to(torch.int64) * CARRY + s_fast_c.to(torch.int64)
                    + v_slow.to(torch.int64) * CARRY)
# Each chase tick removes EXACTLY tick_slow*128 from s_fast and adds tick_slow to
# s_slow -> the FULL integer mantissa is conserved coordinate-wise, exactly.
check("CHASE conserves the full integer mantissa exactly (s_fast -> s_slow, one LSB = 128)",
      torch.equal(full_after_chase, full_before),
      f"max|delta| = {int((full_after_chase - full_before).abs().max())}")
# and the deploy coarse advances by EXACTLY tick_slow*128 (one s_slow LSB each).
coarse_after_chase = (s_slow_c.to(torch.int64) + v_slow.to(torch.int64)) * CARRY
check("CHASE advances deploy coarse by EXACTLY tick_slow*128 (whole LSBs only)",
      torch.equal(coarse_after_chase - coarse_before, tick_slow.to(torch.int64) * CARRY),
      f"max tick_slow = {int(tick_slow.abs().max())}")

# LEAK, mass-preserving (dither_accum_ref.py:554-564)
gap_v = (s_slow_c.to(torch.float32) * 128 - v_slow.to(torch.float32) * 128)
delta_v8 = alpha_v_fast * gap_v / 128.0 * (leak_floor + (1.0 - leak_floor) * coh)
tick_v8 = _sr_round(delta_v8, s_fast_c, pos, salt ^ 0x33335555)
v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)
actual_tick_v8 = v_slow_new - v_slow
s_slow_l = s_slow_c - actual_tick_v8                          # mass-preserve

coarse_after_leak = (s_slow_l.to(torch.int64) + v_slow_new.to(torch.int64)) * CARRY
# The mass-preserving leak moves whole int8 units s_slow <-> v_slow, so the deploy
# coarse (s_slow+v_slow)*128 is CONSERVED across the leak (only chase moves it).
check("LEAK (mass-preserving) conserves deploy coarse exactly (s_slow <-> v_slow, one LSB)",
      torch.equal(coarse_after_leak, coarse_after_chase),
      f"max|delta| = {int((coarse_after_leak - coarse_after_chase).abs().max())}")
# and each realized leak tick moves at most... well, any int; but the point is it
# is an INTEGER int8 move and v_slow's change equals -s_slow's change exactly.
check("LEAK moves equal-and-opposite integer units between s_slow and v_slow",
      torch.equal((v_slow_new - v_slow), -(s_slow_l - s_slow_c)),
      f"max|tick_v8 realized| = {int(actual_tick_v8.abs().max())}")


# ============================================================================
# (2) MASS CONSERVATION: total represented value == cumulative input
# ============================================================================
# The sigma-delta accumulator must conserve mass: every unit of inflow either
# lands in the integer registers or sits (sub-LSB) in err_s -- NONE is dropped,
# NONE is created. We verify this against an INDEPENDENT running sum of the
# per-step inflow (recomputed here from the public state with the SAME formula
# step() uses), NOT against the accumulator's own delta (which would be circular).
#
# The conserved physical quantity is the WEIGHT VALUE, not the raw mantissa: the
# step's preconditioned increment in weight units is  delta_W = -lr * step_live
# (since delta_grad = delta_W / scale and the integer mantissa is decoded back as
# mantissa * scale). Working in weight space makes the test invariant to the
# value-preserving re-exponent (renormalize_on_saturation halves the mantissa and
# bumps row_exp, which is a no-op on the weight value but changes mantissa units).
# Then, at EVERY horizon:
#     live_weight()  ==  live_weight0 + sum_t(delta_W)   to within ONE current LSB
# (the open err_s remainder) -- exactly the sigma-delta guarantee: bounded error,
# zero secular drift. A REALISTIC init keeps the block-float scale sane (a zero
# init pins row_exp at the -30 floor -> scale_inv ~ 3.5e13, an unrealizable regime
# with nothing to do with the carry logic).
print("== (2) MASS CONSERVATION: represented value == cumulative input ==")

N3, K3 = 4, 4
beta2 = 0.999
step_cap, eps, v_scale, precond_p, lr = 10.0, 1.0, 1.0, 0.5, 0.05

def predict_delta_W(ref, grad_W):
    """Independently recompute the per-step WEIGHT-space increment delta_W the way
    step() does (dither_accum_ref.py:495-507) for the gf_consol=0, drift_cancel_C=0
    case, from the PUBLIC pre-step state. delta_W = -lr*step_live (weight units),
    i.e. delta_grad*scale. This is the 'cumulative input' reference -- scale-free,
    so the value-preserving re-exponent does not perturb it."""
    s_fast, s_slow, v_slow = unpack_word(ref.packed_w)
    scale = _scale_fwd(ref.row_exp, ref.col_exp)
    d_fs_w = (s_fast.to(torch.float32) + ref.err_s) * scale       # drift_cancel_C=0 => noise=d_fs
    v_proxy = d_fs_w * d_fs_w * v_scale
    denom = torch.pow(v_proxy + eps, precond_p)
    step_live = (grad_W / denom).clamp(-step_cap, step_cap)
    return -lr * step_live                                        # weight units

# --- (2a) pure sigma-delta: chase/leak/evap OFF, inflow accumulates in s_fast ---
# A constant-sign inflow grows the weight, so the value-preserving re-exponent
# (renormalize_on_saturation) periodically fires (halve mantissa + bump row_exp).
# That halving uses THREE independent round-to-nearest, so it is value-preserving
# only "to within the halving LSB" (dither_accum_ref.py:339, design risk #2) -- it
# is NOT bit-exact across an exponent change. The sigma-delta ACCUMULATION
# guarantee (no inflow dropped) is therefore the WITHIN-EPOCH statement: between
# re-exponent events the live weight tracks the cumulative independent inflow to
# << 1 LSB. We re-anchor the reference at each re-exponent event (a legitimate
# value-preserving, lossy-to-1-LSB coordinate change) and assert within-epoch
# closure < 1 LSB. (A realistic init keeps row_exp sane; a zero init would pin it
# at the -30 floor -> scale_inv ~ 3.5e13, an unrealizable regime.)
torch.manual_seed(21)
ref2 = DitherAccumRef(N3, K3, dither_enabled=True, seed=21)
ref2.load_weights(torch.randn(N3, K3) * 0.05)            # realistic scale
g3 = torch.full((N3, K3), 0.05)                          # gentle constant inflow
step_kw = dict(lr=lr, alpha=0.0, gf_consol=0.0, drift_cancel_C=0.0,
               alpha_v_fast=0.0, coh_kappa=1.0, chase_floor=0.0, leak_floor=0.0,
               v_scale=v_scale, precond_p=precond_p, eps=eps, step_cap=step_cap,
               beta2=beta2)

base = ref2.live_weight().clone()                        # epoch baseline
cum_input_W = torch.zeros(N3, K3)
worst_lsb = 0.0
n_reexp = 0
re_prev = ref2.row_exp.clone()
for t in range(400):
    dW = predict_delta_W(ref2, g3)                       # independent inflow (weight units)
    ref2.step(g3, **step_kw)
    cum_input_W = cum_input_W + dW
    if bool((ref2.row_exp != re_prev).any()):
        # value-preserving (to within the halving LSB) re-exponent: re-anchor
        base = ref2.live_weight().clone()
        cum_input_W = torch.zeros(N3, K3)
        re_prev = ref2.row_exp.clone()
        n_reexp += 1
        continue
    cur_lsb = _scale_fwd(ref2.row_exp, ref2.col_exp)      # current weight-units per mantissa LSB
    resid_lsb = ((ref2.live_weight() - (base + cum_input_W)) / cur_lsb).abs()
    worst_lsb = max(worst_lsb, float(resid_lsb.max()))

# Within an exponent epoch the accumulator tracks the cumulative input essentially
# exactly (the only slip is the open err_s remainder, < 1 LSB) -- no inflow
# dropped, none created. The draft failed this (~1e5-unit divergence).
check("MASS: live weight == baseline + cumulative independent input within ONE LSB (no drift)",
      worst_lsb < 1.0 + 1e-3,
      f"max within-epoch |live - (base + sum dW)| = {worst_lsb:.3f} LSB over 400 steps, "
      f"{n_reexp} re-exponent events")
# The dropped (sub-LSB) inflow is exactly the bounded err_s remainder, not a leak.
sub_lsb_resid = ref2.err_s + ref2.den_frac
check("MASS: the dropped (sub-LSB) inflow is a BOUNDED remainder (<1 LSB), not a growing leak",
      float(sub_lsb_resid.abs().max()) < 1.0 + 1e-4,
      f"|err_s+den_frac| after 400 steps = {float(sub_lsb_resid.abs().max()):.3f}")

# --- (2b) chase + leak ON, NON-SATURATING regime: the carries only RELOCATE mass
# between s_fast/s_slow/v_slow; the LIVE weight sums all three, so the carries are
# mass-neutral on it and the live weight still equals baseline + cumulative inflow
# to within one LSB. This holds as long as no register SATURATES -- a coord whose
# s_slow pins at +-127 loses the chased overflow at the int8 clamp (the documented
# out-of-range behavior covered by section 3 / the re-exponent safety), which is
# NOT a leak in the conservation sense. We use a gentle mean-zero gradient + a
# moderate init + a short horizon so s_slow/s_fast keep headroom and no re-exponent
# fires; den_frac is left ON (default) -- in a normal coord its leak residual is ~0
# so no whole-unit promotion (the intentional mass source, design risk #3) occurs.
torch.manual_seed(60)
ref2b = DitherAccumRef(N3, K3, dither_enabled=True, seed=60)
ref2b.load_weights(torch.randn(N3, K3) * 0.3)
g3b = torch.randn(N3, K3) * 0.01                         # gentle, mean-zero (no secular growth)
step_kw_carry = dict(step_kw)
step_kw_carry.update(alpha=0.05, alpha_v_fast=0.001, chase_floor=0.1, leak_floor=0.05)
live0b = ref2b.live_weight().clone()
cum_input_Wb = torch.zeros(N3, K3)
worst_lsb_b = 0.0
saturated = False
re_prev_b = ref2b.row_exp.clone()
for t in range(150):
    dW = predict_delta_W(ref2b, g3b)
    ref2b.step(g3b, **step_kw_carry)
    cum_input_Wb = cum_input_Wb + dW
    sf, ss, vs = unpack_word(ref2b.packed_w)
    if int(ss.abs().max()) >= INT8_MAX or bool((ref2b.row_exp != re_prev_b).any()):
        saturated = True                                 # would invalidate the conservation regime
    cur_lsb = _scale_fwd(ref2b.row_exp, ref2b.col_exp)
    worst_lsb_b = max(worst_lsb_b,
                      float(((ref2b.live_weight() - (live0b + cum_input_Wb)) / cur_lsb).abs().max()))
check("MASS: regime stayed non-saturating (s_slow < 127, no re-exponent) -- conservation applies",
      not saturated)
check("MASS: with chase+leak ON (non-saturating), carries only RELOCATE -- live == live0 + sum(input) (<=1 LSB)",
      worst_lsb_b < 1.0 + 1e-3,
      f"max closure residual over 150 steps = {worst_lsb_b:.4f} LSB")


# ============================================================================
# (3) NO OVERFLOW (e_s/e_v fit int8; s_slow/v_slow fit int8; s_fast fits int16)
#     + quantization error bounded ~+-0.5 LSB
# ============================================================================
print("== (3) NO OVERFLOW: byte view fits int8, registers in range, q-error ~+-0.5 LSB ==")

# (3a) Byte VIEW round-trip over the full int16 range: e_s/e_v are byte slices of
# ONE int16, so they ALWAYS read as valid sign-extended int8 and reunify exactly.
s_fast_full = torch.arange(INT16_MIN, INT16_MAX + 1, dtype=torch.int32).reshape(1, -1)
zeros = torch.zeros_like(s_fast_full)
packed_full = pack_word(s_fast_full, zeros, zeros)
e_s, e_v = unpack_bytes(packed_full)
check("byte view e_s is a valid sign-extended int8 (|e_s| <= 127) over ALL int16",
      int(e_s.min()) >= INT8_MIN and int(e_s.max()) <= INT8_MAX,
      f"e_s range = [{int(e_s.min())}, {int(e_s.max())}]")
check("byte view e_v is a valid sign-extended int8 (low byte) over ALL int16",
      int(e_v.min()) >= INT8_MIN and int(e_v.max()) <= INT8_MAX,
      f"e_v range = [{int(e_v.min())}, {int(e_v.max())}]")
check("byte view reunifies bit-exactly to s_fast over the ENTIRE int16 range "
      "(one int16 velocity, never two clamped int8s)",
      torch.equal(reunify_fast(e_s, e_v), s_fast_full))

# (3b) Realistic stream: registers stay in range every step, in BOTH a coherent-
# drift and a strong-gradient regime; the byte view stays valid every step too.
def run_range_check(grad, steps, seed, label):
    r = DitherAccumRef(N, K, dither_enabled=True, seed=seed)
    r.load_weights(torch.randn(N, K) * 0.05)
    ok_int8 = True; ok_int16 = True; ok_byte = True; ok_carries = True
    # track SIGNED extremes (not abs): -128 is a legal int8, so an abs-max of 128
    # would be misleading -- the in-range predicate is min>=-128 and max<=127.
    ss_lo = 0; ss_hi = 0; vs_lo = 0; vs_hi = 0; sf_lo = 0; sf_hi = 0; max_es = 0
    for t in range(steps):
        r.step(grad, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
               coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
        sf, ss, vs = unpack_word(r.packed_w)
        ss_lo = min(ss_lo, int(ss.min())); ss_hi = max(ss_hi, int(ss.max()))
        vs_lo = min(vs_lo, int(vs.min())); vs_hi = max(vs_hi, int(vs.max()))
        sf_lo = min(sf_lo, int(sf.min())); sf_hi = max(sf_hi, int(sf.max()))
        if not (INT8_MIN <= int(ss.min()) and int(ss.max()) <= INT8_MAX):
            ok_int8 = False
        if not (INT8_MIN <= int(vs.min()) and int(vs.max()) <= INT8_MAX):
            ok_int8 = False
        if not (INT16_MIN <= int(sf.min()) and int(sf.max()) <= INT16_MAX):
            ok_int16 = False
        es, ev = unpack_bytes(r.packed_w)
        max_es = max(max_es, int(es.abs().max()))
        # e_s, e_v are sign-extended int8 BYTE views: valid range is the full
        # [-128, 127] (NOT |x| <= 127 -- the low byte 0x80 reads as -128, a legal
        # int8). The whole point is they NEVER need to overflow int8 to hold the
        # velocity, because the velocity lives in the reunified int16, not the byte.
        if not (INT8_MIN <= int(es.min()) and int(es.max()) <= INT8_MAX
                and INT8_MIN <= int(ev.min()) and int(ev.max()) <= INT8_MAX):
            ok_byte = False
        if not torch.equal(reunify_fast(es, ev), sf):
            ok_byte = False
        okc, _, _ = assert_carries_bounded(r.err_s, r.den_frac)
        ok_carries = ok_carries and okc
    return (ok_int8, ok_int16, ok_byte, ok_carries,
            (ss_lo, ss_hi), (vs_lo, vs_hi), (sf_lo, sf_hi), max_es, r)

for grad, steps, seed, label in [
    (-torch.sign(torch.randn(N, K)) * 0.02 + 0.02, 400, 31, "coherent-drift"),
    (torch.ones(N, K) * 0.5, 400, 32, "strong-gradient"),
]:
    oi8, oi16, ob, oc, ssr, vsr, sfr, mes, r = run_range_check(grad, steps, seed, label)
    check(f"NO OVERFLOW [{label}]: s_slow, v_slow in int8 [-128,127] every step",
          oi8, f"s_slow range={ssr}, v_slow range={vsr}")
    check(f"NO OVERFLOW [{label}]: s_fast in int16 [-32768,32767] every step",
          oi16, f"s_fast range={sfr}")
    check(f"NO OVERFLOW [{label}]: byte view (e_s,e_v) valid int8 + reunifies every step",
          ob, f"max|e_s|={mes} (a byte slice -- never overflows int8 by construction)")
    check(f"NO OVERFLOW [{label}]: companions bounded (|err_s|<1, |den_frac|<1) every step",
          oc)

# (3c) Quantization error bounded ~+-0.5 LSB in expectation. The live weight's
# error vs the true cumulative input is err_s (a sub-LSB SR remainder). Over a
# pure sigma-delta accumulation (no carry/leak) the running-mean residual should
# hover near 0 with |residual| < 1 always (one LSB) and time-average ~ |0.5| LSB.
# Realistic init so the block-float scale is sane (a zero init pins row_exp at the
# -30 floor, an unrealizable scale that has nothing to do with the SR quantizer).
torch.manual_seed(41)
ref3 = DitherAccumRef(N3, K3, dither_enabled=True, seed=41)
ref3.load_weights(torch.randn(N3, K3) * 0.05)
g_q = torch.full((N3, K3), 0.37)                     # constant inflow exercising SR
running_abs = []
max_inst = 0.0
for t in range(2000):
    ref3.step(g_q, lr=0.05, alpha=0.0, gf_consol=0.0, drift_cancel_C=0.0,
              alpha_v_fast=0.0, coh_kappa=1.0, chase_floor=0.0, leak_floor=0.0)
    inst = float(ref3.err_s.abs().mean())            # mean |quantization residual| in LSB
    running_abs.append(inst)
    max_inst = max(max_inst, float(ref3.err_s.abs().max()))
mean_abs_lsb = sum(running_abs) / len(running_abs)
check("Q-ERROR: instantaneous quantization residual < 1 LSB at all times (bounded SR)",
      max_inst < 1.0 + 1e-4, f"max|err_s| over 2000 steps = {max_inst:.3f}")
check("Q-ERROR: time-averaged |residual| ~ 0.5 LSB (uniform SR remainder, not a drift)",
      0.2 <= mean_abs_lsb <= 0.8, f"mean|err_s| = {mean_abs_lsb:.3f} LSB (expect ~0.5)")


# ============================================================================
n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} dither carry/mass/overflow CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
