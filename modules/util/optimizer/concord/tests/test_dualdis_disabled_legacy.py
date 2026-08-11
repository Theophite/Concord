"""CPU bit-exactness test: DISABLED dual-dissipation == LEGACY single-int16 (point 7).

Reference under test:  modules/util/optimizer/concord/dual_dissipation_ref.py
Legacy source of truth: modules/util/optimizer/concord/prototype_packed_b.py:986-1110
                         (SR-tick 988-992, chase 1010/1026-1042, leak 1044-1060,
                          int16 clamp 1104, repack 1106-1110, consolidated_weight 2793-2800)

WHAT THIS FILE ASSERTS (point 7 / DISABLED == LEGACY, BIT-EXACT)
----------------------------------------------------------------
With the dual-dissipation feature OFF, the corrected reference must read the 16 fine
bits [31:16] as a SINGLE int16 s_fast (a mode switch, packed>>16) and run the legacy
chase / leak / evap / int16-clamp / SR dynamics VERBATIM; the deploy weight must be
pure coarse (s_slow+v_slow)*128. It must be BIT-EXACT to legacy:

  1. packed word AND live weight match a LITERAL legacy single-int16 reimplementation
     (written from scratch in THIS file -- it shares NO path-under-test code with the
     reference; only the immutable format primitives _hash_uniform / _sr_round /
     _scale_fwd / pack_legacy / unpack_legacy, which define the on-disk bit layout and
     the dither stream, are reused so the SR ticks line up) -- over MANY steps, from
     identical re-synced state each step.

  2. THE PER-BYTE-INT8-CLAMP CATCH (the headline of this file): values with
     |s_fast| in (127, 32767] must round-trip and evolve as ONE int16, NOT as two
     independent int8 bytes. An accidental per-byte int8 clamp (the WRONG #2 layout's
     failure mode) would saturate the high byte at +-127 and corrupt any s_fast whose
     magnitude exceeds 127. We seed s_fast directly to 128, 200, 1000, 16384, 32767,
     -128, -200, -1000, -32768 and assert:
        (a) unpack_legacy round-trips the int16 EXACTLY (no per-byte clamp on read);
        (b) one DISABLED step keeps the reunified s_fast bit-exact to the literal
            legacy int16 tick (no per-byte clamp on write);
        (c) the only clamp that ever bites the fine field is the int16 clamp at
            +-32767 (line 1104), never the int8 clamp at +-127.

  3. DEPLOY in disabled mode is pure coarse (s_slow+v_slow)*128*scale -- byte-identical
     to legacy consolidated_weight (2793-2800), with the e_H fraction term dropped.

  4. The M-guard collapse (M < M_MIN coerces enabled->False) is ALSO bit-exact legacy.

NO GPU. CPU-only (assume CUDA_VISIBLE_DEVICES=""). Pure torch + plain asserts, a
__main__ that prints PASS/FAIL per check. No pytest dependency.

Run:  CUDA_VISIBLE_DEVICES="" venv/Scripts/python.exe \
        modules/util/optimizer/concord/tests/test_dualdis_disabled_legacy.py
"""
import sys
from pathlib import Path

import torch

# ── import the reference (same convention as the sibling CPU tests) ──
OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import dual_dissipation_ref as ref
from dual_dissipation_ref import (
    DualDissipationLayer,
    # immutable FORMAT primitives only (bit layout + dither stream); NOT the
    # path under test. Reusing these is what makes the SR ticks line up so the
    # comparison is a HARD bit-exact check rather than a statistical one.
    pack_legacy,
    unpack_legacy,
    _scale_fwd,
    _hash_uniform,
    _sr_round,
    _pos_hash,
    INT8_MIN, INT8_MAX,
    INT16_MIN, INT16_MAX,
    CARRY,
    MANTISSA_BIAS,
    M_MIN,
)

torch.manual_seed(0)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# =====================================================================
# A LITERAL legacy single-int16 tick, re-implemented FROM SCRATCH in this
# test file (independent of the reference's own DualDissipationLayer._legacy_tick
# and of its legacy_single_int16_tick helper). It mirrors the kernel s_fast path
# prototype_packed_b.py:986-1110 expression-for-expression so the disabled
# reference can be asserted bit-exact against an outside witness.
#
# It deliberately reuses ONLY the format primitives (hash/SR/scale/pack) -- the
# things that DEFINE the bit layout and the dither stream. The dynamics
# (velocity, coherence, AdamW step, evap, chase, leak, mass-preserve, clamps,
# line order) are written out here independently.
# =====================================================================
@torch.no_grad()
def _coherence_legacy(d_fs, d_sv, vhat, drift_cancel_C, coh_kappa, sum_v_inv, N, K,
                      scale_fwd, use_coh_vhat):
    """Independent copy of the USE_FIXED_COH / USE_COH_VHAT coherence (823-846)."""
    sig_w = drift_cancel_C * d_sv * scale_fwd
    sig2 = sig_w * sig_w
    noise_w = (d_fs - drift_cancel_C * d_sv) * scale_fwd
    coh_n2 = noise_w * noise_w
    coh_raw = sig2 / (sig2 + coh_n2 + 1e-30)                 # 826 (dissipation)
    if use_coh_vhat:
        vhat_fl = torch.clamp(vhat, min=0.03 / (sum_v_inv * N * K))  # 838
        cf = sig2 / (drift_cancel_C * drift_cancel_C * vhat_fl + 1e-30)  # 839
        coh_n2 = coh_n2 * coh_kappa / (cf + coh_kappa)      # 840
    coh = sig2 / (sig2 + coh_n2 + 1e-30)                    # 841
    return torch.clamp(coh, 0.0, 1.0), torch.clamp(coh_raw, 0.0, 1.0)


@torch.no_grad()
def literal_legacy_tick(packed, row_exp, col_exp, grad_W, lr, *, alpha, gf_consol,
                        drift_cancel_C, alpha_v_fast, coh_kappa, v_scale, precond_p,
                        eps, step_cap, min_leak, evap_build_min, beta1, beta2,
                        use_coh_vhat, mass_preserve, chase_floor, leak_floor, consf,
                        v_row, v_col, step, seed, mantissa_bias=MANTISSA_BIAS):
    """ONE step of the literal legacy single-int16-s_fast path. Returns
    (packed, v_row, v_col). Reproduces prototype_packed_b.py:986-1110 expression order
    and -- crucially for this test -- clamps the FINE field as ONE int16 at +-32767
    (line 1104), never per byte. Written independently of the reference's own legacy
    code so the bit-exactness comparison has an outside witness."""
    N, K = packed.shape
    salt = (int(seed) ^ (int(step) * 0x9E3779B1)) & 0x7FFFFFFF
    pos = _pos_hash(N, K)

    # ── Adafactor rank-1 vhat (mirror 802-808) ──
    g2 = grad_W * grad_W
    v_row = beta2 * v_row + (1 - beta2) * g2.mean(dim=1)
    v_col = beta2 * v_col + (1 - beta2) * (g2.mean(dim=0) / (v_row.mean() + 1e-30))
    sum_v_inv = 1.0 / (v_row.sum() + 1e-30)
    v_bc = 1.0 / (1.0 - beta2 ** step)
    vhat = (v_row[:, None] * v_col[None, :] * sum_v_inv) * v_bc

    # ── unpack: [31:16] read as ONE int16 (line 758) ──
    s_fast, s_slow, v_slow = unpack_legacy(packed)
    s_fast = s_fast.to(torch.int32)
    s_slow = s_slow.to(torch.int32)
    v_slow = v_slow.to(torch.int32)

    scale_fwd = _scale_fwd(row_exp, col_exp, mantissa_bias)
    scale_inv = 1.0 / scale_fwd

    d_fs = s_fast.to(torch.float32)                                          # 778
    d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY     # 779
    coh, coh_raw = _coherence_legacy(d_fs, d_sv, vhat, drift_cancel_C, coh_kappa,
                                     sum_v_inv, N, K, scale_fwd, use_coh_vhat)

    noise_w = (d_fs - drift_cancel_C * d_sv) * scale_fwd                     # 787-788
    v_proxy = noise_w * noise_w * v_scale                                    # 789
    step_live = (grad_W / torch.pow(v_proxy + eps, precond_p)).clamp(-step_cap, step_cap)  # 869-871
    delta_grad = -lr * step_live * scale_inv                                 # 970

    if gf_consol > 0.0:
        evap_frac = torch.clamp(lr * gf_consol * (1.0 - coh_raw), max=1.0 - min_leak)  # 936
        p_build = torch.clamp(d_fs.abs() / (evap_build_min + 1e-30), max=1.0)          # 948
        r_build = _hash_uniform(s_fast, pos, salt ^ 0x42424242)                        # 949
        build_ok = (r_build < p_build).to(torch.float32)                               # 951
        evap_mantissa = evap_frac * d_fs * build_ok                                    # 952
    else:
        evap_mantissa = torch.zeros_like(d_fs)

    delta_t = delta_grad + consf * (beta1 * coh * d_fs - evap_mantissa)                 # 984
    # SR-tick the DELTA into s_fast (988-991), hash seeded from s_fast.
    tick_fast = _sr_round(delta_t, s_fast, pos, salt)
    s_fast = s_fast + tick_fast                                                          # 992

    # chase (1010 affine gate, 1026-1042). gate_gain = 1.
    gate = chase_floor + (1.0 - chase_floor) * coh                                       # 1010
    chase_mantissa = alpha * gate * s_fast.to(torch.float32) * consf                     # 1026
    tick_slow = _sr_round(chase_mantissa / float(CARRY), s_fast, pos, salt ^ 0x5A5A5A5A)  # 1028-1031
    s_slow = s_slow + tick_slow                                                          # 1041
    s_fast = s_fast - tick_slow * CARRY                                                  # 1042

    # leak -> v_slow (1044-1060)
    gap_v = (s_slow.to(torch.float32) * 128 - v_slow.to(torch.float32) * 128)            # 1048
    delta_v8 = alpha_v_fast * gap_v / 128.0 * consf                                      # 1049
    delta_v8 = delta_v8 * (leak_floor + (1.0 - leak_floor) * coh)                        # 1051
    tick_v8 = _sr_round(delta_v8, s_fast, pos, salt ^ 0x33335555)                        # 1052-1055
    v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)                       # 1056
    if mass_preserve:
        s_slow = s_slow - (v_slow_new - v_slow)                                          # 1060

    # ── THE clamp that matters for this test: int16 on the fine field (1104),
    #    int8 on s_slow (1105). NEVER per-byte int8 on s_fast. ──
    s_fast = s_fast.clamp(INT16_MIN, INT16_MAX)                                          # 1104
    s_slow = s_slow.clamp(INT8_MIN, INT8_MAX)                                            # 1105
    packed = pack_legacy(s_fast, s_slow, v_slow_new)                                     # 1106-1110
    return packed, v_row, v_col


# common optimizer kwargs shared by the Layer and the witness, with consolidation
# ON (gf_consol>0, chase/leak active) so the full legacy dynamics are exercised.
KW = dict(alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02, alpha_v_fast=0.001,
          coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
          min_leak=0.05, evap_build_min=128.0, beta1=0.0, beta2=0.999,
          use_coh_vhat=True, mass_preserve=True, chase_floor=0.1, leak_floor=0.05,
          consf=1.0)


def _live_weight_from_packed(packed, row_exp, col_exp):
    """Disabled-mode live weight = (s_slow*128 + s_fast + v_slow*128) * scale,
    reading [31:16] as ONE int16 (mirror get_weight 2772-2780, with s_fast=packed>>16).
    Independent of the reference's decode so the live-weight check has an outside witness."""
    s_fast, s_slow, v_slow = unpack_legacy(packed)
    m_eff = (s_slow.to(torch.float32) * 128
             + s_fast.to(torch.float32)
             + v_slow.to(torch.float32) * 128)
    return m_eff * _scale_fwd(row_exp, col_exp)


def _coarse_deploy_from_packed(packed, row_exp, col_exp):
    """Pure-coarse deploy = (s_slow + v_slow)*128*scale (mirror consolidated_weight
    2793-2800). The disabled reference deploy must equal this byte-for-byte."""
    _, s_slow, v_slow = unpack_legacy(packed)
    m_slow = (s_slow.to(torch.float32) + v_slow.to(torch.float32)) * 128
    return m_slow * _scale_fwd(row_exp, col_exp)


# =====================================================================
# CHECK 1 -- DISABLED Layer == literal legacy tick, packed word AND live weight,
#            per-step from re-synced identical state, over many steps.
# =====================================================================
def check_1_disabled_bitexact(seed=11, N=8, K=16, steps=80):
    torch.manual_seed(seed)
    W = torch.randn(N, K) * 0.05
    # enabled=False forces the disabled path by the FLAG (M>=M_MIN so it is not the
    # guard doing the collapse). The disabled path must never touch the e_L/e_H split.
    layer = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=seed)
    assert layer.enabled is False, "enabled=False must keep the disabled path"
    layer.load_weights(W)
    col_exp = layer.col_exp.clone()
    g = torch.randn(N, K) * 0.02

    max_packed_gap = 0
    max_sfast_gap = 0
    max_weight_gap = 0.0
    n_compared = 0
    for t in range(steps):
        packed_in = layer.packed_w.clone()
        re_in = layer.row_exp.clone()
        vr_in = layer.v_row.clone()
        vc_in = layer.v_col.clone()

        layer.step(g, lr=0.05, **KW)

        # re-exponent has no analogue in the literal tick; skip steps where it fired.
        if bool((layer.row_exp != re_in).any()):
            continue

        packed_leg, _, _ = literal_legacy_tick(
            packed_in, re_in, col_exp, g, 0.05,
            v_row=vr_in, v_col=vc_in, step=layer._step, seed=seed, **KW)

        packed_gap = int((layer.packed_w - packed_leg).abs().max())
        sf_a, _, _ = unpack_legacy(layer.packed_w)
        sf_b, _, _ = unpack_legacy(packed_leg)
        sfast_gap = int((sf_a - sf_b).abs().max())

        w_a = _live_weight_from_packed(layer.packed_w, re_in, col_exp)
        w_b = _live_weight_from_packed(packed_leg, re_in, col_exp)
        weight_gap = float((w_a - w_b).abs().max())

        max_packed_gap = max(max_packed_gap, packed_gap)
        max_sfast_gap = max(max_sfast_gap, sfast_gap)
        max_weight_gap = max(max_weight_gap, weight_gap)
        n_compared += 1

    ok = (max_packed_gap == 0 and max_sfast_gap == 0 and max_weight_gap == 0.0
          and n_compared > 0)
    check("disabled packed word + live weight bit-exact to legacy (per-step, many steps)",
          ok,
          f"compared {n_compared}/{steps} steps, max packed gap={max_packed_gap}, "
          f"max |s_fast diff|={max_sfast_gap}, max |W diff|={max_weight_gap:.1e}")


# =====================================================================
# CHECK 2 -- THE PER-BYTE-INT8-CLAMP CATCH.
#   Seed s_fast directly to |s_fast| in (127, 32767]; assert it survives as ONE
#   int16 on BOTH read and write, never split / saturated per byte at +-127.
# =====================================================================
S_FAST_BIG = [128, 129, 200, 255, 256, 1000, 4096, 16384, 32767,
              -128, -129, -200, -256, -1000, -4096, -16384, -32768]


def check_2a_unpack_int16_roundtrip():
    """unpack_legacy(pack_legacy(s_fast,...)) must round-trip the FULL int16 with NO
    per-byte int8 clamp on READ -- the headline values |s_fast| in (127, 32767]."""
    vals = torch.tensor(S_FAST_BIG, dtype=torch.int32)
    s_fast = vals.reshape(1, -1)
    s_slow = torch.full_like(s_fast, 7)
    v_slow = torch.full_like(s_fast, -9)
    packed = pack_legacy(s_fast, s_slow, v_slow)
    sf_back, ss_back, vs_back = unpack_legacy(packed)
    ok = (torch.equal(sf_back, s_fast)
          and torch.equal(ss_back, s_slow)
          and torch.equal(vs_back, v_slow))
    # an accidental per-byte int8 clamp would map e.g. 200 -> high byte 0, low byte -56.
    over127 = int((sf_back.abs() > 127).sum())
    check("unpack_legacy round-trips |s_fast| in (127,32767] as ONE int16 (no per-byte clamp on read)",
          ok, f"{over127}/{sf_back.numel()} test values have |s_fast|>127, all preserved exactly")


def check_2b_disabled_step_preserves_big_s_fast(seed=23, steps=40):
    """One+ DISABLED step from a state whose s_fast is seeded large must keep the
    reunified s_fast bit-exact to the literal legacy int16 tick. If the disabled path
    clamped per-byte int8, any |s_fast|>127 would corrupt immediately. We use a tiny
    gradient and a frozen big-s_fast init so the values genuinely live in (127,32767]."""
    N, K = 1, len(S_FAST_BIG)
    # Build a packed word DIRECTLY with the big s_fast values. Keep s_slow/v_slow
    # small and equal-ish so d_sv is modest and the chase doesn't instantly drain
    # everything; the point is that the int16 magnitude survives the step machinery.
    s_fast0 = torch.tensor(S_FAST_BIG, dtype=torch.int32).reshape(N, K)
    s_slow0 = torch.full((N, K), 3, dtype=torch.int32)
    v_slow0 = torch.full((N, K), 3, dtype=torch.int32)
    packed0 = pack_legacy(s_fast0, s_slow0, v_slow0)

    layer = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=seed)
    assert layer.enabled is False
    # Give the layer a coherent exponent (load a weight to set row/col exp), then
    # overwrite the packed word with our big-s_fast init.
    layer.load_weights(torch.randn(N, K) * 0.05)
    row_exp = layer.row_exp.clone()
    col_exp = layer.col_exp.clone()
    layer.packed_w = packed0.clone()
    layer.v_row.zero_()
    layer.v_col.zero_()
    layer._step = 0

    g = torch.randn(N, K) * 1e-4   # tiny grad: don't swamp the seeded s_fast

    saw_over127 = False
    max_packed_gap = 0
    max_sfast_gap = 0
    only_int16_clamp_ever = True
    n_compared = 0
    for t in range(steps):
        packed_in = layer.packed_w.clone()
        re_in = layer.row_exp.clone()
        vr_in = layer.v_row.clone()
        vc_in = layer.v_col.clone()

        # the value being carried into this step, read as ONE int16
        sf_pre, _, _ = unpack_legacy(packed_in)
        if int(sf_pre.abs().max()) > 127:
            saw_over127 = True
        # the fine field must never have left the int16 range. BUG-5: use a SIGNED range
        # check, NOT abs() > INT16_MAX -- the valid boundary value s_fast == INT16_MIN
        # (-32768) has abs 32768 > INT16_MAX (32767), so the abs() form wrongly flags the
        # exact int16 floor as "unbounded". The reference clamps to [INT16_MIN, INT16_MAX],
        # so -32768 is a legitimate int16 (and bit-exact to legacy); accept it.
        if int(sf_pre.min()) < INT16_MIN or int(sf_pre.max()) > INT16_MAX:
            only_int16_clamp_ever = False

        layer.step(g, lr=1e-3, **KW)

        if bool((layer.row_exp != re_in).any()):
            continue
        packed_leg, _, _ = literal_legacy_tick(
            packed_in, re_in, col_exp, g, 1e-3,
            v_row=vr_in, v_col=vc_in, step=layer._step, seed=seed, **KW)

        packed_gap = int((layer.packed_w - packed_leg).abs().max())
        sf_a, _, _ = unpack_legacy(layer.packed_w)
        sf_b, _, _ = unpack_legacy(packed_leg)
        sfast_gap = int((sf_a - sf_b).abs().max())
        # the reference's fine field must stay within int16 (the legitimate clamp),
        # and must NOT be silently pinned at the int8 ceiling when the witness is larger.
        # BUG-5: signed range check (the valid INT16_MIN==-32768 boundary has abs 32768 >
        # INT16_MAX; accept it as in-range rather than flagging it via abs() > INT16_MAX).
        if int(sf_a.min()) < INT16_MIN or int(sf_a.max()) > INT16_MAX:
            only_int16_clamp_ever = False
        max_packed_gap = max(max_packed_gap, packed_gap)
        max_sfast_gap = max(max_sfast_gap, sfast_gap)
        n_compared += 1

    ok = (saw_over127 and n_compared > 0
          and max_packed_gap == 0 and max_sfast_gap == 0 and only_int16_clamp_ever)
    check("DISABLED step preserves |s_fast| in (127,32767] as int16, bit-exact to legacy "
          "(catches a per-byte int8 clamp on write)",
          ok,
          f"saw |s_fast|>127 = {saw_over127}, compared {n_compared}/{steps}, "
          f"max packed gap={max_packed_gap}, max |s_fast diff|={max_sfast_gap}, "
          f"fine field stayed int16-bounded = {only_int16_clamp_ever}")


def check_2c_int16_clamp_not_int8(seed=31):
    """Drive s_fast straight at the int16 ceiling and confirm the ONLY clamp that
    bites the fine field is the int16 clamp at +-32767 -- never the int8 clamp at
    +-127. We push a strong same-sign grad with chase OFF (gf_consol=0, alpha tiny)
    so s_fast accumulates toward saturation, and assert it climbs WELL past 127 and
    pins at the int16 boundary, matching the literal legacy tick."""
    N, K = 1, 4
    layer = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=seed)
    layer.load_weights(torch.full((N, K), 0.5))   # set a sane exponent
    row_exp = layer.row_exp.clone()
    col_exp = layer.col_exp.clone()
    # seed s_fast near the ceiling so a few same-sign steps push it to clamp.
    s_fast0 = torch.full((N, K), 32000, dtype=torch.int32)
    s_slow0 = torch.zeros((N, K), dtype=torch.int32)
    v_slow0 = torch.zeros((N, K), dtype=torch.int32)
    layer.packed_w = pack_legacy(s_fast0, s_slow0, v_slow0)
    layer.v_row.zero_(); layer.v_col.zero_(); layer._step = 0

    # accumulation kwargs: NO consolidation (chase/leak/evap off) so s_fast is the
    # only thing that moves and it ratchets toward the int16 ceiling.
    kw_accum = dict(KW)
    kw_accum.update(gf_consol=0.0, alpha=0.0, alpha_v_fast=0.0)
    g = torch.full((N, K), -50.0)   # large same-sign grad -> push s_fast up

    saw_above127 = False
    max_sfast_gap = 0
    pinned_at_int16 = False
    for t in range(30):
        packed_in = layer.packed_w.clone()
        re_in = layer.row_exp.clone()
        vr_in = layer.v_row.clone(); vc_in = layer.v_col.clone()
        layer.step(g, lr=0.5, **kw_accum)
        sf_now, _, _ = unpack_legacy(layer.packed_w)
        if int(sf_now.abs().max()) > 127:
            saw_above127 = True
        if int(sf_now.abs().max()) == INT16_MAX:
            pinned_at_int16 = True
        if bool((layer.row_exp != re_in).any()):
            continue
        packed_leg, _, _ = literal_legacy_tick(
            packed_in, re_in, col_exp, g, 0.5,
            v_row=vr_in, v_col=vc_in, step=layer._step, seed=seed, **kw_accum)
        sf_a, _, _ = unpack_legacy(layer.packed_w)
        sf_b, _, _ = unpack_legacy(packed_leg)
        max_sfast_gap = max(max_sfast_gap, int((sf_a - sf_b).abs().max()))

    # the fine field must never have been clamped to int8 (it visibly exceeds 127 and
    # reaches the int16 ceiling), and it tracks the literal legacy tick bit-exactly.
    sf_final, _, _ = unpack_legacy(layer.packed_w)
    final_mag = int(sf_final.abs().max())
    ok = (saw_above127 and pinned_at_int16 and final_mag == INT16_MAX
          and max_sfast_gap == 0)
    check("fine field clamps at int16 (+-32767), NOT int8 (+-127); bit-exact to legacy",
          ok,
          f"saw |s_fast|>127 = {saw_above127}, reached int16 ceiling = {pinned_at_int16}, "
          f"final |s_fast|={final_mag} (==32767), max |s_fast diff| vs legacy={max_sfast_gap}")


# =====================================================================
# CHECK 3 -- disabled deploy is PURE COARSE (s_slow+v_slow)*128 (byte-identical legacy).
# =====================================================================
def check_3_disabled_deploy_pure_coarse(seed=41, N=6, K=10, steps=50):
    torch.manual_seed(seed)
    layer = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=seed)
    layer.load_weights(torch.randn(N, K) * 0.05)
    g = torch.randn(N, K) * 0.02
    for t in range(steps):
        layer.step(g, lr=0.05, **KW)

    dep = layer.deploy_weight()                              # disabled -> pure coarse
    coarse = _coarse_deploy_from_packed(layer.packed_w, layer.row_exp, layer.col_exp)
    # atol=0: must be byte-identical, no fraction term, no s_fast contribution.
    ok = torch.equal(dep, coarse)
    gap = float((dep - coarse).abs().max())
    check("disabled deploy == pure coarse (s_slow+v_slow)*128*scale (byte-identical legacy)",
          ok, f"max |deploy - coarse| = {gap:.1e} (== 0)")

    # and it must NOT equal the live weight whenever s_fast != 0 (deploy DROPS s_fast).
    live = _live_weight_from_packed(layer.packed_w, layer.row_exp, layer.col_exp)
    sf, _, _ = unpack_legacy(layer.packed_w)
    if int(sf.abs().sum()) > 0:
        differs = not torch.allclose(dep.to(torch.float32), live.to(torch.float32), atol=0)
        check("disabled deploy DROPS s_fast (deploy != live when s_fast != 0)",
              differs, f"max|s_fast|={int(sf.abs().max())}")
    else:
        check("disabled deploy DROPS s_fast (deploy != live when s_fast != 0)",
              True, "s_fast settled to 0 (vacuously holds)")


# =====================================================================
# CHECK 4 -- M-guard collapse (M < M_MIN coerces enabled->False) is ALSO bit-exact
#            legacy. Confirms the collapse path is the SAME disabled int16 dynamics.
# =====================================================================
def check_4_mguard_collapse_bitexact(seed=53, N=6, K=10, steps=40):
    torch.manual_seed(seed)
    layer = DualDissipationLayer(N, K, enabled=True, grad_accum_M=2, seed=seed)  # M<M_MIN
    coerced = (layer.enabled is False)
    layer.load_weights(torch.randn(N, K) * 0.05)
    col_exp = layer.col_exp.clone()
    g = torch.randn(N, K) * 0.02

    max_packed_gap = 0
    n_compared = 0
    for t in range(steps):
        packed_in = layer.packed_w.clone()
        re_in = layer.row_exp.clone()
        vr_in = layer.v_row.clone(); vc_in = layer.v_col.clone()
        layer.step(g, lr=0.05, **KW)
        if bool((layer.row_exp != re_in).any()):
            continue
        packed_leg, _, _ = literal_legacy_tick(
            packed_in, re_in, col_exp, g, 0.05,
            v_row=vr_in, v_col=vc_in, step=layer._step, seed=seed, **KW)
        max_packed_gap = max(max_packed_gap, int((layer.packed_w - packed_leg).abs().max()))
        n_compared += 1

    ok = (coerced and n_compared > 0 and max_packed_gap == 0)
    check("M-guard (M<M_MIN) collapses to legacy AND is bit-exact to the literal tick",
          ok,
          f"enabled coerced to False = {coerced} (M_MIN={M_MIN}), compared {n_compared}/{steps}, "
          f"max packed gap={max_packed_gap}")


# =====================================================================
if __name__ == "__main__":
    print("test_dualdis_disabled_legacy.py  (CPU; DISABLED dual-dissipation == legacy int16)")
    print("-" * 78)

    print("CHECK 1: disabled bit-exact to literal legacy tick (packed + live weight)")
    check_1_disabled_bitexact()

    print("CHECK 2: per-byte-int8-clamp catch -- |s_fast| in (127, 32767]")
    check_2a_unpack_int16_roundtrip()
    check_2b_disabled_step_preserves_big_s_fast()
    check_2c_int16_clamp_not_int8()

    print("CHECK 3: disabled deploy is pure coarse")
    check_3_disabled_deploy_pure_coarse()

    print("CHECK 4: M-guard collapse is bit-exact legacy")
    check_4_mguard_collapse_bitexact()

    print("-" * 78)
    n_pass = sum(results)
    n_tot = len(results)
    if all(results):
        print(f"ALL {n_tot} CHECKS PASS")
        sys.exit(0)
    else:
        print(f"{n_pass}/{n_tot} CHECKS PASS -- {n_tot - n_pass} FAILED")
        sys.exit(1)
