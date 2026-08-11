"""Concord 2-fast engine: the split-tick bracket in the production word.

Storage layout (little-endian):
    bits [31:24]  e_L        int8  — low-friction fine arm,  arm units
    bits [23:16]  e_H        int8  — high-friction fine arm, arm units
    bits [15:8]   s_slow_i8  int8  — coarse position bearer, × 128
    bits [ 7:0]   v_slow_i8  int8  — long-time anchor,       × 128

The two arms replace prototype_packed_b's int16 s_fast. Every
preconditioned tick deposits HALF into each arm (one backward — the arms
share the gradient); the arms dissipate at bracketed rates
lr·gf·(1∓d)·(1−coh). The gap e_L−e_H is a per-coordinate dW/dlam meter
and the sum's variance a noise-power meter (READ-ONLY in v1 — no control
law; CPU validation: research branch exps 15–22, concord/core2.py).
d = 0 collapses to the single-accumulator dynamics.

Arm-plane exponents: one arm unit = 2^(arm_row_exp + arm_col_exp) fine
units (per-row + per-col int8 deltas, mirroring the coarse plane). The
joint cap ar+ac ≤ +7 keeps one coarse unit an integer number of arm
units, so every cross-plane transfer is integer-exact; negative sums put
the arm grid BELOW one fine unit — sub-scale accumulation held in value.
The ratchet: the apply kernel watermarks the PRE-clamp per-row/col arm
max; GatedArmRatchet ticks up at 96 (SR-halve) urgently and down at 24
(lossless double) on a lazy cadence, 4× hysteresis. Two in-kernel valves
handle single-step spikes: the carry (arm-sum ≥ 160 moves whole coarse
units into s_slow — mass-preserving, integer-exact, live-weight-
invariant, so it is safe mid-accumulation under a frozen weight_buf) and
the gap fold (|gap| ≥ 96 halves the meter plane, sum-preserving).

ARCHITECTURE: this module contains ONLY the 2-fast kernels and layers.
All shared machinery — the module-global schedules (_MIN_LEAK,
_EVAP_BUILD_MIN, _LAZY_THRESH, _COH_KAPPA, _EVAP_SLACK, _GATE_GAIN,
ratio floors, sigmag noise, consolidate flag), the boil/memgap meters,
LAMB trust buffers, setters, and the autotuner — lives in
prototype_packed_b and is read from it AT LAUNCH TIME, so the existing
winner_step schedule and every set_* call drive both kernels from one
source of truth. prototype_packed_b itself is untouched (file-level A/B).

v1 limitations (enforced, not silent): AdamW only; apply_grad_step
(the embedding self-step path) raises; the fused dequant-matmul IS
supported (fused_packed2_linear/gradx — required at the 24 GB ceiling:
the cached path's per-layer bf16 buffers oversubscribe the card); the
GAP_FEEDBACK experimental path runs the arms UNBRACKETED (the conserved
pass/evap split and the bracket both modulate the same rate; composing
them is unvalidated).

New stochastic-rounding salts: arm-H tick 0x1F2F3F4F, chase-debit split
0x4D4D2B2B, wd_sf arm H 0x7777FFFF, wd_anchor arm H 0x2468FFFF, carry
split 0x6E6E1111, gap fold 0x3C3C9999. All inherited salts unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import prototype_packed_b as _pb
from prototype_packed_b import (
    INT8_MIN, INT8_MAX, INT16_MIN, INT16_MAX,
    _hash_uniform, compute_drift_cancel_C,
    _get_step_counter, _get_consolidate_flag, _get_arm_sel_flag,
    _ensure_lr_tensor, _ensure_eps_tensor,
    _ensure_floor_tensors, _v_bc_buf, _vhat_mean_buf,
    _lamb_scale_buf, _lamb_wnorm_sq_buf, _lamb_stepnorm_sq_buf,
    _boil_buf, _memgap_buf, _lookup_layer_meters,
    ConcordLinearPackedB,
)

CARRY_AT = 160
FOLD_AT = 96
ARM_UP_AT, ARM_DN_AT = 96, 24     # ratchet thresholds (4x hysteresis)
# -- GMR+ gap-ratchet constants (both channels OFF by default; see
# docs/GAP_RATCHET_DESIGN.md in the research repo) --
ARM_GAP_UP_MED = 32   # Phase A up: row/col |gap| MEDIAN at/above this = CHRONIC fold
                      # pressure. Derivation: under fold recycling (SR-halve at 96,
                      # re-inject ~+/-48) the stationary |gap| median is ~36; healthy
                      # chase-active rows sit at a few LSB (OU gap). Model-fitted (wd/evap
                      # contraction shifts the ~36) -- the USE_GAP_WM watermark channel is
                      # the model-independent backstop.
ARM_GAP_DN_MED = 8    # = ARM_GAP_UP_MED/4: the plane's own 4x hysteresis ratio (96/24).
GAP_DN_SAFE = FOLD_AT // 2 - 1  # DERIVED (=47): largest |gap| whose lossless down-tick
                      # double (94) lands strictly below FOLD_AT -- a down-tick can never
                      # place a value inside the fold trigger. Moves with FOLD_AT.
# -- Flag 1: slow-plane rail guard (docs/SPLIT_EXPONENT_DESIGN.md, research
# repo; OFF by default). exp80l measured a 16% s_slow rail episode invisible
# to the combined-mantissa trigger: |s_slow| alone maxes at 127*128 = 16256
# < MAX_M = 24000, so a railed settled field can never trip dispatch. --
SLOW_RAIL_AT = 96     # |s_slow| at/above this = rail proximity; the 31-LSB
                      # protected band above it sizes Flag 2's transfer chunks
RAIL_SENTINEL = 24001  # MAX_M + 1 for the repo's fixed MAX_M=24000 word
                      # budget (coupled constant, CLAUDE.md invariant 3): the
                      # sentinel trips the peak > MAX_M dispatch gate AND
                      # blocks the decide's row_max <= MAX_M down path.
ARM_EXP_MIN, ARM_EXP_MAX = -8, 7  # per-axis bounds
ARM_SUM_MAX = 7                   # joint cap: transfer integrality
ARM_RATCHET_EVERY = 32            # ratchet cadence (steps); see GatedArmRatchet


# ============================================================
# Materialize (2-fast unpack)
# ============================================================

@triton.jit
def _materialize_packed2_bf16_kernel(
    packed_ptr, weight_ptr, row_exp_ptr, col_exp_ptr,
    arm_row_exp_ptr, arm_col_exp_ptr,
    N, K, mantissa_bias,
    stride_pn, stride_pk, stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    e_L       = packed >> 24
    e_H       = (packed << 8) >> 24
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24
    ar = tl.load(arm_row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    ac = tl.load(arm_col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    ascale = tl.exp2((ar[:, None] + ac[None, :]).to(tl.float32))
    m_eff_f = ((s_slow_i8 + v_slow_i8) * 128).to(tl.float32) \
        + (e_L + e_H).to(tl.float32) * ascale

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_ptr + w_off,
             (m_eff_f * tl.exp2(exp)).to(tl.bfloat16), mask=nk_mask)


def materialize_packed2_bf16(packed_w, row_exp, col_exp,
                             arm_row_exp, arm_col_exp, out,
                             mantissa_bias=15):
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.shape == packed_w.shape
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _materialize_packed2_bf16_kernel[grid](
        packed_w, out, row_exp, col_exp, arm_row_exp, arm_col_exp,
        N, K, int(mantissa_bias),
        packed_w.stride(0), packed_w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out


# ============================================================
# Apply kernel (AdamW; mirrors _pb._apply_packed_adamw_kernel with
# the arm surgery — see the module docstring for the map)
# ============================================================

@triton.jit
def _apply_packed2_adamw_kernel(
    packed_ptr, grad_W_ptr, weight_buf_ptr,
    row_exp_ptr, col_exp_ptr,
    arm_row_exp_ptr, arm_col_exp_ptr,
    arm_row_max_ptr, arm_col_max_ptr,
    row_max_ptr, col_max_ptr,
    v_row_ptr, v_col_ptr, sum_v_inv_ptr,
    coh_pre_ptr,
    v_full_ptr, vhat_mean_ptr,
    N, K,
    lr_ptr, mantissa_bias, alpha, beta1_ptr,
    weight_decay, eps_ptr, step_cap,
    lazy_thresh,
    v_scale, precond_p, gf_consol_ptr,
    d_ptr,               # *fp32[1]: bracket half-width (per-layer device
                         # tensor; live under CUDA graphs; 0 = legacy rule)
    drift_cancel_C, alpha_v_fast,
    coh_kappa, evap_slack, perfcoh_tau,
    wd_sv, wd_sf, wd_anchor,
    gf_trust_delta_sq,
    min_leak, evap_build_min,
    gate_gain,
    chase_floor_ptr, leak_floor_ptr,
    v_bc_ptr,
    memgap_ptr, boil_ptr,
    cohq_ptr,          # *fp32[16]: realized-coherence CDF sketch (15 thresholds + count)
    preq_ptr,            # *fp32[4]: prequential servo meter (WRITE_PREQ; [3]=gross)
    gap_inv_scale,
    step_salt_ptr,
    consolidate_ptr,
    arm_sel_ptr,         # *fp32[1]: held-out router selector (1=tick to L, 0=to H);
                         # trainer-filled per micro OUTSIDE capture (invariant 5)
    step_scale_ptr, wnorm_sq_ptr, stepnorm_sq_ptr,
    stride_pn, stride_pk,
    stride_gn, stride_gk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    APPLY_CHASE: tl.constexpr,
    TRACK_REBALANCE: tl.constexpr,
    USE_GF_TRUST_REGION: tl.constexpr,
    USE_GF_CONSOLIDATION: tl.constexpr,
    USE_COHPRE: tl.constexpr,
    USE_FIXED_COH: tl.constexpr,
    USE_COH_VHAT: tl.constexpr,
    USE_RATIO_COH: tl.constexpr,
    USE_GAP_FEEDBACK: tl.constexpr,
    USE_LAZY_GATE: tl.constexpr,
    USE_GRAD_ACTIVITY: tl.constexpr,
    WRITE_BOIL: tl.constexpr,
    WRITE_PREQ: tl.constexpr,
    USE_EVICT: tl.constexpr,
    EVICT_GAIN: tl.constexpr,
    USE_EVICT_CF: tl.constexpr,
    WRITE_LAMB_NORMS: tl.constexpr,
    USE_FULL_V: tl.constexpr,
    USE_HELDOUT_ROUTER: tl.constexpr,
    USE_ROUTER_NOISE: tl.constexpr,
    USE_PERFCOH: tl.constexpr,
    USE_GAP_WM: tl.constexpr,    # GMR+ Phase B: pre-valve gap-demand watermark term
    USE_RAIL_GUARD: tl.constexpr,  # Flag 1: rail sentinel + headroom clamps
    USE_RAIL_SPILL: tl.constexpr,  # Flag 1 sub-flag: clamp-spill on ALL paths
    USE_SOVEREIGN: tl.constexpr,   # Flag 2: sovereign arm scale (converter live)
    CF_PER_ROW: tl.constexpr,
    LF_PER_ROW: tl.constexpr,
    GF_PER_ROW: tl.constexpr,     # True -> gf_consol_ptr is [N] per-row kappas; False -> [1] scalar
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    lr = tl.load(lr_ptr).to(tl.float32)
    step_scale = tl.load(step_scale_ptr)
    lr_eff = lr * step_scale
    eps = tl.load(eps_ptr).to(tl.float32)
    beta1 = tl.load(beta1_ptr).to(tl.float32)
    if not GF_PER_ROW:
        gf_consol = tl.load(gf_consol_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    if not CF_PER_ROW:
        chase_floor = tl.load(chase_floor_ptr).to(tl.float32)
    if not LF_PER_ROW:
        leak_floor = tl.load(leak_floor_ptr).to(tl.float32)
    cons = tl.load(consolidate_ptr).to(tl.int32)
    consf = cons.to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]
    # per-row incoherent-admission floors: the gate/leak floor becomes an [N]
    # device buffer read per row ([BLOCK_N,1] broadcasts over the gate math).
    # ABSOLUTE values, not multipliers -- an armed layer's floors are owned by
    # whoever armed them (refresh by device-tensor fill, graph-safe); the
    # winner schedule's scalar no longer drives that layer. Unarmed layers
    # compile this branch out and are byte-identical to the scalar path.
    if CF_PER_ROW:
        chase_floor = tl.load(chase_floor_ptr + offs_n, mask=n_mask,
                              other=0.0).to(tl.float32)[:, None]
    if LF_PER_ROW:
        leak_floor = tl.load(leak_floor_ptr + offs_n, mask=n_mask,
                             other=0.0).to(tl.float32)[:, None]
    if GF_PER_ROW:
        # per-ROW dissipation (packed_b's emb pattern, now for UNet rows): the layer's
        # _gf_consol_buf is [N], selected by BUFFER SIZE at launch (numel > 1) -- no new
        # argument; [:, None] broadcasts over the evap base exactly as the scalar did
        gf_consol = tl.load(gf_consol_ptr + offs_n, mask=n_mask,
                            other=0.0).to(tl.float32)[:, None]

    # ── load packed + unpack (2-fast word) ─────────────────────
    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    e_L       = packed >> 24
    e_H       = (packed << 8) >> 24
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24
    s_slow_full = s_slow_i8 * 128
    v_slow_full = v_slow_i8 * 128

    g_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + g_off, mask=nk_mask, other=0.0).to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_fwd = tl.exp2(total_exp)
    scale_inv = tl.exp2(-total_exp)
    # Arm-plane exponent: one arm unit = 2^(ar+ac) fine units; dynamics
    # compute in FINE units and convert at the arm boundary (exact powers
    # of two; ar+ac <= ARM_SUM_MAX keeps cross-plane transfers integer).
    ar = tl.load(arm_row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    ac = tl.load(arm_col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    arm_e = (ar[:, None] + ac[None, :]).to(tl.float32)
    ascale = tl.exp2(arm_e)
    ainv = tl.exp2(-arm_e)

    # ── velocity (the ARM SUM plays s_fast's role) + drift-cancel ──
    fine = e_L + e_H
    d_fs = fine.to(tl.float32) * ascale
    d_sv = (s_slow_full - v_slow_full).to(tl.float32)

    # memorization-gap meter (first-order L_live - L_deploy; unchanged
    # expression — d_fs is the fine-unit velocity here as there).
    tl.atomic_add(memgap_ptr, tl.sum(grad_W * d_fs * scale_fwd))
    if WRITE_PREQ:
        # Prequential servo meter: <g, A_gap> with the gap in WEIGHT units
        # at the live arm scale -- scale-honest across reticks (SR-halve
        # preserves value). CORRECTED 2026-07-25 (code-read audit): an
        # earlier version of this comment claimed the carry debit is
        # complement-split and gap-preserving -- SOURCE-FALSE. The carry
        # debit splits PROPORTIONALLY to arm content (wL2 = e_L/fine), so
        # E[gap'] = gap*(1 - debit2/fine): up to ~5x gap collapse per fire
        # at the |sum|>=160 threshold. Only the EVICTION credit is a true
        # 0.5-complement. Readings of this meter (or the hybrid G^2 floor)
        # taken just after a carry transient are biased LOW. Sign law for the (1-/+d) rate bracket: +A_gap = less
        # dissipation, so window cosine s < 0 => lower kappa reduces
        # in-stream loss, s > 0 => raise. THIS SIGN LAW ASSUMES THE RATE
        # BRACKET AUTHORS THE GAP: under USE_HELDOUT_ROUTER the gap is
        # dominated by the cross-split deposit term instead, and the
        # cosine is no longer a dissipation derivative -- never feed a
        # dissipation controller from a routed gap. The in-stream signal is
        # sign-faithful vs a true holdout but ~30% attenuated by
        # cross-epoch memorization (exp 24) -- consume with a margin >=
        # the exp-23 servo margin. Slots: [0] sum g*gap, [1] sum g^2,
        # [2] sum gap^2; cosine = [0]/sqrt([1]*[2]). Slot [3] is the GROSS
        # alignment sum|g*gap|: |slot0| << slot3 means the per-coordinate
        # signal cancels across the layer (a per-block readout could recover
        # it); slot0 ~ slot3 both tiny means genuinely no signal. Pure reads +
        # atomics into a separate buffer: the update math is untouched, and
        # with no meter registered this branch compiles out entirely.
        gap_w = (e_L - e_H).to(tl.float32) * ascale * scale_fwd
        gp = grad_W * gap_w
        tl.atomic_add(preq_ptr, tl.sum(gp))
        tl.atomic_add(preq_ptr + 1, tl.sum(grad_W * grad_W))
        tl.atomic_add(preq_ptr + 2, tl.sum(gap_w * gap_w))
        tl.atomic_add(preq_ptr + 3, tl.sum(tl.abs(gp)))     # gross alignment -> cancellation probe
    noise = d_fs - drift_cancel_C * d_sv
    noise_in_w = noise * scale_fwd
    v_proxy = noise_in_w * noise_in_w * v_scale

    coh = 0.0
    ev_cf_warm = 1.0   # eviction cf-gate warmth (exp 26f); 1.0 = no gate unless USE_EVICT_CF
    coh_raw = 0.0
    if (USE_GF_TRUST_REGION or USE_GF_CONSOLIDATION) or (USE_COHPRE or USE_RATIO_COH) or USE_LAZY_GATE:
        if USE_FULL_V:
            v_full_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            v_hat = tl.load(v_full_ptr + v_full_off,
                            mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            sum_v_inv = tl.load(sum_v_inv_ptr).to(tl.float32)
        else:
            v_row_tile = tl.load(v_row_ptr + offs_n,
                                 mask=n_mask, other=0.0).to(tl.float32)
            v_col_tile = tl.load(v_col_ptr + offs_k,
                                 mask=k_mask, other=0.0).to(tl.float32)
            sum_v_inv = tl.load(sum_v_inv_ptr).to(tl.float32)
            v_hat = v_row_tile[:, None] * v_col_tile[None, :] * sum_v_inv
        v_hat = v_hat * tl.load(v_bc_ptr).to(tl.float32)
    if USE_GF_TRUST_REGION:
        v_proxy = v_proxy + gf_trust_delta_sq * v_hat
    if USE_GF_CONSOLIDATION or (USE_COHPRE or USE_RATIO_COH):
        if USE_FIXED_COH:
            sig_w = drift_cancel_C * d_sv * scale_fwd
            sig2 = sig_w * sig_w
            coh_n2 = noise_in_w * noise_in_w
            coh_raw = sig2 / (sig2 + coh_n2 + 1e-30)
            if USE_COH_VHAT:
                if USE_FULL_V:
                    vhat_fl = tl.maximum(v_hat, 0.03 * tl.load(vhat_mean_ptr).to(tl.float32))
                else:
                    vhat_fl = tl.maximum(v_hat, 0.03 / (sum_v_inv * N * K))
                cf = sig2 / (drift_cancel_C * drift_cancel_C * vhat_fl + 1e-30)
                coh_n2 = coh_n2 * coh_kappa / (cf + coh_kappa)
                if USE_EVICT_CF:
                    # gate the WHOLE cf-eviction by CF warmth: x the window-fill
                    # 1/v_bc = (1-beta2^t) so it ramps from 0 (CF cold) to the cf-gate
                    # (warm) -- trust neither eviction nor suppression until CF fills its
                    # window. Neutral at a production 1-epoch v_hat window (exp 26g).
                    ev_cf_warm = (cf / (cf + coh_kappa)) / tl.load(v_bc_ptr).to(tl.float32)
            if USE_ROUTER_NOISE:
                # Router cross-split disagreement as an UN-DISCOUNTABLE noise floor,
                # added AFTER the cf discount: cf may lift coherence by SNR magnitude
                # but cannot erase a live contradiction between the two data halves.
                # Temporal coherence is foolable by persistent (memorized) pushes;
                # cross-split agreement is not -- a memorized example's ticks land in
                # one arm and the other, integrating the rest of the data, never
                # corroborates. Only meaningful with the held-out router on (the gap
                # is then a data-split meter, not the rate-bracket derivative).
                # Weight units, matching noise_in_w. exp47b [epic-williamson lineage].
                gap_noise = (e_L - e_H).to(tl.float32) * ascale * scale_fwd
                coh_n2 = coh_n2 + gap_noise * gap_noise
            coh = sig2 / (sig2 + coh_n2 + 1e-30)
        else:
            mean_grad_w = alpha_v_fast * d_sv * scale_fwd
            coh = mean_grad_w * mean_grad_w / (v_hat + 1e-12)
            coh_raw = coh
        coh = tl.minimum(tl.maximum(coh, 0.0), 1.0)
        coh_raw = tl.minimum(tl.maximum(coh_raw, 0.0), 1.0)
        # Realized-coherence CDF sketch (the tau-calibration meter,
        # docs/COH_TIMESTEP_CALIBRATION.md): counts of coh below 15 uniform
        # thresholds i/16, accumulated from a fixed 1-in-16 sample of tiles
        # (pid hash -- deterministic, capture-stable). This is the GATE'S OWN
        # input distribution (post cf discount, post router floor): its
        # quantiles are what any anchored knee would read, and its ceiling
        # measures the timestep-mixture compression against the ~1 ideal.
        # ~16 atomics per sampled tile; the host reads-and-zeros on the
        # health-line cadence (read_cohq).
        if ((pid_n * 1103515245 ^ pid_k) & 15) == 0:   # int32-safe LCG hash
            for _q in tl.static_range(15):
                _th = (_q + 1) * 0.0625
                tl.atomic_add(cohq_ptr + _q,
                              tl.sum(((coh < _th) & nk_mask).to(tl.float32)))
            tl.atomic_add(cohq_ptr + 15, tl.sum(nk_mask.to(tl.float32)))
        if USE_GAP_FEEDBACK:
            c_pass = tl.minimum(coh + tl.exp(-tl.abs(d_sv) * gap_inv_scale), 1.0)

    # ── AdamW step (shared tick; deposited half per arm below) ─
    denom_p = tl.exp2(precond_p * tl.log2(v_proxy + eps))
    step_live = grad_W / denom_p
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    if WRITE_LAMB_NORMS:
        w_deploy = (s_slow_full + v_slow_full).to(tl.float32) * scale_fwd
        tl.atomic_add(wnorm_sq_ptr, tl.sum(w_deploy * w_deploy))
        tl.atomic_add(stepnorm_sq_ptr, tl.sum(step_live * step_live))
    s_fast_in_w = d_fs * scale_fwd
    g_active = 1.0
    if USE_GRAD_ACTIVITY:
        g_active = (grad_W != 0.0).to(tl.float32)
    elif USE_LAZY_GATE:
        g_active = (s_fast_in_w * s_fast_in_w > lazy_thresh * v_hat).to(tl.float32)
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    eL_f = e_L.to(tl.float32)
    eH_f = e_H.to(tl.float32)
    if USE_GAP_FEEDBACK:
        # Arms UNBRACKETED on this experimental path (see module docstring).
        evap_L = (1.0 - c_pass) * alpha * eL_f * g_active
        evap_H = (1.0 - c_pass) * alpha * eH_f * g_active
        killed_fine = (evap_L + evap_H) * ascale
    elif USE_GF_CONSOLIDATION:
        # BRACKETED evaporation: each arm dissipates its own value at its
        # own rate lr_eff·gf·(1∓d)·(1−coh_evap); the mean rate is the
        # legacy rate, the differential feeds the gap meter. min_leak
        # bounds EACH arm's per-step survival (the valve never fully
        # shuts on either arm).
        coh_evap = tl.minimum(coh, coh_raw + evap_slack)
        base = lr_eff * gf_consol * (1.0 - coh_evap)
        d_eff = d
        # Matched-pair bracketed evaporation: L permissive base*(1-d), H aggressive base*(1+d).
        # Under the held-out router the gap e_L-e_H is a cross-split data meter (read by the coherence noise floor and the per-row arm meter); it is NEVER a dissipation derivative and must not drive a lam controller.
        evap_frac_L = tl.minimum(base * (1.0 - d_eff), 1.0 - min_leak)
        evap_frac_H = tl.minimum(base * (1.0 + d_eff), 1.0 - min_leak)
        # Hypothesis-infancy guard, keyed on the DYNAMIC LSB gap between the two accumulators.
        # s_slow's LSB sits log2(S_SLOW_FACTOR)=7 bits above the fine base; the arm's LSB sits
        # arm_e=(ar+ac) bits up (the main plane cancels in the difference). Protect arm OCCUPANCY
        # |fine| below that gap -- the meter-unresolvable few-count tail -- and open above it, so
        # the gate tracks the arm plane instead of a fixed deploy-tick (128 fine units) the int8
        # arms cannot reach once the plane ratchets fine (ascale < 1) -- the build_ok~0 pathology.
        # One shared stream; not an arm-level term. arm_e=7 -> shift 0 -> all-open (arm sits at the
        # s_slow LSB, so every count is committable). evap_build_min now feeds only the band_inf meter.
        arm_shift = tl.maximum(7.0 - arm_e, 0.0)
        p_build = tl.minimum(tl.abs(fine.to(tl.float32)) / (arm_shift + 1e-30), 1.0)
        r_build = _hash_uniform(fine, pos_hash, step_salt ^ 0x42424242)
        build_ok = (r_build < p_build).to(tl.float32)
        evap_L = evap_frac_L * eL_f * g_active * build_ok
        evap_H = evap_frac_H * eH_f * g_active * build_ok
        killed_fine = (evap_L + evap_H) * ascale
        killed_w = killed_fine * consf * scale_fwd
        killed_sq = killed_w * killed_w
        if WRITE_BOIL:
            tl.atomic_add(boil_ptr, tl.sum(killed_sq * coh_raw))
            tl.atomic_add(boil_ptr + 1, tl.sum(killed_sq))
            tl.atomic_add(boil_ptr + 3, tl.sum(killed_sq * coh_evap))
            band_inf = (tl.abs(fine.to(tl.float32)) < arm_shift).to(tl.float32)
            s_slow_w = s_slow_full.to(tl.float32) * scale_fwd
            tl.atomic_add(boil_ptr + 4, tl.sum(killed_sq * coh_raw * band_inf))
            tl.atomic_add(boil_ptr + 5, tl.sum(s_slow_w * s_slow_w))
    else:
        step_live = step_live + weight_decay * s_fast_in_w
        evap_L = eL_f * 0.0
        evap_H = eH_f * 0.0
    delta_grad = -lr_eff * step_live * scale_inv     # FINE units, TOTAL
    # Tick deposits half per arm (ARM units); the consolidation terms
    # (coh_raw-gated momentum + evaporation) fire only on the consolidate
    # step, exactly as in the winner kernel. Under the held-out router the
    # FULL tick routes to one arm per micro (arm_sel: 1->L, 0->H); the arm
    # SUM's deposit is algebraically unchanged (arm_sel + (1-arm_sel) = 1)
    # while the GAP gains the cross-split disagreement term. Evaporation
    # still reads the PRE-tick snapshots (eL_f/eH_f): a fresh tick is never
    # evaporated in the launch that deposits it (load-bearing; the exp21
    # first-build mistake cost 4.9 points on clean data).
    if USE_HELDOUT_ROUTER:
        arm_sel = tl.load(arm_sel_ptr).to(tl.float32)
        delta_L = delta_grad * arm_sel * ainv \
            + consf * (beta1 * coh_raw * eL_f * step_scale - evap_L)
        delta_H = delta_grad * (1.0 - arm_sel) * ainv \
            + consf * (beta1 * coh_raw * eH_f * step_scale - evap_H)
    else:
        delta_L = delta_grad * 0.5 * ainv \
            + consf * (beta1 * coh_raw * eL_f * step_scale - evap_L)
        delta_H = delta_grad * 0.5 * ainv \
            + consf * (beta1 * coh_raw * eH_f * step_scale - evap_H)

    # ── SR-tick the arms (separate salts: shared streams bias) ─
    rL = _hash_uniform(e_L, pos_hash, step_salt)
    floor_L = tl.floor(delta_L)
    tick_L = (floor_L + (rL < (delta_L - floor_L)).to(tl.float32)).to(tl.int32)
    e_L = e_L + tick_L
    rH = _hash_uniform(e_H, pos_hash, step_salt ^ 0x1F2F3F4F)
    floor_H = tl.floor(delta_H)
    tick_H = (floor_H + (rH < (delta_H - floor_H)).to(tl.float32)).to(tl.int32)
    e_H = e_H + tick_H

    new_v_int8 = v_slow_i8

    if APPLY_CHASE:
        gate = 1.0
        if USE_GAP_FEEDBACK:
            gate = c_pass
        elif USE_RATIO_COH:
            gate = chase_floor + (1.0 - chase_floor) * coh
        elif USE_COHPRE:
            coh_pre = tl.load(coh_pre_ptr + p_off,
                              mask=nk_mask, other=1.0).to(tl.float32)
            gate = coh + coh_pre * (1.0 - coh)
            coh_pre_ema = (1.0 - alpha_v_fast) * coh_pre + alpha_v_fast * coh
            coh_pre_new = coh_pre + consf * (coh_pre_ema - coh_pre)
            tl.store(coh_pre_ptr + p_off, coh_pre_new, mask=nk_mask)
        fine_post = e_L + e_H
        if USE_EVICT:
            # Eviction valve decision (exp 25): mirror of the chase law,
            # from the same snapshot (pre-chase s_slow_i8, fine_post).
            # Trigger: sign disagreement, both nonzero. Strength: the
            # forward rate law in fine units, SR'd to coarse ticks,
            # capped at |s_slow| (the clamp runs BEFORE the subtraction
            # -- the position cannot overshoot zero by more than the
            # co-firing chase annihilation, exactly as in the reference).
            # DELTA-GATED (exp 26): disagreement with the learned delta
            # d_sv = s_slow - v_slow, drain moves s_slow toward the anchor.
            # The common-mode prior (s_slow == v_slow) never triggers, so the
            # pretrained prior is protected by construction (raw-gating eroded
            # it -- the SDXL scramble). Scaled by EVICT_GAIN (exp 26d knee).
            ev_dsv = s_slow_i8 - v_slow_i8
            ev_dis = (fine_post * ev_dsv) < 0
            ev_dir = tl.where(ev_dsv > 0, 1, -1)
            ev_u = EVICT_GAIN * alpha * gate * gate_gain \
                * tl.abs(fine_post.to(tl.float32)) * ascale * consf / 128.0
            if USE_EVICT_CF:
                ev_u = ev_u * ev_cf_warm   # cf-gate: fire only where coherence SNR is warm (exp 26f)
            if USE_PERFCOH:
                # perfcoh partition, REVERSAL half: eviction reverts marginal admissions
                # freely and protects only near-perfect coherence -- the COMPLEMENT of the
                # leak's commit factor below. One coherence-quality measure, opposite
                # roles, so commit and reversal PARTITION mass by coherence quality; the
                # same-factor pairing cancels (it suppresses eviction exactly on its
                # targets: disagreeing coords read imperfect coh). CPU: exp47b/flux.
                ev_u = ev_u * (1.0 - tl.exp(-(1.0 - coh) / perfcoh_tau))
            rEv = _hash_uniform(s_slow_i8, pos_hash, step_salt ^ 0x1B9B5555)  # salt: TOP BIT CLEAR (int32)
            floor_ev = tl.floor(ev_u)
            ev_tick = (floor_ev + (rEv < (ev_u - floor_ev)).to(tl.float32)).to(tl.int32)
            ev_tick = tl.minimum(ev_tick, tl.abs(ev_dsv))   # cap at |d_sv|: no overshoot past the anchor
            ev_tick = tl.where(ev_dis, ev_tick, 0)
        chase_mantissa = alpha * gate * gate_gain \
            * fine_post.to(tl.float32) * ascale * consf     # FINE units
        chase_int8_f = chase_mantissa / 128.0
        r2 = _hash_uniform(fine_post, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(chase_int8_f)
        frac_s = chase_int8_f - floor_s
        tick_slow_i8 = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        if USE_RAIL_GUARD:
            # Headroom clamp (clamp-FIRST, invariant 2): never credit past
            # the int8 rail. The un-transferred remainder stays in the ARMS
            # for the ratchet/rebalance to re-scale -- the terminal s_slow
            # clamp becomes defense-in-depth instead of a mass sink (the
            # exp80l finding: it is the one non-mass-preserving valve).
            tick_slow_i8 = tl.minimum(tl.maximum(tick_slow_i8,
                                                 -128 - s_slow_i8),
                                      127 - s_slow_i8)
        if USE_SOVEREIGN:
            # Flag 2 converter, chase leg (docs/SPLIT_EXPONENT_DESIGN.md):
            # where the arm grid is COARSER than a slow LSB (r_e > 7),
            # quantize the transfer on the ARM grid (own salt; top bit
            # clear -- the design text's 0x9D... salt violated its own
            # convention, corrected here) and move the exact integer
            # complement tick_arm * 2^(r_e-7) on the slow grid. Where
            # r_e <= 7 the legacy law above stands bit-for-bit (its SR
            # stream untouched; the extra hash below is stateless).
            # Live weight preserved EXACTLY in both regimes -- CPU model
            # exp80m: M1 exact, bias < 0.2% q, var 1/6 q^2.
            fineq = arm_e <= 7.0
            sm_f = ascale / 128.0                # = 2^(r_e-7)
            chase_arm_f = chase_mantissa / ascale
            rSv = _hash_uniform(fine_post, pos_hash, step_salt ^ 0x1D2C7E45)
            floor_a = tl.floor(chase_arm_f)
            tick_arm = (floor_a
                        + (rSv < (chase_arm_f - floor_a)).to(tl.float32)
                        ).to(tl.int32)
            sm_i = tl.maximum(sm_f.to(tl.int32), 1)
            max_t = tl.floor((127 - s_slow_i8).to(tl.float32)
                             / sm_i.to(tl.float32)).to(tl.int32)
            min_t = -tl.floor((128 + s_slow_i8).to(tl.float32)
                              / sm_i.to(tl.float32)).to(tl.int32)
            tick_arm = tl.minimum(tl.maximum(tick_arm, min_t), max_t)
            tick_slow_i8 = tl.where(fineq, tick_slow_i8, tick_arm * sm_i)
            # (the arm debit below, tick * 128 * ainv, is exact fp power-of-
            # two arithmetic in BOTH regimes once the credit is selected:
            # at r_e > 7 it evaluates to tick_arm exactly)
        if WRITE_BOIL:
            realized_chase_w = (tick_slow_i8.to(tl.float32) * 128.0) * scale_fwd
            tl.atomic_add(boil_ptr + 2, tl.sum(realized_chase_w * realized_chase_w))
        s_slow_i8 = s_slow_i8 + tick_slow_i8
        # Debit the arms by EXACTLY 128 fine units per s_slow tick =
        # tick·(128·ainv) ARM units (integer by the +7 joint cap). SR the
        # L share, give H the exact complement — live weight preserved
        # exactly; the split noise lands only in the gap meter.
        debit = (tick_slow_i8.to(tl.float32) * (128.0 * ainv)).to(tl.int32)
        fine_pf = fine_post.to(tl.float32)
        wL = tl.where(tl.abs(fine_pf) > 0.5,
                      e_L.to(tl.float32) / fine_pf, 0.5)
        wL = tl.minimum(tl.maximum(wL, 0.0), 1.0)
        debit_L_f = debit.to(tl.float32) * wL
        r2b = _hash_uniform(e_L, pos_hash, step_salt ^ 0x4D4D2B2B)
        floor_dL = tl.floor(debit_L_f)
        debit_L = (floor_dL
                   + (r2b < (debit_L_f - floor_dL)).to(tl.float32)).to(tl.int32)
        e_L = e_L - debit_L
        e_H = e_H - (debit - debit_L)
        if USE_EVICT:
            # Apply the eviction: -ev coarse from the position, +ev*128
            # fine INTO the arms (integer by the +7 joint cap). SR the L
            # half, give H the exact complement: the sum is exact (live
            # weight preserved bit-for-bit) and the split noise is
            # zero-mean, so the gap meter A_bar stays clean.
            s_slow_i8 = s_slow_i8 - ev_tick * ev_dir
            if WRITE_BOIL:
                # REFUND: eviction debits the slot the chase credits, so
                # slot [2] single-counts disposition: an evict->re-chase
                # bounce no longer double-counts as carried, and mass that
                # evicts then evaporates ends up counted as KILLED, not
                # both. On contested pools this shifts the audited
                # disposition toward destruction (CPU ref: 0.86 -> 0.92 on
                # pure noise) -- the honest reading the servo needs. The
                # signed chase itself still counts (an annihilating chase
                # IS adjudicated evidence); slot [2] can still run
                # NEGATIVE in a window -- consumers clamp at 0 (waste->1).
                realized_evict_w = (ev_tick.to(tl.float32) * 128.0) * scale_fwd
                tl.atomic_add(boil_ptr + 2,
                              -tl.sum(realized_evict_w * realized_evict_w))
            ev_credit = (ev_tick.to(tl.float32) * (128.0 * ainv)).to(tl.int32)
            if USE_SOVEREIGN:
                # Flag 2 evict leg: the legacy truncation above is the
                # panel's :597 mass-destruction bug at r_e > 7 (sub-unit
                # credit truncates to 0 after s_slow was debited). SR on
                # the arm grid; at r_e <= 7 the value is an exact integer
                # and SR is the identity -- one law, both regimes.
                ev_cf = ev_tick.to(tl.float32) * (128.0 * ainv)
                rEsv = _hash_uniform(s_slow_i8, pos_hash,
                                     step_salt ^ 0x6B1A8D33)
                floor_ec2 = tl.floor(ev_cf)
                ev_credit = (floor_ec2
                             + (rEsv < (ev_cf - floor_ec2)).to(tl.float32)
                             ).to(tl.int32)
            ev_half = ev_credit.to(tl.float32) * 0.5
            rEs = _hash_uniform(e_H, pos_hash, step_salt ^ 0x25A54444)
            floor_eh = tl.floor(ev_half)
            ev_L = (floor_eh + (rEs < (ev_half - floor_eh)).to(tl.float32)).to(tl.int32)
            e_L = e_L + ev_L * ev_dir
            e_H = e_H + (ev_credit - ev_L) * ev_dir

        # ── v_slow_i8 leak (unchanged; hash values use the fine ints) ──
        s_slow_full_post = s_slow_i8 * 128
        gap_v_full = (s_slow_full_post - v_slow_full).to(tl.float32)
        delta_v8 = alpha_v_fast * gap_v_full / 128.0 * consf * g_active
        if USE_RATIO_COH:
            delta_v8 = delta_v8 * (leak_floor + (1.0 - leak_floor) * coh)
        if USE_PERFCOH:
            # perfcoh partition, COMMIT half: the anchor commits only near PERFECT
            # coherence, discounted by the nats of divergence exp(-(1-coh)/tau). A coord
            # merely past cf's acceptance boundary stays on probation -- telescope gap
            # open, still evictable -- instead of being baked into the reference (the
            # cf ratchet). Pairs with the eviction complement above. CPU: exp47b/flux.
            delta_v8 = delta_v8 * tl.exp(-(1.0 - coh) / perfcoh_tau)
        r3 = _hash_uniform(e_L + e_H, pos_hash, step_salt ^ 0x33335555)
        floor_v = tl.floor(delta_v8)
        frac_v = delta_v8 - floor_v
        tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
        new_v_int8 = tl.minimum(tl.maximum(v_slow_i8 + tick_v8, -128), 127)
        if MASS_PRESERVE:
            actual_tick_v8 = new_v_int8 - v_slow_i8
            s_slow_i8 = s_slow_i8 - actual_tick_v8

        # ── Bayesian-anchored wd ──
        v_slow_full_post = new_v_int8 * 128
        d_sv_full_post = (s_slow_i8 * 128 - v_slow_full_post).to(tl.float32)
        wd_sv_delta_int8 = lr_eff * wd_sv * d_sv_full_post / 128.0 * consf
        r4 = _hash_uniform(e_L + e_H, pos_hash, step_salt ^ 0x66665555)
        floor_wd_sv = tl.floor(wd_sv_delta_int8)
        frac_wd_sv = wd_sv_delta_int8 - floor_wd_sv
        tick_wd_sv = (floor_wd_sv + (r4 < frac_wd_sv).to(tl.float32)).to(tl.int32)
        s_slow_i8 = s_slow_i8 - tick_wd_sv

        # wd_sf: velocity pull toward the anchor, half per arm in ARM
        # units (independent SR — wd is a sink, not a transfer).
        s_fast_logical_post = (s_slow_i8 * 128).to(tl.float32) \
            + (e_L + e_H).to(tl.float32) * ascale
        d_sf_full_post = s_fast_logical_post - v_slow_full_post.to(tl.float32)
        wd_sf_half = lr_eff * wd_sf * d_sf_full_post * consf * 0.5 * ainv
        r5L = _hash_uniform(e_L, pos_hash, step_salt ^ 0x77770000)
        floor_wL = tl.floor(wd_sf_half)
        tick_wL = (floor_wL
                   + (r5L < (wd_sf_half - floor_wL)).to(tl.float32)).to(tl.int32)
        e_L = e_L - tick_wL
        r5H = _hash_uniform(e_H, pos_hash, step_salt ^ 0x7777FFFF)
        tick_wH = (floor_wL
                   + (r5H < (wd_sf_half - floor_wL)).to(tl.float32)).to(tl.int32)
        e_H = e_H - tick_wH

        # ── anchor decay (wd_anchor): delta toward 0, per arm ──
        anc_s8 = lr_eff * wd_anchor * s_slow_i8.to(tl.float32) * consf
        r6a = _hash_uniform(e_L + e_H, pos_hash, step_salt ^ 0x13571357)
        floor_a8 = tl.floor(anc_s8)
        tick_a8 = (floor_a8 + (r6a < (anc_s8 - floor_a8)).to(tl.float32)).to(tl.int32)
        s_slow_i8 = s_slow_i8 - tick_a8
        anc_L = lr_eff * wd_anchor * eL_f * consf
        r7L = _hash_uniform(e_L, pos_hash, step_salt ^ 0x2468ACE1)
        floor_aL = tl.floor(anc_L)
        tick_aL = (floor_aL + (r7L < (anc_L - floor_aL)).to(tl.float32)).to(tl.int32)
        e_L = e_L - tick_aL
        anc_H = lr_eff * wd_anchor * eH_f * consf
        r7H = _hash_uniform(e_H, pos_hash, step_salt ^ 0x2468FFFF)
        floor_aH = tl.floor(anc_H)
        tick_aH = (floor_aH + (r7H < (anc_H - floor_aH)).to(tl.float32)).to(tl.int32)
        e_H = e_H - tick_aH

    # ── saturation valves (single-step spikes; the ratchet handles
    # sustained pressure one step later). Carry: whole coarse units of
    # arm-sum pressure move into s_slow — integer-exact and live-weight-
    # invariant, hence safe mid-accumulation under the frozen weight_buf.
    # (Thresholds in ARM units; 160 = CARRY_AT, 96 = FOLD_AT.) ──
    fine_now = e_L + e_H
    if USE_GAP_WM:
        # GMR+ Phase B: capture the gap DEMAND before the carry's
        # proportional debit and the fold halve it. |G| >= 96 here means
        # "a fold-magnitude event occurred this window" (FOLD_AT ==
        # ARM_UP_AT -- no new constant); the post-valve arm max is
        # structurally blind to it (pure-noise gap 96 = arm max ~48).
        gap_dem = tl.abs(e_L - e_H)
    abs_sum = tl.abs(fine_now)
    coarse_in_arm = (128.0 * ainv).to(tl.int32)
    carry_mag = tl.where(abs_sum >= 160,
                         ((abs_sum - 32).to(tl.float32)
                          / (128.0 * ainv)).to(tl.int32), 0)
    carry = tl.where(fine_now < 0, -carry_mag, carry_mag)
    if USE_SOVEREIGN:
        # Flag 2 carry leg: at r_e > 7 the legacy debit (carry *
        # coarse_in_arm) is the panel's verified mass-CREATION bug --
        # coarse_in_arm truncates to 0 while s_slow is still credited.
        # The sovereign law works in ARM units: whole arm counts move,
        # each worth 2^(r_e-7) slow LSBs (integer, deterministic -- no
        # salt per the deterministic-writes rule); headroom is clamped in
        # ARM units so the clamp cannot break the multiple property.
        fineq2 = arm_e <= 7.0
        sm2_i = tl.maximum((ascale / 128.0).to(tl.int32), 1)
        carry_arm_m = tl.where(abs_sum >= 160, abs_sum - 32, 0)
        carry_arm = tl.where(fine_now < 0, -carry_arm_m, carry_arm_m)
    if USE_RAIL_GUARD:
        # Headroom clamp, same law as the chase credit: the carry may only
        # move what the rail can hold; the rest stays in the arms (the rail
        # sentinel below fires rebalance at the next consolidation).
        carry = tl.minimum(tl.maximum(carry, -128 - s_slow_i8),
                           127 - s_slow_i8)
        if USE_SOVEREIGN:
            max_a2 = tl.floor((127 - s_slow_i8).to(tl.float32)
                              / sm2_i.to(tl.float32)).to(tl.int32)
            min_a2 = -tl.floor((128 + s_slow_i8).to(tl.float32)
                               / sm2_i.to(tl.float32)).to(tl.int32)
            carry_arm = tl.minimum(tl.maximum(carry_arm, min_a2), max_a2)
    if USE_SOVEREIGN:
        carry = tl.where(fineq2, carry, carry_arm * sm2_i)
    s_slow_i8 = s_slow_i8 + carry
    debit2 = carry * coarse_in_arm
    if USE_SOVEREIGN:
        debit2 = tl.where(fineq2, debit2, carry_arm)
    fine_nf = fine_now.to(tl.float32)
    wL2 = tl.where(tl.abs(fine_nf) > 0.5,
                   e_L.to(tl.float32) / fine_nf, 0.5)
    wL2 = tl.minimum(tl.maximum(wL2, 0.0), 1.0)
    debit2_L_f = debit2.to(tl.float32) * wL2
    r8 = _hash_uniform(e_L, pos_hash, step_salt ^ 0x6E6E1111)
    floor_d2 = tl.floor(debit2_L_f)
    debit2_L = (floor_d2
                + (r8 < (debit2_L_f - floor_d2)).to(tl.float32)).to(tl.int32)
    e_L = e_L - debit2_L
    e_H = e_H - (debit2 - debit2_L)
    # Gap fold: move SR(gap/4) rich→poor — the gap halves in expectation,
    # the sum (the weight) is untouched; the meter loses half its history.
    gap_now = e_L - e_H
    fold_t_f = gap_now.to(tl.float32) * 0.25
    r9 = _hash_uniform(gap_now, pos_hash, step_salt ^ 0x3C3C9999)
    floor_ft = tl.floor(fold_t_f)
    fold_t = (floor_ft
              + (r9 < (fold_t_f - floor_ft)).to(tl.float32)).to(tl.int32)
    fold_t = tl.where(tl.abs(gap_now) >= 96, fold_t, 0)
    e_L = e_L - fold_t
    e_H = e_H + fold_t

    # ── arm-ratchet watermarks: PRE-clamp max(|arm|) per row and col,
    # tracked on EVERY launch (accumulation pressure counts) ──
    arm_pre = tl.maximum(tl.abs(e_L), tl.abs(e_H))
    if USE_GAP_WM:
        # Phase B rides the EXISTING graph-wired watermark buffers: zero
        # new state crosses the capture boundary. Entanglement is bounded
        # and one-directional (max(a,g) >= a: up set grows; gap pressure
        # >= 96 also BLOCKS down via the am_r < ARM_UP_AT guard).
        arm_pre = tl.maximum(arm_pre, gap_dem)
    arm_pre = tl.where(nk_mask, arm_pre, 0)
    tl.atomic_max(arm_row_max_ptr + offs_n, tl.max(arm_pre, axis=1),
                  mask=n_mask)
    tl.atomic_max(arm_col_max_ptr + offs_k, tl.max(arm_pre, axis=0),
                  mask=k_mask)

    # ── clamp and repack ──────────────────────────────────────
    eL_c = tl.minimum(tl.maximum(e_L, -128), 127)
    eH_c = tl.minimum(tl.maximum(e_H, -128), 127)
    if USE_HELDOUT_ROUTER or USE_RAIL_SPILL:
        # Mass-preserve the arm clamp (invariant 2: clamp FIRST, transfer the
        # realized remainder). Under the legacy 0.5/0.5 split both arms track
        # ~half of fine and this clamp is dormant; under the router one arm
        # carries the full per-micro tick and can rail routinely -- discarding
        # the truncation silently depletes the arm SUM that coherence, the
        # build gate, the chase, and the eviction trigger all treat as exact.
        # Spill into s_slow at the carry valve's unit conversion (128*ainv arm
        # counts per s_slow LSB), SR'd with its OWN salt (invariant 6); the
        # live weight is preserved through the clamp in expectation.
        trunc = (e_L - eL_c) + (e_H - eH_c)
        spill_f = trunc.to(tl.float32) / (128.0 * ainv)
        r10 = _hash_uniform(trunc, pos_hash, step_salt ^ 0x5A17C1A3)
        floor_sp = tl.floor(spill_f)
        spill = (floor_sp
                 + (r10 < (spill_f - floor_sp)).to(tl.float32)).to(tl.int32)
        s_slow_i8 = s_slow_i8 + spill
    s_slow_c = tl.minimum(tl.maximum(s_slow_i8, -128), 127)
    packed_new = (
        ((eL_c & 0xFF) << 24)
        | ((eH_c & 0xFF) << 16)
        | ((s_slow_c & 0xFF) << 8)
        | (new_v_int8 & 0xFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── materialize-merge (float m_eff: the arm plane can sit below one
    # fine unit); weight_buf frozen mid-cycle exactly as the winner ──
    new_m_eff_f = ((s_slow_c + new_v_int8) * 128).to(tl.float32) \
        + (eL_c + eH_c).to(tl.float32) * ascale
    new_weight = new_m_eff_f * scale_fwd
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_buf_ptr + w_off,
             new_weight.to(tl.bfloat16), mask=nk_mask & (cons == 1))

    # ── rebalance tracking: full mantissa only (the arm plane self-
    # manages via its ratchet; the D1 |s_fast|-alone fold is superseded) ──
    if TRACK_REBALANCE:
        abs_eff = tl.abs(new_m_eff_f).to(tl.int32)
        abs_eff = tl.where(nk_mask, abs_eff, 0)
        if USE_RAIL_GUARD:
            # Slow-plane rail term (Flag 1): inject RAIL_SENTINEL = MAX_M+1
            # so a railed s_slow reaches the trip-gated dispatcher (which a
            # railed field alone cannot: 16256 < 24000) and blocks the
            # decide's down path. Rides the SAME cons gating as the
            # existing term -- mid-window fires stay structurally
            # impossible (the no-mid-window invariant the carry comment
            # states); a mid-window rail from the consf-ungated carry is
            # detected one consolidation late, which the headroom clamps
            # make safe (mass strands recoverably in the arms).
            railed = (tl.abs(s_slow_c) >= 96) & nk_mask
            abs_eff = tl.maximum(abs_eff, tl.where(railed, 24001, 0))
        tile_row_max = tl.max(abs_eff, axis=1)
        tile_col_max = tl.max(abs_eff, axis=0)
        tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask & (cons == 1))
        tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask & (cons == 1))


def apply_packed2_adamw(packed_w, grad_W, weight_buf, row_exp, col_exp,
                        arm_row_exp, arm_col_exp, arm_row_max, arm_col_max,
                        d_buf,
                        row_max, col_max,
                        lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                        # beta1=0 is LOAD-BEARING, not a missing feature: any momentum
                        # (an EMA of the update stream) amplifies temporally-consistent
                        # gradients at full strength while honest noise averages out --
                        # persistent label corruption rides it to ~100% memorization and
                        # no admission policy catches it in time (held-out degradation
                        # lags consolidation; detection-theoretic, not fixable here).
                        # Opt-in only for curated data, guarded by data policy, never by
                        # the gap floor. See docs/DUAL_MUON_REPORT.md (mechanics branch).
                        weight_decay=0.0, eps=1.0, step_cap=10.0,
                        v_scale=1.0, precond_p=0.5, gf_consol=0.0,
                        drift_cancel_C=None,
                        alpha_v_fast=0.001,
                        wd_sv=0.0, wd_sf=0.0, wd_anchor=0.0,
                        mass_preserve=False, apply_chase=True,
                        track_rebalance=True,
                        v_row=None, v_col=None, sum_v_inv=None,
                        v_full=None, use_full_v=False,
                        gf_trust_delta_sq=0.0, coh_pre=None,
                        gf_consol_buf=None, beta1_buf=None,
                        boil_buf=None, memgap_buf=None, preq_buf=None,
                        grad_activity=False):
    """2-fast mirror of _pb.apply_packed_adamw. All shared schedule state
    (_MIN_LEAK, _EVAP_BUILD_MIN, _LAZY_THRESH, _COH_KAPPA, _EVAP_SLACK,
    _GATE_GAIN, ratio floors, consolidate flag, LAMB bufs, v_bc) is read
    from prototype_packed_b AT LAUNCH TIME, so winner_step and every
    set_* call drive this kernel identically to the winner's."""
    if drift_cancel_C is None:
        drift_cancel_C = compute_drift_cancel_C(alpha, alpha_v_fast,
                                                mass_preserve=mass_preserve)
    write_boil = (abs(float(drift_cancel_C)) > 0.0) and (float(wd_anchor) <= 0.0)
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16
    assert weight_buf.dtype == torch.bfloat16
    assert weight_buf.shape == packed_w.shape
    if step_cap is None or float(step_cap) <= 0.0:
        step_cap = 1e30
    use_gf_trust = (gf_trust_delta_sq > 0)
    use_gf_consol = (gf_consol_buf is not None) or (gf_consol > 0)
    use_cohpre = (coh_pre is not None)
    coh_pre_arg = coh_pre if coh_pre is not None else packed_w
    step_scale_ptr = _lamb_scale_buf(packed_w)
    lamb_wnsq_ptr = _lamb_wnorm_sq_buf(packed_w)
    lamb_snsq_ptr = _lamb_stepnorm_sq_buf(packed_w)
    if use_gf_trust or use_gf_consol or use_cohpre:
        assert v_row is not None and v_col is not None \
            and sum_v_inv is not None, \
            "gf_trust/gf_consol/coh_pre require v_row, v_col, sum_v_inv"
    else:
        v_row = v_row if v_row is not None else packed_w
        v_col = v_col if v_col is not None else packed_w
        sum_v_inv = sum_v_inv if sum_v_inv is not None else packed_w
    use_full_v = bool(use_full_v)
    if use_full_v:
        assert v_full is not None and v_full.shape == packed_w.shape \
            and v_full.is_contiguous(), "use_full_v requires a contiguous [N,K] v_full"
        vhat_mean_ptr = _vhat_mean_buf(packed_w.device)
        vhat_mean_ptr.copy_(
            (v_full.float().mean().clamp(min=1e-30)
             * _v_bc_buf(packed_w.device).reshape(())).reshape(1))
    else:
        v_full = packed_w
        vhat_mean_ptr = sum_v_inv
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    consolidate_ptr = _get_consolidate_flag(packed_w.device)
    arm_sel_ptr = _get_arm_sel_flag(packed_w.device)
    _nsr = _lookup_nsr(packed_w)
    if _nsr is not None:
        # Noise-scale meter: mean-gradient sketch (k fixed coords) + micro count +
        # full ||g||^2, per micro, RAW grad. vector_norm reduces without a full
        # squared temp. Captured torch ops; no kernel change, no SR involvement.
        _nbuf, _nidx = _nsr
        _k = _nidx.numel()
        _nbuf[:_k] += grad_W.reshape(-1)[_nidx].float()
        _nbuf[_k] += 1.0
        _nbuf[_k + 1] += torch.linalg.vector_norm(grad_W, dtype=torch.float32) ** 2
    lr_ptr = _ensure_lr_tensor(lr, packed_w.device)
    eps_ptr = _ensure_eps_tensor(eps, packed_w.device)
    gf_consol_ptr = gf_consol_buf if gf_consol_buf is not None \
        else _pb._ensure_named_scalar("gf_consol", gf_consol, packed_w.device)
    beta1_ptr = beta1_buf if beta1_buf is not None \
        else _pb._ensure_named_scalar("beta1", beta1, packed_w.device)
    chase_floor_ptr, leak_floor_ptr = _ensure_floor_tensors(packed_w.device)
    # per-row floor arming (registry above): an armed buffer REPLACES the
    # scheduled scalar for this layer and selects the per-row kernel variant
    _rf = _lookup_row_floors(packed_w)
    cf_per_row = _rf is not None and _rf[0] is not None
    lf_per_row = _rf is not None and _rf[1] is not None
    if cf_per_row:
        chase_floor_ptr = _rf[0]
    if lf_per_row:
        leak_floor_ptr = _rf[1]
    bc_buf = _v_bc_buf(packed_w.device)
    gap_inv = (1.0 / _pb._GAP_SCALE) if (_pb._GAP_FEEDBACK and _pb._GAP_SCALE > 0) else 0.0
    # 32x32, NOT the winner's 32x64: this kernel holds ~15 more live
    # per-block fp32 tiles (arms, exponent scales, per-arm evap/deltas/
    # ticks, valves) and out-of-resources at the winner's tile size.
    BLOCK_N, BLOCK_K = 32, 32
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_packed2_adamw_kernel[grid](
        packed_w, grad_W, weight_buf, row_exp, col_exp,
        arm_row_exp, arm_col_exp, arm_row_max, arm_col_max,
        row_max, col_max,
        v_row, v_col, sum_v_inv,
        coh_pre_arg,
        v_full, vhat_mean_ptr,
        N, K,
        lr_ptr, int(mantissa_bias), float(alpha), beta1_ptr,
        float(weight_decay), eps_ptr, float(step_cap),
        float(_pb._LAZY_THRESH),
        float(v_scale), float(precond_p), gf_consol_ptr,
        d_buf,
        float(drift_cancel_C),
        float(alpha_v_fast),
        float(_pb._COH_KAPPA),
        float(_pb._EVAP_SLACK),
        float(max(_pb._PERFCOH_TAU, 1e-6)),
        float(wd_sv), float(wd_sf), float(wd_anchor),
        float(gf_trust_delta_sq),
        float(_pb._MIN_LEAK),
        float(_pb._EVAP_BUILD_MIN),
        float(_pb._GATE_GAIN),
        chase_floor_ptr, leak_floor_ptr,
        bc_buf,
        memgap_buf if memgap_buf is not None else _memgap_buf(packed_w.device),
        boil_buf if boil_buf is not None else _boil_buf(packed_w.device),
        _cohq_buf(packed_w.device),
        preq_buf if preq_buf is not None else _memgap_buf(packed_w.device),
        float(gap_inv),
        step_counter,
        consolidate_ptr,
        arm_sel_ptr,
        step_scale_ptr,
        lamb_wnsq_ptr, lamb_snsq_ptr,
        packed_w.stride(0), packed_w.stride(1),
        grad_W.stride(0), grad_W.stride(1),
        weight_buf.stride(0), weight_buf.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        MASS_PRESERVE=bool(mass_preserve),
        APPLY_CHASE=bool(apply_chase),
        TRACK_REBALANCE=bool(track_rebalance),
        USE_GF_TRUST_REGION=bool(use_gf_trust),
        USE_GF_CONSOLIDATION=bool(use_gf_consol),
        USE_COHPRE=bool(use_cohpre),
        USE_FIXED_COH=bool(_pb._USE_FIXED_COH),
        USE_COH_VHAT=bool(_pb._USE_COH_VHAT),
        USE_RATIO_COH=bool(_pb._RATIO_COH),
        USE_GAP_FEEDBACK=bool(_pb._GAP_FEEDBACK),
        USE_LAZY_GATE=bool(_pb._LAZY_GATE),
        USE_GRAD_ACTIVITY=bool(grad_activity),
        WRITE_BOIL=bool(write_boil),
        WRITE_PREQ=preq_buf is not None,
        USE_EVICT=bool(_pb._EVICT_VALVE),
        EVICT_GAIN=float(_pb._EVICT_GAIN),
        USE_EVICT_CF=bool(_pb._EVICT_CF_GATE),
        WRITE_LAMB_NORMS=bool(_pb._LAMB_TRUST),
        USE_FULL_V=use_full_v,
        USE_HELDOUT_ROUTER=bool(_pb._HELDOUT_ROUTER),
        USE_ROUTER_NOISE=bool(_pb._ROUTER_NOISE),
        USE_PERFCOH=bool(_pb._PERFCOH),
        USE_GAP_WM=bool(_pb._GAP_WM),
        USE_RAIL_GUARD=bool(_pb._ARM_RAIL_GUARD),
        USE_RAIL_SPILL=bool(_pb._RAIL_SPILL_ALL),
        USE_SOVEREIGN=bool(_pb._ARM_SOVEREIGN),
        CF_PER_ROW=bool(cf_per_row),
        LF_PER_ROW=bool(lf_per_row),
        GF_PER_ROW=bool(gf_consol_ptr.numel() > 1),
    )


# ============================================================
# Word-level rebalance (four fields; no residual migration — an int8
# arm cannot absorb a ±96-unit residual; SR keeps the shift unbiased)
# and the arm-plane retick.
# ============================================================

@triton.jit
def _rebalance_packed2_decide_kernel(
    packed_ptr,
    row_exp_ptr, col_exp_ptr,
    row_max_ptr, col_max_ptr,
    row_i8med_ptr, col_i8med_ptr,
    arm_row_exp_ptr, arm_col_exp_ptr,
    N, K, MAX_M, EXP_MAX, EXP_MIN,
    seed_ptr,
    stride_pn, stride_pk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TICKDOWN: tl.constexpr,
    SOVEREIGN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    seed = tl.load(seed_ptr).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    row_max = tl.load(row_max_ptr + offs_n, mask=n_mask, other=0)
    col_max = tl.load(col_max_ptr + offs_k, mask=k_mask, other=0)
    row_i8med = tl.load(row_i8med_ptr + offs_n, mask=n_mask, other=127)
    col_i8med = tl.load(col_i8med_ptr + offs_k, mask=k_mask, other=127)
    row_exp = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0)
    col_exp = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0)

    row_up = (row_max > MAX_M) & (row_exp < EXP_MAX)
    col_up = (col_max > MAX_M) & (col_exp < EXP_MAX)
    if ALLOW_TICKDOWN:
        row_dn = (row_max <= MAX_M) & (row_i8med <= 31) & (row_exp > EXP_MIN)
        col_dn = (col_max <= MAX_M) & (col_i8med <= 31) & (col_exp > EXP_MIN)
    else:
        row_dn = row_max < 0
        col_dn = col_max < 0
    row_t = row_up.to(tl.int32) - row_dn.to(tl.int32)
    col_t = col_up.to(tl.int32) - col_dn.to(tl.int32)
    net = row_t[:, None] + col_t[None, :]

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    e_L       = packed >> 24
    e_H       = (packed << 8) >> 24
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24

    rand_off = offs_n[:, None] * K + offs_k[None, :]
    rsh = tl.maximum(net, 0)
    lsh = tl.maximum(-net, 0)
    take_left = net < 0
    two_pos = tl.exp2(rsh.to(tl.float32))

    q_eL = e_L >> rsh
    rem_eL = (e_L - (q_eL << rsh)).to(tl.float32)
    up_eL = (tl.rand(seed, rand_off) * two_pos < rem_eL).to(tl.int32)
    eL_up = q_eL + up_eL
    q_eH = e_H >> rsh
    rem_eH = (e_H - (q_eH << rsh)).to(tl.float32)
    up_eH = (tl.rand(seed, rand_off + 3 * N * K) * two_pos < rem_eH).to(tl.int32)
    eH_up = q_eH + up_eH
    q_slow = s_slow_i8 >> rsh
    rem_slow = (s_slow_i8 - (q_slow << rsh)).to(tl.float32)
    up_slow = (tl.rand(seed, rand_off + N * K) * two_pos < rem_slow).to(tl.int32)
    s_slow_up = q_slow + up_slow
    q_v = v_slow_i8 >> rsh
    rem_v = (v_slow_i8 - (q_v << rsh)).to(tl.float32)
    up_v = (tl.rand(seed, rand_off + 2 * N * K) * two_pos < rem_v).to(tl.int32)
    v_up = q_v + up_v

    if SOVEREIGN:
        # Flag 2: the arm plane's ABSOLUTE scale is invariant across word
        # reticks -- arms pass through untouched; the stored int8 DELTAS
        # are compensated at the exponent stores below (delta - tick).
        # This DELETES the per-word-tick arm SR-shift (a chronic variance
        # class, <= 1/4 arm-unit^2 per coordinate per tick) and makes the
        # gap meters exactly continuous across reticks. Compensation may
        # legally push stored deltas outside the decide band [-8, +5]:
        # the band is decide-time-only (kernel header note).
        eL_new = e_L
        eH_new = e_H
    else:
        eL_new = tl.where(take_left, e_L << lsh, eL_up)
        eH_new = tl.where(take_left, e_H << lsh, eH_up)
    s_slow_new = tl.where(take_left, s_slow_i8 << lsh, s_slow_up)
    v_new = tl.where(take_left, v_slow_i8 << lsh, v_up)

    eL_c = tl.minimum(tl.maximum(eL_new, -128), 127)
    eH_c = tl.minimum(tl.maximum(eH_new, -128), 127)
    s_slow_c = tl.minimum(tl.maximum(s_slow_new, -128), 127)
    v_c = tl.minimum(tl.maximum(v_new, -128), 127)
    packed_new = (
        ((eL_c & 0xFF) << 24)
        | ((eH_c & 0xFF) << 16)
        | ((s_slow_c & 0xFF) << 8)
        | (v_c & 0xFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    if pid_k == 0:
        tl.store(row_exp_ptr + offs_n, row_exp + row_t, mask=n_mask)
        if SOVEREIGN:
            ar_c = tl.load(arm_row_exp_ptr + offs_n, mask=n_mask,
                           other=0).to(tl.int32)
            tl.store(arm_row_exp_ptr + offs_n, (ar_c - row_t).to(tl.int8),
                     mask=n_mask)
    if pid_n == 0:
        tl.store(col_exp_ptr + offs_k, col_exp + col_t, mask=k_mask)
        if SOVEREIGN:
            ac_c = tl.load(arm_col_exp_ptr + offs_k, mask=k_mask,
                           other=0).to(tl.int32)
            tl.store(arm_col_exp_ptr + offs_k, (ac_c - col_t).to(tl.int8),
                     mask=k_mask)


def rebalance_packed2(packed_w, row_exp, col_exp, row_max, col_max,
                      MAX_M=24000, EXP_MAX=7, EXP_MIN=-8, seed_buf=None,
                      allow_tickdown=False,
                      arm_row_exp=None, arm_col_exp=None):
    if seed_buf is None:
        seed_buf = _pb._ensure_reb_seed_tensor(packed_w.device) \
            if hasattr(_pb, "_ensure_reb_seed_tensor") else None
    N, K = packed_w.shape
    s_slow_i8 = (packed_w << 16) >> 24
    v_slow_i8 = (packed_w << 24) >> 24
    i8mag = torch.maximum(s_slow_i8.abs(), v_slow_i8.abs())
    row_i8med = i8mag.median(dim=1).values.to(torch.int32)
    col_i8med = i8mag.median(dim=0).values.to(torch.int32)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    sov = bool(_pb._ARM_SOVEREIGN) and arm_row_exp is not None
    _rebalance_packed2_decide_kernel[grid](
        packed_w, row_exp, col_exp, row_max, col_max,
        row_i8med, col_i8med,
        arm_row_exp if arm_row_exp is not None else row_exp,
        arm_col_exp if arm_col_exp is not None else col_exp,
        N, K, int(MAX_M), int(EXP_MAX), int(EXP_MIN),
        seed_buf,
        packed_w.stride(0), packed_w.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ALLOW_TICKDOWN=bool(allow_tickdown),
        SOVEREIGN=sov,
    )


@triton.jit
def _arm_retick_kernel(
    packed_ptr, rflag_ptr, cflag_ptr, seed_ptr,
    N, K,
    stride_pn, stride_pk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Shift the two arms by net = row_flag + col_flag in {-2..+2}. Net
    up: SR-right-shift per arm on its own stream (expectation-
    preserving). Net down: lossless left shift (decide guarantees the
    affected values are <= ~24). Coarse fields untouched."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    seed = tl.load(seed_ptr).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    rflag = tl.load(rflag_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    cflag = tl.load(cflag_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    net = rflag[:, None] + cflag[None, :]
    rsh = tl.maximum(net, 0)
    lsh = tl.maximum(-net, 0)
    take_left = net < 0
    two_pos = tl.exp2(rsh.to(tl.float32))

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    e_L = packed >> 24
    e_H = (packed << 8) >> 24
    rest = packed & 0xFFFF

    rand_off = offs_n[:, None] * K + offs_k[None, :]
    q_L = e_L >> rsh
    rem_L = (e_L - (q_L << rsh)).to(tl.float32)
    up_L = (tl.rand(seed, rand_off) * two_pos < rem_L).to(tl.int32)
    q_H = e_H >> rsh
    rem_H = (e_H - (q_H << rsh)).to(tl.float32)
    up_H = (tl.rand(seed, rand_off + N * K) * two_pos < rem_H).to(tl.int32)

    eL_new = tl.where(take_left, e_L << lsh, q_L + up_L)
    eH_new = tl.where(take_left, e_H << lsh, q_H + up_H)
    eL_c = tl.minimum(tl.maximum(eL_new, -128), 127)
    eH_c = tl.minimum(tl.maximum(eH_new, -128), 127)
    packed_new = ((eL_c & 0xFF) << 24) | ((eH_c & 0xFF) << 16) | rest
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)


# ── Eviction valve (exp 25) ──────────────────────────────────────────
# Detailed balance at the consolidation boundary: on sign disagreement
# between the fine mass and the position, demote position mass into the
# arms at PRECISELY the forward chase rate law alpha*gate*gate_gain*
# |fine| -- same gates, same strength, opposite direction. W-invariant
# escrow (not destruction): demoted mass re-consolidates if evidence
# returns, evaporates if the contradiction sustains. With the signed
# chase's annihilation also biting on disagreement, position persists
# only under >2/3 sign-agreement; silent-gradient coordinates pay
# nothing (unlike lambda*W). CPU-validated: exp 25 (identity exact,
# bitwise-inert on sign-consistent learning, norms plateau, deploy
# IMPROVES clean +0.19 / 40%-noise +0.47). Baked as a kernel constexpr:
# set BEFORE capture (the controller sets it before the self-test);
# flipping it mid-run needs a recapture. Default OFF = bit-identical.
# The flag lives in prototype_packed_b (single source of truth --
# shared-globals rule): BOTH kernels read _pb._EVICT_VALVE at launch;
# set via _pb.set_evict_valve.


# ── Per-layer prequential servo meter (exp 24) ────────────────────────
# fp32[3] per registered layer: [sum g*gap_w, sum g^2, sum gap_w^2],
# atomic-added by the apply kernel on EVERY launch (each microbatch's
# gradient is measured BEFORE its own tick -- prequential validity).
# Keyed by packed_w storage pointer like _pb's boil/memgap registry:
# register BEFORE CUDA-graph capture (the captured launch bakes the
# pointer), and RE-register whenever packed_w is reallocated (the
# sampling eviction's restore does). Absent key => WRITE_PREQ=False =>
# the kernel branch compiles out (bit-identical launch).
_PERLAYER_PREQ = {}


def register_preq_meter(packed_w, buf):
    _PERLAYER_PREQ[packed_w.data_ptr()] = buf


def _lookup_preq(packed_w):
    return _PERLAYER_PREQ.get(packed_w.data_ptr())


# Noise-scale meter (the set-don't-hunt seeder, exp49/exp50): per-layer
# [k coord-sums | micro count | sum ||g||^2] fp32 buffer + fixed random
# coords. The wrapper accumulates the RAW gradient (pre-preconditioner --
# exp50: the raw stream calibrates better and never rails) with plain torch
# ops, so a CUDA graph captures them with the registered buffer baked. Same
# registry discipline as preq: data_ptr-keyed, register pre-capture,
# re-register on reallocation (the before_step self-heal covers both).
_COHQ_BUFS = {}


def _cohq_key(device):
    # normalize 'cuda' vs 'cuda:0': un-indexed cuda devices resolve to the current one
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return str(d)


def _cohq_buf(device):
    """Per-device [16] fp32 CDF sketch: slots 0..14 = counts of coh < (i+1)/16 over the
    sampled tiles, slot 15 = sampled-coordinate count. Shared across layers on purpose:
    the gate's tau acts on the whole net's realized distribution."""
    key = _cohq_key(device)
    b = _COHQ_BUFS.get(key)
    if b is None:
        b = torch.zeros(16, dtype=torch.float32, device=device)
        _COHQ_BUFS[key] = b
    return b


def read_cohq(device):
    """Read-and-zero the realized-coherence quantiles (q10, q50, q90, n) accumulated
    since the last read. Linear interpolation over the 15-threshold CDF; returns None
    when nothing accumulated (coherence block not running)."""
    b = _COHQ_BUFS.get(_cohq_key(device))
    if b is None:
        return None
    v = b.detach().cpu().tolist()
    b.zero_()
    n = v[15]
    if n <= 0:
        return None
    cdf = [c / n for c in v[:15]]
    ths = [(i + 1) * 0.0625 for i in range(15)]

    def q(p):
        lo_t, lo_c = 0.0, 0.0
        for t, c in zip(ths, cdf):
            if c >= p:
                return lo_t + (t - lo_t) * (p - lo_c) / max(c - lo_c, 1e-9)
            lo_t, lo_c = t, c
        return 1.0
    return q(0.10), q(0.50), q(0.90), int(n)


_PERLAYER_NSR = {}


def register_nsr_meter(packed_w, buf, idx):
    _PERLAYER_NSR[packed_w.data_ptr()] = (buf, idx)


def _lookup_nsr(packed_w):
    return _PERLAYER_NSR.get(packed_w.data_ptr())


def reregister_all_nsr(pairs):
    """Identity-based REBUILD with purge. Stale data_ptr keys are never safe to leave:
    a reallocated layer can land on another layer's old pointer, and an is-None heal
    then cross-wires the two (same-size: silent NSR corruption; different-size: the
    captured gather indexes out of bounds -- device assert, run dies). Observed live as
    'nsr=0' re-registrations after a 722-layer sampling eviction."""
    _PERLAYER_NSR.clear()
    for pw, buf, idx in pairs:
        _PERLAYER_NSR[pw.data_ptr()] = (buf, idx)


def reregister_all_preq(pairs):
    """Same purge-and-rebuild for the preq meters (same collision class, telemetry tier)."""
    _PERLAYER_PREQ.clear()
    for pw, meter in pairs:
        _PERLAYER_PREQ[pw.data_ptr()] = meter


def reregister_all_row_floors(entries):
    """Same purge-and-rebuild for the per-row floor registry ((pw, chase, leak) triples)."""
    _PERLAYER_ROWFLOORS.clear()
    for pw, chase, leak in entries:
        _PERLAYER_ROWFLOORS[pw.data_ptr()] = (chase, leak)


# Per-row incoherent-admission floors (data_ptr-keyed, like the NSR meters).
# Arming a layer flips its compiled kernel variant (CF/LF_PER_ROW constexpr),
# so under CUDA graphs arm BEFORE capture -- flipping later means recapture.
# Buffers are ABSOLUTE floor values, fp32 [N] on the layer's device; the
# arming owner refreshes them (device-tensor fills cross the graph boundary;
# the winner schedule's scalar stops driving an armed layer). data_ptr moves
# (reallocation) orphan the entry -- the arming owner re-registers, mirroring
# the concord_ot pre-capture self-heal for the NSR/preq meters.
_PERLAYER_ROWFLOORS = {}


def register_row_floors(packed_w, chase_rows=None, leak_rows=None):
    for b in (chase_rows, leak_rows):
        if b is not None:
            assert b.dtype == torch.float32 and b.device.type == packed_w.device.type \
                and b.numel() == packed_w.shape[0], "row floor buf: fp32 [N] on-device"
    if chase_rows is None and leak_rows is None:
        _PERLAYER_ROWFLOORS.pop(packed_w.data_ptr(), None)
    else:
        _PERLAYER_ROWFLOORS[packed_w.data_ptr()] = (chase_rows, leak_rows)


def _lookup_row_floors(packed_w):
    return _PERLAYER_ROWFLOORS.get(packed_w.data_ptr())


def read_layer_preq(layer, reset=True):
    """One layer's (sum_dot, sum_g2, sum_gap2) since the last reset; zeros
    if no meter is registered. Window cosine = dot / sqrt(g2 * gap2)."""
    buf = getattr(layer, "_preq_meter", None)
    if buf is None:
        return 0.0, 0.0, 0.0
    a, b, c = float(buf[0]), float(buf[1]), float(buf[2])
    if reset:
        buf.zero_()
    return a, b, c


class GatedArmRatchet:
    """Arm-plane ratchet dispatcher: PURE CADENCE, no per-step trigger.

    An earlier version fired on (global watermark max >= ARM_UP_AT) --
    at SDXL scale that max runs over ~5e7 coordinates, some tail
    coordinate exceeds 96 essentially every step, and the "gate"
    degenerated into 794 layers x two full-tensor medians + a retick
    launch EVERY step, outside the CUDA graph (the observed massive
    slowdown). The per-step trigger is also unnecessary: the in-kernel
    valves (carry + gap fold) own within-step clamp pressure, so the
    exponent ratchet is only responsible for SUSTAINED scale mismatch
    -- a slow phenomenon. Every ARM_RATCHET_EVERY steps, every 2-fast
    layer runs its device-side decide (up on watermark max, down on the
    bulk median); amortized ~25 layer-ratchets per step at cadence 32,
    zero host syncs, zero per-step reductions. Between fires the valves
    + clamps bound the arms; the watermark accumulates the window max,
    which is exactly what the up decide wants."""

    def __init__(self, layers):
        self.layers = [m for m in (layers or [])
                       if hasattr(m, "arm_ratchet")]
        self.calls = 0
        self.fires = 0

    def __call__(self):
        if not self.layers:
            return False
        self.calls += 1
        if self.calls % ARM_RATCHET_EVERY != 0:
            return False
        for m in self.layers:
            m.arm_ratchet()
        self.fires += 1
        return True


# ============================================================
# Autograd Functions (mirror the winner's; always the weight_buf
# forward — the fused-matmul dequant path is 16-bit-layout only)
# ============================================================

class FusedConcordLinear2Fast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, packed_w, row_exp, col_exp,
                arm_row_exp, arm_col_exp, arm_row_max, arm_col_max, d_buf,
                bias,
                lr, alpha, beta1, mantissa_bias,
                weight_decay, eps, step_cap,
                v_scale, precond_p, gf_consol, drift_cancel_C, alpha_v_fast,
                wd_sv, wd_sf, wd_anchor,
                mass_preserve, apply_chase, track_rebalance,
                weight_buf, row_max_buf, col_max_buf,
                v_row, v_col, sum_v_inv,
                adafactor_beta2, track_adafactor_v,
                gf_trust_delta_sq, coh_pre,
                gf_consol_buf, beta1_buf):
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        ctx.fused = _pb._FUSED_MATMUL
        if ctx.fused:
            y = fused_packed2_linear(x, packed_w, row_exp, col_exp,
                                     arm_row_exp, arm_col_exp,
                                     bias_bf16, mantissa_bias)
            ctx.save_for_backward(x)
        else:
            y = F.linear(x, weight_buf, bias_bf16)
            ctx.save_for_backward(x, weight_buf)
        ctx.args = (packed_w, row_exp, col_exp,
                    arm_row_exp, arm_col_exp, arm_row_max, arm_col_max,
                    d_buf,
                    lr, alpha, beta1, mantissa_bias,
                    weight_decay, eps, step_cap,
                    v_scale, precond_p, gf_consol, drift_cancel_C,
                    alpha_v_fast, wd_sv, wd_sf, wd_anchor,
                    mass_preserve, apply_chase, track_rebalance,
                    weight_buf, row_max_buf, col_max_buf,
                    v_row, v_col, sum_v_inv,
                    adafactor_beta2, track_adafactor_v,
                    gf_trust_delta_sq, coh_pre,
                    gf_consol_buf, beta1_buf)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if ctx.fused:
            (x,) = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
        (packed_w, row_exp, col_exp,
         arm_row_exp, arm_col_exp, arm_row_max, arm_col_max,
         d_buf,
         lr, alpha, beta1, mantissa_bias,
         weight_decay, eps, step_cap,
         v_scale, precond_p, gf_consol, drift_cancel_C,
         alpha_v_fast, wd_sv, wd_sf, wd_anchor,
         mass_preserve, apply_chase, track_rebalance,
         weight_buf, row_max_buf, col_max_buf,
         v_row, v_col, sum_v_inv,
         adafactor_beta2, track_adafactor_v,
         gf_trust_delta_sq, coh_pre,
         gf_consol_buf, beta1_buf) = ctx.args
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()
        if ctx.fused:
            grad_x = fused_packed2_gradx(grad_y, packed_w, row_exp, col_exp,
                                         arm_row_exp, arm_col_exp,
                                         mantissa_bias)
        else:
            grad_x = grad_y @ weight
        out_features, in_features = packed_w.shape
        x_flat = x.reshape(-1, in_features)
        grad_y_flat = grad_y.reshape(-1, out_features)
        grad_W = grad_y_flat.transpose(0, 1) @ x_flat
        # Sigma_g-shaped fluctuation noise: same injection as the winner
        # (module globals + schedule shared via _pb). With the legacy 0.5/0.5
        # deposit the noise enters BOTH arms equally through the shared tick,
        # preserving the matched-pair attribution of the gap. Under the
        # held-out router that attribution does NOT hold -- a micro's injected
        # noise routes in full to the on-duty arm and lands in the gap as an
        # extra zero-mean noise source (uncharacterized; concord_ot warns at
        # construction when both are enabled).
        #
        # sigma==0 traffic skip (audit C-2). While the schedule holds sigma at
        # exactly 0.0 the chain computes grad_W + noise*0 -- a numeric no-op
        # that still moves ~44 B/elt. The NUMERIC gate stays branchless and
        # device-resident (the sig_t multiply inside the chain), so a captured
        # graph's replays are live at whatever sigma the schedule fills later;
        # the traffic skip below reads only the HOST schedule mirror and is
        # FORCED OFF while a CUDA graph is capturing, so the full chain is
        # always recorded and no python branch on per-step state ever bakes
        # into a capture (invariant 5 -- baking the skip arm would silently
        # zero the validated injection for the whole run). Check order matters:
        # sigma != 0 short-circuits first, so the steady noise-on path never
        # pays the capture query. Named caveat (same class as the seeder's
        # randperm->randint): while the skip is active, randn_like no longer
        # advances the default CUDA generator, so a sigma==0 stretch is not
        # same-seed-reproducible against pre-skip runs; the per-micro update
        # itself is exactly identical.
        if _pb._SIGMAG_NOISE and (
                _pb._SIGMAG_SIGMA != 0.0
                or torch.cuda.is_current_stream_capturing()):
            with torch.no_grad():
                gwf = grad_W.float()
                if _pb._SIGMAG_ISO:
                    noise = torch.randn_like(gwf)
                else:
                    gyf = grad_y_flat.float()
                    M = gyf.shape[0]
                    eps_v = torch.randn(M, device=gyf.device, dtype=torch.float32)
                    gbar = gwf / max(M, 1)
                    noise = (eps_v[:, None] * gyf).transpose(0, 1) @ x_flat.float() \
                        - eps_v.sum() * gbar
                sig_t = _pb._get_sigmag_sigma(gwf.device)
                nrm = noise.norm().clamp_min(1e-12)
                scaled = noise * (sig_t * gwf.norm() / nrm)
                if _pb._LAZY_GATE:
                    gp = gwf * gwf
                    scaled = scaled * (gp > _pb._LAZY_THRESH * gp.mean()).to(gwf.dtype)
                grad_W = gwf + scaled
        if grad_W.dtype != torch.bfloat16:
            grad_W = grad_W.to(torch.bfloat16)
        if not grad_W.is_contiguous():
            grad_W = grad_W.contiguous()
        if track_rebalance:
            row_max_buf.zero_()
            col_max_buf.zero_()
        if track_adafactor_v and v_row is not None:
            with torch.no_grad():
                g2 = grad_W.float() ** 2
                if _pb._COH_WEIGHTED_V and coh_pre is not None:
                    w = coh_pre / coh_pre.mean().clamp(min=1e-12)
                    g2 = g2 * w
                g2_row = g2.sum(dim=1)
                g2_col = g2.sum(dim=0)
                b2 = adafactor_beta2
                _warm = getattr(v_row, '_concord_warm', None)
                if _warm is None:
                    v_row.mul_(b2).add_(g2_row, alpha=1.0 - b2)
                    v_col.mul_(b2).add_(g2_col, alpha=1.0 - b2)
                else:
                    _cv = b2 * (1.0 - _warm)
                    _cg = _warm + (1.0 - b2) * (1.0 - _warm)
                    v_row.mul_(_cv).add_(g2_row.mul(_cg))
                    v_col.mul_(_cv).add_(g2_col.mul(_cg))
                    _warm.fill_(0.0)
                sum_v = v_row.sum().clamp(min=1e-30)
                sum_v_inv.fill_(0).add_(1.0 / sum_v)
        _boil_m, _memgap_m = _lookup_layer_meters(packed_w)
        apply_packed2_adamw(
            packed_w, grad_W, weight_buf, row_exp, col_exp,
            arm_row_exp, arm_col_exp, arm_row_max, arm_col_max, d_buf,
            row_max_buf, col_max_buf,
            lr=lr, mantissa_bias=mantissa_bias, alpha=alpha, beta1=beta1,
            weight_decay=weight_decay, eps=eps, step_cap=step_cap,
            v_scale=v_scale, precond_p=precond_p,
            gf_consol=gf_consol,
            gf_consol_buf=gf_consol_buf, beta1_buf=beta1_buf,
            boil_buf=_boil_m, memgap_buf=_memgap_m,
            preq_buf=_lookup_preq(packed_w),
            drift_cancel_C=drift_cancel_C,
            alpha_v_fast=alpha_v_fast,
            wd_sv=wd_sv, wd_sf=wd_sf, wd_anchor=wd_anchor,
            mass_preserve=mass_preserve,
            apply_chase=apply_chase,
            track_rebalance=track_rebalance,
            v_row=v_row, v_col=v_col, sum_v_inv=sum_v_inv,
            gf_trust_delta_sq=gf_trust_delta_sq, coh_pre=coh_pre)
        grad_bias = grad_y_flat.sum(0) if ctx.has_bias else None
        # 40 forward args; x (slot 0) and bias (slot 9) get grads.
        return (grad_x,) + (None,) * 8 + (grad_bias,) + (None,) * 30


class FusedConcordConv2d2Fast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, packed_w, row_exp, col_exp,
                arm_row_exp, arm_col_exp, arm_row_max, arm_col_max, d_buf,
                bias,
                in_channels, out_channels, kh, kw, stride, padding,
                lr, alpha, beta1, mantissa_bias,
                weight_decay, eps, step_cap,
                v_scale, precond_p, gf_consol, drift_cancel_C, alpha_v_fast,
                wd_sv, wd_sf, wd_anchor,
                mass_preserve, apply_chase, track_rebalance,
                weight_buf, row_max_buf, col_max_buf,
                v_row, v_col, sum_v_inv,
                adafactor_beta2, track_adafactor_v,
                gf_trust_delta_sq, coh_pre,
                gf_consol_buf, beta1_buf,
                v_full, use_full_v):
        weight_4d = weight_buf.view(out_channels, in_channels, kh, kw)
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        if not x_bf16.is_contiguous():
            x_bf16 = x_bf16.contiguous()
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = F.conv2d(x_bf16, weight_4d, bias=bias_bf16,
                     stride=stride, padding=padding)
        ctx.fused = _pb._FUSED_MATMUL
        if ctx.fused:
            # weight_4d is a view of the SHARED scratch (overwritten by
            # later layers) -> don't save it; the backward re-materializes
            # from packed_w (mirrors the winner's fused-conv contract).
            ctx.save_for_backward(x_bf16)
        else:
            ctx.save_for_backward(x_bf16, weight_4d)
        ctx.conv = (in_channels, out_channels, kh, kw, stride, padding)
        ctx.args = (packed_w, row_exp, col_exp,
                    arm_row_exp, arm_col_exp, arm_row_max, arm_col_max,
                    d_buf,
                    lr, alpha, beta1, mantissa_bias,
                    weight_decay, eps, step_cap,
                    v_scale, precond_p, gf_consol, drift_cancel_C,
                    alpha_v_fast, wd_sv, wd_sf, wd_anchor,
                    mass_preserve, apply_chase, track_rebalance,
                    weight_buf, row_max_buf, col_max_buf,
                    v_row, v_col, sum_v_inv,
                    adafactor_beta2, track_adafactor_v,
                    gf_trust_delta_sq, coh_pre,
                    gf_consol_buf, beta1_buf,
                    v_full, use_full_v)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        in_channels, out_channels, kh, kw, stride, padding = ctx.conv
        if ctx.fused:
            (x_bf16,) = ctx.saved_tensors
        else:
            x_bf16, weight_4d = ctx.saved_tensors
        (packed_w, row_exp, col_exp,
         arm_row_exp, arm_col_exp, arm_row_max, arm_col_max,
         d_buf,
         lr, alpha, beta1, mantissa_bias,
         weight_decay, eps, step_cap,
         v_scale, precond_p, gf_consol, drift_cancel_C,
         alpha_v_fast, wd_sv, wd_sf, wd_anchor,
         mass_preserve, apply_chase, track_rebalance,
         weight_buf, row_max_buf, col_max_buf,
         v_row, v_col, sum_v_inv,
         adafactor_beta2, track_adafactor_v,
         gf_trust_delta_sq, coh_pre,
         gf_consol_buf, beta1_buf,
         v_full, use_full_v) = ctx.args
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()
        if ctx.fused:
            wbuf = _pb._get_fused_scratch(out_channels,
                                          in_channels * kh * kw,
                                          packed_w.device)
            materialize_packed2_bf16(packed_w, row_exp, col_exp,
                                     arm_row_exp, arm_col_exp, out=wbuf,
                                     mantissa_bias=mantissa_bias)
            weight_4d = wbuf.view(out_channels, in_channels, kh, kw)
        grad_x = torch.nn.grad.conv2d_input(
            x_bf16.shape, weight_4d, grad_y,
            stride=stride, padding=padding)
        grad_W_4d = torch.nn.grad.conv2d_weight(
            x_bf16, (out_channels, in_channels, kh, kw), grad_y,
            stride=stride, padding=padding)
        grad_W_2d = grad_W_4d.reshape(out_channels, -1).contiguous()
        if grad_W_2d.dtype != torch.bfloat16:
            grad_W_2d = grad_W_2d.to(torch.bfloat16)
        # (No Sigma_g injection on the conv path — matches the winner,
        # whose noise block lives only in the Linear backward.)
        if track_rebalance:
            row_max_buf.zero_()
            col_max_buf.zero_()
        if track_adafactor_v and v_row is not None:
            with torch.no_grad():
                g2 = grad_W_2d.float() ** 2
                if _pb._COH_WEIGHTED_V and coh_pre is not None:
                    w = coh_pre / coh_pre.mean().clamp(min=1e-12)
                    g2 = g2 * w
                b2 = adafactor_beta2
                v_row.mul_(b2).add_(g2.sum(dim=1), alpha=1.0 - b2)
                v_col.mul_(b2).add_(g2.sum(dim=0), alpha=1.0 - b2)
                if use_full_v and v_full is not None:
                    v_full.mul_(b2).add_(g2, alpha=1.0 - b2)
                sum_v = v_row.sum().clamp(min=1e-30)
                sum_v_inv.fill_(0).add_(1.0 / sum_v)
        _boil_m, _memgap_m = _lookup_layer_meters(packed_w)
        apply_packed2_adamw(
            packed_w, grad_W_2d, weight_buf, row_exp, col_exp,
            arm_row_exp, arm_col_exp, arm_row_max, arm_col_max, d_buf,
            row_max_buf, col_max_buf,
            lr=lr, mantissa_bias=mantissa_bias, alpha=alpha, beta1=beta1,
            weight_decay=weight_decay, eps=eps, step_cap=step_cap,
            v_scale=v_scale, precond_p=precond_p,
            gf_consol=gf_consol,
            gf_consol_buf=gf_consol_buf, beta1_buf=beta1_buf,
            boil_buf=_boil_m, memgap_buf=_memgap_m,
            preq_buf=_lookup_preq(packed_w),
            drift_cancel_C=drift_cancel_C,
            alpha_v_fast=alpha_v_fast,
            wd_sv=wd_sv, wd_sf=wd_sf, wd_anchor=wd_anchor,
            mass_preserve=mass_preserve,
            apply_chase=apply_chase,
            track_rebalance=track_rebalance,
            v_row=v_row, v_col=v_col, sum_v_inv=sum_v_inv,
            v_full=v_full, use_full_v=use_full_v,
            gf_trust_delta_sq=gf_trust_delta_sq, coh_pre=coh_pre)
        grad_bias = grad_y.sum(dim=(0, 2, 3)) if ctx.has_bias else None
        # 48 forward args; x (slot 0) and bias (slot 9) get grads.
        return (grad_x,) + (None,) * 8 + (grad_bias,) + (None,) * 38




# ============================================================
# Fused dequant-matmul (2-fast word): y = x @ dequant2(packed)^T and
# grad_x = grad_y @ dequant2(packed) without materializing bf16 weights.
# Two dots per tile (coarse mantissa + arm plane); both exponent planes
# are separable, so the scales fold into x / grad_y and the output.
# ============================================================

@triton.autotune(configs=_pb._FUSED_AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_packed2_linear_kernel(
    x_ptr, packed_ptr, row_exp_ptr, col_exp_ptr,
    arm_row_exp_ptr, arm_col_exp_ptr, bias_ptr, y_ptr,
    M, N, K, mantissa_bias,
    stride_xm, stride_xk, stride_pn, stride_pk, stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    arm_re = tl.load(arm_row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    row_scale = tl.exp2((row_e - mantissa_bias).to(tl.float32))      # per-N
    arm_rscale = tl.exp2(arm_re.to(tl.float32))                      # per-N
    acc_c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
        arm_ce = tl.load(arm_col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
        col_scale = tl.exp2(col_e.to(tl.float32))
        acol_scale = tl.exp2((col_e + arm_ce).to(tl.float32))
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0).to(tl.float32)
        x_c = (x * col_scale[None, :]).to(tl.bfloat16)
        x_a = (x * acol_scale[None, :]).to(tl.bfloat16)
        packed = tl.load(packed_ptr + offs_k[:, None] * stride_pk + offs_n[None, :] * stride_pn,
                         mask=(k_mask[:, None]) & (n_mask[None, :]), other=0).to(tl.int32)
        e_L = packed >> 24
        e_H = (packed << 8) >> 24
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        coarse = ((s_slow + v_slow) * 128).to(tl.bfloat16)
        arm = (e_L + e_H).to(tl.bfloat16)
        acc_c += tl.dot(x_c, coarse)
        acc_a += tl.dot(x_a, arm)
    acc = row_scale[None, :] * (acc_c + arm_rscale[None, :] * acc_a)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (n_mask[None, :]))


@triton.autotune(configs=_pb._FUSED_AUTOTUNE_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_packed2_gradx_kernel(
    gy_ptr, packed_ptr, row_exp_ptr, col_exp_ptr,
    arm_row_exp_ptr, arm_col_exp_ptr, gx_ptr,
    M, N, K, mantissa_bias,
    stride_gym, stride_gyn, stride_pn, stride_pk, stride_gxm, stride_gxk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    arm_ce = tl.load(arm_col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    col_scale = tl.exp2((col_e - mantissa_bias).to(tl.float32))      # per-K
    arm_cscale = tl.exp2(arm_ce.to(tl.float32))                      # per-K
    acc_c = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
        arm_re = tl.load(arm_row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
        row_scale = tl.exp2(row_e.to(tl.float32))
        arow_scale = tl.exp2((row_e + arm_re).to(tl.float32))
        gy = tl.load(gy_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn,
                     mask=(offs_m[:, None] < M) & (n_mask[None, :]), other=0.0).to(tl.float32)
        gy_c = (gy * row_scale[None, :]).to(tl.bfloat16)
        gy_a = (gy * arow_scale[None, :]).to(tl.bfloat16)
        packed = tl.load(packed_ptr + offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk,
                         mask=(n_mask[:, None]) & (k_mask[None, :]), other=0).to(tl.int32)
        e_L = packed >> 24
        e_H = (packed << 8) >> 24
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        coarse = ((s_slow + v_slow) * 128).to(tl.bfloat16)
        arm = (e_L + e_H).to(tl.bfloat16)
        acc_c += tl.dot(gy_c, coarse)
        acc_a += tl.dot(gy_a, arm)
    acc = col_scale[None, :] * acc_c + (col_scale * arm_cscale)[None, :] * acc_a
    tl.store(gx_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (k_mask[None, :]))


def fused_packed2_linear(x, packed_w, row_exp, col_exp,
                         arm_row_exp, arm_col_exp, bias=None, mantissa_bias=15):
    *lead, K = x.shape
    N = packed_w.shape[0]
    x2d = x.reshape(-1, K).contiguous() if not x.is_contiguous() else x.reshape(-1, K)
    M = x2d.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _fused_packed2_linear_kernel[grid](
        x2d, packed_w, row_exp, col_exp, arm_row_exp, arm_col_exp,
        bias if bias is not None else x2d, y,
        M, N, K, int(mantissa_bias),
        x2d.stride(0), x2d.stride(1), packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None,
    )
    return y.reshape(*lead, N)


def fused_packed2_gradx(grad_y, packed_w, row_exp, col_exp,
                        arm_row_exp, arm_col_exp, mantissa_bias=15):
    *lead, N = grad_y.shape
    K = packed_w.shape[1]
    gy2d = grad_y.reshape(-1, N).contiguous() if not grad_y.is_contiguous() else grad_y.reshape(-1, N)
    M = gy2d.shape[0]
    gx = torch.empty((M, K), dtype=torch.bfloat16, device=grad_y.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(K, META['BLOCK_K']))
    _fused_packed2_gradx_kernel[grid](
        gy2d, packed_w, row_exp, col_exp, arm_row_exp, arm_col_exp, gx,
        M, N, K, int(mantissa_bias),
        gy2d.stride(0), gy2d.stride(1), packed_w.stride(0), packed_w.stride(1),
        gx.stride(0), gx.stride(1),
    )
    # Graph-native gradient-SNR sketch (registered layers only; absent-key -> no launch,
    # bit-identical to before). Rides gy here because the apply kernel only sees grad_W.
    # The key check is a capture-time constant (registration is fixed before capture), so
    # the launch is baked for the chosen layers and never appears for the rest.
    _sk_buf, _sk_coord = _lookup_sketch_meter(packed_w)
    if _sk_buf is not None:
        _k_each = _sk_coord.numel()
        _sketch_gy_kernel[(_k_each,)](
            gy2d, _sk_coord, _sk_buf, M, N, _k_each,
            gy2d.stride(0), gy2d.stride(1), BLOCK_M=256,
        )
    return gx.reshape(*lead, K)


# ============================================================
# Graph-native gradient-SNR sketch (feat-only)
# ============================================================
# A representative handful of UNet layers each carry a device buffer + fixed random
# FEATURE coordinates. Inside the captured grad_x launch, _sketch_gy_kernel sums the
# output gradient gy over ALL rows (batch x tokens) at those columns and atomic-adds
# into the buffer; because the buffer is zeroed once per OPTIMIZER step (not per
# micro-batch), the accumulation pools the gradient-accumulation micro-batches into one
# larger effective batch. Buffer layout: [k_each signal sums, 1 row-count]; the host
# forms the feat-only batch-mean = sums / count post-replay and feeds the CSNR meter.
# Feat-only (not the eager token*feat sketch) is the layout the de-risk picked: it
# concentrates the signal and averages more noise, and it is a plain column reduction
# here rather than a strided gather. Register BEFORE capture so the pointer is baked.

_SKETCH_METERS = {}     # packed_w.data_ptr() -> (buf [k_each+1] f32, coord [k_each] int32)


def register_sketch_meter(packed_w, k_each, seed):
    """Attach a feat-only sketch to this layer. Fixed feature coords are drawn once
    (baked before capture); the buffer accumulates across micro-batches. Idempotent."""
    dev = packed_w.device
    N = int(packed_w.shape[0])                      # gy has N output-feature columns
    g = torch.Generator().manual_seed(int(seed))
    coord = torch.randint(0, N, (int(k_each),), generator=g).to(torch.int32).to(dev)
    buf = torch.zeros(int(k_each) + 1, dtype=torch.float32, device=dev)
    _SKETCH_METERS[packed_w.data_ptr()] = (buf, coord)
    return buf, coord


def _lookup_sketch_meter(packed_w):
    return _SKETCH_METERS.get(packed_w.data_ptr(), (None, None))


def clear_sketch_meters():
    _SKETCH_METERS.clear()


@triton.jit
def _sketch_gy_kernel(gy_ptr, coord_ptr, out_ptr, M, N, K_EACH,
                      stride_gym, stride_gyn, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)                           # one program per sketch coordinate
    coord = tl.load(coord_ptr + pid)                 # a fixed feature column in [0, N)
    acc = 0.0
    for m0 in range(0, M, BLOCK_M):
        offs = m0 + tl.arange(0, BLOCK_M)
        mask = offs < M
        vals = tl.load(gy_ptr + offs * stride_gym + coord * stride_gyn,
                       mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals)
    tl.atomic_add(out_ptr + pid, acc)                # pooled over accumulation micro-batches
    # row-count slot (pid 0 only): host forms the batch-mean = sums / count
    tl.atomic_add(out_ptr + K_EACH, tl.where(pid == 0, M, 0).to(tl.float32))


# ============================================================
# Layer classes
# ============================================================

class ConcordLinear2Fast(ConcordLinearPackedB):
    """2-fast layer: subclasses the production layer (inherits every
    knob, property, and buffer) and overrides only the fine-plane
    surface. New per-layer state: two int8 arm-exponent vectors, two
    watermark vectors, and the bracket-width device tensor. AdamW only;
    the SGD kind is refused (8-bit arms without the preconditioner
    saturate)."""

    def __init__(self, in_features, out_features, bias=True,
                 device='cuda', alpha=0.1, beta1=0.0, lr=0.01,
                 bracket_d=0.25):
        self._bracket_d_value = float(bracket_d)
        super().__init__(in_features, out_features, bias=bias,
                         device=device, alpha=alpha, beta1=beta1, lr=lr)
        self.register_buffer('arm_row_exp',
            torch.zeros(out_features, dtype=torch.int8, device=device))
        self.register_buffer('arm_col_exp',
            torch.zeros(in_features, dtype=torch.int8, device=device))
        self.register_buffer('_arm_row_max',
            torch.zeros(out_features, dtype=torch.int32, device=device))
        self.register_buffer('_arm_col_max',
            torch.zeros(in_features, dtype=torch.int32, device=device))
        self.register_buffer('_d_buf',
            torch.full((1,), self._bracket_d_value,
                       dtype=torch.float32, device=device))
        # __init__ ran _init_weight/_ensure_buffers BEFORE the arm buffers
        # existed (the overrides tolerate that); re-run now for a correct
        # first materialize.
        self._load_prior_pending()

    # ── bracket width (device-tensor knob, lr pattern) ─────────
    @property
    def bracket_d(self):
        return self._bracket_d_value

    @bracket_d.setter
    def bracket_d(self, value):
        v = float(value)
        self._bracket_d_value = v
        buf = getattr(self, '_d_buf', None)
        if buf is not None:
            buf.fill_(v)

    def set_optimizer_kind(self, kind, weight_decay=0.0, eps=1.0,
                           step_cap=10.0):
        if kind != 'adamw':
            raise ValueError("ConcordLinear2Fast is AdamW-only (8-bit "
                             "arms without the preconditioner saturate)")
        return super().set_optimizer_kind(kind, weight_decay=weight_decay,
                                          eps=eps, step_cap=step_cap)

    # ── init: production's coarse split + Kaiming gap seed, with the
    # ±64 fine residual landing in the arms (half each, gap ≤ 1) ──
    def _init_weight(self):
        # A bare 2-fast layer constructs to the ZERO state (packed_w is
        # already zeros; exponents sane) -- NOT the base's random Kaiming
        # field. Every real consumer (the UNet swap, the self-test) calls
        # load_weights immediately, and the base behavior cost 794 layers
        # of randn + clone + a discarded full pack pass at load time.
        # Call load_weights yourself if you construct one bare.
        self.row_exp.zero_()
        self.col_exp.zero_()

    def _load_prior_pending(self):
        w = getattr(self, '_pending_W', None)
        if w is not None:
            kw = getattr(self, '_pending_kw', {})
            del self._pending_W
            self._pending_kw = {}
            self.load_weights(w, **kw)

    @torch.no_grad()
    def load_weights(self, W, gap=0.0, kaiming_init=False, kaiming_scale=0.0):
        if not hasattr(self, 'arm_row_exp'):
            # called from the base __init__ before the arm buffers exist;
            # defer until they do (see __init__).
            self._pending_W = W.detach().clone()
            self._pending_kw = dict(gap=gap, kaiming_init=kaiming_init,
                                    kaiming_scale=kaiming_scale)
            return
        W = W.to(device=self.packed_w.device, dtype=torch.float32)
        max_abs = W.abs().max(dim=1).values.clamp(min=1e-30)
        self.row_exp.copy_(
            torch.ceil(torch.log2(max_abs) + 1.0)
            .clamp(self.EXP_MIN, self.EXP_MAX).to(torch.int8))
        self.col_exp.zero_()
        exp = (self.row_exp[:, None].to(torch.float32)
               + self.col_exp[None, :].to(torch.float32)
               - self.MANTISSA_BIAS)
        scale = torch.pow(2.0, exp)
        m_total = (W / scale).round().to(torch.int32).clamp(INT16_MIN, INT16_MAX)
        coarse = (m_total.to(torch.float32) / 128.0).round().to(torch.int32).clamp(
            2 * INT8_MIN, 2 * INT8_MAX)
        v_slow_i8 = (coarse.to(torch.float32) * (1.0 - gap) / 2.0).round() \
            .to(torch.int32).clamp(INT8_MIN, INT8_MAX)
        s_slow_i8 = (coarse - v_slow_i8).clamp(INT8_MIN, INT8_MAX)
        if kaiming_init and kaiming_scale > 0.0:
            # identical to the winner's mass-preserving symmetric gap seed
            # (coarse fields only; deploy bit-identical at any scale).
            row_rms = W.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)
            k_i8 = (kaiming_scale * row_rms * torch.randn_like(W) / (128.0 * scale)
                    ).round().to(torch.int32)
            k_lo = torch.maximum(INT8_MIN - s_slow_i8, v_slow_i8 - INT8_MAX)
            k_hi = torch.minimum(INT8_MAX - s_slow_i8, v_slow_i8 - INT8_MIN)
            k_i8 = torch.minimum(torch.maximum(k_i8, k_lo), k_hi)
            s_slow_i8 = s_slow_i8 + k_i8
            v_slow_i8 = coarse - s_slow_i8
        # fine residual (|.| <= 64 < evap_build_min => never an evaporation
        # target) splits across the arms, half each, init gap <= 1.
        resid = m_total - (s_slow_i8 + v_slow_i8) * 128
        e_L = (resid + 1).div(2, rounding_mode='floor').to(torch.int32)
        e_H = resid - e_L
        packed = (
            ((e_L & 0xFF) << 24)
            | ((e_H & 0xFF) << 16)
            | ((s_slow_i8 & 0xFF) << 8)
            | (v_slow_i8 & 0xFF)
        )
        self.packed_w.copy_(packed)
        self.arm_row_exp.zero_()      # residual stored exactly at grid 1
        self.arm_col_exp.zero_()
        self._arm_row_max.zero_()
        self._arm_col_max.zero_()
        self._resync_weight_buf()

    @torch.no_grad()
    def load_weights_finetune(self, W):
        self.load_weights(W)

    @torch.no_grad()
    def load_weights_anchor(self, W):
        raise NotImplementedError(
            "2-fast: the frozen-anchor TE packing is not ported; anchored "
            "text encoders stay on the winner kernel")

    @torch.no_grad()
    def apply_grad_step(self, grad_W, v_stats_from=None):
        raise NotImplementedError(
            "2-fast: the embedding self-step path is not ported; packed "
            "embeddings stay on the winner kernel")

    # ── buffers / materialize ──────────────────────────────────
    @torch.no_grad()
    def _resync_weight_buf(self):
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if wbuf is None or wbuf.shape != self.packed_w.shape:
            return    # fused mode: no per-layer cache to resync
        if not hasattr(self, 'arm_row_exp'):
            return   # base __init__ path; re-materialized after arm bufs exist
        materialize_packed2_bf16(self.packed_w, self.row_exp, self.col_exp,
                                 self.arm_row_exp, self.arm_col_exp,
                                 out=wbuf, mantissa_bias=self.MANTISSA_BIAS)

    def _ensure_buffers(self):
        N, K = self.packed_w.shape
        if _pb._FUSED_MATMUL and not hasattr(self, 'kh'):
            # Fused dequant-matmul (Linear): no per-layer bf16 cache. The
            # apply kernel still writes the materialized weight as a side
            # effect, so hand it the ONE shared throwaway scratch. Same
            # accumulation caveat as the winner: no freeze buffer, the
            # arms tick through the accumulation cycle.
            wbuf = _pb._get_fused_scratch(N, K, self.packed_w.device)
        elif _pb._FUSED_MATMUL:
            # Fused conv: cuDNN cannot dequant inside -- materialize this
            # conv's weight into the SHARED scratch each forward.
            wbuf = _pb._get_fused_scratch(N, K, self.packed_w.device)
            if hasattr(self, 'arm_row_exp'):
                materialize_packed2_bf16(
                    self.packed_w, self.row_exp, self.col_exp,
                    self.arm_row_exp, self.arm_col_exp,
                    out=wbuf, mantissa_bias=self.MANTISSA_BIAS)
        else:
            wbuf = getattr(self, '_bf16_weight_buf', None)
            if wbuf is None or wbuf.shape != self.packed_w.shape:
                wbuf = torch.empty(self.packed_w.shape, dtype=torch.bfloat16,
                                   device=self.packed_w.device)
                self._bf16_weight_buf = wbuf
                if hasattr(self, 'arm_row_exp'):
                    materialize_packed2_bf16(
                        self.packed_w, self.row_exp, self.col_exp,
                        self.arm_row_exp, self.arm_col_exp,
                        out=wbuf, mantissa_bias=self.MANTISSA_BIAS)
        rmbuf = getattr(self, '_row_max_buf', None)
        if rmbuf is None or rmbuf.shape[0] != N:
            self._row_max_buf = torch.zeros(N, dtype=torch.int32,
                                            device=self.packed_w.device)
        cmbuf = getattr(self, '_col_max_buf', None)
        if cmbuf is None or cmbuf.shape[0] != K:
            self._col_max_buf = torch.zeros(K, dtype=torch.int32,
                                            device=self.packed_w.device)
        return wbuf, self._row_max_buf, self._col_max_buf

    # ── state readouts (fine plane) ────────────────────────────
    @torch.no_grad()
    def get_state(self):
        """(e_L, e_H, s_slow_i8, v_slow_i8)."""
        e_L       = (self.packed_w >> 24)
        e_H       = ((self.packed_w << 8) >> 24)
        s_slow_i8 = ((self.packed_w << 16) >> 24)
        v_slow_i8 = ((self.packed_w << 24) >> 24)
        return e_L, e_H, s_slow_i8, v_slow_i8

    def _arm_scale(self):
        ar = self.arm_row_exp.to(torch.float32)[:, None]
        ac = self.arm_col_exp.to(torch.float32)[None, :]
        return torch.pow(2.0, ar + ac)

    @torch.no_grad()
    def arm_gap(self):
        """(e_L − e_H)·2^(ar+ac) in FINE units: the integrated
        dissipation differential (dW/dlam per coordinate). READ-ONLY."""
        e_L, e_H, _, _ = self.get_state()
        return (e_L - e_H).to(torch.float32) * self._arm_scale()

    @torch.no_grad()
    def fine_sum(self):
        """(e_L + e_H)·2^(ar+ac) in FINE units (s_fast's role)."""
        e_L, e_H, _, _ = self.get_state()
        return (e_L + e_H).to(torch.float32) * self._arm_scale()

    @torch.no_grad()
    def get_weight(self):
        e_L, e_H, ss, vs = self.get_state()
        m_eff_f = ((ss + vs) * 128).to(torch.float32) \
            + (e_L + e_H).to(torch.float32) * self._arm_scale()
        exp = (self.row_exp[:, None].to(torch.int32)
               + self.col_exp[None, :].to(torch.int32)
               - self.MANTISSA_BIAS).to(torch.float32)
        return (m_eff_f * torch.pow(2.0, exp)).to(torch.bfloat16)

    # consolidated_weight is INHERITED: it reads only the coarse fields,
    # whose bit positions are unchanged.

    # ── forward / rebalance / ratchet ──────────────────────────
    def forward(self, x):
        in_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        wbuf, rmbuf, cmbuf = self._ensure_buffers()
        fg = self.fast_gain
        if fg < 1.0:
            with torch.no_grad():
                e_L, e_H, ss, vs = self.get_state()
                exp = (self.row_exp.float()[:, None]
                       + self.col_exp.float()[None, :] - self.MANTISSA_BIAS)
                fine_f = (e_L + e_H).float() * self._arm_scale()
                m_gated = (ss.float() * 128.0 + fg * fine_f
                           + vs.float() * 128.0) * torch.exp2(exp)
                wbuf.copy_(m_gated.to(wbuf.dtype))
        y = FusedConcordLinear2Fast.apply(
            x, self.packed_w, self.row_exp, self.col_exp,
            self.arm_row_exp, self.arm_col_exp,
            self._arm_row_max, self._arm_col_max, self._d_buf,
            self.bias,
            self._lr_buf, self.alpha, self.beta1, self.MANTISSA_BIAS,
            self.weight_decay, self._eps_buf, self.step_cap,
            self.v_scale, self.precond_p, self.gf_consol,
            self.drift_cancel_C, self.alpha_v_fast,
            self.wd_sv, self.wd_sf, self.wd_anchor,
            self.mass_preserve_v, self.apply_chase, self.track_rebalance,
            wbuf, rmbuf, cmbuf,
            self.v_row, self.v_col, self._sum_v_inv,
            float(self.adafactor_beta2),
            bool(self.track_adafactor_v),
            float(self.gf_trust_delta_sq), self._coh_pre,
            self._gf_consol_buf, self._beta1_buf)
        return y.to(in_dtype)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Flag-2 loader guard (both directions -- docs/SPLIT_EXPONENT_
        # DESIGN.md ladder cell 4): a flag-OFF build must refuse any
        # checkpoint whose joint arm exponents exceed the legacy cap,
        # because the legacy transfer arithmetic silently truncates (and
        # the carry CREATES mass) in that regime.
        joint = int((self.arm_row_exp.to(torch.int32)[:, None]
                     + self.arm_col_exp.to(torch.int32)[None, :]).max())
        if joint > 7 and not _pb._ARM_SOVEREIGN:
            raise RuntimeError(
                f"checkpoint carries joint arm exponent {joint} > 7: this "
                f"state was written under _ARM_SOVEREIGN (Flag 2) and the "
                f"legacy transfer arithmetic is invalid for it -- enable "
                f"set_arm_sovereign(True) (with its Flag-1/GMR+ "
                f"preconditions) before loading.")

    @torch.no_grad()
    def rebalance(self):
        """Word-level exponent ratchet (GatedRebalance calls this on a
        MAX_M trip). The ARM ratchet is separate — arm_ratchet(), driven
        by GatedArmRatchet — because the word gate would starve it."""
        rmbuf = getattr(self, '_row_max_buf', None)
        cmbuf = getattr(self, '_col_max_buf', None)
        if rmbuf is None or cmbuf is None:
            return
        if _pb._ARM_RAIL_GUARD:
            # Flag 1 (1d): keep the hwm in MANTISSA semantics -- mask the
            # RAIL_SENTINEL before folding it in (else health lines read
            # rail windows as mantissa pressure) and count rail events on
            # their own meter.
            if not hasattr(self, "_rail_events"):
                self._rail_events = 0
            self._rail_events += int((rmbuf > 24000).sum().item()
                                     + (cmbuf > 24000).sum().item())
            torch.maximum(self._row_max_hwm, rmbuf.clamp(max=24000),
                          out=self._row_max_hwm)
            torch.maximum(self._col_max_hwm, cmbuf.clamp(max=24000),
                          out=self._col_max_hwm)
        else:
            torch.maximum(self._row_max_hwm, rmbuf, out=self._row_max_hwm)
            torch.maximum(self._col_max_hwm, cmbuf, out=self._col_max_hwm)
        self._reb_seed.add_(1)
        rebalance_packed2(
            self.packed_w, self.row_exp, self.col_exp,
            rmbuf, cmbuf,
            MAX_M=self.MAX_M, EXP_MAX=self.EXP_MAX, EXP_MIN=self.EXP_MIN,
            seed_buf=self._reb_seed, allow_tickdown=self.allow_tickdown,
            arm_row_exp=self.arm_row_exp, arm_col_exp=self.arm_col_exp,
        )

    @torch.no_grad()
    def arm_ratchet(self):
        """Arm-plane exponent ratchet: decide from the pre-clamp
        watermarks (device-side, no host sync), rows first then cols
        against the updated row max so the joint cap ar+ac <= ARM_SUM_MAX
        (transfer integrality) holds by construction. Up SR-halves; down
        doubles losslessly; 4x hysteresis."""
        # ASYMMETRIC gating, mirroring the word plane's reasoning: UP is
        # gated by the pre-clamp watermark MAX (any element's demand is
        # urgent -- it is about to clip). DOWN is gated by the CURRENT
        # per-row/col MEDIAN of the arm magnitudes (bulk underuse) -- a
        # max-gated down-tick starves on one hot coordinate, and in the
        # fine-tune regime the down direction (sub-fine-unit grids) is
        # where the bulk of the delta tensor lives. Current max <= 63
        # keeps the lossless double from clipping the bulk (down/down
        # crossing outliers clip by design, as on the word plane); a
        # watermark spike >= the up threshold converts to an up-tick,
        # never a down-tick into returning pressure.
        am_r, am_c = self._arm_row_max, self._arm_col_max
        ar = self.arm_row_exp.to(torch.int32)
        ac = self.arm_col_exp.to(torch.int32)
        # Bulk statistics with BOUNDED transients: arms-only unpack (not
        # get_state -- that materializes four [N,K] tensors) and chunked
        # reductions, ~10 MB peak instead of ~130 MB on the largest conv
        # (which OOMed at the VRAM ceiling). Row and col medians are
        # independent per row/col, so chunking is exact.
        CH = 256
        pw = self.packed_w
        N, K = pw.shape
        gapr = bool(_pb._ARM_GAP_RATCHET)   # GMR+ Phase A; off = legacy decide
        r_med, r_now = [], []
        rg_med, rg_now = [], []
        for i in range(0, N, CH):
            eL = (pw[i:i + CH] >> 24)
            eH = ((pw[i:i + CH] << 8) >> 24)
            a = torch.maximum(eL.abs(), eH.abs())
            r_med.append(a.median(dim=1).values)
            r_now.append(a.amax(dim=1))
            if gapr:
                g = (eL - eH).abs()
                rg_med.append(g.median(dim=1).values)
                rg_now.append(g.amax(dim=1))
        row_med = torch.cat(r_med).to(torch.int32)
        row_now = torch.cat(r_now).to(torch.int32)
        c_med, c_now = [], []
        cg_med, cg_now = [], []
        for j in range(0, K, CH):
            eL = (pw[:, j:j + CH] >> 24)
            eH = ((pw[:, j:j + CH] << 8) >> 24)
            a = torch.maximum(eL.abs(), eH.abs())
            c_med.append(a.median(dim=0).values)
            c_now.append(a.amax(dim=0))
            if gapr:
                g = (eL - eH).abs()
                cg_med.append(g.median(dim=0).values)
                cg_now.append(g.amax(dim=0))
        col_med = torch.cat(c_med).to(torch.int32)
        col_now = torch.cat(c_now).to(torch.int32)
        sov = bool(_pb._ARM_SOVEREIGN)
        if sov:
            # Flag 2 guard, RE-SCOPED per the pre-registered stop rule
            # (exp80e sovereign cell run 1): the panel's per-axis bands
            # [-8,+5] CUT row headroom vs legacy (rows reached 7 with cols
            # at 0; the band capped them at 5 and sigma discrimination
            # FELL, 0.99 vs 1.11 -- attribution: band-bound, NOT MAX_M).
            # The derivation's real constraint is the JOINT r_e <= +10
            # (transfer chunk 2^(r_e-7) <= 8 slow LSBs fits the 31-LSB
            # protected band above SLOW_RAIL_AT) -- so the guard keeps the
            # legacy joint FORM with the constant raised 7 -> 10. The
            # hot-column veto survives; its cost is mild against the
            # measured cost of per-axis caps. Decide-time only -- retick
            # compensation may push stored deltas past any band legally.
            r_cap = (ar + 1 + ac.max() <= 10)
        else:
            r_cap = (ar < ARM_EXP_MAX) & (ar + 1 + ac.max() <= ARM_SUM_MAX)
        r_up = (am_r >= ARM_UP_AT) & r_cap
        r_dn = (row_med <= ARM_DN_AT) & (row_now <= 63) \
            & (am_r < ARM_UP_AT) & (ar > ARM_EXP_MIN)
        if gapr:
            # GMR+ gap-median channel. UP: a chronically-folding row's
            # |gap| median parks at ~36 (fold-recycle stationary state --
            # a STANDING property a cadence snapshot sees); healthy rows
            # sit at a few LSB. The legacy arm-max metric is structurally
            # blind here (pure-noise gap 96 reads arm max ~48 < 96).
            row_gmed = torch.cat(rg_med).to(torch.int32)
            row_gnow = torch.cat(rg_now).to(torch.int32)
            r_up = r_up | ((row_gmed >= ARM_GAP_UP_MED) & r_cap)
            # DOWN guards, BOTH load-bearing (do not relax):
            #  - gap median <= ARM_GAP_DN_MED: without it the post-up-tick
            #    chronic-noise state (arms ~8-16, gap median ~16-18)
            #    satisfies the legacy clause -> period-2-cadence up/down
            #    limit cycle.
            #  - gap amax <= GAP_DN_SAFE: row_now <= 63 bounds ARMS only;
            #    an outlier gap up to 126 would double to >= FOLD_AT and
            #    refold immediately. 47 doubles to 94 < 96. Also closes
            #    the LATENT LEGACY BUG: a fold-cycling noise row
            #    (post-fold arm median ~24, arm max < 96) satisfies
            #    legacy r_dn and down-ticks INTO fold pressure.
            r_dn = r_dn & (row_gmed <= ARM_GAP_DN_MED) \
                & (row_gnow <= GAP_DN_SAFE)
        rflag = r_up.to(torch.int32) - r_dn.to(torch.int32)
        ar_new_max = (ar + rflag).max()
        if sov:
            c_cap = (ac + 1 + ar_new_max <= 10)
        else:
            c_cap = (ac < ARM_EXP_MAX) & (ac + 1 + ar_new_max <= ARM_SUM_MAX)
        c_up = (am_c >= ARM_UP_AT) & c_cap
        c_dn = (col_med <= ARM_DN_AT) & (col_now <= 63) \
            & (am_c < ARM_UP_AT) & (ac > ARM_EXP_MIN)
        if gapr:
            col_gmed = torch.cat(cg_med).to(torch.int32)
            col_gnow = torch.cat(cg_now).to(torch.int32)
            c_up = c_up | ((col_gmed >= ARM_GAP_UP_MED) & c_cap)
            c_dn = c_dn & (col_gmed <= ARM_GAP_DN_MED) \
                & (col_gnow <= GAP_DN_SAFE)
        cflag = c_up.to(torch.int32) - c_dn.to(torch.int32)
        self._reb_seed.add_(1)
        BLOCK_N, BLOCK_K = 32, 64
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
        _arm_retick_kernel[grid](
            self.packed_w, rflag, cflag, self._reb_seed, N, K,
            self.packed_w.stride(0), self.packed_w.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        self.arm_row_exp.add_(rflag.to(torch.int8))
        self.arm_col_exp.add_(cflag.to(torch.int8))
        am_r.zero_()
        am_c.zero_()


class ConcordConv2d2Fast(ConcordLinear2Fast):
    """Conv2d variant: packed state as (out_channels, in*kh*kw)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, device='cuda',
                 alpha=0.1, beta1=0.0, lr=0.01, bracket_d=0.25):
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = kh, kw
        self.stride = stride
        self.padding = padding
        super().__init__(in_features=in_channels * kh * kw,
                         out_features=out_channels,
                         bias=bias, device=device,
                         alpha=alpha, beta1=beta1, lr=lr,
                         bracket_d=bracket_d)
        self.register_buffer('v_full',
            torch.zeros(out_channels, in_channels * kh * kw,
                        dtype=torch.float32, device=device))

    def forward(self, x):
        in_dtype = x.dtype
        wbuf, rmbuf, cmbuf = self._ensure_buffers()
        y = FusedConcordConv2d2Fast.apply(
            x, self.packed_w, self.row_exp, self.col_exp,
            self.arm_row_exp, self.arm_col_exp,
            self._arm_row_max, self._arm_col_max, self._d_buf,
            self.bias,
            self.in_channels, self.out_channels, self.kh, self.kw,
            self.stride, self.padding,
            self._lr_buf, self.alpha, self.beta1, self.MANTISSA_BIAS,
            self.weight_decay, self._eps_buf, self.step_cap,
            self.v_scale, self.precond_p, self.gf_consol,
            self.drift_cancel_C, self.alpha_v_fast,
            self.wd_sv, self.wd_sf, self.wd_anchor,
            self.mass_preserve_v, self.apply_chase, self.track_rebalance,
            wbuf, rmbuf, cmbuf,
            self.v_row, self.v_col, self._sum_v_inv,
            float(self.adafactor_beta2),
            bool(self.track_adafactor_v),
            float(self.gf_trust_delta_sq), self._coh_pre,
            self._gf_consol_buf, self._beta1_buf,
            self.v_full, bool(self.use_full_v))
        return y.to(in_dtype)


# ============================================================
# Gate helpers (2-fast unpack; autotuner via measure_fn)
# ============================================================

@torch.no_grad()
def gate_coherence_from_fields2(e_L, e_H, s_slow_i8, v_slow_i8,
                                drift_cancel_C,
                                arm_row_exp=None, arm_col_exp=None):
    d_fs = (e_L + e_H).to(torch.float32)
    if arm_row_exp is not None:
        ae = arm_row_exp.to(torch.float32)[:, None]
        if arm_col_exp is not None:
            ae = ae + arm_col_exp.to(torch.float32)[None, :]
        d_fs = d_fs * torch.pow(2.0, ae)
    d_sv = (s_slow_i8.to(torch.float32) - v_slow_i8.to(torch.float32)) * 128.0
    sig = drift_cancel_C * d_sv
    nse = d_fs - sig
    return (sig * sig) / (sig * sig + nse * nse + 1e-30)


@torch.no_grad()
def measure_coherence2(layer):
    """The autotuner probe for 2-fast layers (pass as measure_fn — the
    default 16-bit unpack would misread the arms as one int16 field)."""
    e_L, e_H, ss, vs = layer.get_state()
    coh = gate_coherence_from_fields2(e_L, e_H, ss, vs,
                                      float(layer.drift_cancel_C),
                                      arm_row_exp=layer.arm_row_exp,
                                      arm_col_exp=layer.arm_col_exp)
    return float(coh.mean().item())


# ============================================================
# Startup self-test (the unguard): compile + sanity-run the 2-fast
# kernels on one small layer BEFORE any UNet swap. The kernels in this
# module have never executed until a GPU runs this — fail loudly at
# t=0, not mid-run.
# ============================================================

def _self_test_key(device):
    import hashlib
    import triton as _tr
    h = hashlib.sha256()
    h.update(open(__file__, 'rb').read())
    h.update(torch.__version__.encode())
    h.update(getattr(_tr, '__version__', '?').encode())
    h.update(torch.cuda.get_device_name(device).encode())
    return h.hexdigest()


def two_fast_self_test(device='cuda', verbose=True):
    import os as _os
    import time
    marker = __file__ + '.selftest-ok'
    key = _self_test_key(device)
    if _os.environ.get('CONCORD_2FAST_SELFTEST', '') != 'force':
        try:
            if open(marker).read().strip() == key:
                if verbose:
                    print('[concord] 2-fast self-test: cached PASS for this '
                          'engine hash (CONCORD_2FAST_SELFTEST=force to rerun)')
                return True
        except OSError:
            pass
    t0 = time.time()
    torch.manual_seed(0)
    n_in, n_out, bsz, steps = 64, 48, 16, 30
    if _pb._HELDOUT_ROUTER:
        # Routed arms run ~2x occupancy BY DESIGN, and the system that manages
        # sustained occupancy is the arm ratchet (up-shifts the arm exponent on
        # the watermark) -- which fires only every ARM_RATCHET_EVERY calls. A
        # 30-step test never lets it fire once, so it reads normal pre-ratchet
        # rail pressure as failure: it tests the router with its safety system
        # disabled. Span two ratchet periods plus settling so the decide runs.
        steps = 2 * ARM_RATCHET_EVERY + 8
    target_W = torch.randn(n_out, n_in, device=device) * 0.3
    x_all = torch.randn(256, n_in, device=device)
    y_all = F.linear(x_all, target_W)
    m = ConcordLinear2Fast(n_in, n_out, device=device, lr=0.01)
    m.gf_consol = 50.0
    m.load_weights(torch.randn(n_out, n_in, device=device) * 0.1)
    ratchet = GatedArmRatchet([m])
    losses = []
    for t in range(steps):
        if _pb._HELDOUT_ROUTER:
            # Mirror the trainer's per-micro alternation (GenericTrainer fills arm_sel
            # by global_step parity). Without this the selector sits at its default and
            # EVERY tick lands in one arm -- the degenerate configuration the accum
            # guard forbids -- railing that arm and failing the range check spuriously.
            _pb.set_arm_sel(device, (t & 1) == 0)
        idx = torch.randint(0, 256, (bsz,), device=device)
        y = m(x_all[idx])
        loss = F.mse_loss(y.float(), y_all[idx])
        loss.backward()
        m.rebalance()
        ratchet()
        losses.append(float(loss.detach()))
    l0 = sum(losses[:5]) / 5
    l1 = sum(losses[-5:]) / 5
    ok_conv = l1 < l0
    dep = m.consolidated_weight().float()
    ok_dep = bool(torch.isfinite(dep).all())
    gap_mag = float(m.arm_gap().abs().mean())
    e_L, e_H, _, _ = m.get_state()
    if _pb._HELDOUT_ROUTER:
        # Routed arms brush the rail between ratchet fires by design (occupancy ~2x;
        # the clamp spill keeps rail contact mass-preserving). The test window spans
        # two ratchet periods, so the exponent has had its chances to up-shift: fail
        # only on BROAD saturation that survives the ratchet.
        at_rail = 0.5 * (float((e_L.abs() >= 127).float().mean())
                         + float((e_H.abs() >= 127).float().mean()))
        ok_range = at_rail <= 0.05
    else:
        ok_range = int(e_L.abs().max()) <= 127 and int(e_H.abs().max()) <= 127
    if not (ok_conv and ok_dep and ok_range):
        _rail = (f" (rail fraction {at_rail:.3f} > 0.05, router on)"
                 if _pb._HELDOUT_ROUTER else
                 f" (|e_L|max={int(e_L.abs().max())}, |e_H|max={int(e_H.abs().max())})")
        raise RuntimeError(
            f"concord 2-fast startup self-test FAILED: loss {l0:.4f}->"
            f"{l1:.4f} (must decrease), deploy finite {ok_dep}, arms in "
            f"range {ok_range}{_rail}. The 2-fast kernels are not safe to swap "
            f"into the UNet — aborting before any model surgery.")
    if verbose:
        print(f"[concord] 2-fast self-test PASS: loss {l0:.4f}->{l1:.4f}, "
              f"|gap| {gap_mag:.2f}, arm_exp "
              f"[{int(m.arm_row_exp.min())},{int(m.arm_row_exp.max())}] "
              f"({time.time()-t0:.1f}s)")

    # ── realized-coherence CDF sketch sanity (read_cohq) ──
    _cq = read_cohq(device)
    ok_cq = _cq is not None and 0.0 <= _cq[0] <= _cq[1] <= _cq[2] <= 1.0 and _cq[3] > 0
    if not ok_cq:
        raise RuntimeError(
            f"concord 2-fast startup self-test FAILED: coherence CDF sketch returned "
            f"{_cq} (expected ordered quantiles in [0,1] with n>0).")
    if verbose:
        print(f"[concord] 2-fast self-test cohq PASS: q10/50/90="
              f"{_cq[0]:.2f}/{_cq[1]:.2f}/{_cq[2]:.2f} (n={_cq[3]})")

    # ── per-row floor plumbing (registry + CF/LF_PER_ROW kernel variant) ──
    # Parity is SINGLE-STEP: the armed and scalar paths are DIFFERENT compiled
    # binaries, so trajectory comparison amplifies FMA-order LSB noise through
    # stochastic-rounding threshold flips (measured ~1% after 24 hot steps --
    # variant jitter, not wrong values). One apply from identical packed state
    # bounds the true variant delta: bare compilation noise, orders below any
    # real plumbing error. Distinct rows then run a full trajectory for
    # finiteness (dynamics-level sanity; broadcast cannot see indexing bugs
    # anyway since every row holds the same value).
    mW0 = torch.randn(n_out, n_in, device=device) * 0.1

    def _floor_run(arm, steps_n):
        torch.manual_seed(2)
        mm = ConcordLinear2Fast(n_in, n_out, device=device, lr=0.01)
        mm.gf_consol = 50.0
        mm.load_weights(mW0.clone())
        cf_t, lf_t = _ensure_floor_tensors(mm.packed_w.device)
        cf_v, lf_v = float(cf_t.item()), float(lf_t.item())
        if arm == 'broadcast':
            register_row_floors(
                mm.packed_w,
                torch.full((n_out,), cf_v, dtype=torch.float32, device=mm.packed_w.device),
                torch.full((n_out,), lf_v, dtype=torch.float32, device=mm.packed_w.device))
        elif arm == 'distinct':
            register_row_floors(
                mm.packed_w,
                torch.linspace(0.0, min(2.0 * cf_v, 1.0), n_out,
                               dtype=torch.float32, device=mm.packed_w.device),
                torch.linspace(0.0, min(2.0 * lf_v, 1.0), n_out,
                               dtype=torch.float32, device=mm.packed_w.device))
        rat = GatedArmRatchet([mm])
        for tt in range(steps_n):
            if _pb._HELDOUT_ROUTER:
                _pb.set_arm_sel(device, (tt & 1) == 0)
            idx2 = torch.randint(0, 256, (bsz,), device=device)
            F.mse_loss(mm(x_all[idx2]).float(), y_all[idx2]).backward()
            mm.rebalance()
            rat()
        register_row_floors(mm.packed_w)       # disarm: registry hygiene
        return mm.consolidated_weight().float()

    dep_s = _floor_run(None, 1)
    dep_b = _floor_run('broadcast', 1)
    dep_d = _floor_run('distinct', 24)
    rel = float((dep_b - dep_s).abs().mean()
                / dep_s.abs().mean().clamp_min(1e-9))
    ok_rf = rel < 1e-3 and bool(torch.isfinite(dep_d).all())
    if not ok_rf:
        raise RuntimeError(
            f"concord 2-fast startup self-test FAILED: per-row floor plumbing "
            f"(single-step broadcast-vs-scalar rel dev {rel:.4g} must be < 1e-3; "
            f"distinct-rows finite {bool(torch.isfinite(dep_d).all())}). The per-row "
            f"gate-floor variant is not safe -- aborting before any model surgery.")
    if verbose:
        print(f"[concord] 2-fast self-test row-floor PASS: single-step broadcast "
              f"rel dev {rel:.2e}")
    try:
        with open(marker, 'w') as f:
            f.write(key)
    except OSError:
        pass                       # read-only install: just rerun next time
    return True
