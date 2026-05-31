"""Prototype B: packed int32 with int16 s_fast + int8×128 s_slow + int8×128 v_slow.

Storage layout (little-endian):
    bits [31:16]  s_fast      int16   — fine SR-tick accumulator, scale × 1
    bits [15:8]   s_slow_i8   int8    — coarse position bearer,    scale × 128
    bits [ 7:0]   v_slow_i8   int8    — long-time anchor,          scale × 128

Live weight:
    m_eff = s_slow_i8 × 128 + s_fast + v_slow_i8 × 128
    weight = m_eff × 2^(row_exp + col_exp - mantissa_bias)

Same shared-exponent envelope as the original int16 path: row_exp / col_exp
small per-row/col int8 tensors carry the per-row+col scale. Rebalance not
implemented in this prototype — for the smoke test, s_fast / s_slow stay
within range without it.

Compared to Option A (bf16 + s_fast_delta i8 + v_slow_i8):
  + s_fast at int16 — virtually non-saturating under heavy gradients
    (Option A's int8 delta saturated at 115-128 within 5-15 steps in the
    MLP smoke test, capping the SR-tick precision from then on)
  + Materialize is arithmetic-driven (not just memcpy + bitshift) but the
    optimizer dynamic doesn't need a per-element bf16 exponent — the
    shared row_exp/col_exp envelope is enough
  + Inherits the existing int16 path's rebalance + materialize logic
    structure (just operating on packed int32 instead of three buffers)
  + Architectural property still holds: 32 bits/param total. No bf16 in
    the persistent state — it's all int.

v_slow_i8 dynamics (leak + Bayesian wd) are skipped for prototype simplicity.
Set v_slow_i8 = 0 at load and leave untouched. Same is true for Option A.

Run:
    python prototype_packed_b.py
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


MANTISSA_BIAS = 15
INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
S_SLOW_FACTOR = 128
V_SLOW_FACTOR = 128


def compute_drift_cancel_C(alpha, alpha_v_fast,
                              alpha_v_slow=0.0, refit_period=1):
    """Drift-cancellation coefficient C* that zeroes E[noise] under
    a pure-drift gradient stream (μ = E[a_t], no noise).

    Derivation. With β1=0 and the kernel's update order, let
        L = (1−α)/α        (d_fs steady-state drift-lag ratio)
        ρ = α_vf + α_vs/T_r  (effective per-step v-leak: per-step + amortized periodic)
    Under pure drift μ the buffers all ride a ramp at rate μ; solving
    for the lags gives
        E[d_fs] = μ·L
        E[d_sv] = μ·(1 − L·α_vf) / ρ
    Setting E[noise] = E[d_fs] − C·E[d_sv] = 0:

        C* = L·ρ / (1 − L·α_vf)

    At packed-B defaults (α=0.1, α_v_fast=0.001, no periodic leak):
        L = 9, ρ = 0.001, L·α_vf = 0.009  →  C* ≈ 0.00908.
    With a periodic v-refit (α_v_slow > 0, T_r > 0) ρ grows and C*
    grows with it; e.g. α_v_slow/T_r = 0.00125 → ρ = 0.00225 → C* ≈ 0.0204.

    The shipped default of 0.1 was ~11× too large at the no-periodic-leak
    rates and ~5× too large even with the periodic leak; the README's
    0.196 formula dropped T_r from the denominator (had α − T_r·α_vf
    where the honest denominator is ∝ T_r·α), which is the bulk of the
    error. With C off, drift cancellation fails and the noise estimate
    picks up the signal — flattening the per-weight variance map to a
    near-constant and breaking the "which weights have converged?"
    diagnostic that the estimator is structurally capable of.
    """
    L = (1.0 - alpha) / max(alpha, 1e-12)
    rho = alpha_v_fast + alpha_v_slow / max(refit_period, 1)
    return L * rho / (1.0 - L * alpha_v_fast)


# ============================================================
# Triton kernels
# ============================================================

@triton.jit
def _hash_uniform(x, pos, salt):
    """xorshift: (x, pos, salt) int32 tensors → uniform [0,1) float32."""
    h = x ^ salt ^ pos
    h = h ^ (h << 13)
    h = h ^ (h >> 17)
    h = h ^ (h << 5)
    h = h ^ (h >> 7)
    return (h & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _materialize_packed_bf16_kernel(
    packed_ptr,        # [N, K] int32
    weight_ptr,        # [N, K] bf16
    row_exp_ptr,       # [N] int8
    col_exp_ptr,       # [K] int8
    N, K,
    mantissa_bias,
    stride_pn, stride_pk,
    stride_wn, stride_wk,
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

    # Unpack (arith shifts sign-extend each segment)
    s_fast    = packed >> 16
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24
    m_eff = s_slow_i8 * 128 + s_fast + v_slow_i8 * 128

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    weight_fp32 = m_eff.to(tl.float32) * tl.exp2(exp)

    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_ptr + w_off, weight_fp32.to(tl.bfloat16), mask=nk_mask)


def materialize_packed_bf16(packed_w, row_exp, col_exp, out,
                              mantissa_bias=15):
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.shape == packed_w.shape
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _materialize_packed_bf16_kernel[grid](
        packed_w, out, row_exp, col_exp, N, K, int(mantissa_bias),
        packed_w.stride(0), packed_w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


@triton.jit
def _apply_packed_sgd_kernel(
    packed_ptr,        # [N, K] int32, mutated in place
    grad_W_ptr,        # [N, K] bf16
    weight_buf_ptr,    # [N, K] bf16 — emit updated weight here (materialize-merge)
    row_exp_ptr,       # [N] int8
    col_exp_ptr,       # [K] int8
    row_max_ptr,       # [N] int32, pre-zeroed (rebalance bookkeeping)
    col_max_ptr,       # [K] int32, pre-zeroed
    N, K,
    lr_ptr, mantissa_bias, alpha, beta1, alpha_v_fast,
    step_salt_ptr,
    stride_pn, stride_pk,
    stride_gn, stride_gk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    TRACK_REBALANCE: tl.constexpr,
):
    """Packed apply kernel (SGD path).

    One int32 load → unpack (s_fast i16, s_slow_i8, v_slow_i8) → SR-tick
    on s_fast → mass-preserve chase moves α·s_fast into s_slow at int8
    granularity (each int8 tick = 128 mantissa units, SR-rounded) →
    v_slow_i8 leak chases s_slow_full at rate alpha_v_fast (mass-preserve
    toggleable: when True, the actual int8-clamped tick is subtracted
    from s_slow so the live mantissa is exactly conserved per step) →
    repack and store one int32."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    # lr is read from device tensor — lets us update it from Python
    # between CUDA-graph replays without re-capturing the graph.
    lr = tl.load(lr_ptr).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # ── load ────────────────────────────────────────────────────
    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast    = packed >> 16
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24

    g_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + g_off, mask=nk_mask, other=0.0).to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_inv = tl.exp2(-total_exp)

    # ── SR-tick on s_fast ───────────────────────────────────────
    delta_grad = -lr * grad_W * scale_inv     # mantissa units
    delta_t = delta_grad - beta1 * s_fast.to(tl.float32)

    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # ── mass-preserve chase ────────────────────────────────────
    # α·s_fast is the chase amount in MANTISSA units. The s_slow_i8
    # tick is the same amount at × 128 scale: chase_in_int8 = chase_mantissa / 128.
    # SR-round to int8 units. Mass-preserve: s_fast loses
    # actual_tick * 128 mantissa.
    chase_mantissa = alpha * s_fast.to(tl.float32)
    chase_int8_f = chase_mantissa / 128.0
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_s = tl.floor(chase_int8_f)
    frac_s = chase_int8_f - floor_s
    tick_slow_i8 = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
    actual_tick_mantissa = tick_slow_i8 * 128
    s_slow_i8 = s_slow_i8 + tick_slow_i8
    s_fast = s_fast - actual_tick_mantissa

    # ── v_slow_i8 leak ─ target s_slow_full (the position).
    s_slow_full_post = s_slow_i8 * 128
    v_slow_full = v_slow_i8 * 128
    gap_v_full = (s_slow_full_post - v_slow_full).to(tl.float32)
    delta_v8 = alpha_v_fast * gap_v_full / 128.0
    r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
    floor_v = tl.floor(delta_v8)
    frac_v = delta_v8 - floor_v
    tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
    # Clamp v_slow_i8 to int8 range BEFORE mass-preserve subtraction —
    # otherwise the saturating-clamp would lose mass that should have
    # gone into s_slow.
    new_v_int8 = tl.minimum(tl.maximum(v_slow_i8 + tick_v8, -128), 127)
    if MASS_PRESERVE:
        # The actual int8 tick after clamping (could differ from the
        # unclamped tick_v8 near saturation). Subtract its mantissa
        # equivalent from s_slow so the live mantissa
        # (s_slow*128 + s_fast + v_slow*128) is conserved per step.
        actual_tick_v8 = new_v_int8 - v_slow_i8
        s_slow_i8 = s_slow_i8 - actual_tick_v8

    # ── clamp and repack ───────────────────────────────────────
    s_fast_c = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    s_slow_c = tl.minimum(tl.maximum(s_slow_i8, -128), 127)
    v_slow_c = new_v_int8
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | ((s_slow_c & 0xFF) << 8)
        | (v_slow_c & 0xFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── materialize-merge: emit the NEW bf16 weight for next step's
    # forward, eliminating the separate _materialize_packed_bf16 kernel
    # launch. We already have all the data in registers — total_exp,
    # scale_fwd, and the new s_fast/s_slow/v_slow values. One extra
    # fp32→bf16 cast + store per element.
    new_m_eff = s_slow_c * 128 + s_fast_c + v_slow_c * 128
    scale_fwd = tl.exp2(total_exp)
    new_weight = new_m_eff.to(tl.float32) * scale_fwd
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_buf_ptr + w_off,
             new_weight.to(tl.bfloat16), mask=nk_mask)

    # ── atomic-max for rebalance (gated — typical CIFAR run never
    # triggers rebalance, so skipping the reduction + atomics is a
    # ~5-15% win on the apply kernel) ──────────────────────────
    if TRACK_REBALANCE:
        # Full live mantissa: all three accumulators contribute.
        # The previous version used only (s_slow*128 + s_fast), which
        # underestimated the true saturation risk by ~2x because v_slow
        # tracks s_slow tightly and contributes another ~equal mass.
        abs_eff = tl.abs(s_slow_c * 128 + s_fast_c + v_slow_c * 128)
        abs_eff = tl.where(nk_mask, abs_eff, 0)
        tile_row_max = tl.max(abs_eff, axis=1)
        tile_col_max = tl.max(abs_eff, axis=0)
        tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask)
        tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask)


_STEP_COUNTERS = {}


def _get_step_counter(device):
    key = str(device)
    if key not in _STEP_COUNTERS:
        _STEP_COUNTERS[key] = torch.zeros(1, dtype=torch.int32, device=device)
    return _STEP_COUNTERS[key]


_LR_SCALAR_CACHE = {}


def _ensure_lr_tensor(lr, device):
    """Return a 1-elem fp32 tensor holding `lr`. If `lr` is already a
    tensor, return it. Otherwise reuse a per-device cached tensor and
    refill it — this lets us call the kernel with a scalar lr without
    forcing the caller to manage a tensor."""
    if isinstance(lr, torch.Tensor):
        return lr
    key = str(device)
    buf = _LR_SCALAR_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _LR_SCALAR_CACHE[key] = buf
    buf.fill_(float(lr))
    return buf


_EPS_SCALAR_CACHE = {}


def _ensure_eps_tensor(eps, device):
    """Like _ensure_lr_tensor but with a SEPARATE per-device cache (sharing
    the lr cache would alias the two scalars). If `eps` is already a tensor
    (the layer passes its _eps_buf), return it unchanged."""
    if isinstance(eps, torch.Tensor):
        return eps
    key = str(device)
    buf = _EPS_SCALAR_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _EPS_SCALAR_CACHE[key] = buf
    buf.fill_(float(eps))
    return buf


def apply_packed_sgd(packed_w, grad_W, weight_buf, row_exp, col_exp,
                       row_max, col_max,
                       lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                       alpha_v_fast=0.001, mass_preserve=False,
                       track_rebalance=True):
    """SGD apply kernel. weight_buf is a [N, K] bf16 tensor that the
    kernel updates with the new live weight (materialize-merge) so the
    next step's forward can skip a separate materialize_packed_bf16
    call. Must be the same shape and dtype as packed_w but in bf16."""
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16
    assert weight_buf.dtype == torch.bfloat16
    assert weight_buf.shape == packed_w.shape
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    lr_ptr = _ensure_lr_tensor(lr, packed_w.device)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_packed_sgd_kernel[grid](
        packed_w, grad_W, weight_buf, row_exp, col_exp,
        row_max, col_max,
        N, K,
        lr_ptr, int(mantissa_bias), float(alpha), float(beta1),
        float(alpha_v_fast),
        step_counter,
        packed_w.stride(0), packed_w.stride(1),
        grad_W.stride(0), grad_W.stride(1),
        weight_buf.stride(0), weight_buf.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        MASS_PRESERVE=bool(mass_preserve),
        TRACK_REBALANCE=bool(track_rebalance),
    )


# ============================================================
# AdamW three-accumulator apply kernel (packed).
#
# Drift-cancel variance from the three stored quantities:
#   d_fs = s_fast              — velocity (delta from s_slow position)
#   d_sv = s_slow*128 - v_slow*128
#   noise = d_fs - drift_cancel_C * d_sv
# AdamW step preconditioned by noise² → v_proxy. Step clamped by step_cap.
#
# drift_cancel_C is the C* from compute_drift_cancel_C(alpha, alpha_v_fast,
# alpha_v_slow, refit_period) — the analytic coefficient that zeroes
# E[noise] under pure drift. Layer's __init__ wires it up; passing it
# explicitly to apply_packed_adamw also accepts None (auto-compute).
# With C wrong, the drift-cancel fails and the estimator picks up the
# signal as well as the noise, flattening the per-weight variance map.
# SR-tick lands in s_fast. Mass-preserve chase moves α·s_fast from
# s_fast into s_slow_i8 (at int8 × 128 quantization, SR-rounded).
# v_slow_i8 leak chases s_slow_full at rate alpha_v_fast. Bayesian-
# anchored wd ticks pull s_slow toward v_slow_full (wd_sv) and
# s_fast_logical toward v_slow_full (wd_sf).
#
# All three accumulators packed into one int32 word; this kernel does
# one int32 load + all math in registers + one int32 store.
# ============================================================

@triton.jit
def _apply_packed_adamw_kernel(
    packed_ptr,        # [N, K] int32, mutated in place
    grad_W_ptr,        # [N, K] bf16
    weight_buf_ptr,    # [N, K] bf16 — emit updated weight here (materialize-merge)
    row_exp_ptr,       # [N] int8
    col_exp_ptr,       # [K] int8
    row_max_ptr,       # [N] int32, pre-zeroed
    col_max_ptr,       # [K] int32, pre-zeroed
    v_row_ptr,         # [N] fp32 — Adafactor row second-moment EMA (g²)
    v_col_ptr,         # [K] fp32 — Adafactor col second-moment EMA (g²)
    sum_v_inv_ptr,     # [1] fp32 — 1 / Σ_k v_row_k (precomputed by caller)
    coh_pre_ptr,       # [N,K] fp32 — per-coord established-coherence EMA
    N, K,
    lr_ptr, mantissa_bias, alpha, beta1,
    weight_decay, eps_ptr, step_cap,
    v_scale, precond_p, gf_consol, drift_cancel_C, alpha_v_fast,
    wd_sv, wd_sf,
    gf_trust_delta_sq,   # fp32: δ² in step = grad/√(Var + δ²·v̂);
                          # 0 disables (legacy: clamp by step_cap only)
    gate_gain,           # fp32: scalar cosine schedule on the commitment gate
    chase_floor,         # fp32: ratio-coh fast->slow floor (cosine ->0 over ~1 epoch)
    leak_floor,          # fp32: ratio-coh slow->v_slow floor
                          # (1.0 = off). Anneals α·gate·s_fast consolidation.
    step_salt_ptr,
    stride_pn, stride_pk,
    stride_gn, stride_gk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,     # for v_slow leak (mass out of s_slow)
    APPLY_CHASE: tl.constexpr,
    TRACK_REBALANCE: tl.constexpr,
    USE_GF_TRUST_REGION: tl.constexpr,  # True → add δ²·v̂ floor to v_proxy
    USE_GF_CONSOLIDATION: tl.constexpr,  # True → gf-gated evaporation routing
    USE_COHPRE: tl.constexpr,  # True → coh_pre-gated acceptance (chase)
    USE_FIXED_COH: tl.constexpr,  # True → Wiener coh = S/(S+noise²) (units-correct)
    USE_RATIO_COH: tl.constexpr,  # True -> gate chase+leak by live coh, no coh_pre
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    # lr is read from a device tensor so it can change between
    # CUDA-graph replays without re-capturing the graph.
    lr = tl.load(lr_ptr).to(tl.float32)
    # eps likewise read from a device tensor — lets an eps warmup
    # schedule update it between graph replays (e.g. SGD->precond handoff).
    eps = tl.load(eps_ptr).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # ── load packed + unpack ───────────────────────────────────
    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast    = packed >> 16
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

    # ── live weight + drift-cancel variance ───────────────────
    m_eff = s_slow_full + s_fast + v_slow_full
    current_weight = m_eff.to(tl.float32) * scale_fwd

    # Delta convention: s_fast IS the velocity directly.
    d_fs = s_fast.to(tl.float32)
    d_sv = (s_slow_full - v_slow_full).to(tl.float32)
    noise = d_fs - drift_cancel_C * d_sv
    noise_in_w = noise * scale_fwd
    v_proxy = noise_in_w * noise_in_w * v_scale

    # Garbage-fraction trust region: add δ²·v̂ floor to v_proxy so the
    # implied step is bounded by ~1/δ at gf→0 (no step_cap needed) and
    # decays to 0 at gf→1 (no spurious noise-floor motion on converged
    # weights). v̂ is Adafactor rank-1 — same scale (g² = W²) as v_proxy,
    # so the sum is dimensionally sound. v̂ here is a *typical-gradient-
    # magnitude reference*, not a variance estimate — different role
    # from the drift-cancel noise² that stays the primary preconditioner.
    # v̂ (Adafactor rank-1, W² units) feeds both the trust region and the
    # consolidation-coherence gate, so load it if either is active.
    coh = 0.0
    if USE_GF_TRUST_REGION or USE_GF_CONSOLIDATION or USE_COHPRE or USE_RATIO_COH:
        v_row_tile = tl.load(v_row_ptr + offs_n,
                              mask=n_mask, other=0.0).to(tl.float32)
        v_col_tile = tl.load(v_col_ptr + offs_k,
                              mask=k_mask, other=0.0).to(tl.float32)
        sum_v_inv = tl.load(sum_v_inv_ptr).to(tl.float32)
        v_hat = v_row_tile[:, None] * v_col_tile[None, :] * sum_v_inv
    if USE_GF_TRUST_REGION:
        v_proxy = v_proxy + gf_trust_delta_sq * v_hat
    if USE_GF_CONSOLIDATION or USE_COHPRE or USE_RATIO_COH:
        # Momentum = displacement between two time-lagged positions: the
        # mean gradient over v_slow's window is α_v·d_sv (mantissa units),
        # → W units via scale_fwd. Coherence = (mean grad)² / E[g²] ∈ [0,1]
        # (Cauchy–Schwarz). Reconstructed from s_slow−v_slow + v̂ — no new
        # buffer. coh→1 = coherent signal, coh→0 = incoherent noise.
        if USE_FIXED_COH:
            # Dimensionally-correct Wiener/Kalman SNR gate. Both terms from the
            # SAME velocity decomposition d_fs = signal + noise: signal =
            # drift_cancel_C·d_sv (the drift), noise = noise_in_w (already
            # computed). coh = S²/(S²+N²) ∈[0,1]; the lr/scale cancels -> true
            # gradient-SNR (vs the broken α_v·d_sv vs E[g²] units mismatch).
            sig_w = drift_cancel_C * d_sv * scale_fwd
            sig2 = sig_w * sig_w
            coh = sig2 / (sig2 + noise_in_w * noise_in_w + 1e-30)
        else:
            mean_grad_w = alpha_v_fast * d_sv * scale_fwd
            coh = mean_grad_w * mean_grad_w / (v_hat + 1e-12)
        coh = tl.minimum(tl.maximum(coh, 0.0), 1.0)

    # ── AdamW step ─────────────────────────────────────────────
    # Weight decay enters via step_live as `wd * current_weight`. After
    # the -lr * scale_inv multiplication this becomes a mantissa tick
    # of -lr * wd * m_eff that lands in s_fast only — the chase
    # naturally migrates it to s_slow over ~1/α steps. s_slow and
    # v_slow are NOT decayed directly; they're tracking accumulators.
    # Decoupled from the grad step in the sense that wd is not divided
    # by (v+eps)^p; only the gradient term is preconditioned.
    # precond_p is the Padam-style partial-adaptivity exponent: p=0.5 is
    # the usual (unjustified) sqrt; p=0 gives (denom)^0=1 => step=grad
    # (pure SGD); p in (0,0.5) interpolates between the linear (SGD) and
    # fully-smoothed (trust-region) regimes. Use exp2/log2 for the pow;
    # v_proxy+eps > 0 always (eps>0) so the log is safe.
    denom_p = tl.exp2(precond_p * tl.log2(v_proxy + eps))
    step_live = grad_W / denom_p
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    # Cautious weight decay: decay s_fast (the velocity) toward 0,
    # equivalently decay the live weight W toward (s_slow + v_slow)·
    # scale_fwd (the persistent "slow position"), NOT toward 0. The
    # rationale: the slow accumulators already carry the "correct"
    # long-term weight via the chase + v_slow leak dynamics; wd's job
    # is to damp the recent gradient velocity, not to crush the
    # underlying position. This solves the "uniform wd crushes conv
    # features" problem because conv weights are pulled toward their
    # natural learned magnitude, not toward 0.
    #
    # wd reference is d_fs · scale_fwd (= s_fast in W units) instead of
    # current_weight (= full m_eff in W units). After -lr·scale_inv,
    # the wd contribution to s_fast is -lr·wd·s_fast — a simple
    # exponential decay of s_fast with rate lr·wd per step.
    s_fast_in_w = d_fs * scale_fwd
    if USE_GF_CONSOLIDATION:
        # gf-gated evaporation REPLACES uniform cautious wd: drain only the
        # INCOHERENT part of the velocity. coh→1 (signal) is preserved and
        # flows to s_slow via the unconditional chase; coh→0 (noise) decays
        # out of s_fast before it can consolidate. gf_consol = κ < α so the
        # chase outruns the skim and d_sv (hence coherence) can bootstrap.
        # lr-proportional, like the cautious wd this replaces (lr·wd·s_fast):
        # the cosine lr→0 auto-fades the skim in the tail so the accumulated
        # position can settle (constant-κ over-skimmed late, when small
        # late-stage signal reads as incoherent). gf_consol is now the rate
        # at unit lr; at peak lr=0.1, gf_consol=0.3 ≈ the old κ=0.03.
        evap_mantissa = lr * gf_consol * (1.0 - coh) * d_fs
    else:
        step_live = step_live + weight_decay * s_fast_in_w
        evap_mantissa = 0.0
    delta_grad = -lr * step_live * scale_inv     # mantissa units
    delta_t = delta_grad - beta1 * d_fs - evap_mantissa

    # ── SR-tick s_fast ────────────────────────────────────────
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Default v_slow_i8 value (used in repack regardless of APPLY_CHASE).
    new_v_int8 = v_slow_i8

    if APPLY_CHASE:
        # ── mass-preserve chase (coh_pre-gated acceptance) ──
        # Gate the chase (= acceptance into the slow accumulator) by
        # coh + coh_pre·(1-coh): coherent signal (coh→1) accepts fully;
        # incoherent (coh→0) accepts only at the per-coord established floor
        # coh_pre. coh_pre is an EMA of coh (init 1, rate λ=alpha_v_fast),
        # so it self-terminates for never-coherent (noise) coords -> bounds
        # the diffusion of noise into s_slow at the source, while holding
        # converged coords (high coh_pre memory) that plain coh would drop.
        gate = 1.0
        if USE_RATIO_COH:
            gate = chase_floor + (1.0 - chase_floor) * coh   # fast->slow, floored
        elif USE_COHPRE:
            coh_pre = tl.load(coh_pre_ptr + p_off,
                              mask=nk_mask, other=1.0).to(tl.float32)
            gate = coh + coh_pre * (1.0 - coh)
            coh_pre_new = (1.0 - alpha_v_fast) * coh_pre + alpha_v_fast * coh
            tl.store(coh_pre_ptr + p_off, coh_pre_new, mask=nk_mask)
        chase_mantissa = alpha * gate * gate_gain * s_fast.to(tl.float32)
        chase_int8_f = chase_mantissa / 128.0
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(chase_int8_f)
        frac_s = chase_int8_f - floor_s
        tick_slow_i8 = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow_i8 = s_slow_i8 + tick_slow_i8
        s_fast = s_fast - tick_slow_i8 * 128

        # ── v_slow_i8 leak ─ target s_slow_full (the position)
        # In delta convention, the leak target is the cumulative position
        # (s_slow_full), not s_fast (which is now a small velocity).
        s_slow_full_post = s_slow_i8 * 128
        gap_v_full = (s_slow_full_post - v_slow_full).to(tl.float32)
        delta_v8 = alpha_v_fast * gap_v_full / 128.0
        if USE_RATIO_COH:
            delta_v8 = delta_v8 * (leak_floor + (1.0 - leak_floor) * coh)   # slow->v_slow, floored
        r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
        floor_v = tl.floor(delta_v8)
        frac_v = delta_v8 - floor_v
        tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
        new_v_int8 = tl.minimum(tl.maximum(v_slow_i8 + tick_v8, -128), 127)
        if MASS_PRESERVE:
            # Mass out of s_slow (the position), not s_fast (small velocity).
            actual_tick_v8 = new_v_int8 - v_slow_i8
            s_slow_i8 = s_slow_i8 - actual_tick_v8

        # ── Bayesian-anchored wd ──
        v_slow_full_post = new_v_int8 * 128
        # wd_sv: pull s_slow_i8 toward v_slow_full_post (at int8 scale).
        d_sv_full_post = (s_slow_i8 * 128 - v_slow_full_post).to(tl.float32)
        wd_sv_delta_int8 = lr * wd_sv * d_sv_full_post / 128.0
        r4 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x66665555)
        floor_wd_sv = tl.floor(wd_sv_delta_int8)
        frac_wd_sv = wd_sv_delta_int8 - floor_wd_sv
        tick_wd_sv = (floor_wd_sv + (r4 < frac_wd_sv).to(tl.float32)).to(tl.int32)
        s_slow_i8 = s_slow_i8 - tick_wd_sv

        # wd_sf: pull s_fast_logical (= s_slow_full_post + s_fast) toward
        # v_slow_full_post. Tick goes to s_fast (mantissa units).
        s_fast_logical_post = s_slow_i8 * 128 + s_fast
        d_sf_full_post = (s_fast_logical_post - v_slow_full_post).to(tl.float32)
        wd_sf_delta = lr * wd_sf * d_sf_full_post
        r5 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x77770000)
        floor_wd_sf = tl.floor(wd_sf_delta)
        frac_wd_sf = wd_sf_delta - floor_wd_sf
        tick_wd_sf = (floor_wd_sf + (r5 < frac_wd_sf).to(tl.float32)).to(tl.int32)
        s_fast = s_fast - tick_wd_sf

    # ── clamp and repack ──────────────────────────────────────
    s_fast_c = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    s_slow_c = tl.minimum(tl.maximum(s_slow_i8, -128), 127)
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | ((s_slow_c & 0xFF) << 8)
        | (new_v_int8 & 0xFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── materialize-merge: emit the NEW bf16 weight for next step's
    # forward. scale_fwd is already in registers from earlier. One
    # extra fp32→bf16 cast + store eliminates the separate
    # _materialize_packed_bf16 kernel launch per forward.
    new_m_eff = s_slow_c * 128 + s_fast_c + new_v_int8 * 128
    new_weight = new_m_eff.to(tl.float32) * scale_fwd
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_buf_ptr + w_off,
             new_weight.to(tl.bfloat16), mask=nk_mask)

    # ── atomic-max for rebalance (gated — skipped when caller knows
    # row/col exponents will never trip the threshold). ──
    if TRACK_REBALANCE:
        # Full live mantissa: s_slow + s_fast + v_slow. Previous version
        # was missing v_slow*128 which underestimated saturation risk
        # by ~2x (v_slow tracks s_slow tightly via the leak, so the two
        # contribute roughly equal magnitudes).
        abs_eff = tl.abs(s_slow_c * 128 + s_fast_c + new_v_int8 * 128)
        abs_eff = tl.where(nk_mask, abs_eff, 0)
        tile_row_max = tl.max(abs_eff, axis=1)
        tile_col_max = tl.max(abs_eff, axis=0)
        tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask)
        tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask)


# ── Denominator-term diagnostic ───────────────────────────────────
# When enabled, apply_packed_adamw recomputes the step denominator in
# torch (exactly mirroring the kernel) and prints the median magnitude
# of each term, so we can SEE which of {drift-cancel noise², gf floor
# δ²·v̂, eps} dominates, whether step_cap binds, and the actual mantissa
# delta. Keyed per-tensor so each layer prints at its own step counts.
_DENOM_DIAG = {"enabled": False, "at_steps": (1, 3, 5, 10, 20, 50, 100, 200),
               "counts": {}}


@torch.no_grad()
def _denom_diagnostic(packed_w, grad_W, row_exp, col_exp,
                       v_row, v_col, sum_v_inv, use_gf,
                       lr, eps, v_scale, drift_cancel_C, gf_trust_delta_sq,
                       step_cap, mantissa_bias):
    key = id(packed_w)
    c = _DENOM_DIAG["counts"].get(key, 0) + 1
    _DENOM_DIAG["counts"][key] = c
    if c not in _DENOM_DIAG["at_steps"]:
        return
    N, K = packed_w.shape
    s_fast = (packed_w >> 16).to(torch.float32)
    s_slow_i8 = ((packed_w << 16) >> 24).to(torch.float32)
    v_slow_i8 = ((packed_w << 24) >> 24).to(torch.float32)
    exp = (row_exp[:, None].to(torch.float32)
           + col_exp[None, :].to(torch.float32) - mantissa_bias)
    scale_fwd = torch.pow(2.0, exp)
    scale_inv = torch.pow(2.0, -exp)
    d_fs = s_fast
    d_sv = s_slow_i8 * 128.0 - v_slow_i8 * 128.0
    noise_in_w = (d_fs - drift_cancel_C * d_sv) * scale_fwd
    drift_term = noise_in_w * noise_in_w * v_scale
    if use_gf:
        vhat = (v_row[:, None].to(torch.float32)
                * v_col[None, :].to(torch.float32)
                * sum_v_inv.to(torch.float32))
        gf_term = gf_trust_delta_sq * vhat
    else:
        gf_term = torch.zeros_like(drift_term)
    denom = drift_term + gf_term + eps
    g = grad_W.to(torch.float32)
    step_raw = g / torch.sqrt(denom)
    step_capped = step_raw.clamp(-step_cap, step_cap)
    delta = -lr * step_capped * scale_inv

    def med(t):           # median: for dense terms (drift, denom)
        return t.abs().float().median().item()

    def mn(t):            # mean: for ReLU-sparse grad-derived terms
        return t.abs().float().mean().item()
    # cap% over the NONZERO-grad elements (dead-ReLU zeros would dilute it).
    nz = g.abs() > 0
    nz_frac = nz.float().mean().item()
    cap_frac = ((step_raw.abs() > step_cap) & nz).float().sum().item() \
        / max(nz.float().sum().item(), 1.0)
    print(f"[denom-diag {N}x{K} step{c:>3}] "
          f"drift={med(drift_term):.2e} gf={mn(gf_term):.2e} "
          f"eps={eps:.0e} denom={med(denom):.2e} "
          f"|g|av={mn(g):.2e} nz={nz_frac*100:4.0f}% "
          f"|step_raw|av={mn(step_raw):.2e} "
          f"cap%nz={cap_frac*100:5.1f} |delta|av={mn(delta):.2e}",
          flush=True)

    # gf-bucketed breakdown (one layer only, to keep output readable):
    # split NONZERO-grad weights into clean (gf<0.1) vs noisy (gf>0.9),
    # where gf = noise²/v̂ in [0,1]. Shows whether the trust region gives
    # clean weights a healthy step and suppresses noisy ones — and at
    # what radius the clean-bucket |delta| lands at the SGD scale (~100).
    if use_gf and N == 512 and K == 4096:
        gf_elem = (drift_term / (vhat + 1e-30)).clamp(0.0, 1.0)
        clean = nz & (gf_elem < 0.1)
        noisy = nz & (gf_elem > 0.9)

        def bmn(t, m):
            cnt = m.float().sum().item()
            return (t.abs().float() * m.float()).sum().item() / max(cnt, 1.0)
        fc = clean.float().mean().item()
        fn = noisy.float().mean().item()
        print(f"[denom-diag {N}x{K} step{c:>3}]   gf-buckets: "
              f"clean(gf<.1)={fc*100:4.1f}% |step_raw|={bmn(step_raw, clean):.2e} "
              f"|delta|={bmn(delta, clean):.2e}  ||  "
              f"noisy(gf>.9)={fn*100:4.1f}% |step_raw|={bmn(step_raw, noisy):.2e} "
              f"|delta|={bmn(delta, noisy):.2e}",
              flush=True)


# (Free deviation-preconditioner removed 2026-05-31: the "dev" source was
# self-degraded / lost on CIFAR, and the "grad" source is superseded by the
# rank-1 v-hat that is now the baked default. grad_W flows straight to apply.)


# Use the units-correct Wiener coh = S/(S+noise²) for the coherence gate (vs the
# broken S/v̂ that reads ~0). Module-global so it bakes into the kernel constexpr
# without threading through the 30-arg autograd Function. Set once before training.
_USE_FIXED_COH = True   # validated default: Wiener coh S/(S+noise^2). False=legacy(broken units)
# Scalar cosine schedule on the commitment gate (1.0 = off). Set per-step.
_GATE_GAIN = 1.0


def set_fixed_coh(enabled):
    global _USE_FIXED_COH
    _USE_FIXED_COH = bool(enabled)


def set_gate_gain(g):
    global _GATE_GAIN
    _GATE_GAIN = float(g)


# EXPERIMENTAL (default OFF, so the validated default is untouched): gate the
# rank-1 variance ACCUMULATION by established coherence (coh_pre). The per-element
# weight is consumed in the row/col marginal sums, so v_row/v_col stay rank-1
# (O(N+K)); v_hat becomes a rank-1 fit to coherent gradient power, not raw g^2.
# Read in the autograd Function backward (Python), so no kernel/constexpr change.
# Demotes v_hat to scale-equalization over the active set, leaning on the
# coherence gate for noise rejection. The weight is NORMALIZED by its per-layer mean
# so v_hat only RESHAPES onto coherent power without shrinking magnitude. (The
# un-normalized/raw form effectively raised the LR on the still-training coords and
# HURT the enwik8 tail by ~+0.01 at 10k; normalized is NEUTRAL there and churn-free.
# enwik8 is data>>capacity, so the spared noise-suppression role has nothing to
# reclaim -- candidate-useful only in the overfitting regime.) Read in the autograd
# Function backward (Python); no kernel/constexpr change.
_COH_WEIGHTED_V = False


def set_coh_weighted_v(enabled):
    global _COH_WEIGHTED_V
    _COH_WEIGHTED_V = bool(enabled)


# EXPERIMENTAL (default OFF): ratio-coherence gate. Gate BOTH the chase
# (s_fast->s_slow) and the leak (s_slow->v_slow) by the live Wiener coh and DROP
# coh_pre: the per-coord s_fast:s_slow:v_slow ratio then carries the accumulated
# (established) coherence -- coherent mass settles into v_slow, noise stays in
# s_fast -- so no fp32 coh_pre buffer (back to 32 bits/param). Pair with
# disable_cohpre() on the layers.
_RATIO_COH = False
# Per-transition bootstrap floors: min flow at coh=0 so the gated cascade ignites
# from the all-s_fast init (d_sv=0 -> coh=0 would deadlock). Cosine-decay both to 0
# over ~one epoch -> pure coherence gating, no permanent noise leak. Global scalars.
_RATIO_CHASE_FLOOR = 0.9     # fast -> slow  (beta1-like timescale)
_RATIO_LEAK_FLOOR = 0.999    # slow -> v_slow (beta2-like timescale)


def set_ratio_coh(enabled):
    global _RATIO_COH
    _RATIO_COH = bool(enabled)


def set_ratio_coh_floors(chase, leak):
    """Per-step setter for the two bootstrap floors. Schedule: cosine from 0.9
    (fast->slow) / 0.999 (slow->v_slow) to 0 over ~one epoch."""
    global _RATIO_CHASE_FLOOR, _RATIO_LEAK_FLOOR
    _RATIO_CHASE_FLOOR = float(chase)
    _RATIO_LEAK_FLOOR = float(leak)


def apply_packed_adamw(packed_w, grad_W, weight_buf, row_exp, col_exp,
                         row_max, col_max,
                         lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                         weight_decay=0.0, eps=1.0, step_cap=10.0,
                         v_scale=1.0, precond_p=0.5, gf_consol=0.0,
                         drift_cancel_C=None,
                         alpha_v_fast=0.001,
                         wd_sv=0.0, wd_sf=0.0,
                         mass_preserve=False, apply_chase=True,
                         track_rebalance=True,
                         v_row=None, v_col=None, sum_v_inv=None,
                         gf_trust_delta_sq=0.0, coh_pre=None):
    """Wrapper for the AdamW three-accumulator packed apply kernel.
    weight_buf is updated with the new live weight (materialize-merge),
    so the next forward can read directly without a separate materialize
    kernel call.

    `drift_cancel_C=None` (default) computes C* from (alpha, alpha_v_fast)
    via compute_drift_cancel_C — the analytic value that zeroes E[noise]
    under pure drift. Pass an explicit float to override.

    `gf_trust_delta_sq > 0` enables the garbage-fraction trust region:
    `step = grad / √(noise²·v_scale + δ²·v̂ + ε)`. v̂ from Adafactor
    rank-1 (v_row, v_col, sum_v_inv required). δ²=1/step_cap² gives the
    same asymptotic max step as the legacy hard clamp, smoothly.
    """
    if drift_cancel_C is None:
        drift_cancel_C = compute_drift_cancel_C(alpha, alpha_v_fast)
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16
    assert weight_buf.dtype == torch.bfloat16
    assert weight_buf.shape == packed_w.shape
    use_gf_trust = (gf_trust_delta_sq > 0)
    use_gf_consol = (gf_consol > 0)
    use_cohpre = (coh_pre is not None)
    coh_pre_arg = coh_pre if coh_pre is not None else packed_w  # stub if off
    if use_gf_trust or use_gf_consol or use_cohpre:
        assert v_row is not None and v_col is not None \
            and sum_v_inv is not None, \
            "gf_trust/gf_consol/coh_pre require v_row, v_col, sum_v_inv"
    else:
        # Stub unused pointers; kernel branch never reads them.
        v_row = v_row if v_row is not None else packed_w
        v_col = v_col if v_col is not None else packed_w
        sum_v_inv = sum_v_inv if sum_v_inv is not None else packed_w
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    if _DENOM_DIAG["enabled"]:
        _denom_diagnostic(packed_w, grad_W, row_exp, col_exp,
                          v_row, v_col, sum_v_inv, use_gf_trust,
                          lr, float(eps), float(v_scale),
                          float(drift_cancel_C), float(gf_trust_delta_sq),
                          float(step_cap), int(mantissa_bias))
    lr_ptr = _ensure_lr_tensor(lr, packed_w.device)
    eps_ptr = _ensure_eps_tensor(eps, packed_w.device)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_packed_adamw_kernel[grid](
        packed_w, grad_W, weight_buf, row_exp, col_exp,
        row_max, col_max,
        v_row, v_col, sum_v_inv,
        coh_pre_arg,
        N, K,
        lr_ptr, int(mantissa_bias), float(alpha), float(beta1),
        float(weight_decay), eps_ptr, float(step_cap),
        float(v_scale), float(precond_p), float(gf_consol),
        float(drift_cancel_C),
        float(alpha_v_fast),
        float(wd_sv), float(wd_sf),
        float(gf_trust_delta_sq),
        float(_GATE_GAIN),
        float(_RATIO_CHASE_FLOOR),
        float(_RATIO_LEAK_FLOOR),
        step_counter,
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
        USE_FIXED_COH=bool(_USE_FIXED_COH),
        USE_RATIO_COH=bool(_RATIO_COH),
    )


# ============================================================
# Rebalance kernel: tick-up exponent + SR-right-shift of all 3
# accumulators when row/col max exceeds threshold.
# ============================================================

@triton.jit
def _rebalance_packed_decide_kernel(
    packed_ptr,
    row_exp_ptr, col_exp_ptr,
    row_max_ptr, col_max_ptr,
    row_i8med_ptr, col_i8med_ptr,
    N, K, MAX_M, EXP_MAX, EXP_MIN,
    seed_ptr,
    stride_pn, stride_pk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALLOW_TICKDOWN: tl.constexpr,
):
    """Decide per-row/col exponent tick-ups from row_max/col_max and
    rebalance the packed accumulators by `pos` ∈ {0, +1, +2}.

    PRECISION-PRESERVING REBALANCE: instead of independently right-
    shifting each accumulator (which loses 1 bit of s_slow/v_slow
    precision per fire), we capture the mantissa-unit residual of the
    SR-rounded right-shift on s_slow_i8 and v_slow_i8 and add it into
    s_fast (which has 1-mantissa granularity). The consolidated state
    precision isn't lost — it migrates into s_fast where the natural
    chase dynamics will bleed it back into s_slow_i8 over ~10 steps.

    Math (expectation):
      E[new_m_eff] = E[new_s_slow_full] + E[new_s_fast] + E[new_v_slow_full]
                   = s_slow_full/2^pos + s_fast/2^pos + v_slow_full/2^pos
                   = m_eff/2^pos
      E[new_live]  = E[new_m_eff] × 2^(exp+pos) = live   ✓
    """
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

    # ASYMMETRIC ratchet (matches the asymmetric stakes):
    #   tick UP -- eager, MAX-driven: any single element overflowing (row_max >
    #     MAX_M) saturates the block, so bump the exponent immediately.
    #   tick DOWN -- lazy, MEDIAN-driven: only when the BULK underflows (i8med,
    #     the per-row/col median |s_slow|/|v_slow|, <= 31) does the block read as
    #     genuinely underused. Left-shift the whole block; the MINORITY of large
    #     elements (> ~31) saturate on the shift (we do NOT require a leading
    #     zero on every mantissa -- the bulk reclaims precision, outliers clip).
    #   The median (vs max) is what kills the growth-phase churn: the max dips
    #   transiently during growth (-> false tick-down -> oscillation -> extra
    #   lossy tick-ups), but the median only drops when the block has settled low.
    # Up/down mutually exclusive per dim (row_max can't be both > and <= MAX_M).
    row_up = (row_max > MAX_M) & (row_exp < EXP_MAX)
    col_up = (col_max > MAX_M) & (col_exp < EXP_MAX)
    # tick-down OFF by default: empirically it HURT the packed v-hat (1.43->1.17
    # no-td vs 1.26 max-gated vs 1.35 median-gated) -- per-step tick-down
    # oscillates with the v-hat chase's exponent ratcheting, and the median gate
    # additionally clips legit max weights (median<max for any spread). Kept
    # behind ALLOW_TICKDOWN for a future settled-phase-only experiment.
    if ALLOW_TICKDOWN:
        row_dn = (row_max <= MAX_M) & (row_i8med <= 31) & (row_exp > EXP_MIN)
        col_dn = (col_max <= MAX_M) & (col_i8med <= 31) & (col_exp > EXP_MIN)
    else:
        row_dn = row_max < 0      # all-False (row_max >= 0 always)
        col_dn = col_max < 0
    row_t = row_up.to(tl.int32) - row_dn.to(tl.int32)   # {-1, 0, +1}
    col_t = col_up.to(tl.int32) - col_dn.to(tl.int32)
    net = row_t[:, None] + col_t[None, :]               # in {-2, ..., +2}

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast    = packed >> 16
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24

    # Bidirectional rebalance per element by sign of `net`:
    #   net > 0: SR-RIGHT-shift all 3 (tick-up); int8 s_slow/v_slow
    #            quantization residual migrated into s_fast (fine granularity).
    #   net < 0: LOSSLESS LEFT-shift all 3 (tick-down) — exponent decreased so
    #            the mantissa must grow; no rounding/residual. The upstream int8
    #            headroom gate (i8max<=31) guarantees the <<2 worst case stays
    #            in int8 range.
    #   net = 0: unchanged (rsh=0 makes the up-path an identity).
    rand_off = offs_n[:, None] * K + offs_k[None, :]
    rsh = tl.maximum(net, 0)              # right-shift amount (tick-up) >= 0
    lsh = tl.maximum(-net, 0)             # left-shift amount (tick-down) >= 0
    take_left = net < 0
    two_pos = tl.exp2(rsh.to(tl.float32))

    # --- tick-UP path (SR-right-shift) ---
    q_fast = s_fast >> rsh
    rem_fast = (s_fast - (q_fast << rsh)).to(tl.float32)
    up_fast = (tl.rand(seed, rand_off) * two_pos < rem_fast).to(tl.int32)
    s_fast_shifted = q_fast + up_fast

    q_slow = s_slow_i8 >> rsh
    rem_slow = (s_slow_i8 - (q_slow << rsh)).to(tl.float32)
    up_slow = (tl.rand(seed, rand_off + N * K) * two_pos < rem_slow).to(tl.int32)
    s_slow_up = q_slow + up_slow
    # Mantissa residual in int math (rsh in {0,1,2}): (s_slow*128)/2^rsh - new*128
    s_slow_residual = (s_slow_i8 << (7 - rsh)) - (s_slow_up << 7)

    q_v = v_slow_i8 >> rsh
    rem_v = (v_slow_i8 - (q_v << rsh)).to(tl.float32)
    up_v = (tl.rand(seed, rand_off + 2 * N * K) * two_pos < rem_v).to(tl.int32)
    v_up = q_v + up_v
    v_residual = (v_slow_i8 << (7 - rsh)) - (v_up << 7)

    # Combined residual bounded by ±192 (rsh=2), within int16 added to s_fast.
    s_fast_up = s_fast_shifted + s_slow_residual + v_residual

    # --- tick-DOWN path (lossless left-shift) + select per element ---
    s_fast_new = tl.where(take_left, s_fast << lsh, s_fast_up)
    s_slow_new = tl.where(take_left, s_slow_i8 << lsh, s_slow_up)
    v_new      = tl.where(take_left, v_slow_i8 << lsh, v_up)

    # Clamp + repack.
    s_fast_c = tl.minimum(tl.maximum(s_fast_new, -32768), 32767)
    s_slow_c = tl.minimum(tl.maximum(s_slow_new, -128), 127)
    v_c = tl.minimum(tl.maximum(v_new, -128), 127)
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | ((s_slow_c & 0xFF) << 8)
        | (v_c & 0xFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    if pid_k == 0:
        tl.store(row_exp_ptr + offs_n, row_exp + row_t, mask=n_mask)
    if pid_n == 0:
        tl.store(col_exp_ptr + offs_k, col_exp + col_t, mask=k_mask)


# Optional rebalance instrumentation (set to a dict via reset_reb_stats() to
# accumulate tick-up/down fire counts + left-shift clip counts; None = off).
_REB_STATS = None


def reset_reb_stats():
    global _REB_STATS
    _REB_STATS = dict(calls=0, tickup_dim=0, tickdown_dim=0, dim_total=0,
                      clip_elems=0, elem_total=0)


def get_reb_stats():
    return _REB_STATS


_REB_SEED_CACHE = {}


def _ensure_reb_seed_tensor(device):
    """Process-global rebalance-seed tensor (one per device). Used by
    callers that don't have their own per-layer seed buffer; the kernel
    reads from a pointer so the SR-rounding inside rebalance is graph-
    capturable (no host-side .item() needed)."""
    key = str(device)
    buf = _REB_SEED_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _REB_SEED_CACHE[key] = buf
    return buf


def rebalance_packed(packed_w, row_exp, col_exp, row_max, col_max,
                       MAX_M=24000, EXP_MAX=7, EXP_MIN=-8, seed_buf=None,
                       allow_tickdown=False):
    """Apply the rebalance ratchet: tick-UP the exponent if row/col live
    mantissa exceeds MAX_M (regain headroom), tick-DOWN to reclaim precision
    if the row/col has headroom AND its int8 s_slow/v_slow magnitudes are
    small enough that a left-shift can't overflow. Assumes row_max/col_max
    already populated by a prior apply kernel.

    The per-row/col int8 magnitude max is computed here (torch, capturable:
    no host sync) from packed_w and gates tick-down -- m_eff being small is
    NOT sufficient (s_fast can cancel a large s_slow), so we check the actual
    int8 accumulators that would overflow on the left-shift.

    seed_buf is a 1-elem int32 device tensor. Caller should bump it in
    place (e.g., `seed_buf.add_(1)`) before each call so successive
    rebalances draw different SR-rounding streams. If None, a process-
    global per-device buffer is used (but the caller must still bump it
    if running multiple rebalance() calls back-to-back outside a graph).
    """
    if seed_buf is None:
        seed_buf = _ensure_reb_seed_tensor(packed_w.device)
    N, K = packed_w.shape
    # int8 s_slow / v_slow magnitudes (sign-extended via int32 arithmetic
    # shifts), reduced to per-row/col MEDIAN -> the "bulk underflow" signal that
    # gates the asymmetric tick-down (median small = block genuinely underused;
    # robust to the transient-max dips that churn a max-gated tick-down).
    s_slow_i8 = (packed_w << 16) >> 24
    v_slow_i8 = (packed_w << 24) >> 24
    i8mag = torch.maximum(s_slow_i8.abs(), v_slow_i8.abs())        # [N,K] int32
    row_i8med = i8mag.median(dim=1).values.to(torch.int32)        # [N]
    col_i8med = i8mag.median(dim=0).values.to(torch.int32)        # [K]
    if _REB_STATS is not None:
        re = row_exp.to(torch.int32); ce = col_exp.to(torch.int32)
        ru = (row_max > MAX_M) & (re < EXP_MAX)
        cu = (col_max > MAX_M) & (ce < EXP_MAX)
        if allow_tickdown:
            rd = (row_max <= MAX_M) & (row_i8med <= 31) & (re > EXP_MIN)
            cd = (col_max <= MAX_M) & (col_i8med <= 31) & (ce > EXP_MIN)
        else:
            rd = torch.zeros_like(ru); cd = torch.zeros_like(cu)
        rt = ru.to(torch.int32) - rd.to(torch.int32)
        ct = cu.to(torch.int32) - cd.to(torch.int32)
        lsh = (-(rt[:, None] + ct[None, :])).clamp(min=0).to(torch.float32)
        clip = ((i8mag.to(torch.float32) * torch.exp2(lsh)) > 127).sum().item()
        _REB_STATS['calls'] += 1
        _REB_STATS['tickup_dim'] += int(ru.sum() + cu.sum())
        _REB_STATS['tickdown_dim'] += int(rd.sum() + cd.sum())
        _REB_STATS['dim_total'] += N + K
        _REB_STATS['clip_elems'] += clip
        _REB_STATS['elem_total'] += N * K
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _rebalance_packed_decide_kernel[grid](
        packed_w, row_exp, col_exp, row_max, col_max,
        row_i8med, col_i8med,
        N, K, int(MAX_M), int(EXP_MAX), int(EXP_MIN),
        seed_buf,
        packed_w.stride(0), packed_w.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ALLOW_TICKDOWN=bool(allow_tickdown),
    )


# ============================================================
# Autograd Function + nn.Module
# ============================================================

class FusedConcordLinearPackedB(torch.autograd.Function):
    """Dispatches to SGD or AdamW apply kernel based on optimizer_kind."""

    @staticmethod
    def forward(ctx, x, packed_w, row_exp, col_exp, bias,
                lr, alpha, beta1, mantissa_bias,
                optimizer_kind,
                weight_decay, eps, step_cap,
                v_scale, precond_p, gf_consol, drift_cancel_C, alpha_v_fast,
                wd_sv, wd_sf,
                mass_preserve, apply_chase, track_rebalance,
                weight_buf, row_max_buf, col_max_buf,
                v_row, v_col, sum_v_inv,
                adafactor_beta2, track_adafactor_v,
                gf_trust_delta_sq, coh_pre):
        # weight_buf is kept fresh by the *previous* step's apply kernel
        # (materialize-merge). On the very first forward, the layer's
        # _ensure_buffers materializes it once. So we read it directly
        # without launching a separate materialize_packed_bf16 here.
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = F.linear(x, weight_buf, bias_bf16)
        ctx.save_for_backward(x, weight_buf)
        ctx.packed_w = packed_w
        ctx.row_exp = row_exp
        ctx.col_exp = col_exp
        ctx.lr = lr
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.mantissa_bias = mantissa_bias
        ctx.optimizer_kind = optimizer_kind
        ctx.weight_decay = weight_decay
        ctx.eps = eps
        ctx.step_cap = step_cap
        ctx.v_scale = v_scale
        ctx.precond_p = precond_p
        ctx.gf_consol = gf_consol
        ctx.drift_cancel_C = drift_cancel_C
        ctx.alpha_v_fast = alpha_v_fast
        ctx.wd_sv = wd_sv
        ctx.wd_sf = wd_sf
        ctx.mass_preserve = mass_preserve
        ctx.apply_chase = apply_chase
        ctx.track_rebalance = track_rebalance
        ctx.has_bias = bias is not None
        ctx.weight_buf = weight_buf
        ctx.row_max_buf = row_max_buf
        ctx.col_max_buf = col_max_buf
        ctx.v_row = v_row
        ctx.v_col = v_col
        ctx.sum_v_inv = sum_v_inv
        ctx.adafactor_beta2 = adafactor_beta2
        ctx.track_adafactor_v = track_adafactor_v
        ctx.gf_trust_delta_sq = gf_trust_delta_sq
        ctx.coh_pre = coh_pre
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()
        # grad_x: broadcasts batch dims naturally over weight [N,K].
        grad_x = grad_y @ weight
        # grad_W: flatten leading batch dims so grad_y_flat is [B*..., N]
        # and x_flat is [B*..., K]. grad_y_flat.T @ x_flat = [N, K].
        # The unflattened form `grad_y.transpose(-1,-2) @ x` only works
        # for 2D inputs — for 3D (e.g., transformer (B, L, K)) it gives
        # a per-batch slice, not the reduced [N, K] we need.
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        x_flat = x.reshape(-1, in_features)
        grad_y_flat = grad_y.reshape(-1, out_features)
        grad_W = grad_y_flat.transpose(0, 1) @ x_flat
        if grad_W.dtype != torch.bfloat16:
            grad_W = grad_W.to(torch.bfloat16)
        if not grad_W.is_contiguous():
            grad_W = grad_W.contiguous()
        # Only zero the rebalance bookkeeping buffers if we're going to
        # write to them — skipping the zeros saves two more kernel
        # launches per layer per backward.
        if ctx.track_rebalance:
            ctx.row_max_buf.zero_()
            ctx.col_max_buf.zero_()
        # Adafactor row/col second-moment EMA update — done BEFORE apply
        # so the kernel sees this step's freshest v̂ when the gf-trust
        # region floor is active. Also feeds the garbage-fraction
        # diagnostic (passive use). β2=0.999 matches v_slow EMA rate.
        if ctx.track_adafactor_v and ctx.v_row is not None:
            with torch.no_grad():
                g2 = grad_W.float() ** 2
                if _COH_WEIGHTED_V and ctx.coh_pre is not None:
                    # normalized: reshape onto coherent power, preserve magnitude
                    # (eff. LR). Weight consumed in the marginals below.
                    w = ctx.coh_pre / ctx.coh_pre.mean().clamp(min=1e-12)
                    g2 = g2 * w
                g2_row = g2.sum(dim=1)
                g2_col = g2.sum(dim=0)
                b2 = ctx.adafactor_beta2
                ctx.v_row.mul_(b2).add_(g2_row, alpha=1.0 - b2)
                ctx.v_col.mul_(b2).add_(g2_col, alpha=1.0 - b2)
                # Precompute 1/Σv_row for the kernel.
                sum_v = ctx.v_row.sum().clamp(min=1e-30)
                ctx.sum_v_inv.fill_(0).add_(1.0 / sum_v)
        # Apply kernel emits the updated bf16 weight into ctx.weight_buf
        # as a side effect — that's what the next forward will read.
        if ctx.optimizer_kind == 'adamw':
            apply_packed_adamw(
                ctx.packed_w, grad_W, ctx.weight_buf,
                ctx.row_exp, ctx.col_exp,
                ctx.row_max_buf, ctx.col_max_buf,
                lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                alpha=ctx.alpha, beta1=ctx.beta1,
                weight_decay=ctx.weight_decay, eps=ctx.eps,
                step_cap=ctx.step_cap,
                v_scale=ctx.v_scale, precond_p=ctx.precond_p,
                gf_consol=ctx.gf_consol,
                drift_cancel_C=ctx.drift_cancel_C,
                alpha_v_fast=ctx.alpha_v_fast,
                wd_sv=ctx.wd_sv, wd_sf=ctx.wd_sf,
                mass_preserve=ctx.mass_preserve,
                apply_chase=ctx.apply_chase,
                track_rebalance=ctx.track_rebalance,
                v_row=ctx.v_row, v_col=ctx.v_col,
                sum_v_inv=ctx.sum_v_inv,
                gf_trust_delta_sq=ctx.gf_trust_delta_sq,
                coh_pre=ctx.coh_pre,
            )
        else:
            apply_packed_sgd(
                ctx.packed_w, grad_W, ctx.weight_buf,
                ctx.row_exp, ctx.col_exp,
                ctx.row_max_buf, ctx.col_max_buf,
                lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                alpha=ctx.alpha, beta1=ctx.beta1,
                alpha_v_fast=ctx.alpha_v_fast,
                mass_preserve=ctx.mass_preserve,
                track_rebalance=ctx.track_rebalance)
        if ctx.has_bias:
            # Sum over all leading dims (handles both 2D and 3D inputs).
            grad_bias = grad_y_flat.sum(0)
        else:
            grad_bias = None
        # 30 forward args. Only x (slot 0) and bias (slot 4) get grads.
        return (grad_x, None, None, None, grad_bias,
                None, None, None, None,
                None,
                None, None, None,
                None, None, None, None, None,
                None, None,
                None, None, None,
                None, None, None,
                None, None, None,
                None, None, None, None)


class ConcordLinearPackedB(nn.Module):
    """Single int32 buffer per layer. int16 s_fast + int8×128 s_slow +
    int8×128 v_slow, all in one word. Shared exponent via row_exp/col_exp.

    Supports both SGD and AdamW (three_accum). v_slow_i8 leak and
    Bayesian-anchored wd are active in AdamW mode.

    DEFAULT (2026-05-31+) = the validated production optimizer:
    optimizer_kind='adamw' with rank-1 v-hat (v_scale=0, gf_trust_delta_sq=1,
    eps=1e-10, precond_p=0.5) and the fixed coherence gate ON (coh_pre
    allocated in __init__; module global _USE_FIXED_COH=True). A bare
    ConcordLinearPackedB(in, out) IS that optimizer — no knob-setting needed,
    which is what makes it a drop-in for OneTrainer / any trainer. Recover the
    legacy SGD-chase via set_optimizer_kind('sgd'); ablate the gate via
    disable_cohpre(); per-knob attributes override the rest.
    """

    MANTISSA_BIAS = 15
    EXP_MIN = -8
    EXP_MAX = 7
    MAX_M = 24000   # rebalance threshold (matches int16 path)

    def __init__(self, in_features, out_features, bias=True,
                 device='cuda', alpha=0.1, beta1=0.0, lr=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta1 = beta1
        # lr is stored as both a scalar (self._lr_value) and a 1-elem
        # device tensor (self._lr_buf). The kernel reads the tensor so we
        # can update lr between CUDA-graph replays without re-capturing.
        # Setting `m.lr = X` (via the property) updates both.
        self._lr_value = float(lr)
        # Optimizer kind + hyperparameters. DEFAULTS = the validated production
        # recipe: rank-1 v-hat AdamW (v_scale=0, gf_trust=1, eps<<1, precond=0.5)
        # + fixed coherence gate (coh_pre allocated below). A bare
        # ConcordLinearPackedB IS the validated optimizer; override for ablations.
        self.optimizer_kind = 'adamw'
        self.weight_decay = 0.0
        self._eps_value = 1e-10   # backing store for `eps`; <<1 engages the v-hat denom
        self.step_cap = 10.0
        self.v_scale = 0.0     # 0 = kill the (proven-dead) velocity-noise v_proxy precond
        self.precond_p = 0.5   # Padam-style preconditioner power: 0.5=sqrt
                               # (default), 0=SGD, (0,0.5)=partial adaptivity
        self.gf_consol = 0.0   # gf-gated consolidation evaporation rate κ.
                               # 0 = off (uniform cautious wd). >0 enables the
                               # routing (evaporate incoherent s_fast, keep κ<α).
        self._coh_pre = None   # coherence-gate EMA; ALLOCATED BELOW (default ON,
        self.fast_gain = 1.0   # gate s_fast share of the FORWARD weight (1.0=full
                               # m_eff; ->0 deploys slow+v_slow). Packed keeps full s_fast.
                               # validated). disable_cohpre() sets it back to None.
        self.alpha_v_fast = 0.001
        # Derived from rates (see compute_drift_cancel_C docstring).
        # Setting `m.drift_cancel_C = X` after construction overrides.
        # If callers change alpha or alpha_v_fast (or add periodic leak),
        # they should recompute via compute_drift_cancel_C.
        self.drift_cancel_C = compute_drift_cancel_C(
            alpha, self.alpha_v_fast)
        # Adafactor row/col second-moment EMAs in g² units. Used for
        # the per-weight v̂_ij = v_row_i·v_col_j / Σ_k v_row_k denominator
        # of the garbage fraction Var(ḡ_ij) / E[ḡ_ij²] ∈ [0,1]. β2=0.999
        # matches the packed-B v_slow EMA rate (1 − alpha_v_fast).
        # Tracked passively in backward; feeds the per-weight v̂ that
        # the garbage-fraction diagnostic uses as the E[g²] denominator
        # AND the gf-trust-region floor δ²·v̂ when enabled. β2=0.999
        # matches v_slow's EMA rate. Set `track_adafactor_v=False` to
        # skip the EMA if you want neither diagnostic nor trust region.
        self.track_adafactor_v = True
        self.adafactor_beta2 = 0.999
        # Garbage-fraction trust region: when > 0, adds δ²·v̂ to v_proxy
        # so step is implicitly bounded by ~1/δ at gf→0 and → 0 at gf→1
        # — replaces the hard step_cap clamp with a smooth SNR gate.
        # δ²=1/step_cap² matches the legacy asymptotic max step.
        self.gf_trust_delta_sq = 1.0   # validated: 1 => v_hat IS the denom (rank-1 Adam)
        # Bidirectional rebalance tick-down (reclaim exponent precision). OFF by
        # default: it hurt the packed v-hat (oscillates with the chase). Set True
        # to experiment (e.g. a settled-phase-only schedule).
        self.allow_tickdown = False
        self.wd_sv = 0.0
        self.wd_sf = 0.0
        self.mass_preserve_v = True   # default to mass-preserving v_slow leak
        self.apply_chase = True
        # Default True for backward-compat. Set False in the cifar driver
        # for ~5-15% speedup on the apply kernel when we know rebalance
        # won't fire (state magnitudes stay well under MAX_M=24000).
        self.track_rebalance = True
        # Process-global rebalance counter; used to key SR stream.
        self.register_buffer('_reb_seed',
            torch.zeros(1, dtype=torch.int32, device=device))

        self.register_buffer('packed_w',
            torch.zeros(out_features, in_features,
                        dtype=torch.int32, device=device))
        self.register_buffer('row_exp',
            torch.zeros(out_features, dtype=torch.int8, device=device))
        self.register_buffer('col_exp',
            torch.zeros(in_features, dtype=torch.int8, device=device))
        # Adafactor row/col second-moment EMAs in g² units. Σv_row=Σv_col.
        self.register_buffer('v_row',
            torch.zeros(out_features, dtype=torch.float32, device=device))
        self.register_buffer('v_col',
            torch.zeros(in_features, dtype=torch.float32, device=device))
        # 1/Σv_row, updated per step; kernel reads it to compute v̂ per
        # element without a per-launch reduction. Initialized to 1 to
        # avoid div-by-zero on the first step (when v_row is still 0).
        self.register_buffer('_sum_v_inv',
            torch.ones(1, dtype=torch.float32, device=device))
        # Coherence gate ON by default (validated recipe): per-coord coh_pre EMA
        # (fp32, init 1.0). Apply kernel gates the chase by coh + coh_pre*(1-coh).
        self._coh_pre = torch.ones_like(self.packed_w, dtype=torch.float32)
        # Per-row/col running high-watermark of |s_slow*128 + s_fast +
        # v_slow*128| (the full live mantissa). Diagnostic: tells us
        # how close we ever came to MAX_M=24000 across training, which
        # answers "is rebalance not firing because we never approach
        # the threshold, or because the threshold is wrong?".
        self.register_buffer('_row_max_hwm',
            torch.zeros(out_features, dtype=torch.int32, device=device))
        self.register_buffer('_col_max_hwm',
            torch.zeros(in_features, dtype=torch.int32, device=device))
        # Device-side lr tensor — read by the apply kernel each step so
        # we can update lr between CUDA-graph replays.
        self.register_buffer('_lr_buf',
            torch.full((1,), self._lr_value,
                       dtype=torch.float32, device=device))
        # Device-side eps tensor — same trick, lets an eps warmup schedule
        # update it between graph replays (SGD -> preconditioner handoff).
        self.register_buffer('_eps_buf',
            torch.full((1,), self._eps_value,
                       dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,
                                                  dtype=torch.bfloat16,
                                                  device=device))
        else:
            self.register_parameter('bias', None)
        self._init_weight()
        # Eagerly allocate + populate weight_buf so external code (like
        # transformers) can read `.weight` before the first forward call.
        # _ensure_buffers also materializes the bf16 weight from packed_w
        # so it's immediately readable.
        self._ensure_buffers()

    @property
    def weight(self):
        """Compatibility shim for code that introspects nn.Linear's
        `.weight` attribute (e.g., T5 does an isinstance/dtype check).
        Returns the bf16 materialized weight buffer (kept fresh by the
        apply kernel). Read-only — assigning to it has no effect on the
        packed state."""
        return getattr(self, '_bf16_weight_buf', None)

    @property
    def lr(self):
        return self._lr_value

    @lr.setter
    def lr(self, value):
        """Set lr and (if the device buffer exists) sync it to the GPU.
        The .fill_() is a kernel launch but lives OUTSIDE any captured
        CUDA graph, so subsequent g.replay() calls see the fresh value."""
        v = float(value)
        self._lr_value = v
        buf = getattr(self, '_lr_buf', None)
        if buf is not None:
            buf.fill_(v)

    @property
    def eps(self):
        return self._eps_value

    @eps.setter
    def eps(self, value):
        """Set eps and sync to the device buffer (read by the apply kernel),
        so an eps schedule propagates to subsequent CUDA-graph replays —
        same pattern as lr."""
        v = float(value)
        self._eps_value = v
        buf = getattr(self, '_eps_buf', None)
        if buf is not None:
            buf.fill_(v)

    @torch.no_grad()
    def enable_cohpre(self):
        """Allocate the per-coord established-coherence EMA buffer (fp32,
        init 1.0, contiguous [N,K] to share packed_w's stride layout). Once
        set, the apply kernel gates acceptance by coh + coh_pre·(1-coh) and
        EMA-updates coh_pre at rate alpha_v_fast. fp32 (not bf16) because the
        ~1e-3 per-step EMA increment is below bf16 resolution near 1.0."""
        self._coh_pre = torch.ones_like(self.packed_w, dtype=torch.float32)
        return self._coh_pre

    @torch.no_grad()
    def disable_cohpre(self):
        """Turn the coherence gate OFF (no-gate ablation): _coh_pre -> None, so
        the apply kernel takes the ungated chase path."""
        self._coh_pre = None

    def set_optimizer_kind(self, kind, weight_decay=0.0, eps=1.0,
                              step_cap=10.0):
        if kind not in ('sgd', 'adamw'):
            raise ValueError(f"optimizer kind must be 'sgd' or 'adamw'")
        self.optimizer_kind = kind
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.step_cap = float(step_cap)

    def _init_weight(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        w = torch.randn(self.out_features, self.in_features,
                        device=self.packed_w.device) * std
        self.load_weights(w)

    @torch.no_grad()
    def load_weights(self, W):
        """Put the full mantissa in s_fast at init. s_slow_i8 starts at 0
        and the mass-preserve chase fills it in over the first ~1/alpha
        steps. v_slow_i8 stays at 0 (unused in this prototype).

        This matches Option A's init pattern (bf16 = position, delta = 0)
        and Option-A-int16-path's _init pattern (s_slow = full, s_fast = 0)
        — except here the "fast" side carries the init mantissa, since
        s_slow at × 128 quantization can't represent fine mantissa values
        exactly. The chase will redistribute over the first ~10 steps.
        """
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
        m_total = (W / scale).round().to(torch.int32).clamp(
            INT16_MIN, INT16_MAX)
        s_fast = m_total                          # int32 in int16 range
        s_slow_i8 = torch.zeros_like(s_fast)
        v_slow_i8 = torch.zeros_like(s_fast)
        packed = (
            ((s_fast & 0xFFFF) << 16)
            | ((s_slow_i8 & 0xFF) << 8)
            | (v_slow_i8 & 0xFF)
        )
        self.packed_w.copy_(packed)
        self._resync_weight_buf()

    @torch.no_grad()
    def _resync_weight_buf(self):
        """Re-materialize the bf16 weight buffer from the current packed_w
        state. Call after any external mutation of packed_w (e.g.,
        load_weights or load_weights_finetune) so that the next forward
        sees the fresh weight without launching a separate materialize."""
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if wbuf is not None:
            materialize_packed_bf16(self.packed_w, self.row_exp,
                                      self.col_exp, out=wbuf,
                                      mantissa_bias=self.MANTISSA_BIAS)

    @torch.no_grad()
    def get_weight(self):
        """Read live bf16 weight from packed state (CPU-side recon, for
        diagnostics + state introspection)."""
        s_fast    = (self.packed_w >> 16)
        s_slow_i8 = ((self.packed_w << 16) >> 24)
        v_slow_i8 = ((self.packed_w << 24) >> 24)
        m_eff = (s_slow_i8 * S_SLOW_FACTOR + s_fast
                 + v_slow_i8 * V_SLOW_FACTOR)
        exp = (self.row_exp[:, None].to(torch.int32)
               + self.col_exp[None, :].to(torch.int32)
               - self.MANTISSA_BIAS).to(torch.float32)
        w_fp32 = m_eff.to(torch.float32) * torch.pow(2.0, exp)
        return w_fp32.to(torch.bfloat16)

    @torch.no_grad()
    def get_state(self):
        """Diagnostic: return (s_fast, s_slow_i8, v_slow_i8) tensors."""
        s_fast    = (self.packed_w >> 16)
        s_slow_i8 = ((self.packed_w << 16) >> 24)
        v_slow_i8 = ((self.packed_w << 24) >> 24)
        return s_fast, s_slow_i8, v_slow_i8

    @torch.no_grad()
    def get_rebalance_watermark_stats(self):
        """Returns (row_hwm, col_hwm) tensors — the all-time per-row /
        per-col max of |s_slow*128 + s_fast + v_slow*128| (the full
        live mantissa) seen during training. Compare these to
        self.MAX_M (rebalance trigger threshold = 24000) to see how
        close training came to firing rebalance.

        Returns None,None if the layer hasn't run any apply with
        track_rebalance=True (so the buffers were never populated)."""
        rmbuf = getattr(self, '_row_max_buf', None)
        if rmbuf is None:
            return None, None
        if not getattr(self, 'track_rebalance', True):
            return None, None
        return self._row_max_hwm.clone(), self._col_max_hwm.clone()

    @torch.no_grad()
    def get_garbage_fraction_stats(self):
        """Per-weight garbage fraction Var(ḡ_ij) / E[ḡ_ij²] ∈ [0,1].

        Numerator: variance from the drift-cancelled noise residual in
        W² units (= mantissa² × scale_fwd²).
        Denominator: Adafactor rank-1 reconstruction of E[g²]:
            v̂_ij = v_row_i · v_col_j / Σ_k v_row_k
        — also in W² units (we EMA'd grad_W² directly).

        Returns dict with median, p25, p50, p75, mean of the per-weight
        garbage fraction. Values near 0 = signal-dominated (weight has
        clear gradient direction); near 1 = noise-dominated (weight is
        effectively converged, remaining motion is noise).

        Returns None if v_row hasn't been populated yet.
        """
        if (self.v_row is None or float(self.v_row.sum().item()) <= 0.0):
            return None
        # Numerator: per-weight noise variance in W² units.
        # Unpack accumulators (matching the apply kernel's view).
        s_fast    = (self.packed_w >> 16).to(torch.float32)
        s_slow_i8 = ((self.packed_w << 16) >> 24).to(torch.float32)
        v_slow_i8 = ((self.packed_w << 24) >> 24).to(torch.float32)
        s_slow_full = s_slow_i8 * S_SLOW_FACTOR
        v_slow_full = v_slow_i8 * V_SLOW_FACTOR
        d_fs = s_fast                                  # velocity
        d_sv = s_slow_full - v_slow_full               # slow-vs-very-slow
        noise_mantissa = d_fs - float(self.drift_cancel_C) * d_sv
        # Convert to W² units.
        exp = (self.row_exp[:, None].to(torch.float32)
               + self.col_exp[None, :].to(torch.float32)
               - self.MANTISSA_BIAS)
        scale_fwd = torch.pow(2.0, exp)               # [N, K]
        var_W = (noise_mantissa * scale_fwd) ** 2     # [N, K]
        # Denominator: rank-1 v̂ in W² units.
        v_row = self.v_row.float()
        v_col = self.v_col.float()
        sum_v = v_row.sum().clamp(min=1e-30)
        v_hat = v_row[:, None] * v_col[None, :] / sum_v   # [N, K]
        # Garbage fraction, clamped to [0, 1].
        eps = 1e-30
        gf = (var_W / (v_hat + eps)).clamp(min=0.0, max=1.0)
        flat = gf.flatten().float()
        # Quantiles + mean. .quantile is cheap on contiguous flat tensors.
        q = torch.tensor([0.25, 0.5, 0.75], device=flat.device)
        q_vals = torch.quantile(flat, q)
        return {
            'mean': flat.mean().item(),
            'p25': q_vals[0].item(),
            'median': q_vals[1].item(),
            'p75': q_vals[2].item(),
            'frac_signal_dominated': (flat < 0.1).float().mean().item(),
            'frac_noise_dominated': (flat > 0.9).float().mean().item(),
        }

    def _ensure_buffers(self):
        N, K = self.packed_w.shape
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if wbuf is None or wbuf.shape != self.packed_w.shape:
            wbuf = torch.empty(self.packed_w.shape, dtype=torch.bfloat16,
                                device=self.packed_w.device)
            self._bf16_weight_buf = wbuf
            # One-shot materialize: from now on, the apply kernel keeps
            # wbuf in sync with packed_w (materialize-merge), so the
            # forward path doesn't need a separate materialize launch.
            materialize_packed_bf16(self.packed_w, self.row_exp,
                                      self.col_exp, out=wbuf,
                                      mantissa_bias=self.MANTISSA_BIAS)
        rmbuf = getattr(self, '_row_max_buf', None)
        if rmbuf is None or rmbuf.shape[0] != N:
            rmbuf = torch.zeros(N, dtype=torch.int32,
                                 device=self.packed_w.device)
            self._row_max_buf = rmbuf
        cmbuf = getattr(self, '_col_max_buf', None)
        if cmbuf is None or cmbuf.shape[0] != K:
            cmbuf = torch.zeros(K, dtype=torch.int32,
                                 device=self.packed_w.device)
            self._col_max_buf = cmbuf
        return wbuf, rmbuf, cmbuf

    def forward(self, x):
        in_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        wbuf, rmbuf, cmbuf = self._ensure_buffers()
        fg = self.fast_gain
        if fg < 1.0:                       # smooth-gate s_fast OUT of the forward weight
            with torch.no_grad():          # packed keeps full s_fast; grad flows through
                pw = self.packed_w         # the gated weight -> loss drives signal to slow
                sf = (pw >> 16).to(torch.float32)
                ss = ((pw << 16) >> 24).to(torch.float32)
                vs = ((pw << 24) >> 24).to(torch.float32)
                exp = (self.row_exp.float()[:, None] + self.col_exp.float()[None, :]
                       - self.MANTISSA_BIAS)
                m_gated = (ss * 128.0 + fg * sf + vs * 128.0) * torch.exp2(exp)
                wbuf.copy_(m_gated.to(wbuf.dtype))
        # Pass the lr DEVICE TENSOR (not scalar) so it's the same tensor
        # across forward calls — the captured CUDA graph sees a stable
        # pointer, and updates via `m.lr = X` (which .fill_()'s the buf)
        # propagate to subsequent graph replays.
        y = FusedConcordLinearPackedB.apply(
            x, self.packed_w, self.row_exp, self.col_exp, self.bias,
            self._lr_buf, self.alpha, self.beta1, self.MANTISSA_BIAS,
            self.optimizer_kind,
            self.weight_decay, self._eps_buf, self.step_cap,
            self.v_scale, self.precond_p, self.gf_consol, self.drift_cancel_C, self.alpha_v_fast,
            self.wd_sv, self.wd_sf,
            self.mass_preserve_v, self.apply_chase, self.track_rebalance,
            wbuf, rmbuf, cmbuf,
            self.v_row, self.v_col, self._sum_v_inv,
            float(self.adafactor_beta2),
            bool(self.track_adafactor_v),
            float(self.gf_trust_delta_sq), self._coh_pre)
        return y.to(in_dtype)

    @torch.no_grad()
    def rebalance(self):
        """Tick-up rebalance using the row_max/col_max populated by the
        last apply kernel call. Graph-capturable: bumps the seed buffer
        in-place and passes the pointer to the kernel — no host-side
        .item() needed. Also folds the current step's row/col max into
        a high-watermark buffer for post-training diagnostics."""
        rmbuf = getattr(self, '_row_max_buf', None)
        cmbuf = getattr(self, '_col_max_buf', None)
        if rmbuf is None or cmbuf is None:
            return  # No apply has run yet
        # Update the all-time high-watermark BEFORE the rebalance kernel
        # consumes / resets the per-step buffers (the apply kernel will
        # re-zero them on the next backward).
        torch.maximum(self._row_max_hwm, rmbuf, out=self._row_max_hwm)
        torch.maximum(self._col_max_hwm, cmbuf, out=self._col_max_hwm)
        self._reb_seed.add_(1)
        rebalance_packed(
            self.packed_w, self.row_exp, self.col_exp,
            rmbuf, cmbuf,
            MAX_M=self.MAX_M, EXP_MAX=self.EXP_MAX, EXP_MIN=self.EXP_MIN,
            seed_buf=self._reb_seed, allow_tickdown=self.allow_tickdown,
        )

    @torch.no_grad()
    def load_weights_finetune(self, W):
        """Pretrained-weight init: put the live weight at the
        zero-gradient steady state of the three-accumulator dynamic.

        live_mantissa = s_slow_full + s_fast + v_slow_full
        Equilibrium (no grad, no drift):
          v_slow_full ≈ s_slow_full ≈ live / 2 (each carries half)
          s_fast ≈ 0 (no velocity)
        So d_fs = s_fast = 0 and d_sv = s_slow_full - v_slow_full = 0
        → drift-cancel noise = 0 → variance estimator starts clean.
        wd_sv / wd_sf both have zero gap → no spurious decay at step 1.
        """
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
        m_total = (W / scale).round().to(torch.int32)
        # Half to v_slow_i8 (at × 128 scale).
        target_v_full = (m_total.float() / 2.0).round().to(torch.int32)
        v_slow_i8 = (target_v_full.float() / 128.0).round().to(torch.int32).clamp(-128, 127)
        actual_v_full = v_slow_i8 * 128
        # Remaining mantissa goes into s_slow_i8 (at × 128 scale).
        remaining = m_total - actual_v_full
        s_slow_i8 = (remaining.float() / 128.0).round().to(torch.int32).clamp(-128, 127)
        actual_s_slow_full = s_slow_i8 * 128
        # Quantization residual lands in s_fast (mantissa units).
        s_fast = (remaining - actual_s_slow_full).clamp(INT16_MIN, INT16_MAX)
        packed = (
            ((s_fast & 0xFFFF) << 16)
            | ((s_slow_i8 & 0xFF) << 8)
            | (v_slow_i8 & 0xFF)
        )
        self.packed_w.copy_(packed)
        self._resync_weight_buf()


# ============================================================
# Conv2d wrapper (packed B)
# ============================================================

class FusedConcordConv2dPackedB(torch.autograd.Function):
    """Conv2d that uses cuDNN for forward + grad_x + grad_W, and the
    packed apply kernel for the state update. Materializes a bf16
    weight from packed int32 once per forward, freed after backward."""

    @staticmethod
    def forward(ctx, x, packed_w, row_exp, col_exp, bias,
                in_channels, out_channels, kh, kw, stride, padding,
                lr, alpha, beta1, mantissa_bias,
                optimizer_kind,
                weight_decay, eps, step_cap,
                v_scale, precond_p, gf_consol, drift_cancel_C, alpha_v_fast,
                wd_sv, wd_sf,
                mass_preserve, apply_chase, track_rebalance,
                weight_buf, row_max_buf, col_max_buf,
                v_row, v_col, sum_v_inv,
                adafactor_beta2, track_adafactor_v,
                gf_trust_delta_sq, coh_pre):
        # weight_buf kept fresh by previous apply (materialize-merge).
        # Initial population happens once in _ensure_buffers.
        weight_4d = weight_buf.view(out_channels, in_channels, kh, kw)
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        if not x_bf16.is_contiguous():
            x_bf16 = x_bf16.contiguous()
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = torch.nn.functional.conv2d(x_bf16, weight_4d, bias=bias_bf16,
                                         stride=stride, padding=padding)
        ctx.save_for_backward(x_bf16, weight_4d)
        ctx.packed_w = packed_w
        ctx.row_exp = row_exp
        ctx.col_exp = col_exp
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.kh, ctx.kw = kh, kw
        ctx.stride = stride
        ctx.padding = padding
        ctx.lr = lr
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.mantissa_bias = mantissa_bias
        ctx.optimizer_kind = optimizer_kind
        ctx.weight_decay = weight_decay
        ctx.eps = eps
        ctx.step_cap = step_cap
        ctx.v_scale = v_scale
        ctx.precond_p = precond_p
        ctx.gf_consol = gf_consol
        ctx.drift_cancel_C = drift_cancel_C
        ctx.alpha_v_fast = alpha_v_fast
        ctx.wd_sv = wd_sv
        ctx.wd_sf = wd_sf
        ctx.mass_preserve = mass_preserve
        ctx.apply_chase = apply_chase
        ctx.track_rebalance = track_rebalance
        ctx.has_bias = bias is not None
        ctx.weight_buf = weight_buf
        ctx.row_max_buf = row_max_buf
        ctx.col_max_buf = col_max_buf
        ctx.v_row = v_row
        ctx.v_col = v_col
        ctx.sum_v_inv = sum_v_inv
        ctx.adafactor_beta2 = adafactor_beta2
        ctx.track_adafactor_v = track_adafactor_v
        ctx.gf_trust_delta_sq = gf_trust_delta_sq
        ctx.coh_pre = coh_pre
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_bf16, weight_4d = ctx.saved_tensors
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()
        grad_x = torch.nn.grad.conv2d_input(
            x_bf16.shape, weight_4d, grad_y,
            stride=ctx.stride, padding=ctx.padding)
        grad_W_4d = torch.nn.grad.conv2d_weight(
            x_bf16,
            (ctx.out_channels, ctx.in_channels, ctx.kh, ctx.kw),
            grad_y,
            stride=ctx.stride, padding=ctx.padding,
        )
        grad_W_2d = grad_W_4d.reshape(ctx.out_channels, -1).contiguous()
        if grad_W_2d.dtype != torch.bfloat16:
            grad_W_2d = grad_W_2d.to(torch.bfloat16)
        if ctx.track_rebalance:
            ctx.row_max_buf.zero_()
            ctx.col_max_buf.zero_()
        # Adafactor row/col second-moment EMA — same as Linear path. Done
        # BEFORE apply so the kernel sees this step's fresh v̂ for the
        # gf trust region (when active). β2=0.999 matches v_slow EMA rate.
        if ctx.track_adafactor_v and ctx.v_row is not None:
            with torch.no_grad():
                g2 = grad_W_2d.float() ** 2
                if _COH_WEIGHTED_V and ctx.coh_pre is not None:
                    w = ctx.coh_pre / ctx.coh_pre.mean().clamp(min=1e-12)
                    g2 = g2 * w             # reshape onto coherent power (mag-preserving)
                g2_row = g2.sum(dim=1)
                g2_col = g2.sum(dim=0)
                b2 = ctx.adafactor_beta2
                ctx.v_row.mul_(b2).add_(g2_row, alpha=1.0 - b2)
                ctx.v_col.mul_(b2).add_(g2_col, alpha=1.0 - b2)
                sum_v = ctx.v_row.sum().clamp(min=1e-30)
                ctx.sum_v_inv.fill_(0).add_(1.0 / sum_v)
        # Apply kernel emits the updated bf16 weight into ctx.weight_buf
        # (the same buffer the next forward will read from).
        if ctx.optimizer_kind == 'adamw':
            apply_packed_adamw(
                ctx.packed_w, grad_W_2d, ctx.weight_buf,
                ctx.row_exp, ctx.col_exp,
                ctx.row_max_buf, ctx.col_max_buf,
                lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                alpha=ctx.alpha, beta1=ctx.beta1,
                weight_decay=ctx.weight_decay, eps=ctx.eps,
                step_cap=ctx.step_cap,
                v_scale=ctx.v_scale, precond_p=ctx.precond_p,
                gf_consol=ctx.gf_consol,
                drift_cancel_C=ctx.drift_cancel_C,
                alpha_v_fast=ctx.alpha_v_fast,
                wd_sv=ctx.wd_sv, wd_sf=ctx.wd_sf,
                mass_preserve=ctx.mass_preserve,
                apply_chase=ctx.apply_chase,
                track_rebalance=ctx.track_rebalance,
                v_row=ctx.v_row, v_col=ctx.v_col,
                sum_v_inv=ctx.sum_v_inv,
                gf_trust_delta_sq=ctx.gf_trust_delta_sq,
                coh_pre=ctx.coh_pre,
            )
        else:
            # SGD path doesn't have a variance preconditioner, so the gf
            # trust region wouldn't have anything to floor; v_row/v_col
            # still EMA above for the diagnostic.
            apply_packed_sgd(
                ctx.packed_w, grad_W_2d, ctx.weight_buf,
                ctx.row_exp, ctx.col_exp,
                ctx.row_max_buf, ctx.col_max_buf,
                lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                alpha=ctx.alpha, beta1=ctx.beta1,
                alpha_v_fast=ctx.alpha_v_fast,
                mass_preserve=ctx.mass_preserve,
                track_rebalance=ctx.track_rebalance)
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.sum(dim=(0, 2, 3))
        # 36 forward args: only x (slot 0) and bias (slot 4) receive grads.
        return (grad_x, None, None, None, grad_bias,
                None, None, None, None, None, None,
                None, None, None, None,
                None,
                None, None, None,
                None, None, None, None, None,
                None, None,
                None, None, None,
                None, None, None,
                None, None, None,
                None, None, None, None)


class ConcordConv2dPackedB(ConcordLinearPackedB):
    """Conv2d variant of ConcordLinearPackedB. Same packed int32 storage,
    treated as a (out_channels, in_channels * kh * kw) matrix internally."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, device='cuda',
                 alpha=0.1, beta1=0.0, lr=0.01):
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
                         alpha=alpha, beta1=beta1, lr=lr)

    def forward(self, x):
        in_dtype = x.dtype
        wbuf, rmbuf, cmbuf = self._ensure_buffers()
        # Pass the lr device tensor (same as Linear path) for graph-capture
        # compatibility — see ConcordLinearPackedB.forward.
        y = FusedConcordConv2dPackedB.apply(
            x, self.packed_w, self.row_exp, self.col_exp, self.bias,
            self.in_channels, self.out_channels, self.kh, self.kw,
            self.stride, self.padding,
            self._lr_buf, self.alpha, self.beta1, self.MANTISSA_BIAS,
            self.optimizer_kind,
            self.weight_decay, self._eps_buf, self.step_cap,
            self.v_scale, self.precond_p, self.gf_consol, self.drift_cancel_C, self.alpha_v_fast,
            self.wd_sv, self.wd_sf,
            self.mass_preserve_v, self.apply_chase, self.track_rebalance,
            wbuf, rmbuf, cmbuf,
            self.v_row, self.v_col, self._sum_v_inv,
            float(self.adafactor_beta2),
            bool(self.track_adafactor_v),
            float(self.gf_trust_delta_sq), self._coh_pre)
        return y.to(in_dtype)


# ============================================================
# Smoke test (same shape as prototype_packed.py for direct A/B comparison)
# ============================================================

def _run_one(model_factory, lr, n_steps=500, bsz=32):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    in_features = 32
    hid = 64
    out_features = 16
    N = 256
    target_W1 = torch.randn(hid, in_features, device=device) * 0.3
    target_W2 = torch.randn(out_features, hid, device=device) * 0.3
    def target(x):
        return F.linear(F.relu(F.linear(x, target_W1)), target_W2)
    x_all = torch.randn(N, in_features, device=device)
    y_all = target(x_all)

    model = model_factory(in_features, hid, out_features, lr, device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    opt = getattr(model, '_torch_opt', None)

    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, N, (bsz,), device=device)
        xb = x_all[idx]
        yb = y_all[idx]
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = F.mse_loss(pred.float(), yb.float())
        loss.backward()
        if opt is not None:
            opt.step()
        losses.append(loss.item())
    return losses


def _packed_b_mlp(in_f, hid, out_f, lr, device):
    return nn.Sequential(
        ConcordLinearPackedB(in_f, hid, device=device, lr=lr, alpha=0.1),
        nn.ReLU(),
        ConcordLinearPackedB(hid, out_f, device=device, lr=lr, alpha=0.1),
    )


def _packed_b_mlp_adamw(in_f, hid, out_f, lr, device):
    m = nn.Sequential(
        ConcordLinearPackedB(in_f, hid, device=device, lr=lr, alpha=0.1),
        nn.ReLU(),
        ConcordLinearPackedB(hid, out_f, device=device, lr=lr, alpha=0.1),
    )
    for layer in m:
        if isinstance(layer, ConcordLinearPackedB):
            layer.set_optimizer_kind('adamw', weight_decay=0.01,
                                       eps=1.0, step_cap=10.0)
            layer.wd_sv = 1e-5
            layer.wd_sf = 1e-5
    return m


def _torch_mlp_adamw(in_f, hid, out_f, lr, device):
    model = nn.Sequential(
        nn.Linear(in_f, hid, device=device),
        nn.ReLU(),
        nn.Linear(hid, out_f, device=device),
    )
    model._torch_opt = torch.optim.AdamW(model.parameters(),
                                          lr=lr, weight_decay=0.01)
    return model


def _torch_mlp(in_f, hid, out_f, lr, device):
    model = nn.Sequential(
        nn.Linear(in_f, hid, device=device),
        nn.ReLU(),
        nn.Linear(hid, out_f, device=device),
    )
    model._torch_opt = torch.optim.SGD(model.parameters(), lr=lr)
    return model


def diagnose():
    """Trace single-layer + MLP dynamics."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    layer = ConcordLinearPackedB(32, 16, device=device, lr=0.05, alpha=0.1)
    w_init = layer.get_weight()
    print(f"[diag] init weight: |w|max={w_init.abs().max().item():.4f} "
          f"|w|mean={w_init.float().abs().mean().item():.4f}")
    s_fast0, s_slow0, _ = layer.get_state()
    print(f"[diag] init state: s_fast|max={s_fast0.abs().max().item()} "
          f"s_slow_i8|max={s_slow0.abs().max().item()}")

    x = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
    y_target = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
    for step in range(20):
        pred = layer(x)
        loss = F.mse_loss(pred.float(), y_target.float())
        loss.backward()
        s_fast, s_slow_i8, _ = layer.get_state()
        if step < 5 or step % 5 == 0:
            print(f"[diag] step {step:2d}: loss={loss.item():.4f}  "
                  f"|s_fast|max={s_fast.abs().max().item():5d}  "
                  f"|s_slow_i8|max={s_slow_i8.abs().max().item():3d}  "
                  f"|w|max={layer.get_weight().abs().max().item():.4f}")


def smoke_test():
    print("=== Storage roundtrip ===")
    device = 'cuda'
    torch.manual_seed(0)
    layer = ConcordLinearPackedB(8, 4, device=device)
    w = layer.get_weight()
    print(f"[OK] get_weight works: shape={tuple(w.shape)} dtype={w.dtype} "
          f"|w|max={w.abs().max().item():.4f}")
    # Compare CPU-side reconstruction to kernel materialization
    wbuf, _, _ = layer._ensure_buffers()
    materialize_packed_bf16(layer.packed_w, layer.row_exp,
                              layer.col_exp, wbuf,
                              mantissa_bias=layer.MANTISSA_BIAS)
    diff = (w.float() - wbuf.float()).abs().max().item()
    print(f"[OK] kernel materialize matches CPU recon: max_diff={diff:.6f}")
    print()

    print("=== Single-layer dynamics ===")
    diagnose()
    print()

    print("=== MLP smoke (sweep over lr) ===")
    results = {}
    for tag, factory, lr in [
        ('packed-B SGD lr=0.1',     _packed_b_mlp, 0.1),
        ('packed-B SGD lr=0.05',    _packed_b_mlp, 0.05),
        ('packed-B SGD lr=0.01',    _packed_b_mlp, 0.01),
        ('packed-B AdamW lr=0.05',  _packed_b_mlp_adamw, 0.05),
        ('packed-B AdamW lr=0.01',  _packed_b_mlp_adamw, 0.01),
        ('packed-B AdamW lr=0.001', _packed_b_mlp_adamw, 0.001),
        ('SGD baseline lr=0.05',    _torch_mlp,    0.05),
        ('SGD baseline lr=0.01',    _torch_mlp,    0.01),
        ('AdamW baseline lr=0.01',  _torch_mlp_adamw,  0.01),
        ('AdamW baseline lr=0.001', _torch_mlp_adamw,  0.001),
    ]:
        L = _run_one(factory, lr=lr)
        nans = sum(1 for v in L if v != v)
        ratio = L[0] / max(L[-1], 1e-30)
        print(f"[smoke] {tag:>22}  init={L[0]:6.3f}  "
              f"s100={L[100]:7.3f}  s500={L[-1]:7.3f}  "
              f"ratio={ratio:6.1f}x"
              + (f"  [NaN x{nans}]" if nans else ""))
        results[tag] = (L, ratio, nans)
    print()

    # Apples-to-apples: packed-B SGD vs torch.SGD baseline (both use
    # plain gradient + chase, no adaptive per-parameter LR). Packed-B
    # AdamW vs torch.AdamW isn't directly comparable because the
    # drift-cancel variance is a coarser estimator than Adam's smooth
    # per-element g² EMA; differences here are expected, and the real
    # comparison is at full CIFAR training time (~80 epochs).
    pkdB_sgd  = max((r[1] for k, r in results.items()
                     if k.startswith('packed-B SGD') and r[2] == 0),
                    default=0.0)
    base_sgd  = max((r[1] for k, r in results.items()
                     if k.startswith('SGD baseline') and r[2] == 0),
                    default=0.0)
    pkdB_adam = max((r[1] for k, r in results.items()
                     if k.startswith('packed-B AdamW') and r[2] == 0),
                    default=0.0)
    base_adam = max((r[1] for k, r in results.items()
                     if k.startswith('AdamW baseline') and r[2] == 0),
                    default=0.0)
    print(f"[smoke] packed-B SGD best:    {pkdB_sgd:.1f}x   (vs torch.SGD baseline {base_sgd:.1f}x)")
    print(f"[smoke] packed-B AdamW best:  {pkdB_adam:.1f}x   (vs torch.AdamW baseline {base_adam:.1f}x)")
    print()

    sgd_ok = pkdB_sgd >= base_sgd * 0.7
    adam_works = pkdB_adam > 1.5   # at least some convergence (not NaN)
    if sgd_ok and adam_works:
        print(f"[PASS] packed-B SGD >= 70% of torch.SGD, AdamW path converges")
        return True
    elif not sgd_ok:
        print(f"[FAIL] packed-B SGD ratio too low ({pkdB_sgd:.1f}x vs "
              f"baseline {base_sgd:.1f}x)")
        return False
    else:
        print(f"[FAIL] packed-B AdamW path failed to converge")
        return False


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    ok = smoke_test()
    print()
    if ok:
        print("=== PROTOTYPE B WORKS ===")
    else:
        print("=== PROTOTYPE B FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
