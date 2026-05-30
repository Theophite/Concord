"""Prototype C: packed int32 with int16 s_fast + int16 position (byte-concat,
shared sign).

Builds on packed-B but exploits the empirical observation that v_slow and
s_slow are same-sign in practice (their magnitudes differ by <0.1% at
80-epoch equilibrium). Collapses v_slow + s_slow into a single signed
int16 "position":
  - High byte of position = "slow / anchor" role (changes only on carries
    from the low byte, naturally lagging at ~1/256 of low-byte updates)
  - Low byte of position = "recent fine" role (changes per chase tick)
  - Sign is shared (it's the int16's sign bit)

Storage layout (little-endian):
    bits [31:16]  s_fast       int16   — fine SR-tick accumulator, scale × 1
    bits [15:0]   position     int16   — combined consolidated state, scale × 1

Live weight:
    m_eff = position + s_fast
    weight = m_eff × 2^(row_exp + col_exp - mantissa_bias)

Comparison to packed-B:
  - Same 32 bits/param, same int32 word.
  - Position granularity 1 (vs 128 in packed-B) — much finer chase ticks.
  - No separate v_slow leak — the high byte naturally lags via carries.
  - Drift-cancel: d_sv = signed-int8(low byte of position), the "recent
    deviation from coarse anchor" emerges as a natural drift signal.
  - wd_sv / wd_sf dropped (the explicit Bayesian anchor concept goes away
    when v_slow merges into position). Standard decoupled wd applies.

Run:
    python prototype_packed_c.py
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


def _torch_spread_bits(x):
    """CPU/torch version of _spread_bits — same Morton encoding, on int32
    tensors. Used by load_weights / get_weight for the bit interleave."""
    x = x & 0xFF
    x = (x | (x << 4)) & 0x0F0F
    x = (x | (x << 2)) & 0x3333
    x = (x | (x << 1)) & 0x5555
    return x


def _torch_gather_even_bits(x):
    """CPU/torch version of _gather_even_bits — inverse of _torch_spread_bits."""
    x = x & 0x5555
    x = (x | (x >> 1)) & 0x3333
    x = (x | (x >> 2)) & 0x0F0F
    x = (x | (x >> 4)) & 0x00FF
    return x


@triton.jit
def _spread_bits(x):
    """Spread 8 bits of x to alternating positions of a 16-bit value:
    output bit (2k) = input bit k, for k in 0..7. Output bits at odd
    positions are zero. x is masked to its low 8 bits first.
    Returns unsigned int representation in tl.int32 (values 0 to 0x5555).
    Classic Morton / Z-order encoding bit-twiddle."""
    x = x & 0xFF
    x = (x | (x << 4)) & 0x0F0F     # bits 0-3 at 0-3, 4-7 at 8-11
    x = (x | (x << 2)) & 0x3333     # bits at 0-1, 4-5, 8-9, 12-13
    x = (x | (x << 1)) & 0x5555     # bits at 0, 2, 4, ..., 14
    return x


@triton.jit
def _gather_even_bits(x):
    """Inverse of _spread_bits: gather bits at even positions {0, 2, ..., 14}
    of x into a single byte. Returns the 8 bits as int32 in [0, 255].
    Used to decompose a combined int16 back into s_slow_i8 and v_slow_i8."""
    x = x & 0x5555                  # keep only bits at 0, 2, ..., 14
    x = (x | (x >> 1)) & 0x3333
    x = (x | (x >> 2)) & 0x0F0F
    x = (x | (x >> 4)) & 0x00FF
    return x


@triton.jit
def _interleave(s_i8, v_i8):
    """Combine two int8s into one int16-as-int32 via Morton/Z-order:
    s_i8's bits land at even positions {0, 2, ..., 14},
    v_i8's bits land at odd positions {1, 3, ..., 15}.
    Returns the SIGNED int16 value (sign-extended to int32 via bit 15
    being v_i8's bit 7)."""
    spread_s = _spread_bits(s_i8)
    spread_v = _spread_bits(v_i8)
    combined_unsigned = spread_s | (spread_v << 1)
    # Sign-extend the 16-bit value to int32: shift up to int32 sign
    # position, then arithmetic right shift back.
    return (combined_unsigned << 16) >> 16


@triton.jit
def _deinterleave(combined_i32):
    """Inverse of _interleave: extract s_i8 and v_i8 from a combined
    int16-as-int32. Returns (s_i8, v_i8) both as signed int8 in int32.
    Sign-extension on each byte is via the high bit of the corresponding
    8-bit gathered value."""
    # Mask to 16 bits (unsigned view of combined)
    unsigned16 = combined_i32 & 0xFFFF
    s_unsigned = _gather_even_bits(unsigned16)         # bits at even positions
    v_unsigned = _gather_even_bits(unsigned16 >> 1)    # bits at odd, shifted to even
    # Sign-extend each 8-bit value to int32 via shift trick:
    s_signed = (s_unsigned << 24) >> 24
    v_signed = (v_unsigned << 24) >> 24
    return s_signed, v_signed


@triton.jit
def _materialize_packed_bf16_kernel(
    packed_ptr,        # [N, K] int32
    weight_ptr,        # [N, K] bf16
    row_exp_ptr,       # [N] int8
    col_exp_ptr,       # [K] int8
    step_salt_ptr,     # scalar int32 step counter (for SR emission)
    N, K,
    mantissa_bias,
    stride_pn, stride_pk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)

    # Scheme E: unpack s_fast (high half) and the two int8 bytes that
    # together encode the consolidated state via bit-interleave.
    s_fast    = packed >> 16
    s_slow_i8 = (packed << 16) >> 24       # byte 1 of packed (signed)
    v_slow_i8 = (packed << 24) >> 24       # byte 0 of packed (signed)
    # 1-mantissa granularity in [-32768, 32767].
    combined = _interleave(s_slow_i8 & 0xFF, v_slow_i8 & 0xFF)
    # Double the combined contribution and add s_fast for live mantissa.
    # 2 × combined ranges ±65k (granularity 2); s_fast fills the gap.
    m_eff = (combined << 1) + s_fast

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
# <<<MISSING 192>>>
# <<<MISSING 193>>>
# <<<MISSING 194>>>
# <<<MISSING 195>>>
# <<<MISSING 196>>>
# <<<MISSING 197>>>
    biased_bits = fp32_bits + dither
    # Truncate low 16 bits → exact bf16 representation
    bf16_bits = biased_bits & 0xFFFF0000
    weight_sr = bf16_bits.to(tl.float32, bitcast=True).to(tl.bfloat16)

    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_ptr + w_off, weight_sr, mask=nk_mask)


def materialize_packed_bf16(packed_w, row_exp, col_exp, out,
                              mantissa_bias=15):
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.shape == packed_w.shape
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _materialize_packed_bf16_kernel[grid](
        packed_w, out, row_exp, col_exp, N, K, int(mantissa_bias),
        packed_w.stride(0), packed_w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


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

    # ── mass-preserve chase (int16 granularity!) ───────────────
    # α·s_fast is the chase amount in mantissa units. SR-round to
    # int (1-mantissa granularity, NOT divided by 128 anymore).
    # The carry from low byte to high byte of position happens
    # naturally via int16 arithmetic — no explicit v_slow leak.
    chase_mantissa = alpha * s_fast.to(tl.float32)
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_c = tl.floor(chase_mantissa)
    frac_c = chase_mantissa - floor_c
    tick_position = (floor_c + (r2 < frac_c).to(tl.float32)).to(tl.int32)
    position = position + tick_position
    s_fast   = s_fast - tick_position

    # ── clamp and repack ───────────────────────────────────────
    s_fast_c   = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    position_c = tl.minimum(tl.maximum(position, -32768), 32767)
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | (position_c & 0xFFFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── atomic-max for rebalance ───────────────────────────────
    abs_eff = tl.abs(position_c + s_fast_c)
    abs_eff = tl.where(nk_mask, abs_eff, 0)
    tile_row_max = tl.max(abs_eff, axis=1)
    tile_col_max = tl.max(abs_eff, axis=0)
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
    # Recompose combined via bit-interleave. The combined int16 (signed,
    # in ±32k) is the "consolidated state" we operate on.
    combined = _interleave(s_slow_i8 & 0xFF, v_slow_i8 & 0xFF)

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

    # ── mass-preserve chase (granularity-2 — for doubled combined) ─
    # α·s_fast in mantissa units. We tick combined by SR(α·s_fast / 2)
    # because m_eff includes 2 × combined (so one unit of combined =
    # two mantissa units in m_eff). s_fast loses 2 × tick_combined.
    chase_mantissa = alpha * s_fast.to(tl.float32)
    chase_combined_f = chase_mantissa * 0.5      # divide by 2 for doubled mapping
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_c = tl.floor(chase_combined_f)
    frac_c = chase_combined_f - floor_c
    tick_combined = (floor_c + (r2 < frac_c).to(tl.float32)).to(tl.int32)
    combined = combined + tick_combined
    s_fast   = s_fast - (tick_combined << 1)     # mass preserve: 2 × tick

    # ── clamp and repack ───────────────────────────────────────
    s_fast_c   = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    combined_c = tl.minimum(tl.maximum(combined, -32768), 32767)
    # Decompose combined back into bit-interleaved (s_slow_i8, v_slow_i8).
    s_slow_new, v_slow_new = _deinterleave(combined_c)
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | ((s_slow_new & 0xFF) << 8)
        | (v_slow_new & 0xFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── atomic-max for rebalance ───────────────────────────────
    # Live mantissa magnitude = |2*combined + s_fast|. Use this.
    abs_eff = tl.abs((combined_c << 1) + s_fast_c)
    abs_eff = tl.where(nk_mask, abs_eff, 0)
    tile_row_max = tl.max(abs_eff, axis=1)
    tile_col_max = tl.max(abs_eff, axis=0)
    tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask)
    tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask)


def apply_packed_sgd(packed_w, grad_W, row_exp, col_exp,
                       row_max, col_max,
                       lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                       alpha_v_fast=0.001):
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_packed_sgd_kernel[grid](
        packed_w, grad_W, row_exp, col_exp,
        row_max, col_max,
        N, K,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        float(alpha_v_fast),
        step_counter,
        packed_w.stride(0), packed_w.stride(1),
        grad_W.stride(0), grad_W.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


# ============================================================
# AdamW three-accumulator apply kernel (packed).
#
# Drift-cancel variance from the three stored quantities:
#   d_fs = s_fast              — velocity (delta from s_slow position)
#   d_sv = s_slow*128 - v_slow*128
#   noise = d_fs - drift_cancel_C * d_sv
    # residual. d_sv signal = how far position has drifted past its
    # last 256-multiple anchor.
    s_slow_role = (position << 24) >> 24     # signed int8 in int32

    g_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + g_off, mask=nk_mask, other=0.0).to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_fwd = tl.exp2(total_exp)
    scale_inv = tl.exp2(-total_exp)

    # ── live weight + drift-cancel variance ───────────────────
    m_eff = position + s_fast
    current_weight = m_eff.to(tl.float32) * scale_fwd

    # Velocity from s_fast. Drift signal from the low byte of position
    # (the residual after the high-byte coarse anchor). Both are in
    # mantissa units; cancel the slow drift from the velocity to get
    # a cleaner variance estimate.
    d_fs = s_fast.to(tl.float32)
    d_sv = s_slow_role.to(tl.float32)
    noise = d_fs - drift_cancel_C * d_sv
    noise_in_w = noise * scale_fwd
    v_proxy = noise_in_w * noise_in_w * v_scale

    # ── AdamW step ─────────────────────────────────────────────
    step_live = grad_W / tl.sqrt(v_proxy + eps) + weight_decay * current_weight
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv     # mantissa units
    delta_t = delta_grad - beta1 * d_fs

    # ── SR-tick s_fast ────────────────────────────────────────
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    if APPLY_CHASE:
        # ── mass-preserve chase (int16 granularity) ──
        # α·s_fast in mantissa units; SR-round to integer; apply to
        # position via int16 arithmetic (carries propagate naturally
        # from low byte to high byte).
        chase_mantissa = alpha * s_fast.to(tl.float32)
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_c = tl.floor(chase_mantissa)
        frac_c = chase_mantissa - floor_c
        tick_position = (floor_c + (r2 < frac_c).to(tl.float32)).to(tl.int32)
        position = position + tick_position
        s_fast   = s_fast - tick_position

    # ── clamp and repack ──────────────────────────────────────
    s_fast_c   = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    position_c = tl.minimum(tl.maximum(position, -32768), 32767)
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | (position_c & 0xFFFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── atomic-max for rebalance ──
    abs_eff = tl.abs(position_c + s_fast_c)
    abs_eff = tl.where(nk_mask, abs_eff, 0)
    tile_row_max = tl.max(abs_eff, axis=1)
    tile_col_max = tl.max(abs_eff, axis=0)
    tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask)
    tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask)


def apply_packed_adamw(packed_w, grad_W, row_exp, col_exp,
                         row_max, col_max,
                         lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                         weight_decay=0.0, eps=1.0, step_cap=10.0,
                         v_scale=1.0, drift_cancel_C=0.1,
                         alpha_v_fast=0.001,
                         wd_sv=0.0, wd_sf=0.0,
                         mass_preserve=False, apply_chase=True):
    """Wrapper for the AdamW three-accumulator packed apply kernel."""
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_packed_adamw_kernel[grid](
        packed_w, grad_W, row_exp, col_exp,
        row_max, col_max,
        N, K,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        float(weight_decay), float(eps), float(step_cap),
# <<<MISSING 464>>>
# <<<MISSING 465>>>
# <<<MISSING 466>>>
# <<<MISSING 467>>>
# <<<MISSING 468>>>
# <<<MISSING 469>>>
# <<<MISSING 470>>>
# <<<MISSING 471>>>
# <<<MISSING 472>>>
# <<<MISSING 473>>>
# <<<MISSING 474>>>
# <<<MISSING 475>>>
# <<<MISSING 476>>>
# <<<MISSING 477>>>
# <<<MISSING 478>>>
# <<<MISSING 479>>>
# <<<MISSING 480>>>
# <<<MISSING 481>>>
# <<<MISSING 482>>>
# <<<MISSING 483>>>
# <<<MISSING 484>>>
# <<<MISSING 485>>>
# <<<MISSING 486>>>
# <<<MISSING 487>>>
# <<<MISSING 488>>>
# <<<MISSING 489>>>
# <<<MISSING 490>>>
# <<<MISSING 491>>>
# <<<MISSING 492>>>
# <<<MISSING 493>>>
# <<<MISSING 494>>>
# <<<MISSING 495>>>
# <<<MISSING 496>>>
# <<<MISSING 497>>>
# <<<MISSING 498>>>
# <<<MISSING 499>>>
# <<<MISSING 500>>>
# <<<MISSING 501>>>
# <<<MISSING 502>>>
# <<<MISSING 503>>>
# <<<MISSING 504>>>
# <<<MISSING 505>>>
# <<<MISSING 506>>>
# <<<MISSING 507>>>
# <<<MISSING 508>>>
# <<<MISSING 509>>>
# <<<MISSING 510>>>
# <<<MISSING 511>>>
# <<<MISSING 512>>>
# <<<MISSING 513>>>
# <<<MISSING 514>>>
# <<<MISSING 515>>>
# <<<MISSING 516>>>
# <<<MISSING 517>>>
# <<<MISSING 518>>>
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    row_max = tl.load(row_max_ptr + offs_n, mask=n_mask, other=0)
    col_max = tl.load(col_max_ptr + offs_k, mask=k_mask, other=0)
    row_exp = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0)
    col_exp = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0)

    row_up = (row_max > MAX_M) & (row_exp < EXP_MAX)
    col_up = (col_max > MAX_M) & (col_exp < EXP_MAX)
    row_t = row_up.to(tl.int32)
    col_t = col_up.to(tl.int32)
    pos = row_t[:, None] + col_t[None, :]   # in {0, 1, 2}

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast   = packed >> 16
    position = (packed << 16) >> 16

    # SR-right-shift both segments. Both are int16 with fine
    # granularity, so each loses LSBs to SR rounding. No mantissa-
    # residual migration needed (scheme C has no coarse storage to
    # preserve precision for — position is already fine).
    rand_off = offs_n[:, None] * K + offs_k[None, :]
    two_pos = tl.exp2(pos.to(tl.float32))

    q_fast = s_fast >> pos
    rem_fast = (s_fast - (q_fast << pos)).to(tl.float32)
    up_fast = (tl.rand(seed, rand_off) * two_pos < rem_fast).to(tl.int32)
    s_fast_new = q_fast + up_fast

    q_position = position >> pos
    rem_position = (position - (q_position << pos)).to(tl.float32)
    up_position = (tl.rand(seed, rand_off + N * K) * two_pos < rem_position).to(tl.int32)
    position_new = q_position + up_position

    # Clamp + repack.
    s_fast_c   = tl.minimum(tl.maximum(s_fast_new, -32768), 32767)
    position_c = tl.minimum(tl.maximum(position_new, -32768), 32767)
    packed_new = (
        ((s_fast_c & 0xFFFF) << 16)
        | (position_c & 0xFFFF)
    )
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    if pid_k == 0:
        tl.store(row_exp_ptr + offs_n, row_exp + row_t, mask=n_mask)
    if pid_n == 0:
        tl.store(col_exp_ptr + offs_k, col_exp + col_t, mask=k_mask)


def rebalance_packed(packed_w, row_exp, col_exp, row_max, col_max,
                       MAX_M=24000, EXP_MAX=7, seed=0):
    """Apply the rebalance ratchet (tick-up exponent if row/col max
    exceeds threshold). Assumes row_max/col_max already populated by a
# <<<MISSING 574>>>
# <<<MISSING 575>>>
# <<<MISSING 576>>>
# <<<MISSING 577>>>
# <<<MISSING 578>>>
# <<<MISSING 579>>>
# <<<MISSING 580>>>
# <<<MISSING 581>>>
# <<<MISSING 582>>>
# <<<MISSING 583>>>
# <<<MISSING 584>>>
# <<<MISSING 585>>>
# <<<MISSING 586>>>
# <<<MISSING 587>>>
# <<<MISSING 588>>>
# <<<MISSING 589>>>
# <<<MISSING 590>>>
# <<<MISSING 591>>>
# <<<MISSING 592>>>
# <<<MISSING 593>>>
# <<<MISSING 594>>>
# <<<MISSING 595>>>
# <<<MISSING 596>>>
# <<<MISSING 597>>>
# <<<MISSING 598>>>
# <<<MISSING 599>>>
# <<<MISSING 600>>>
# <<<MISSING 601>>>
# <<<MISSING 602>>>
# <<<MISSING 603>>>
# <<<MISSING 604>>>
# <<<MISSING 605>>>
# <<<MISSING 606>>>
# <<<MISSING 607>>>
# <<<MISSING 608>>>
# <<<MISSING 609>>>
# <<<MISSING 610>>>
# <<<MISSING 611>>>
# <<<MISSING 612>>>
# <<<MISSING 613>>>
# <<<MISSING 614>>>
# <<<MISSING 615>>>
# <<<MISSING 616>>>
# <<<MISSING 617>>>
# <<<MISSING 618>>>
# <<<MISSING 619>>>
# <<<MISSING 620>>>
# <<<MISSING 621>>>
# <<<MISSING 622>>>
# <<<MISSING 623>>>
# <<<MISSING 624>>>
# <<<MISSING 625>>>
# <<<MISSING 626>>>
# <<<MISSING 627>>>
# <<<MISSING 628>>>
# <<<MISSING 629>>>
# <<<MISSING 630>>>
# <<<MISSING 631>>>
# <<<MISSING 632>>>
# <<<MISSING 633>>>
# <<<MISSING 634>>>
# <<<MISSING 635>>>
# <<<MISSING 636>>>
# <<<MISSING 637>>>
# <<<MISSING 638>>>
# <<<MISSING 639>>>
# <<<MISSING 640>>>
# <<<MISSING 641>>>
# <<<MISSING 642>>>
# <<<MISSING 643>>>
# <<<MISSING 644>>>
# <<<MISSING 645>>>
# <<<MISSING 646>>>
# <<<MISSING 647>>>
# <<<MISSING 648>>>
# <<<MISSING 649>>>
# <<<MISSING 650>>>
# <<<MISSING 651>>>
# <<<MISSING 652>>>
# <<<MISSING 653>>>
# <<<MISSING 654>>>
# <<<MISSING 655>>>
# <<<MISSING 656>>>
# <<<MISSING 657>>>
# <<<MISSING 658>>>
# <<<MISSING 659>>>
# <<<MISSING 660>>>
# <<<MISSING 661>>>
# <<<MISSING 662>>>
# <<<MISSING 663>>>
# <<<MISSING 664>>>
# <<<MISSING 665>>>
# <<<MISSING 666>>>
# <<<MISSING 667>>>
# <<<MISSING 668>>>
# <<<MISSING 669>>>
# <<<MISSING 670>>>
# <<<MISSING 671>>>
# <<<MISSING 672>>>
# <<<MISSING 673>>>
# <<<MISSING 674>>>
# <<<MISSING 675>>>
# <<<MISSING 676>>>
# <<<MISSING 677>>>
# <<<MISSING 678>>>
# <<<MISSING 679>>>
# <<<MISSING 680>>>
# <<<MISSING 681>>>
# <<<MISSING 682>>>
# <<<MISSING 683>>>
# <<<MISSING 684>>>
# <<<MISSING 685>>>
# <<<MISSING 686>>>
# <<<MISSING 687>>>
# <<<MISSING 688>>>
# <<<MISSING 689>>>
# <<<MISSING 690>>>
# <<<MISSING 691>>>
# <<<MISSING 692>>>
# <<<MISSING 693>>>
# <<<MISSING 694>>>
# <<<MISSING 695>>>
# <<<MISSING 696>>>
# <<<MISSING 697>>>
# <<<MISSING 698>>>
# <<<MISSING 699>>>
# <<<MISSING 700>>>
# <<<MISSING 701>>>
# <<<MISSING 702>>>
# <<<MISSING 703>>>
# <<<MISSING 704>>>
# <<<MISSING 705>>>
# <<<MISSING 706>>>
# <<<MISSING 707>>>
# <<<MISSING 708>>>
# <<<MISSING 709>>>
# <<<MISSING 710>>>
# <<<MISSING 711>>>
# <<<MISSING 712>>>
# <<<MISSING 713>>>
# <<<MISSING 714>>>
# <<<MISSING 715>>>
# <<<MISSING 716>>>
# <<<MISSING 717>>>
# <<<MISSING 718>>>
# <<<MISSING 719>>>
# <<<MISSING 720>>>
# <<<MISSING 721>>>
# <<<MISSING 722>>>
# <<<MISSING 723>>>
# <<<MISSING 724>>>
# <<<MISSING 725>>>
# <<<MISSING 726>>>
# <<<MISSING 727>>>
# <<<MISSING 728>>>
# <<<MISSING 729>>>
# <<<MISSING 730>>>
# <<<MISSING 731>>>
# <<<MISSING 732>>>
# <<<MISSING 733>>>
# <<<MISSING 734>>>
# <<<MISSING 735>>>
# <<<MISSING 736>>>
# <<<MISSING 737>>>
# <<<MISSING 738>>>
# <<<MISSING 739>>>
# <<<MISSING 740>>>
# <<<MISSING 741>>>
# <<<MISSING 742>>>
# <<<MISSING 743>>>
# <<<MISSING 744>>>
# <<<MISSING 745>>>
# <<<MISSING 746>>>
# <<<MISSING 747>>>
# <<<MISSING 748>>>
# <<<MISSING 749>>>
# <<<MISSING 750>>>
    @torch.no_grad()
    def load_weights(self, W):
        """Init scheme C: put the full mantissa in position. s_fast = 0.

        Why position (not s_fast): in scheme C the position has fine
        1-mantissa granularity and matches the live weight's natural
        representation. s_fast starts at 0 (no recent velocity).
        Chase + grad will fill s_fast over the first few steps from
        the gradient signal.
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
        position = m_total                        # int32 in int16 range
        s_fast = torch.zeros_like(position)
        packed = (
            ((s_fast & 0xFFFF) << 16)
            | (position & 0xFFFF)
        )
        self.packed_w.copy_(packed)

    @torch.no_grad()
    def get_weight(self):
        """Read live bf16 weight from packed state (scheme C: position +
        s_fast)."""
        s_fast   = (self.packed_w >> 16)
        position = (self.packed_w << 16) >> 16   # sign-extend low 16
        m_eff = position + s_fast
        exp = (self.row_exp[:, None].to(torch.int32)
               + self.col_exp[None, :].to(torch.int32)
               - self.MANTISSA_BIAS).to(torch.float32)
        w_fp32 = m_eff.to(torch.float32) * torch.pow(2.0, exp)
        return w_fp32.to(torch.bfloat16)

    @torch.no_grad()
    def get_state(self):
        """Diagnostic: return (s_fast, position) tensors (scheme C)."""
        s_fast   = (self.packed_w >> 16)
        position = (self.packed_w << 16) >> 16
        return s_fast, position

    def _ensure_buffers(self):
        N, K = self.packed_w.shape
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if wbuf is None or wbuf.shape != self.packed_w.shape:
            wbuf = torch.empty(self.packed_w.shape, dtype=torch.bfloat16,
                                device=self.packed_w.device)
            self._bf16_weight_buf = wbuf
        rmbuf = getattr(self, '_row_max_buf', None)
        if rmbuf is None or rmbuf.shape[0] != N:
            rmbuf = torch.zeros(N, dtype=torch.int32,
