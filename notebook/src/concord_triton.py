"""Triton kernels for ConcordLinear hot paths.

Kernels:
  apply_grad        - mantissa += round(-lr * grad * 2^(MB - row_e - col_e))
  apply_ticks       - shift mantissa[i,j] by row_tick[i] + col_tick[j] (signed,
                      arithmetic right shift for +ve, left shift for -ve)
  sync_weight       - weight = mantissa.float() * 2^(row_e + col_e - MB)

Reductions (row/col mean, max) remain in PyTorch — they are already efficient.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _apply_grad_kernel(
    mantissa_ptr, grad_ptr,
    row_exp_ptr, col_exp_ptr,
    M, N,
    lr, mantissa_bias,
    stride_mm, stride_mn,
    stride_gm, stride_gn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N
    mask = rm[:, None] & cm[None, :]

    row_e = tl.load(row_exp_ptr + rows, mask=rm, other=0)
    col_e = tl.load(col_exp_ptr + cols, mask=cm, other=0)

    m_off = rows[:, None] * stride_mm + cols[None, :] * stride_mn
    g_off = rows[:, None] * stride_gm + cols[None, :] * stride_gn

    mantissa = tl.load(mantissa_ptr + m_off, mask=mask, other=0)
    grad = tl.load(grad_ptr + g_off, mask=mask, other=0.0)

    exp = (mantissa_bias - row_e[:, None] - col_e[None, :]).to(tl.float32)
    scale_inv = tl.exp2(exp)

    d_m_f = -lr * grad * scale_inv
    # Symmetric round-to-nearest (truncate after adding signed 0.5)
    bias = tl.where(d_m_f >= 0.0, 0.5, -0.5)
    d_m = (d_m_f + bias).to(tl.int32)

    new_mantissa = mantissa + d_m
    tl.store(mantissa_ptr + m_off, new_mantissa, mask=mask)


@triton.jit
def _apply_ticks_kernel(
    mantissa_ptr,
    row_ticks_ptr, col_ticks_ptr,
    M, N,
    stride_mm, stride_mn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N
    mask = rm[:, None] & cm[None, :]

    row_t = tl.load(row_ticks_ptr + rows, mask=rm, other=0)
    col_t = tl.load(col_ticks_ptr + cols, mask=cm, other=0)

    offsets = rows[:, None] * stride_mm + cols[None, :] * stride_mn
    m = tl.load(mantissa_ptr + offsets, mask=mask, other=0).to(tl.int32)

    total = row_t[:, None] + col_t[None, :]  # int32, can be negative
    pos = tl.maximum(total, 0)
    neg = tl.maximum(-total, 0)
    # Either pos or neg is 0 (since one of total>=0 or total<=0 holds).
    # Arithmetic right shift on int32 preserves sign.
    m_shifted = (m >> pos) << neg
    # Clamp into int16 range before the narrowing store. A left-shift
    # (tick-down / counter-tick) can transiently exceed int16; callers
    # (qtridiag) clamp right afterwards anyway, so saturating here is
    # equivalent to the int32-buffer behaviour.
    m_shifted = tl.minimum(tl.maximum(m_shifted, -32768), 32767)

    tl.store(mantissa_ptr + offsets, m_shifted.to(tl.int16), mask=mask)


@triton.jit
def _sync_weight_kernel(
    weight_ptr, mantissa_ptr,
    row_exp_ptr, col_exp_ptr,
    M, N, mantissa_bias,
    stride_wm, stride_wn,
    stride_mm, stride_mn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N
    mask = rm[:, None] & cm[None, :]

    row_e = tl.load(row_exp_ptr + rows, mask=rm, other=0)
    col_e = tl.load(col_exp_ptr + cols, mask=cm, other=0)

    m_off = rows[:, None] * stride_mm + cols[None, :] * stride_mn
    w_off = rows[:, None] * stride_wm + cols[None, :] * stride_wn

    mantissa = tl.load(mantissa_ptr + m_off, mask=mask, other=0)

    exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale = tl.exp2(exp)

    weight = mantissa.to(tl.float32) * scale
    tl.store(weight_ptr + w_off, weight, mask=mask)


def _grid(M, N):
    return lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))


def apply_grad_triton(mantissa, grad, row_exp, col_exp, lr, mantissa_bias,
                      block_m=32, block_n=64):
    M, N = mantissa.shape
    assert mantissa.dtype == torch.int32
    assert grad.dtype == torch.float32
    # row/col_exp are int8 at rest (kernel loads then widens to int32
    # in registers). int32 still accepted as a fallback for in-flight
    # buffers from older training runs.
    assert row_exp.dtype in (torch.int8, torch.int32)
    assert col_exp.dtype in (torch.int8, torch.int32)
    _apply_grad_kernel[_grid(M, N)](
        mantissa, grad, row_exp, col_exp,
        M, N,
        float(lr), int(mantissa_bias),
        mantissa.stride(0), mantissa.stride(1),
        grad.stride(0), grad.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )


def apply_ticks_triton(mantissa, row_ticks, col_ticks,
                       block_m=32, block_n=64):
    M, N = mantissa.shape
    assert mantissa.dtype == torch.int16
    assert row_ticks.dtype == torch.int32
    assert col_ticks.dtype == torch.int32
    _apply_ticks_kernel[_grid(M, N)](
        mantissa, row_ticks, col_ticks,
        M, N,
        mantissa.stride(0), mantissa.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )


def sync_weight_triton(weight, mantissa, row_exp, col_exp, mantissa_bias,
                       block_m=32, block_n=64):
    M, N = mantissa.shape
    assert weight.dtype == torch.float32
    assert mantissa.dtype == torch.int32
    _sync_weight_kernel[_grid(M, N)](
        weight, mantissa, row_exp, col_exp,
        M, N, int(mantissa_bias),
        weight.stride(0), weight.stride(1),
        mantissa.stride(0), mantissa.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
    )


# === Pair (s_slow + s_fast) → bf16 weight ===
# Reconstructs bf16 weight on the fly from the two int16-range integer
# accumulators that together form the concord representation:
#     weight[i,j] = (s_slow[i,j] + s_fast[i,j]) * 2^(row_exp[i] + col_exp[j] - BIAS)
# The bf16 conversion is via fp32 intermediate (which Triton/HW does in
# registers); equivalent to direct bit manipulation since bf16 truncation
# of fp32 is just dropping the low 16 mantissa bits.

@triton.jit
def _sync_weight_bf16_pair_kernel(
    weight_ptr,
    s_slow_ptr, s_fast_ptr,
    row_exp_ptr, col_exp_ptr,
    v_slow_ptr,        # [M, N] int8 (gated by USE_V_SLOW). When set,
                       # v_slow_i8 adds additively to the live weight at
                       # shifted scale V_SLOW_FACTOR.
    M, N, mantissa_bias,
    stride_wm, stride_wn,
    stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    SLOW_SCALE: tl.constexpr = 1,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = rows < M
    cm = cols < N
    mask = rm[:, None] & cm[None, :]

    row_e = tl.load(row_exp_ptr + rows, mask=rm, other=0)
    col_e = tl.load(col_exp_ptr + cols, mask=cm, other=0)

    s_off = rows[:, None] * stride_sm + cols[None, :] * stride_sn
    w_off = rows[:, None] * stride_wm + cols[None, :] * stride_wn

    s_slow = tl.load(s_slow_ptr + s_off, mask=mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=mask, other=0).to(tl.int32)
    # SLOW_SCALE=1 default reproduces the canonical live = s_slow + s_fast
    # formula. SLOW_SCALE=2 ("double slow" experiment) makes each s_slow
    # mantissa unit contribute twice to the live weight; equilibrium then
    # has |s_slow| settle at ~half its single-scale value while v_slow
    # absorbs the rest of the position.
    m = SLOW_SCALE * s_slow + s_fast
    if USE_V_SLOW:
        v_slow = tl.load(v_slow_ptr + s_off, mask=mask, other=0).to(tl.int32)
        m = m + v_slow * V_SLOW_FACTOR

    exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale = tl.exp2(exp)
    weight_f32 = m.to(tl.float32) * scale
    tl.store(weight_ptr + w_off, weight_f32.to(tl.bfloat16), mask=mask)


# ============================================================
# Fused rebalance: compute per-row/per-col max in one kernel, with atomics.
# Then a second kernel decides ticks based on those stats and applies the
# bit-shifts to s_slow + s_fast. Replaces ~16 PyTorch ops per pass per layer.
# ============================================================

@triton.jit
def _rebalance_reduce_kernel(
    s_slow_ptr, s_fast_ptr,
    row_max_ptr,    # [N] int32, must be pre-zeroed
    col_max_ptr,    # [K] int32, must be pre-zeroed
    N, K,
    stride_n, stride_k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    s_off = offs_n[:, None] * stride_n + offs_k[None, :] * stride_k
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    abs_eff = tl.abs(s_slow + s_fast)
    # Mask out-of-bounds with 0 so they don't pollute the max
    abs_eff = tl.where(nk_mask, abs_eff, 0)

    row_partial_max = tl.max(abs_eff, axis=1)  # [BLOCK_N]
    col_partial_max = tl.max(abs_eff, axis=0)  # [BLOCK_K]

    tl.atomic_max(row_max_ptr + offs_n, row_partial_max, mask=n_mask)
    tl.atomic_max(col_max_ptr + offs_k, col_partial_max, mask=k_mask)


@triton.jit
def _rebalance_decide_apply_kernel(
    s_slow_ptr, s_fast_ptr,
    row_exp_ptr, col_exp_ptr,
    row_max_ptr, col_max_ptr,
    N, K, MAX_M, EXP_MAX,
    seed,
    stride_n, stride_k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    S_FAST_IS_INT8: tl.constexpr,
):
    """Decide per-row/col exponent tick-ups from row_max/col_max and apply
    the right-shift to s_slow + s_fast.

    Tick-up only (row_max > MAX_M, equivalently max(h) ≥ log2(MAX_M)):
    right-shift the row/col's mantissas to free headroom and bump the
    exponent. The right-shift is stochastically rounded — the dropped low
    bits bump the quotient up with matching probability so the monotone
    ratchet is value-neutral in expectation: E[SR(s >> pos)] = s / 2^pos.

    No tick-down. Pre-CLZ-bitcast, a tick-down left-shift was needed to
    bring small mantissas back into the bf16 mantissa range so they
    weren't lost to the fp32→bf16 cast. With CLZ-bitcast emission the
    per-element leading-zero count `h` is now absorbed directly into the
    bf16 biased exponent, so small mantissas emit at the correct bf16
    value without any rebalance intervention. Removing tick-down also
    removes the left-shift saturation guard, the do_tickdown / dn_axis
    parameters, the row_dn / col_dn logic, and the alternation parity
    that was needed to prevent the -2 row×col exponent collision.

    S_FAST_IS_INT8: when True, s_fast is stored as int8 (delta-storage
    path); the store narrows to int8 instead of int16. The arithmetic
    is identical -- right-shift of a bounded int8 stays in int8 range."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    row_max = tl.load(row_max_ptr + offs_n, mask=n_mask, other=0)
    col_max = tl.load(col_max_ptr + offs_k, mask=k_mask, other=0)

    row_exp = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0)
    col_exp = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0)

    row_up = (row_max > MAX_M) & (row_exp < EXP_MAX)
    col_up = (col_max > MAX_M) & (col_exp < EXP_MAX)

    row_t = row_up.to(tl.int32)             # in {0, +1}
    col_t = col_up.to(tl.int32)             # in {0, +1}
    pos = row_t[:, None] + col_t[None, :]   # in {0, +1, +2}

    s_off = offs_n[:, None] * stride_n + offs_k[None, :] * stride_k
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    # Stochastically-rounded right-shift. The dropped low `pos` bits (rem,
    # in [0, 2^pos)) bump the quotient up with probability rem/2^pos, so
    # E[result] = s / 2^pos — the monotone exponent ratchet costs no
    # mantissa value in expectation. s_slow and s_fast draw from disjoint
    # RNG offsets so their rounding is independent: lower variance on
    # m = s_slow + s_fast and on v = s_fast - s_slow, both still unbiased.
    # pos = 0 makes the SR a no-op.
    rand_off = offs_n[:, None] * K + offs_k[None, :]
    two_pos = tl.exp2(pos.to(tl.float32))      # 2^pos in {1, 2, 4} — exact
    q_slow = s_slow >> pos
    q_fast = s_fast >> pos
    rem_slow = (s_slow - (q_slow << pos)).to(tl.float32)
    rem_fast = (s_fast - (q_fast << pos)).to(tl.float32)
    up_slow = (tl.rand(seed, rand_off) * two_pos < rem_slow).to(tl.int32)
    up_fast = (tl.rand(seed, rand_off + N * K) * two_pos < rem_fast).to(tl.int32)
    s_slow_new = q_slow + up_slow
    s_fast_new = q_fast + up_fast
    # No left-shift now, so no saturation guard is needed — the right-
    # shift can only bring magnitudes down, never up.
    tl.store(s_slow_ptr + s_off, s_slow_new.to(tl.int16), mask=nk_mask)
    if S_FAST_IS_INT8:
        # Bounded int8 delta. The SR-right-shift can only shrink
        # magnitude, so an int8-fitting input stays int8-fitting.
        tl.store(s_fast_ptr + s_off, s_fast_new.to(tl.int8), mask=nk_mask)
    else:
        tl.store(s_fast_ptr + s_off, s_fast_new.to(tl.int16), mask=nk_mask)

    if pid_k == 0:
        tl.store(row_exp_ptr + offs_n, row_exp + row_t, mask=n_mask)
    if pid_n == 0:
        tl.store(col_exp_ptr + offs_k, col_exp + col_t, mask=k_mask)


def rebalance_fused_triton(s_slow, s_fast, row_exp, col_exp,
                            MAX_M, EXP_MAX,
                            max_iters=2, seed=0,
                            block_n=32, block_k=64):
    """Fused rebalance: one (reduce + decide+apply) kernel pair per iter.
    Tick-up only — see _rebalance_decide_apply_kernel for why tick-down
    was removed (CLZ-bitcast emission handles small mantissas natively).

    MAX_M / EXP_MAX are required (no defaults) so the caller's concord-
    format constants stay the single source of truth. MIN_M and EXP_MIN
    are no longer needed and the dn_axis / alternation logic is gone.

    `seed` keys the stochastic rounding of the tick-up right-shift; pass
    a fresh value per call so repeated tick-ups round independently.

    s_fast may be int16 (classic absolute-storage path) or int8
    (delta-storage path). The reduce kernel auto-widens via .to(int32);
    the apply kernel narrows the store to int16 or int8 based on the
    S_FAST_IS_INT8 constexpr (Triton compiles a separate specialization
    per dtype).
    """
    N, K = s_slow.shape
    assert s_fast.dtype in (torch.int16, torch.int8), \
        f"s_fast must be int16 or int8, got {s_fast.dtype}"
    s_fast_is_int8 = s_fast.dtype == torch.int8
    row_max = torch.zeros(N, dtype=torch.int32, device=s_slow.device)
    col_max = torch.zeros(K, dtype=torch.int32, device=s_slow.device)

    for it in range(max_iters):
        row_max.zero_()
        col_max.zero_()
        grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
        _rebalance_reduce_kernel[grid](
            s_slow, s_fast, row_max, col_max,
            N, K,
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_N=block_n, BLOCK_K=block_k,
        )
        _rebalance_decide_apply_kernel[grid](
            s_slow, s_fast, row_exp, col_exp, row_max, col_max,
            N, K, int(MAX_M), int(EXP_MAX),
            int(seed) * max_iters + it,
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_N=block_n, BLOCK_K=block_k,
            S_FAST_IS_INT8=s_fast_is_int8,
        )


def sync_weight_bf16_pair_triton(weight, s_slow, s_fast, row_exp, col_exp,
                                 mantissa_bias,
                                 v_slow=None, v_slow_factor=128,
                                 slow_scale=1,
                                 block_m=32, block_n=64):
    """Materialise live bf16 weight from concord int state into ``weight``
    (must be pre-allocated bf16, same shape as s_slow). When ``v_slow``
    is passed (int8, same shape as s_slow), it adds additively at
    shifted scale ``v_slow_factor``.

    s_fast may be int16 (classic absolute-storage path) or int8
    (delta-storage path, where s_fast holds (s_fast_logical - s_slow)).
    The kernel is dtype-polymorphic on s_fast: Triton compiles a
    separate specialization for each dtype seen, and the load expression
    `tl.load(s_fast_ptr).to(tl.int32)` sign-extends correctly from
    either width. The math `m = s_slow + s_fast` is identical in both
    storage conventions -- in the int8 case s_slow has absorbed the
    full position so the sum still gives the live mantissa."""
    M, N = s_slow.shape
    assert weight.dtype == torch.bfloat16
    assert s_slow.dtype == torch.int16
    assert s_fast.dtype in (torch.int16, torch.int8)
    assert s_slow.shape == s_fast.shape
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    if use_v_slow:
        assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8
    _sync_weight_bf16_pair_kernel[_grid(M, N)](
        weight, s_slow, s_fast, row_exp, col_exp, v_slow_ptr,
        M, N, int(mantissa_bias),
        weight.stride(0), weight.stride(1),
        s_slow.stride(0), s_slow.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
        USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        SLOW_SCALE=int(slow_scale),
    )
