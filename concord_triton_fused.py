"""Stage 2: fused bf16-reconstruction matmul kernels.

The forward kernel reconstructs bf16 weight tiles in registers and feeds
them directly to the tensor-core matmul accumulator. The weight is never
materialized in HBM — only the int state (s_slow, s_fast) plus the tiny
shared row/col_exp vectors are stored. Storage per parameter = 32 bits.

The backward path is fused similarly:
  - grad_x kernel: same recon-matmul, with grad_y replacing x.
  - grad_W + update kernel: accumulates grad_W per tile, immediately
    applies the concord-momentum integer update (stochastic round →
    s_fast tick → chase → s_slow tick), writes int state back. The
    dense grad_W tensor is never written to HBM.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from fused_profiler import PROFILER


# ============================================================
# FORWARD: y = x @ W.T  where W is bf16-reconstructed on the fly
# ============================================================

@triton.jit
def _fused_forward_kernel(
    # Inputs
    x_ptr,          # [M, K]   bf16   (M=batch, K=in_features)
    s_slow_ptr,     # [N, K]   int16  (N=out_features)
    s_fast_ptr,     # [N, K]   int16
    row_exp_ptr,    # [N]      int32
    col_exp_ptr,    # [K]      int32
    bias_ptr,       # [N]      bf16   (optional, set to 0 ptr to skip)
    v_slow_ptr,     # [N, K]   int8   (only read when USE_V_SLOW=True;
                    #                  irrelevant pointer accepted otherwise)
    # Output
    y_ptr,          # [M, N]   bf16
    # Sizes
    M, N, K, mantissa_bias, has_bias: tl.constexpr,
    # Strides
    stride_xm, stride_xk,
    stride_sn, stride_sk,
    stride_ym, stride_yn,
    # Block sizes + v_slow flag
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Load per-output-row exponent (constant w.r.t. the inner-K loop).
    # Kept in int32 — the bf16 reconstruction is done via clz-bitcast in
    # integer domain, no fp32 exp2 round-trip.
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # x tile: [BLOCK_M, BLOCK_K] bf16
        x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptr + x_off,
                         mask=m_mask[:, None] & k_mask[None, :],
                         other=0.0)

        # s_slow, s_fast tile: [BLOCK_N, BLOCK_K] int16 (widened to int32)
        s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
        s_mask = n_mask[:, None] & k_mask[None, :]
        m_eff = (tl.load(s_slow_ptr + s_off, mask=s_mask, other=0).to(tl.int32)
                 + tl.load(s_fast_ptr + s_off, mask=s_mask, other=0).to(tl.int32))
        if USE_V_SLOW:
            # v_slow is stored at int8 with shifted scale V_SLOW_FACTOR
            # (default 128). Each int8 unit represents V_SLOW_FACTOR units
            # of s_slow's mantissa, so the effective range matches s_slow
            # at int8 quantisation cost. v_slow contributes additively to
            # the live weight.
            v_slow_tile = tl.load(v_slow_ptr + s_off,
                                  mask=s_mask, other=0).to(tl.int32)
            m_eff = m_eff + v_slow_tile * V_SLOW_FACTOR

        # col_exp for this K tile.
        col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)

        # ----- clz-bitcast bf16 reconstruction ---------------------------
        # Build the bf16 bit pattern directly from (m_eff, row_e+col_e):
        #   sign | exp_bf16 | mantissa, then bitcast.
        # Equivalent to (m_eff.float() * 2^(r+c-B)).to(bf16) bit-for-bit
        # after RNE, but avoids the fp32 exp2 + multiply round-trip.
        # See clz-bitcast doc in the module header.
        abs_m = tl.abs(m_eff)
        nonzero = abs_m != 0
        safe_abs = tl.where(nonzero, abs_m, 1)
        cz = libdevice.clz(safe_abs).to(tl.int32)
        h = 31 - cz
        shift = h - 7

        # 7-bit mantissa via right-shift (h>=7, RNE-rounded) or left-shift
        # (h<7, exact since no bits are dropped).
        mant_un = (safe_abs >> tl.maximum(shift, 0)) & 0x7F
        round_bit = tl.where(shift >= 1,
                              (safe_abs >> tl.maximum(shift - 1, 0)) & 1, 0)
        sticky_mask = tl.where(shift >= 2,
                                (1 << tl.maximum(shift - 1, 0)) - 1, 0)
        sticky = ((safe_abs & sticky_mask) != 0).to(tl.int32)
        round_up = round_bit & (sticky | (mant_un & 1))
        mant_r = mant_un + round_up
        carry_ge7 = (mant_r >> 7) & 1
        mant_ge7 = mant_r & 0x7F
        mant_lt7 = (safe_abs << tl.maximum(7 - h, 0)) & 0x7F
        mant = tl.where(h >= 7, mant_ge7, mant_lt7)
        h_eff = tl.where(h >= 7, h + carry_ge7, h)

        exp_bf16 = (row_e[:, None] + col_e[None, :]
                    - mantissa_bias + h_eff + 127)
        in_range = (exp_bf16 >= 1) & (exp_bf16 <= 254) & nonzero
        sign_bit = (m_eff < 0).to(tl.int32) << 15
        bits = sign_bit | (exp_bf16 << 7) | mant
        bits = tl.where(in_range, bits, 0).to(tl.uint16)
        w_tile = bits.to(tl.bfloat16, bitcast=True)

        # tl.dot wants [M, K] @ [K, N]. We have x [M, K] and W [N, K];
        # transpose W to [K, N] for the matmul.
        acc += tl.dot(x_tile, tl.trans(w_tile))

    if has_bias:
        b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += b[None, :]

    y_off = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptr + y_off,
             acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def fused_forward(x, s_slow, s_fast, row_exp, col_exp, bias,
                  mantissa_bias=15,
                  v_slow=None, v_slow_factor=128,
                  block_m=64, block_n=64, block_k=32):
    """v_slow: optional int8 (N, K) tensor. When passed, the live weight
    reconstruction adds v_slow * v_slow_factor in mantissa units."""
    M, K = x.shape
    N, K2 = s_slow.shape
    assert K == K2
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else torch.empty(0, dtype=torch.bfloat16, device=x.device)
    use_v_slow = v_slow is not None
    # When v_slow is unused, pass s_slow's pointer as a placeholder --
    # the kernel's USE_V_SLOW constexpr gates the actual load.
    v_slow_ptr = v_slow if use_v_slow else s_slow
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    with PROFILER.time('linear_forward'):
        _fused_forward_kernel[grid](
            x, s_slow, s_fast, row_exp, col_exp, bias_ptr, v_slow_ptr,
            y,
            M, N, K, int(mantissa_bias), has_bias,
            x.stride(0), x.stride(1),
            s_slow.stride(0), s_slow.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        )
    return y


# ============================================================
# BACKWARD-grad-x: dx = grad_y @ W  (W still reconstructed on the fly)
# ============================================================

@triton.jit
def _fused_grad_x_kernel(
    grad_y_ptr,     # [M, N] bf16
    s_slow_ptr,     # [N, K] int16
    s_fast_ptr,     # [N, K] int16
    row_exp_ptr,    # [N] int32
    col_exp_ptr,    # [K] int32
    grad_x_ptr,     # [M, K] bf16 — output
    v_slow_ptr,     # [N, K] int8 (only read when USE_V_SLOW=True)
    M, N, K, mantissa_bias,
    stride_gym, stride_gyn,
    stride_sn, stride_sk,
    stride_gxm, stride_gxk,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    k_mask = offs_k < K

    # col_exp kept in int32 — bf16 W is reconstructed via clz-bitcast
    # below, no fp32 exp2 round-trip.
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        gy_off = offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        gy_tile = tl.load(grad_y_ptr + gy_off,
                          mask=m_mask[:, None] & n_mask[None, :],
                          other=0.0)

        s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
        s_mask = n_mask[:, None] & k_mask[None, :]
        m_eff = (tl.load(s_slow_ptr + s_off, mask=s_mask, other=0).to(tl.int32)
                 + tl.load(s_fast_ptr + s_off, mask=s_mask, other=0).to(tl.int32))
        if USE_V_SLOW:
            v_slow_tile = tl.load(v_slow_ptr + s_off,
                                  mask=s_mask, other=0).to(tl.int32)
            m_eff = m_eff + v_slow_tile * V_SLOW_FACTOR

        row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)

        # clz-bitcast bf16 reconstruction (RNE-rounded, bit-exact match
        # to fp32 m_eff * exp2(r+c-B) → bf16). See _fused_forward_kernel
        # for the long-form comment block.
        abs_m = tl.abs(m_eff)
        nonzero = abs_m != 0
        safe_abs = tl.where(nonzero, abs_m, 1)
        cz = libdevice.clz(safe_abs).to(tl.int32)
        h = 31 - cz
        shift = h - 7
        mant_un = (safe_abs >> tl.maximum(shift, 0)) & 0x7F
        round_bit = tl.where(shift >= 1,
                              (safe_abs >> tl.maximum(shift - 1, 0)) & 1, 0)
        sticky_mask = tl.where(shift >= 2,
                                (1 << tl.maximum(shift - 1, 0)) - 1, 0)
        sticky = ((safe_abs & sticky_mask) != 0).to(tl.int32)
        round_up = round_bit & (sticky | (mant_un & 1))
        mant_r = mant_un + round_up
        carry_ge7 = (mant_r >> 7) & 1
        mant_ge7 = mant_r & 0x7F
        mant_lt7 = (safe_abs << tl.maximum(7 - h, 0)) & 0x7F
        mant = tl.where(h >= 7, mant_ge7, mant_lt7)
        h_eff = tl.where(h >= 7, h + carry_ge7, h)
        exp_bf16 = (row_e[:, None] + col_e[None, :]
                    - mantissa_bias + h_eff + 127)
        in_range = (exp_bf16 >= 1) & (exp_bf16 <= 254) & nonzero
        sign_bit = (m_eff < 0).to(tl.int32) << 15
        bits = sign_bit | (exp_bf16 << 7) | mant
        bits = tl.where(in_range, bits, 0).to(tl.uint16)
        w_tile = bits.to(tl.bfloat16, bitcast=True)

        # grad_x[m, k] = sum_n grad_y[m, n] * W[n, k]
        # tl.dot(grad_y[M, N], W[N, K]) gives [M, K]. Direct call.
        acc += tl.dot(gy_tile, w_tile)

    gx_off = offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk
    tl.store(grad_x_ptr + gx_off,
             acc.to(tl.bfloat16),
             mask=m_mask[:, None] & k_mask[None, :])


# ============================================================
# BACKWARD-grad-W + UPDATE: accumulate grad_W and tick s_slow/s_fast
# in a single kernel, never materializing grad_W in HBM.
# ============================================================

@triton.jit
def _hash_uniform(x, pos, salt):
    """xorshift hash: (x, pos, salt) int32 tensors → uniform [0,1) float32.

    x:    per-element value (typically s_fast bits). Diversifies the
          draws across steps as s_fast accumulates ticks.
    pos:  per-element position term. Diversifies the draws across
          elements within a tile, even when x is uniform (e.g.
          cold-start s_fast=0). Compute once per kernel from offs_n
          and offs_k.
    salt: per-step scalar (the step counter). Diversifies across steps.

    Pure bitwise ops (no large-int multiplications) so Triton stays
    in int32."""
    h = x ^ salt ^ pos
    h = h ^ (h << 13)
    h = h ^ (h >> 17)
    h = h ^ (h << 5)
    h = h ^ (h >> 7)
    return (h & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _fused_grad_W_and_update_kernel(
    grad_y_ptr,     # [M, N] bf16
    x_ptr,          # [M, K] bf16
    s_slow_ptr,     # [N, K] int16, modified in place
    s_fast_ptr,     # [N, K] int16, modified in place
    row_exp_ptr,    # [N] int32
    col_exp_ptr,    # [K] int32
    v_slow_ptr,     # [N, K] int8, modified in place (only when USE_V_SLOW=True)
    M, N, K,
    lr, mantissa_bias, alpha, beta1,
    amplify_aligned,  # fp32: when grad*v_prev > 0, add amplify*v_prev to
                      # delta_t. "Conditional momentum": amplifies the
                      # commitment to s_fast iff the new gradient agrees
                      # with the existing velocity direction.
    alpha_v_fast,   # fp32: per-step v_slow <- s_fast leak rate
    step_salt_ptr,  # int32* [1]: per-step varying salt. Loaded once per
                    # kernel. Tensor-backed (rather than int scalar) so
                    # Dynamo can trace the launcher's in-place counter
                    # bump under HOPs (gradient checkpointing).
    stride_gym, stride_gyn,
    stride_xm, stride_xk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    # When APPLY_CHASE=False, only the SR-tick of s_fast fires. The
    # chase into s_slow and the v_slow leak are skipped so that K
    # consecutive backward calls accumulate K independent SR-ticks
    # into s_fast before any smoothing is applied. The K-th call
    # (APPLY_CHASE=True) does the full update — chase + leak — over
    # the accumulated s_fast drift. This is the Concord-native
    # gradient-accumulation primitive: equivalent in expected sum to
    # one big tick of the summed grad, but with K× the SR-rounding
    # variance, preserving per-microbatch noise structure.
    APPLY_CHASE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Load the per-step salt once; all SR draws in this kernel re-use it
    # via cheap XOR with per-purpose constants (see r1/r2/r3 below).
    step_salt = tl.load(step_salt_ptr).to(tl.int32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # Accumulate grad_W[n,k] = sum_m grad_y[m,n] * x[m,k]
    grad_W = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        gy_off = offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        gy = tl.load(grad_y_ptr + gy_off,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        xt = tl.load(x_ptr + x_off,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # grad_W += gy.T @ xt
        grad_W += tl.dot(tl.trans(gy), xt)

    # === Apply concord momentum update directly ===
    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.float32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.float32)
    inv_exp = mantissa_bias - row_e[:, None] - col_e[None, :]
    scale_inv = tl.exp2(inv_exp)
    delta_grad = -lr * grad_W * scale_inv

    # v_prev = s_fast - s_slow (in mantissa units, as float for the math)
    v_prev = (s_fast - s_slow).to(tl.float32)
    # beta1 damps the existing velocity (pull s_fast back toward s_slow).
    # amplify_aligned does the opposite, but per-element gated on
    # sign(delta_grad) == sign(v_prev) — "conditional momentum" that
    # doubles down on s_fast's velocity only when the new gradient
    # agrees with where s_fast was already heading. Default 0 = off.
    aligned = (delta_grad * v_prev > 0.0).to(tl.float32)
    delta_t = delta_grad - beta1 * v_prev + amplify_aligned * aligned * v_prev

    # Stochastic round → tick on s_fast.
    # Hash from (s_fast bits, position, step salt) — no external RNG
    # tensor needed. pos_hash diversifies across elements within the
    # tile (needed when s_fast is uniform, e.g. cold-start s_fast=0);
    # s_fast bits diversify across steps as the buffer accumulates
    # ticks; step_salt diversifies across steps via the launcher's
    # global counter.
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Chase + v_slow leak — only when APPLY_CHASE=True. Under
    # gradient accumulation (APPLY_CHASE=False on microbatches
    # 0..K-2), s_fast accumulates K SR-ticks of grad and the chase
    # plus leak fire once on the K-th call, draining the accumulated
    # drift into s_slow / v_slow_i8 in one pass.
    if APPLY_CHASE:
        # Chase: alpha * (s_fast_new - s_slow). Use a different salt
        # so the two stochastic-round draws are decorrelated.
        gap = (s_fast - s_slow).to(tl.float32)
        delta_slow_f = alpha * gap
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(delta_slow_f)
        frac_s = delta_slow_f - floor_s
        tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow = s_slow + tick_slow

        # v_slow ← s_fast leak (per-step). v_slow is stored at int8
        # with shifted scale V_SLOW_FACTOR; the leak operates in
        # s_fast's mantissa units, the tick is quantised to int8 (one
        # v_slow unit = V_SLOW_FACTOR mantissa units).
        #
        # When MASS_PRESERVE=True, s_fast loses exactly what v_slow
        # gains in mantissa units, so the live weight is unchanged by
        # the leak — only the gradient changes the weight. When
        # MASS_PRESERVE=False, the leak is a "second chase" (the live
        # weight grows by what v_slow gained), mirroring the existing
        # s_fast/s_slow chase that is also non-mass-preserving and
        # which the CIFAR headline depends on.
        if USE_V_SLOW:
            v_slow_old = tl.load(v_slow_ptr + s_off, mask=nk_mask,
                                   other=0).to(tl.int32)
            gap_v_full = (s_fast - v_slow_old * V_SLOW_FACTOR).to(tl.float32)
            delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
            r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
            floor_v = tl.floor(delta_v8)
            frac_v = delta_v8 - floor_v
            tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
            new_v_int32 = v_slow_old + tick_v8
            new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
            if MASS_PRESERVE:
                actual_tick_v8 = new_v_int8 - v_slow_old
                actual_tick_full = actual_tick_v8 * V_SLOW_FACTOR
                s_fast = s_fast - actual_tick_full
            tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8),
                       mask=nk_mask)

        # s_slow only changes inside the chase branch — write back here.
        # Saturate to int16 range before the narrowing store; clamp
        # rationale unchanged from the original kernel.
        s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
        tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)

    # s_fast is updated every call (tick-only mode included) — write it
    # back unconditionally.
    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)


# ============================================================
# BACKWARD-grad-W + ADAMW UPDATE: single fused kernel.
#
# Variance estimate is the per-element weight magnitude itself,
# derived for free from the bit pattern we already compute for the
# bf16 forward emission:
#
#   h_ij      = 31 - clz(|s_slow_ij + s_fast_ij|)    # leading-1 bit
#   |W|_proxy = 2^(h_ij + r_i + c_j - B)             # log2-quantised
#
# The β2 time-discount role is filled by a per-layer scalar
# discount_t (caller-supplied, recomputed CPU-side at refit cadence
# from the cascade's |W| trajectory — see the lazy-path discussion
# in ConcordLinearFused.update_discount_from_cascade). discount_t
# enters the step as a multiplicative effective-lr modulation:
#
#   step_live = discount_t · grad / (|W|_proxy + eps) + wd · W
#   delta     = -lr · step_live · scale_inv
#
# State budget (per managed weight matrix of shape [N, K]):
#   s_slow + s_fast    : 32 bits/param (live weight, implicit m via chase)
#   row_exp + col_exp  : 16 bits per row/col, amortized ≈ 0/param
#                        (storage scale; refit_envelope re-anchors)
#
# Total: ≈ 32 bits/param. No v_row, no v_col, no g²-EMA tracking,
# no atomic-adds, no two-kernel hand-off. The per-element variance
# is the weight magnitude itself; the time-discount is a per-layer
# scalar that gets updated CPU-side on the lazy path.
#
# Heavy gradient pressure on an element drives |m_eff| up over many
# steps, which grows h_ij, which grows the preconditioner, which
# damps further updates. Self-stabilising: the weight records its
# own cumulative gradient pressure as a side effect of being updated.
# ============================================================

@triton.jit
def _fused_grad_W_and_adamw_update_kernel(
    grad_y_ptr,     # [M, N] bf16
    x_ptr,          # [M, K] bf16
    s_slow_ptr,     # [N, K] int16, modified in place
    s_fast_ptr,     # [N, K] int16, modified in place
    row_exp_ptr,    # [N] int8 (int4 range)
    col_exp_ptr,    # [K] int8
    discount_row_ptr,  # [N] fp32 — per-row factor of the rank-1 discount
    discount_col_ptr,  # [K] fp32 — per-col factor
    M, N, K,
    lr, mantissa_bias, alpha, beta1, weight_decay, eps, step_cap,
    step_salt,      # int32: vary across training steps to decorrelate
    stride_gym, stride_gyn,
    stride_xm, stride_xk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # Accumulate grad_W[n,k] = sum_m grad_y[m,n] * x[m,k] in fp32 regs.
    grad_W = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        gy_off = offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        gy = tl.load(grad_y_ptr + gy_off,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        xt = tl.load(x_ptr + x_off,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        grad_W += tl.dot(tl.trans(gy), xt)

    # Load state.
    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)

    # Mantissa-aware |W|_proxy: use the full live weight |m_eff|*2^(r+c-B),
    # NOT the CLZ floor-log2. The CLZ version ignored the mantissa entirely
    # (only the leading-bit position), so |W|_proxy was power-of-2
    # quantized: |m_eff| ticking 1->2 doubled the proxy, so 1/|W| halved
    # discontinuously. The mantissa carries strictly finer variance info
    # than just the leading bit (|m_eff|=1 vs |m_eff|=127 are very
    # different scales living in the same h=0 binade). Using the full
    # current_weight directly removes the binade discontinuities.
    m_eff = s_slow + s_fast
    abs_m = tl.abs(m_eff)
    nonzero = abs_m != 0

    # Scale conversion: live-weight delta -> mantissa-units delta.
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_inv = tl.exp2(-total_exp)
    scale_fwd = tl.exp2(total_exp)
    current_weight = m_eff.to(tl.float32) * scale_fwd
    abs_W_proxy = tl.abs(current_weight)
    # When m_eff == 0, force a small nonzero floor so 1/|W| stays sane.
    abs_W_proxy = tl.where(nonzero, abs_W_proxy, eps)

    # AdamW-style step. The rank-1 discount = discount_row[i] * discount_col[j]
    # is the per-element effective-lr modulator; both vectors are caller-
    # updated CPU-side at refit cadence from the cascade's per-row /
    # per-col |W| trajectory. Weight-decay applies to the live weight.
    disc_row = tl.load(discount_row_ptr + offs_n, mask=n_mask, other=1.0)
    disc_col = tl.load(discount_col_ptr + offs_k, mask=k_mask, other=1.0)
    discount_ij = disc_row[:, None] * disc_col[None, :]
    step_live = (discount_ij * grad_W / (abs_W_proxy + eps)
                 + weight_decay * current_weight)
    # Bound the per-element step at |step_live| <= step_cap, mirroring real
    # AdamW's m/sqrt(v+eps) -> sign(m) as v -> 0. Without this, an element
    # with |W|_proxy near 0 sees step ~ grad/eps -> unbounded -> int16
    # saturation in one tick -> rebalance bumps row_exp -> the whole row
    # doubles -> the network cascades. Verified load-bearing: 20-epoch
    # CIFAR with no-curve-fit + no-cap collapses at step ~500 just like
    # the original (rank-1 discount was not the cause). Cap costs two
    # min/max ops per element.
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv

    # Velocity feedback (β1). With β1=0 default, this is a no-op and
    # the implicit-m chase carries the momentum dynamics on its own.
    v_prev = (s_fast - s_slow).to(tl.float32)
    delta_t = delta_grad - beta1 * v_prev

    # SR tick s_fast. pos_hash provides per-element decorrelation.
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Chase s_slow toward s_fast.
    gap = (s_fast - s_slow).to(tl.float32)
    delta_slow_f = alpha * gap
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_s = tl.floor(delta_slow_f)
    frac_s = delta_slow_f - floor_s
    tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
    s_slow = s_slow + tick_slow

    # Saturate; the int17 budget for s_slow+s_fast covers ordinary
    # growth, the periodic redistribution handles anything larger.
    s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)


# ============================================================
# Fused EMA close-out for v_rank1: v = beta2*v + g2; g2 = 0
# Replaces three pytorch in-place ops (mul_, add_, zero_) with one
# kernel launch. Tiny but called twice per layer per step, so cuts
# ~12 launches/step down to ~4 on a 2-linear model.
# ============================================================

@triton.jit
def _v_ema_close_kernel(
    v_ptr,   # fp32 [N], read+write
    g2_ptr,  # fp32 [N], read+zero
    N, beta2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    g2 = tl.load(g2_ptr + offs, mask=mask, other=0.0)
    v_new = beta2 * v + g2
    tl.store(v_ptr + offs, v_new, mask=mask)
    tl.store(g2_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


def v_ema_close(v, g2, beta2, BLOCK=128):
    """Fused: v = beta2*v + g2;  g2 = 0. In-place on both buffers.

    Triton requires BLOCK to be a power of 2 and >= 16; pad as needed."""
    N = v.numel()
    grid = (triton.cdiv(N, BLOCK),)
    _v_ema_close_kernel[grid](v, g2, N, float(beta2), BLOCK=BLOCK)


# ============================================================
# BACKWARD-grad-W + ADAMW UPDATE (v_rank1 variant): the variance estimate
# is a rank-1 reconstruction from per-row + per-col EMAs of g² (Adafactor).
#
# Two-buffer scheme avoids the in-kernel atomic_add read/write race:
#   - v_row [N], v_col [K]    : EMA of g² (read-only this kernel)
#   - g2_row [N], g2_col [K]  : scratch, zeroed each step, atomic-summed here
# The python wrapper updates v_row/v_col = β2*v + g2_* AFTER the kernel.
#
# Per element: v_rank1[n,k] = v_row[n] * v_col[k] / mean(v_row)  (Adafactor)
# Bias-corrected: v_hat = v_rank1 / (1 - β2^t)
# step_live = grad / sqrt(v_hat + eps), then clipped at ±step_cap.
# ============================================================

@triton.jit
def _fused_grad_W_and_adamw_v_rank1_update_kernel(
    grad_y_ptr,     # [M, N] bf16
    x_ptr,          # [M, K] bf16
    s_slow_ptr,     # [N, K] int16, modified in place
    s_fast_ptr,     # [N, K] int16, modified in place
    row_exp_ptr,    # [N] int8
    col_exp_ptr,    # [K] int8
    v_row_ptr,      # [N] fp32, EMA of mean_k(g²[n,:]). Read-only here.
    v_col_ptr,      # [K] fp32, EMA of mean_n(g²[:,k]). Read-only here.
    g2_row_ptr,     # [N] fp32, scratch (pre-zeroed). atomic_add (1-β2)/K * tile_g²_row_sum
    g2_col_ptr,     # [K] fp32, scratch (pre-zeroed). atomic_add (1-β2)/N * tile_g²_col_sum
    v_slow_ptr,     # [N, K] int8 (gated by USE_V_SLOW). Mutated.
    M, N, K,
    lr, mantissa_bias, alpha, beta1, weight_decay, eps, step_cap,
    mean_v_row_inv_ptr,    # 1-elem fp32 GPU tensor: 1 / mean(v_row)
    bias_corr_inv,     # 1 / (1 - β2^t)
    inv_K, inv_N,      # 1/K, 1/N  -- per-axis g² mean (not sum)
    one_minus_beta2,   # (1 - β2)  -- scaling the atomic_add contribution
    alpha_v_fast,      # fp32: per-step v_slow ← s_fast leak rate
    step_salt_ptr,
    stride_gym, stride_gyn,
    stride_xm, stride_xk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    # Per-step salt from tensor counter — Dynamo-safe under HOP and
    # actually varies across steps (vs the bias-prone static int salt).
    step_salt = tl.load(step_salt_ptr).to(tl.int32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # 1) Compute grad_W in fp32 registers.
    grad_W = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        gy_off = offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        gy = tl.load(grad_y_ptr + gy_off,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        xt = tl.load(x_ptr + x_off,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        grad_W += tl.dot(tl.trans(gy), xt)

    # 2) Load state and scale factors.
    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    m_eff = s_slow + s_fast
    if USE_V_SLOW:
        # v_slow_i8 contributes additively to the live weight at the
        # shifted scale V_SLOW_FACTOR (each int8 unit = factor mantissa
        # units). current_weight (used for weight_decay) must reflect
        # this, otherwise wd would shrink only s_slow+s_fast and let
        # v_slow drift unchecked.
        v_slow_old = tl.load(v_slow_ptr + s_off,
                              mask=nk_mask, other=0).to(tl.int32)
        m_eff = m_eff + v_slow_old * V_SLOW_FACTOR
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_inv = tl.exp2(-total_exp)
    scale_fwd = tl.exp2(total_exp)
    current_weight = m_eff.to(tl.float32) * scale_fwd

    # 3) Read per-axis variance (lagged: v_row/v_col are the previous step's
    #    completed EMA; this step's contribution goes into g2_row/g2_col).
    v_r = tl.load(v_row_ptr + offs_n, mask=n_mask, other=0.0)
    v_c = tl.load(v_col_ptr + offs_k, mask=k_mask, other=0.0)
    # Adafactor rank-1: v_rank1[n,k] = v_row[n] * v_col[k] / mean(v_row).
    # mean_v_row_inv comes via a 1-elem GPU tensor pointer so the launcher
    # never needs a `.item()` sync.
    mean_v_row_inv = tl.load(mean_v_row_inv_ptr)
    v_rank1 = v_r[:, None] * v_c[None, :] * mean_v_row_inv
    # Bias correction: v_hat = v_rank1 / (1 - β2^t)
    v_hat = v_rank1 * bias_corr_inv

    # 4) AdamW step: grad / sqrt(v_hat + eps). Cap clips the small-v_hat
    #    blowup that's analogous to the small-|W| blowup in the W_proxy path.
    step_live = grad_W / tl.sqrt(v_hat + eps) + weight_decay * current_weight
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv

    # 5) Velocity feedback (β1, default 0).
    v_prev = (s_fast - s_slow).to(tl.float32)
    delta_t = delta_grad - beta1 * v_prev

    # 6) SR tick s_fast and chase s_slow (same as W_proxy path).
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast
    gap = (s_fast - s_slow).to(tl.float32)
    delta_slow_f = alpha * gap
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_s = tl.floor(delta_slow_f)
    frac_s = delta_slow_f - floor_s
    tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
    s_slow = s_slow + tick_slow

    # 6b) v_slow_i8 leak (same as the SGD chase kernel: quantised to
    #     int8 at the shifted scale V_SLOW_FACTOR, optionally mass-
    #     preserving). The int8 quantum (one v_slow_i8 unit = factor
    #     mantissa units) is coarse, so SR rounding here is more lossy
    #     per shift than the int16 path — fine for the slow EMA role
    #     v_slow plays, where most ticks are zero anyway.
    if USE_V_SLOW:
        gap_v_full = (s_fast - v_slow_old * V_SLOW_FACTOR).to(tl.float32)
        delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
        r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
        floor_v = tl.floor(delta_v8)
        frac_v = delta_v8 - floor_v
        tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
        new_v_int32 = v_slow_old + tick_v8
        new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
        if MASS_PRESERVE:
            actual_tick_v8 = new_v_int8 - v_slow_old
            actual_tick_full = actual_tick_v8 * V_SLOW_FACTOR
            s_fast = s_fast - actual_tick_full
        tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8), mask=nk_mask)

    s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)

    # 7) Accumulate per-axis (1-β2)*mean(g²) contribution into g2_row/g2_col.
    #    Each tile contributes its slice; the python wrapper then does
    #    v_row = β2 * v_row + g2_row (full EMA closure).
    g2 = grad_W * grad_W
    g2 = tl.where(nk_mask, g2, 0.0)
    g2_row_tile = tl.sum(g2, axis=1)  # [BLOCK_N]
    g2_col_tile = tl.sum(g2, axis=0)  # [BLOCK_K]
    tl.atomic_add(g2_row_ptr + offs_n,
                  one_minus_beta2 * inv_K * g2_row_tile, mask=n_mask)
    tl.atomic_add(g2_col_ptr + offs_k,
                  one_minus_beta2 * inv_N * g2_col_tile, mask=k_mask)


# ============================================================
# BACKWARD-grad-W + ADAMW UPDATE (v from velocity): the variance estimate
# is read OFF (s_fast - s_slow) directly -- the implicit momentum already
# accumulates gradient updates over the chase window with effective time
# constant rebalance_every / alpha (~80 steps at the defaults), so its
# squared magnitude (in weight space) is an AdamW-second-moment estimator
# at zero extra storage.
#
# Per element:
#   v_int = s_fast - s_slow             (int16, persistent, already stored)
#   scale = 2^(row_exp + col_exp - mantissa_bias)
#   v_proxy = (v_int * scale)^2 * v_scale
#   step_live = grad_W / sqrt(v_proxy + eps)   (then clipped at +/- step_cap)
#
# `v_scale` is a single fp32 scalar that absorbs the lr^2 * T_eff constant
# from the derivation. Tune it like a temperature on the preconditioner.
# ============================================================

@triton.jit
def _fused_grad_W_and_adamw_v_from_velocity_update_kernel(
    grad_y_ptr,     # [M, N] bf16
    x_ptr,          # [M, K] bf16
    s_slow_ptr,     # [N, K] int16, modified in place
    s_fast_ptr,     # [N, K] int16, modified in place
    row_exp_ptr,    # [N] int8
    col_exp_ptr,    # [K] int8
    M, N, K,
    lr, mantissa_bias, alpha, beta1, weight_decay, eps, step_cap,
    v_scale,        # fp32 scalar multiplied into the v_proxy
    step_salt,
    stride_gym, stride_gyn,
    stride_xm, stride_xk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # 1) Compute grad_W in fp32 registers.
    grad_W = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        gy_off = offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        gy = tl.load(grad_y_ptr + gy_off,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        xt = tl.load(x_ptr + x_off,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        grad_W += tl.dot(tl.trans(gy), xt)

    # 2) Load state and scale factors.
    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    m_eff = s_slow + s_fast
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_inv = tl.exp2(-total_exp)
    scale_fwd = tl.exp2(total_exp)
    current_weight = m_eff.to(tl.float32) * scale_fwd

    # 3) Per-element variance from the velocity field. v_int already lives
    #    in s_fast - s_slow (no extra load). In weight-space:
    #        v_proxy = (v_int * scale)^2 * v_scale
    #    No β2 EMA needed -- the chase dynamic IS the EMA.
    v_int = (s_fast - s_slow).to(tl.float32)
    v_in_w = v_int * scale_fwd                     # velocity in weight units
    v_proxy = v_in_w * v_in_w * v_scale

    # 4) AdamW step. step_live = g / sqrt(v + eps), capped at +/- step_cap.
    step_live = grad_W / tl.sqrt(v_proxy + eps) + weight_decay * current_weight
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv

    # 5) Velocity feedback (β1, default 0).
    delta_t = delta_grad - beta1 * v_int

    # 6) SR tick s_fast and chase s_slow (same as the other AdamW paths).
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast
    gap = (s_fast - s_slow).to(tl.float32)
    delta_slow_f = alpha * gap
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_s = tl.floor(delta_slow_f)
    frac_s = delta_slow_f - floor_s
    tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
    s_slow = s_slow + tick_slow
    s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)


# ============================================================
# BACKWARD-grad-W + ADAMW UPDATE (three-accumulator variant).
#
# Adds a third int16 buffer v_slow alongside s_slow / s_fast. v_slow is
# a very-long EMA of (mostly s_slow, partly s_fast) that captures the
# secular drift. The high-pass residual
#     noise = (s_fast - s_slow) - C * (s_slow - v_slow)
# rejects the drift component and leaves zero-mean noise; |noise * scale|^2
# is then a per-element AdamW second-moment estimator in the noise regime,
# without the (s_fast - s_slow) drift-suppression failure mode that
# v_from_velocity hits.
#
# Per-step v_slow leak (here, inside the kernel):
#     v_slow <- (1 - alpha_v_fast) * v_slow + alpha_v_fast * s_fast
# applied via the same SR-rounding primitive s_fast uses.
#
# The per-rebalance v_slow leak (toward s_slow at alpha_v_slow) lives in
# ConcordLinearFused.rebalance().
# ============================================================

@triton.jit
def _fused_grad_W_and_adamw_three_accum_update_kernel(
    grad_y_ptr,     # [M, N] bf16
    x_ptr,          # [M, K] bf16
    s_slow_ptr,     # [N, K] int16, mutated
    s_fast_ptr,     # [N, K] int16, mutated
    v_slow_ptr,     # [N, K] int8, mutated. v_slow_i8 at shifted scale
                    # V_SLOW_FACTOR — each unit represents factor
                    # mantissa units. Plays a dual role: drift-cancel
                    # input for the variance signal AND additive
                    # contributor to the live weight (consistent with
                    # the SGD chase int8 path).
    row_exp_ptr,    # [N] int8
    col_exp_ptr,    # [K] int8
    M, N, K,
    lr, mantissa_bias, alpha, beta1, weight_decay, eps, step_cap,
    v_scale,         # fp32: absorbs lr^2 * T_eff constant
    drift_cancel_C,  # fp32: high-pass coefficient on (s_slow - v_slow*F)
    alpha_v_fast,    # fp32: per-step v_slow <- s_fast leak rate
    wd_sv,           # fp32: decay coefficient pulling s_slow toward
                     # v_slow_full. Non-mass-preserving: live weight
                     # shrinks by lr*wd_sv*(s_slow - v_slow_full) per
                     # step. The "drift signal that hasn't been
                     # confirmed by the long-EMA". 0 = off.
    wd_sf,           # fp32: decay coefficient pulling s_fast toward
                     # v_slow_full. Same form as wd_sv but applied to
                     # s_fast's offset. 0 = off.
    step_salt_ptr,   # 1-elem int32 GPU tensor; tl.load gives a per-step
                     # varying salt (Dynamo-safe via in-place add by
                     # launcher). See _get_step_counter.
    stride_gym, stride_gyn,
    stride_xm, stride_xk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    # See _fused_grad_W_and_update_kernel docstring for APPLY_CHASE
    # semantics — same Concord-native grad-accumulation primitive.
    APPLY_CHASE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # 1) grad_W in fp32 registers.
    grad_W = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        gy_off = offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        x_off = offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        gy = tl.load(grad_y_ptr + gy_off,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        xt = tl.load(x_ptr + x_off,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        grad_W += tl.dot(tl.trans(gy), xt)

    # 2) Load state + scale. v_slow_i8 is stored at int8 with shifted
    #    scale V_SLOW_FACTOR (default 128 = 2^7). Pre-compute its
    #    mantissa-units full-precision form for downstream use.
    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    v_slow_i8 = tl.load(v_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    v_slow_full = v_slow_i8 * V_SLOW_FACTOR
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    # v_slow_i8 contributes to live weight at shifted scale, matching
    # the SGD chase int8 path. weight_decay applies to the FULL live
    # weight so wd shrinks every accumulator (not just s_slow/s_fast).
    m_eff = s_slow + s_fast + v_slow_full
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_inv = tl.exp2(-total_exp)
    scale_fwd = tl.exp2(total_exp)
    current_weight = m_eff.to(tl.float32) * scale_fwd

    # 3) Per-element noise residual = drift-cancelled velocity. In drift
    #    regime, (s_fast - s_slow) and (s_slow - v_slow_full) are
    #    linearly related; drift_cancel_C is the ratio that makes their
    #    drift contributions cancel. What remains is high-frequency
    #    noise — the AdamW second moment estimator.
    d_fs = (s_fast - s_slow).to(tl.float32)
    d_sv = (s_slow - v_slow_full).to(tl.float32)
    noise = d_fs - drift_cancel_C * d_sv
    noise_in_w = noise * scale_fwd
    v_proxy = noise_in_w * noise_in_w * v_scale

    # 4) AdamW step.
    step_live = grad_W / tl.sqrt(v_proxy + eps) + weight_decay * current_weight
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv
    delta_t = delta_grad - beta1 * d_fs

    # 5) SR tick s_fast (same as everywhere else).
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Chase + v_slow leak + Bayesian-anchored wd — all gated on
    # APPLY_CHASE. Under accumulation (APPLY_CHASE=False on
    # microbatches 0..K-2), only the SR-tick on s_fast above fires;
    # s_slow / v_slow_i8 / wd terms wait for the K-th call.
    if APPLY_CHASE:
        # 6) Chase s_slow toward s_fast at alpha.
        gap = (s_fast - s_slow).to(tl.float32)
        delta_slow_f = alpha * gap
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(delta_slow_f)
        frac_s = delta_slow_f - floor_s
        tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow = s_slow + tick_slow

        # 7) v_slow_i8 leak toward s_fast (int8 SR rounding at shifted
        #    scale). Mirrors the SGD chase int8 path so the buffer
        #    semantics are unified. MASS_PRESERVE=True subtracts the
        #    committed tick from s_fast in mantissa units; False makes
        #    the leak a "second chase" that grows live weight (matching
        #    the non-mass-preserving s_fast/s_slow chase).
        gap_v_full = (s_fast - v_slow_full).to(tl.float32)
        delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
        r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
        floor_v = tl.floor(delta_v8)
        frac_v = delta_v8 - floor_v
        tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
        new_v_int32 = v_slow_i8 + tick_v8
        new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
        if MASS_PRESERVE:
            actual_tick_v8 = new_v_int8 - v_slow_i8
            actual_tick_full = actual_tick_v8 * V_SLOW_FACTOR
            s_fast = s_fast - actual_tick_full

        # "Posterior-justified" weight decay (Bayesian-anchored at
        # v_slow_full): v_slow_full is the long-time-averaged gradient
        # signal — the data-supported component of the weight. s_fast
        # and s_slow's offsets from v_slow_full are the less-confirmed
        # transients; decaying them toward v_slow_full (non-mass-
        # preserving) shrinks the live weight by the part not yet
        # justified by the long-time gradient history. v_slow_full
        # itself is exempt (it IS the prior mean).
        #
        # wd_sv = decay rate for (s_slow - v_slow_full).
        # wd_sf = decay rate for (s_fast - v_slow_full).
        # Set either to 0 to disable that axis.
        v_slow_full_post = new_v_int8 * V_SLOW_FACTOR
        d_sv_full_post = (s_slow - v_slow_full_post).to(tl.float32)
        wd_sv_delta = lr * wd_sv * d_sv_full_post
        r4 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x66665555)
        floor_wd_sv = tl.floor(wd_sv_delta)
        frac_wd_sv = wd_sv_delta - floor_wd_sv
        tick_wd_sv = (floor_wd_sv + (r4 < frac_wd_sv).to(tl.float32)).to(tl.int32)
        s_slow = s_slow - tick_wd_sv

        d_sf_full_post = (s_fast - v_slow_full_post).to(tl.float32)
        wd_sf_delta = lr * wd_sf * d_sf_full_post
        r5 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x77770000)
        floor_wd_sf = tl.floor(wd_sf_delta)
        frac_wd_sf = wd_sf_delta - floor_wd_sf
        tick_wd_sf = (floor_wd_sf + (r5 < frac_wd_sf).to(tl.float32)).to(tl.int32)
        s_fast = s_fast - tick_wd_sf

        s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
        tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)
        tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8), mask=nk_mask)

    # s_fast is touched on every call (tick + optional wd_sf when
    # APPLY_CHASE). Write back unconditionally.
    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)


# ============================================================
# Python launchers for backward kernels
# ============================================================

def fused_grad_x(grad_y, s_slow, s_fast, row_exp, col_exp,
                 mantissa_bias=15,
                 v_slow=None, v_slow_factor=128,
                 block_m=64, block_k=64, block_n=32):
    """dx = grad_y @ W  with W reconstructed in registers. When v_slow
    is passed, the weight reconstruction adds v_slow * v_slow_factor."""
    M, N = grad_y.shape
    N2, K = s_slow.shape
    assert N == N2
    grad_x = torch.empty(M, K, dtype=torch.bfloat16, device=grad_y.device)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    grid = (triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    with PROFILER.time('linear_grad_x'):
      _fused_grad_x_kernel[grid](
        grad_y, s_slow, s_fast, row_exp, col_exp,
        grad_x, v_slow_ptr,
        M, N, K, int(mantissa_bias),
        grad_y.stride(0), grad_y.stride(1),
        s_slow.stride(0), s_slow.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_N=block_n,
        USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
    )
    return grad_x


# ============================================================
# FUSED CONV2D FORWARD: implicit-GEMM with bf16 weight recon inline.
# No im2col intermediate; input pixels gathered into register tiles
# alongside weight reconstruction.
# ============================================================

@triton.jit
def _fused_conv2d_forward_kernel(
    x_ptr,           # [B, C_in, H_in, W_in] bf16
    s_slow_ptr,      # [C_out, C_in*KH*KW] int16
    s_fast_ptr,      # [C_out, C_in*KH*KW] int16
    row_exp_ptr,     # [C_out] int32
    col_exp_ptr,     # [C_in*KH*KW] int32
    bias_ptr,        # [C_out] bf16
    v_slow_ptr,      # [C_out, C_in*KH*KW] int8 (gated by USE_V_SLOW)
    y_ptr,           # [B, C_out, H_out, W_out] bf16
    B, C_in, H_in, W_in, C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    mantissa_bias, has_bias: tl.constexpr,
    sx_b, sx_c, sx_h, sx_w,
    sy_b, sy_c, sy_h, sy_w,
    ss_n, ss_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * H_out * W_out
    K = C_in * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < C_out

    # Decompose m → (b, h_out, w_out)
    b_idx = offs_m // (H_out * W_out)
    hw_idx = offs_m % (H_out * W_out)
    h_out = hw_idx // W_out
    w_out = hw_idx % W_out

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Decompose k → (c_in, kh, kw)
        c_in_idx = offs_k // (KH * KW)
        khw_idx = offs_k % (KH * KW)
        kh_idx = khw_idx // KW
        kw_idx = khw_idx % KW

        # Input spatial position for each (m, k)
        h_in_idx = h_out[:, None] * STRIDE + kh_idx[None, :] - PADDING
        w_in_idx = w_out[:, None] * STRIDE + kw_idx[None, :] - PADDING
        valid = ((h_in_idx >= 0) & (h_in_idx < H_in)
                 & (w_in_idx >= 0) & (w_in_idx < W_in)
                 & m_mask[:, None] & k_mask[None, :])

        # Gather input
        x_off = (b_idx[:, None] * sx_b + c_in_idx[None, :] * sx_c
                 + h_in_idx * sx_h + w_in_idx * sx_w)
        x_tile = tl.load(x_ptr + x_off, mask=valid, other=0.0)

        # Load state tile for K chunk
        s_off = offs_n[:, None] * ss_n + offs_k[None, :] * ss_k
        s_mask = n_mask[:, None] & k_mask[None, :]
        m_eff = (tl.load(s_slow_ptr + s_off, mask=s_mask, other=0).to(tl.int32)
                 + tl.load(s_fast_ptr + s_off, mask=s_mask, other=0).to(tl.int32))
        if USE_V_SLOW:
            v_slow_tile = tl.load(v_slow_ptr + s_off,
                                  mask=s_mask, other=0).to(tl.int32)
            m_eff = m_eff + v_slow_tile * V_SLOW_FACTOR

        col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.float32)
        exp = row_e[:, None] + col_e[None, :] - mantissa_bias
        scale = tl.exp2(exp)
        w_tile = (m_eff.to(tl.float32) * scale).to(tl.bfloat16)

        acc += tl.dot(x_tile, tl.trans(w_tile))

    if has_bias:
        b_val = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += b_val[None, :]

    y_off = (b_idx[:, None] * sy_b + offs_n[None, :] * sy_c
             + h_out[:, None] * sy_h + w_out[:, None] * sy_w)
    tl.store(y_ptr + y_off, acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def fused_conv2d_forward(x, s_slow, s_fast, row_exp, col_exp, bias,
                         kh, kw, stride, padding,
                         mantissa_bias=15,
                         v_slow=None, v_slow_factor=128,
                         block_m=64, block_n=64, block_k=32):
    B, C_in, H_in, W_in = x.shape
    C_out = s_slow.shape[0]
    H_out = (H_in + 2 * padding - kh) // stride + 1
    W_out = (W_in + 2 * padding - kw) // stride + 1
    y = torch.empty(B, C_out, H_out, W_out, dtype=torch.bfloat16, device=x.device)
    has_bias = bias is not None
    bias_ptr = bias if has_bias else torch.empty(0, dtype=torch.bfloat16, device=x.device)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    M = B * H_out * W_out
    grid = (triton.cdiv(M, block_m), triton.cdiv(C_out, block_n))
    with PROFILER.time(f'conv_fwd_{C_in}x{C_out}'):
        _fused_conv2d_forward_kernel[grid](
            x, s_slow, s_fast, row_exp, col_exp, bias_ptr, v_slow_ptr, y,
            B, C_in, H_in, W_in, C_out, H_out, W_out,
            kh, kw, stride, padding,
            int(mantissa_bias), has_bias,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        )
    return y


# ============================================================
# FUSED CONV2D grad_x kernel: implicit-deconv with inline bf16 weight recon.
# ============================================================

@triton.jit
def _fused_conv2d_grad_x_kernel(
    grad_y_ptr,        # [B, C_out, H_out, W_out] bf16
    s_slow_ptr, s_fast_ptr,
    row_exp_ptr, col_exp_ptr,
    grad_x_ptr,        # [B, C_in, H_in, W_in] bf16
    v_slow_ptr,        # [C_out, C_in*KH*KW] int8 (gated by USE_V_SLOW)
    B, C_in, H_in, W_in, C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    mantissa_bias,
    sgy_b, sgy_c, sgy_h, sgy_w,
    sgx_b, sgx_c, sgx_h, sgx_w,
    ss_n, ss_k,
    BLOCK_M: tl.constexpr,  # over B*H_in*W_in
    BLOCK_N: tl.constexpr,  # over C_in
    BLOCK_K: tl.constexpr,  # over C_out (reduction)
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * H_in * W_in

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < C_in

    b_idx = offs_m // (H_in * W_in)
    hw_in = offs_m % (H_in * W_in)
    h_in = hw_in // W_in
    w_in = hw_in % W_in

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Single inner loop over K = C_out * KH * KW
    K = C_out * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        c_out_idx = offs_k // (KH * KW)
        khw_idx = offs_k % (KH * KW)
        kh_idx = khw_idx // KW
        kw_idx = khw_idx % KW

        # For each (m, k): determine input position
        # grad_x at (b, c_in, h_in, w_in) receives contributions from
        # grad_y at (b, c_out, h_out, w_out) where
        # h_out * stride + kh - padding = h_in → h_out = (h_in - kh + padding) / stride
        h_num = h_in[:, None] - kh_idx[None, :] + PADDING
        w_num = w_in[:, None] - kw_idx[None, :] + PADDING
        h_q = h_num // STRIDE
        w_q = w_num // STRIDE
        valid = (((h_num - h_q * STRIDE) == 0)
                 & ((w_num - w_q * STRIDE) == 0)
                 & (h_q >= 0) & (h_q < H_out)
                 & (w_q >= 0) & (w_q < W_out)
                 & m_mask[:, None] & k_mask[None, :])

        gy_off = (b_idx[:, None] * sgy_b + c_out_idx[None, :] * sgy_c
                  + h_q * sgy_h + w_q * sgy_w)
        gy_tile = tl.load(grad_y_ptr + gy_off, mask=valid, other=0.0)

        # Weight load: W_2d[c_out_idx[k], offs_n[n] * KH*KW + kh_idx[k]*KW + kw_idx[k]]
        K_w_idx = (offs_n[:, None] * (KH * KW)
                   + kh_idx[None, :] * KW + kw_idx[None, :])  # [BLOCK_N, BLOCK_K]
        w_off = c_out_idx[None, :] * ss_n + K_w_idx * ss_k
        w_mask = n_mask[:, None] & k_mask[None, :]
        m_eff = (tl.load(s_slow_ptr + w_off, mask=w_mask, other=0).to(tl.int32)
                 + tl.load(s_fast_ptr + w_off, mask=w_mask, other=0).to(tl.int32))
        if USE_V_SLOW:
            v_slow_tile = tl.load(v_slow_ptr + w_off,
                                  mask=w_mask, other=0).to(tl.int32)
            m_eff = m_eff + v_slow_tile * V_SLOW_FACTOR

        row_e = tl.load(row_exp_ptr + c_out_idx, mask=k_mask, other=0).to(tl.float32)
        col_e = tl.load(col_exp_ptr + K_w_idx, mask=w_mask, other=0).to(tl.float32)
        exp = col_e + row_e[None, :] - mantissa_bias
        scale = tl.exp2(exp)
        w_tile = (m_eff.to(tl.float32) * scale).to(tl.bfloat16)

        acc += tl.dot(gy_tile, tl.trans(w_tile))

    gx_off = (b_idx[:, None] * sgx_b + offs_n[None, :] * sgx_c
              + h_in[:, None] * sgx_h + w_in[:, None] * sgx_w)
    tl.store(grad_x_ptr + gx_off, acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def fused_conv2d_grad_x(grad_y, s_slow, s_fast, row_exp, col_exp,
                        in_channels, out_channels, H_in, W_in,
                        kh, kw, stride, padding,
                        mantissa_bias=15,
                        v_slow=None, v_slow_factor=128,
                        block_m=64, block_n=32, block_k=32):
    B = grad_y.shape[0]
    H_out, W_out = grad_y.shape[-2:]
    grad_x = torch.empty(B, in_channels, H_in, W_in,
                         dtype=torch.bfloat16, device=grad_y.device)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    M = B * H_in * W_in
    grid = (triton.cdiv(M, block_m), triton.cdiv(in_channels, block_n))
    with PROFILER.time(f'conv_gx_{in_channels}x{out_channels}'):
        _fused_conv2d_grad_x_kernel[grid](
            grad_y, s_slow, s_fast, row_exp, col_exp, grad_x, v_slow_ptr,
            B, in_channels, H_in, W_in, out_channels, H_out, W_out,
            kh, kw, stride, padding, int(mantissa_bias),
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2), grad_x.stride(3),
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        )
    return grad_x


# ============================================================
# FUSED CONV2D grad_W + update kernel: accumulate grad_W per (C_out, K_w)
# tile, apply concord momentum update in place to s_slow/s_fast.
#
# DEPRECATED: no longer called. FusedConcordConv2d.backward routes all
# conv2d backward through cuDNN's conv2d_weight + apply_update_from_grad_W
# (see the comment in that backward for why). Kept here for reference
# only -- removing it would shave a few hundred lines of dead code but
# the module is lazy-compiled, so the dead kernel costs nothing at
# runtime as long as nothing calls fused_conv2d_grad_W_and_update.
# ============================================================

@triton.autotune(
    # Configs pruned for Windows ptxas (CUDA 12.8) stability. The
    # original set included three large-block variants that produced
    # heavily-spilling kernels (255 registers, 700+ byte spill frame) --
    # those tipped ptxas into STATUS_STACK_BUFFER_OVERRUN (0xC0000409)
    # on Windows under concurrent autotune compilation. They were also
    # slow at runtime (register spills hurt throughput), so dropping
    # them is a stability fix AND a perf fix. Pruned: BLOCK_BHW=1024
    # with BLOCK_N=16; BLOCK_BHW=1024 with BLOCK_N=32; BLOCK_N=64/K=64.
    # Kept: 6 small-to-medium configs that span the useful tile range
    # for SDXL conv2d shapes (in_ch 4..1280, out_ch 320..1280,
    # H_out 8..128).
    configs=[
        triton.Config({'BLOCK_N': 16, 'BLOCK_K': 16, 'BLOCK_BHW': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_K': 16, 'BLOCK_BHW': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_BHW': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_BHW': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_BHW': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32, 'BLOCK_BHW': 128}, num_warps=4),
    ],
    key=['C_out', 'C_in', 'H_out', 'W_out'],
    restore_value=['s_slow_ptr', 's_fast_ptr', 'v_slow_ptr'],
)
@triton.jit
def _fused_conv2d_grad_W_and_update_kernel(
    grad_y_ptr,        # [B, C_out, H_out, W_out] bf16
    x_ptr,             # [B, C_in, H_in, W_in] bf16
    s_slow_ptr, s_fast_ptr,
    row_exp_ptr, col_exp_ptr,
    v_slow_ptr,        # [C_out, C_in*KH*KW] int8 (gated by USE_V_SLOW)
    B, C_in, H_in, W_in, C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    lr, mantissa_bias, alpha, beta1, step_salt_ptr,
    alpha_v_fast,      # fp32: per-step v_slow ← s_fast leak rate
    sgy_b, sgy_c, sgy_h, sgy_w,
    sx_b, sx_c, sx_h, sx_w,
    ss_n, ss_k,
    BLOCK_N: tl.constexpr,  # over C_out
    BLOCK_K: tl.constexpr,  # over C_in*KH*KW
    BLOCK_BHW: tl.constexpr,  # inner reduction chunk over B*H_out*W_out
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    # See _fused_grad_W_and_update_kernel docstring for APPLY_CHASE.
    APPLY_CHASE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Per-step salt loaded from tensor counter — Dynamo-safe under HOP.
    step_salt = tl.load(step_salt_ptr).to(tl.int32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < C_out
    k_mask = offs_k < (C_in * KH * KW)
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # k → (c_in, kh, kw)
    c_in_idx = offs_k // (KH * KW)
    khw_idx = offs_k % (KH * KW)
    kh_idx = khw_idx // KW
    kw_idx = khw_idx % KW

    M = B * H_out * W_out
    grad_W = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_BHW):
        offs_m = m_start + tl.arange(0, BLOCK_BHW)
        m_mask = offs_m < M
        b_idx = offs_m // (H_out * W_out)
        hw_out = offs_m % (H_out * W_out)
        h_out = hw_out // W_out
        w_out = hw_out % W_out

        # grad_y[b, c_out, h_out, w_out] — tile [BLOCK_BHW, BLOCK_N]
        gy_off = (b_idx[:, None] * sgy_b + offs_n[None, :] * sgy_c
                  + h_out[:, None] * sgy_h + w_out[:, None] * sgy_w)
        gy = tl.load(grad_y_ptr + gy_off,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)

        # x[b, c_in, h_out*stride + kh - pad, w_out*stride + kw - pad]
        # tile [BLOCK_BHW, BLOCK_K]
        h_in_idx = h_out[:, None] * STRIDE + kh_idx[None, :] - PADDING
        w_in_idx = w_out[:, None] * STRIDE + kw_idx[None, :] - PADDING
        spatial_valid = ((h_in_idx >= 0) & (h_in_idx < H_in)
                         & (w_in_idx >= 0) & (w_in_idx < W_in)
                         & m_mask[:, None] & k_mask[None, :])
        x_off = (b_idx[:, None] * sx_b + c_in_idx[None, :] * sx_c
                 + h_in_idx * sx_h + w_in_idx * sx_w)
        x_tile = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)

        # grad_W += gy.T @ x_tile: [BLOCK_N, BLOCK_K]
        grad_W += tl.dot(tl.trans(gy), x_tile)

    # === Concord momentum update in place ===
    s_off = offs_n[:, None] * ss_n + offs_k[None, :] * ss_k
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.float32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.float32)
    inv_exp = mantissa_bias - row_e[:, None] - col_e[None, :]
    scale_inv = tl.exp2(inv_exp)
    delta_grad = -lr * grad_W * scale_inv

    v_prev = (s_fast - s_slow).to(tl.float32)
    delta_t = delta_grad - beta1 * v_prev

    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Chase + leak gated on APPLY_CHASE — see linear kernel.
    if APPLY_CHASE:
        gap = (s_fast - s_slow).to(tl.float32)
        delta_slow_f = alpha * gap
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(delta_slow_f)
        frac_s = delta_slow_f - floor_s
        tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow = s_slow + tick_slow

        # v_slow ← s_fast leak (per-step). Same as the linear path: tick
        # quantised to int8 (one v_slow unit = V_SLOW_FACTOR mantissa
        # units). MASS_PRESERVE=True subtracts the actual tick from
        # s_fast so the live weight is unchanged by the leak; False
        # makes it a second chase.
        if USE_V_SLOW:
            v_slow_old = tl.load(v_slow_ptr + s_off,
                                 mask=nk_mask, other=0).to(tl.int32)
            gap_v_full = (s_fast - v_slow_old * V_SLOW_FACTOR).to(tl.float32)
            delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
            r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
            floor_v = tl.floor(delta_v8)
            frac_v = delta_v8 - floor_v
            tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
            new_v_int32 = v_slow_old + tick_v8
            new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
            if MASS_PRESERVE:
                actual_tick_v8 = new_v_int8 - v_slow_old
                actual_tick_full = actual_tick_v8 * V_SLOW_FACTOR
                s_fast = s_fast - actual_tick_full
            tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8),
                       mask=nk_mask)

        s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
        tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)

    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)


def fused_conv2d_grad_W_and_update(grad_y, x, s_slow, s_fast, row_exp, col_exp,
                                   in_channels, out_channels,
                                   kh, kw, stride, padding,
                                   lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                                   v_slow=None, v_slow_factor=128,
                                   alpha_v_fast=0.001,
                                   apply_chase=True):
    B = grad_y.shape[0]
    H_out, W_out = grad_y.shape[-2:]
    H_in, W_in = x.shape[-2:]
    K = in_channels * kh * kw
    step_counter = _get_step_counter(grad_y.device)
    step_counter.add_(1)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    if use_v_slow:
        assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8
    # BLOCK_N / BLOCK_K / BLOCK_BHW are chosen by @triton.autotune on the
    # kernel; the grid reads them back from the winning config via META.
    grid = lambda META: (triton.cdiv(out_channels, META['BLOCK_N']),
                         triton.cdiv(K, META['BLOCK_K']))
    with PROFILER.time(f'conv_gW_{in_channels}x{out_channels}'):
      _fused_conv2d_grad_W_and_update_kernel[grid](
        grad_y, x, s_slow, s_fast, row_exp, col_exp, v_slow_ptr,
        B, in_channels, H_in, W_in, out_channels, H_out, W_out,
        kh, kw, stride, padding,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        step_counter,
        float(alpha_v_fast),
        grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        s_slow.stride(0), s_slow.stride(1),
        USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        MASS_PRESERVE=False,
        APPLY_CHASE=bool(apply_chase),
    )


def materialize_bf16_weight(s_slow, s_fast, row_exp, col_exp,
                              mantissa_bias=15,
                              v_slow=None, v_slow_factor=128,
                              out=None):
    """Reconstruct the live bf16 weight from concord int state. If
    ``out`` is given (must match s_slow.shape, bf16), the recon writes
    in place and avoids the per-step torch.empty(). The reused buffer
    is safe because forward and backward of the same step run between
    consecutive forward calls — no other code reads the buffer during
    the window. (Gradient checkpointing recomputes forward; if you mix
    that with weight reuse, allocate per-checkpoint instead.)"""
    from concord_triton import sync_weight_bf16_pair_triton
    if out is None:
        weight = torch.empty(s_slow.shape, dtype=torch.bfloat16,
                              device=s_slow.device)
    else:
        weight = out
    sync_weight_bf16_pair_triton(weight, s_slow, s_fast, row_exp, col_exp,
                                  mantissa_bias=mantissa_bias,
                                  v_slow=v_slow, v_slow_factor=v_slow_factor)
    return weight


# Module-level switch for mass-preserving chase. When True, every
# Concord layer's apply-update kernel subtracts the s_slow chase tick
# from s_fast, making s_fast a true delta (bounded magnitude). Used
# for the CIFAR validation of the mass-preserving-chase dynamics
# before committing to int8 s_fast storage. Test sets this to True
# before training; production runs leave it False (current behavior).
_GLOBAL_MASS_PRESERVE_CHASE = False


def set_mass_preserve_chase(enabled: bool):
    """Toggle mass-preserving chase globally. Affects all Concord
    layers' subsequent backward passes. Used by the CIFAR validation
    script; not exposed in the per-layer or trainer API yet."""
    global _GLOBAL_MASS_PRESERVE_CHASE
    _GLOBAL_MASS_PRESERVE_CHASE = bool(enabled)


_step_counter = [0]   # mutable counter so step_salt changes each call
                       # (legacy Python-int path; kept for kernels that
                       # still expect an int step_salt scalar)


# Tensor-backed step counter — Dynamo-safe AND per-step varying.
#
# History: under OneTrainer's gradient checkpointing, `_step_counter[0] +=
# 1` raised inside the HOP ("mutating a variable not in the current
# scope"), and the original patch swapped it for a static int salt. A
# static salt makes SR rounding deterministic per (element, fractional)
# pair across steps, so the per-step rounding errors stop averaging out
# — quantization bias accumulates instead of cancelling.
#
# The fix below: keep the counter as a 1-element int32 GPU tensor (one
# per device). Launcher does `counter.add_(1)` (Dynamo-traceable), and
# passes the tensor; the kernel does `tl.load(counter_ptr)`. No CPU sync,
# no module-level mutation, decorrelated per step.
_step_counter_tensors: dict = {}


def _get_step_counter(device):
    """Return the 1-element int32 GPU counter tensor for ``device``,
    creating it lazily on first use. The tensor is bumped in place by
    each launcher before kernel dispatch."""
    key = (device.type, device.index)
    t = _step_counter_tensors.get(key)
    if t is None:
        import torch as _torch
        t = _torch.zeros(1, dtype=_torch.int32, device=device)
        _step_counter_tensors[key] = t
    return t


def fused_grad_W_and_update(grad_y, x, s_slow, s_fast, row_exp, col_exp,
                            lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                            v_slow=None, v_slow_factor=128,
                            alpha_v_fast=0.001,
                            apply_chase=True,
                            block_n=64, block_k=64, block_m=32):
    """Compute grad_W = grad_y.T @ x and apply the concord momentum update
    in place to s_slow, s_fast.

    Optional three-accumulator (SGD path): if ``v_slow`` is passed
    (int8, same shape as s_slow), the kernel also runs a v_slow ←
    s_fast leak after the chase. Each int8 unit of v_slow represents
    ``v_slow_factor`` mantissa units; v_slow adds additively to the
    live weight. The leak is non-mass-preserving (a "second chase":
    the live weight grows by what v_slow gained), mirroring the
    s_fast/s_slow chase semantics. The earlier mass-preserving
    variant was ablated as below-baseline."""
    M, N = grad_y.shape
    M2, K = x.shape
    assert M == M2
    # Tensor-backed step counter: in-place add is Dynamo-traceable under
    # HOP gradient checkpointing, and the kernel reads via pointer so we
    # never CPU-sync. The wrap mask ensures the counter stays within
    # int32 positive range (avoids sign weirdness in the XOR).
    step_counter = _get_step_counter(grad_y.device)
    step_counter.add_(1)
    use_v_slow = v_slow is not None
    # When v_slow is unused, pass s_slow as a placeholder pointer; the
    # USE_V_SLOW constexpr gates the actual load/store.
    v_slow_ptr = v_slow if use_v_slow else s_slow
    if use_v_slow:
        assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8
    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    with PROFILER.time('linear_gW_update'):
      _fused_grad_W_and_update_kernel[grid](
        grad_y, x, s_slow, s_fast, row_exp, col_exp, v_slow_ptr,
        M, N, K,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        0.0,  # amplify_aligned (dead path, kept zero)
        float(alpha_v_fast),
        step_counter,
        grad_y.stride(0), grad_y.stride(1),
        x.stride(0), x.stride(1),
        s_slow.stride(0), s_slow.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
        USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        MASS_PRESERVE=False,
        APPLY_CHASE=bool(apply_chase),
    )


def fused_grad_W_and_adamw_v_rank1_update(
    grad_y, x, s_slow, s_fast, row_exp, col_exp,
    v_row, v_col, g2_row, g2_col,
    lr, beta2, v_step,
    mantissa_bias=15, alpha=0.1, beta1=0.0,
    weight_decay=0.0, eps=1e-8, step_cap=10.0,
    v_slow=None, v_slow_factor=128,
    alpha_v_fast=0.001, mass_preserve=True,
    block_n=64, block_k=64, block_m=32,
):
    """Adafactor-style rank-1 v AdamW step. State per layer:
        v_row [N] fp32 — EMA of mean_k(g²[n,:])
        v_col [K] fp32 — EMA of mean_n(g²[:,k])
        g2_row, g2_col — scratch fp32 buffers, zeroed each step.

    Returns nothing; mutates s_slow, s_fast, v_row, v_col, g2_row, g2_col.
    The caller is responsible for: (a) zeroing g2_row/g2_col before the
    call, (b) closing the EMA after the call:
        v_row = beta2 * v_row + g2_row
        v_col = beta2 * v_col + g2_col
    """
    M, N = grad_y.shape
    M2, K = x.shape
    assert M == M2
    assert v_row.shape == (N,) and v_row.dtype == torch.float32
    assert v_col.shape == (K,) and v_col.dtype == torch.float32
    assert g2_row.shape == (N,) and g2_row.dtype == torch.float32
    assert g2_col.shape == (K,) and g2_col.dtype == torch.float32

    step_counter = _get_step_counter(grad_y.device)
    step_counter.add_(1)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    if use_v_slow:
        assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8

    # mean_v_row_inv = 1 / clamp(mean(v_row), eps). Compute on GPU so we
    # don't pay a `.item()` sync per backward call. The kernel reads it
    # from a 1-element fp32 device buffer.
    mean_v_row_inv_t = v_row.mean().clamp_min(eps).reciprocal()
    # Bias correction stays CPU-side (depends only on integer v_step).
    bc = 1.0 - (beta2 ** max(v_step, 1))
    bias_corr_inv = 1.0 / max(bc, 1e-12)
    inv_K = 1.0 / K
    inv_N = 1.0 / N
    one_minus_beta2 = 1.0 - beta2

    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    with PROFILER.time('linear_gW_adamw_v_rank1_update'):
        _fused_grad_W_and_adamw_v_rank1_update_kernel[grid](
            grad_y, x, s_slow, s_fast, row_exp, col_exp,
            v_row, v_col, g2_row, g2_col, v_slow_ptr,
            M, N, K,
            float(lr), int(mantissa_bias), float(alpha), float(beta1),
            float(weight_decay), float(eps), float(step_cap),
            mean_v_row_inv_t, float(bias_corr_inv),
            float(inv_K), float(inv_N), float(one_minus_beta2),
            float(alpha_v_fast),
            step_counter,
            grad_y.stride(0), grad_y.stride(1),
            x.stride(0), x.stride(1),
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
            USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
            MASS_PRESERVE=bool(mass_preserve),
        )


def fused_grad_W_and_adamw_v_from_velocity_update(
    grad_y, x, s_slow, s_fast, row_exp, col_exp,
    lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
    weight_decay=0.0, eps=1e-8, step_cap=10.0, v_scale=1.0,
    block_n=64, block_k=64, block_m=32,
):
    """AdamW step using (s_fast - s_slow) as the variance source. No v_row,
    no v_col, no g2 scratch, no EMA close-out. The chase dynamic already
    accumulates per-step gradient updates with effective time constant
    rebalance_every / alpha (~80 steps at defaults); its squared magnitude
    in weight-space is the AdamW second moment, scaled.

    ``v_scale`` absorbs the lr^2 * T_eff constant from the derivation;
    treat it as a temperature on the preconditioner (default 1.0).

    Mutates s_slow, s_fast in place. No other state.
    """
    M, N = grad_y.shape
    M2, K = x.shape
    assert M == M2

    _step_counter[0] = (_step_counter[0] + 1) & 0x7FFFFFFF
    step_salt = _step_counter[0]

    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    with PROFILER.time('linear_gW_adamw_v_from_velocity'):
        _fused_grad_W_and_adamw_v_from_velocity_update_kernel[grid](
            grad_y, x, s_slow, s_fast, row_exp, col_exp,
            M, N, K,
            float(lr), int(mantissa_bias), float(alpha), float(beta1),
            float(weight_decay), float(eps), float(step_cap),
            float(v_scale),
            int(step_salt),
            grad_y.stride(0), grad_y.stride(1),
            x.stride(0), x.stride(1),
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
        )


def fused_grad_W_and_adamw_three_accum_update(
    grad_y, x, s_slow, s_fast, v_slow, row_exp, col_exp,
    lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
    weight_decay=0.0, eps=1.0, step_cap=10.0,
    v_scale=1.0, drift_cancel_C=0.1, alpha_v_fast=0.001,
    v_slow_factor=128, mass_preserve=False,
    wd_sv=0.0, wd_sf=0.0,
    apply_chase=True,
    block_n=64, block_k=64, block_m=32,
):
    """Three-accumulator AdamW: s_fast / s_slow / v_slow_i8 (int8).

    The high-pass residual
        noise = (s_fast - s_slow)
                - drift_cancel_C * (s_slow - v_slow_i8 * v_slow_factor)
    rejects the secular drift component, leaving the per-element noise
    floor as the AdamW second-moment estimator. ``v_scale`` absorbs the
    lr² · T_eff constant from the derivation; treat as preconditioner
    temperature (default 1.0).

    v_slow_i8 is at shifted scale (each unit = factor mantissa units),
    same convention as the SGD chase path — so one buffer serves both
    roles. v_slow_i8 ALSO contributes additively to the live weight
    (consistent with the SGD path), so weight_decay shrinks every
    accumulator. ``mass_preserve`` toggles the s_fast offset on each
    leak tick (False default mirrors the existing non-mass-preserving
    chase behaviour the CIFAR headline depends on).

    v_slow_i8 receives two leaks: this kernel (per-step toward s_fast at
    alpha_v_fast), and ConcordLinearFused.rebalance (per-rebalance
    toward s_slow at alpha_v_slow). The int8-aware defensive rebalance
    (_v_slow_i8_rebalance) catches saturation.

    Default eps=1.0 is much larger than standard 1e-8 — the noise
    residual lives in mantissa units where O(1) is the natural floor.
    """
    M, N = grad_y.shape
    M2, K = x.shape
    assert M == M2
    assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8

    step_counter = _get_step_counter(grad_y.device)
    step_counter.add_(1)

    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    with PROFILER.time('linear_gW_adamw_three_accum'):
        _fused_grad_W_and_adamw_three_accum_update_kernel[grid](
            grad_y, x, s_slow, s_fast, v_slow, row_exp, col_exp,
            M, N, K,
            float(lr), int(mantissa_bias), float(alpha), float(beta1),
            float(weight_decay), float(eps), float(step_cap),
            float(v_scale), float(drift_cancel_C), float(alpha_v_fast),
            float(wd_sv), float(wd_sf),
            step_counter,
            grad_y.stride(0), grad_y.stride(1),
            x.stride(0), x.stride(1),
            s_slow.stride(0), s_slow.stride(1),
            BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
            V_SLOW_FACTOR=int(v_slow_factor),
            MASS_PRESERVE=bool(mass_preserve),
            APPLY_CHASE=bool(apply_chase),
        )


def fused_grad_W_and_adamw_update(grad_y, x, s_slow, s_fast, row_exp, col_exp,
                                  discount_row, discount_col,
                                  lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                                  weight_decay=0.0, eps=1e-8, step_cap=10.0,
                                  block_n=64, block_k=64, block_m=32):
    """Single-kernel AdamW step using the per-element |W|-via-CLZ proxy
    as the variance estimate. No v_row / v_col tracking; the per-element
    variance signal lives in the weight's bit pattern itself.

    The β2 / temporal-discount role is filled by a rank-1 factored
    discount: per-row scalar discount_row[i] times per-col scalar
    discount_col[j], applied in the kernel as discount_ij. Both vectors
    are caller-updated CPU-side at refit cadence from the cascade's
    per-row / per-col |W| trajectory (see ConcordLinearFused.
    update_discount_from_cascade). All-ones reproduces pure LARS-style.

    State (caller-allocated, modified in place):
        s_slow [N, K] int16  — slow mantissa
        s_fast [N, K] int16  — fast mantissa
        row_exp [N] int8     — per-row exponent (storage scaling)
        col_exp [K] int8     — per-col exponent
        discount_row [N] fp32 — per-row discount factor (rank-1 part)
        discount_col [K] fp32 — per-col discount factor

    Total persistent state: 32 bits/param + O(M+N) for the exponents
    + O(M+N) fp32 for the discount factors. All amortized ≈ 0/param.

    The momentum-equivalent first moment is implicit in (s_fast - s_slow);
    β1 is the chase-feedback coefficient (default 0 → implicit-m via the
    chase α=0.1 only).
    """
    M, N = grad_y.shape
    M2, K = x.shape
    assert M == M2
    assert discount_row.shape == (N,) and discount_row.dtype == torch.float32
    assert discount_col.shape == (K,) and discount_col.dtype == torch.float32

    _step_counter[0] = (_step_counter[0] + 1) & 0x7FFFFFFF
    step_salt = _step_counter[0]

    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    with PROFILER.time('linear_gW_adamw_update'):
      _fused_grad_W_and_adamw_update_kernel[grid](
        grad_y, x, s_slow, s_fast, row_exp, col_exp,
        discount_row, discount_col,
        M, N, K,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        float(weight_decay), float(eps), float(step_cap),
        int(step_salt),
        grad_y.stride(0), grad_y.stride(1),
        x.stride(0), x.stride(1),
        s_slow.stride(0), s_slow.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_M=block_m,
    )


# ============================================================
# autograd.Function — wires the kernels into PyTorch
# ============================================================

# ============================================================
# UPDATE-ONLY kernel: take an externally-computed grad_W (e.g., from cuDNN
# conv backward) and apply the concord momentum update in place.
# ============================================================

@triton.jit
def _apply_update_kernel(
    grad_W_ptr,      # [N, K] bf16
    s_slow_ptr,
    s_fast_ptr,
    row_exp_ptr, col_exp_ptr,
    row_max_ptr,     # [N] int32, pre-zeroed
    col_max_ptr,     # [K] int32, pre-zeroed
    v_slow_ptr,      # [N, K] int8 (gated by USE_V_SLOW)
    N, K,
    lr, mantissa_bias, alpha, beta1, step_salt_ptr,
    alpha_v_fast,
    stride_gn, stride_gk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    # See _fused_grad_W_and_update_kernel docstring for APPLY_CHASE.
    APPLY_CHASE: tl.constexpr,
    # MASS_PRESERVE_CHASE: when True, the s_slow chase tick is ALSO
    # subtracted from s_fast (whatever mass flowed into s_slow exits
    # s_fast). Makes s_fast a true delta accumulator: it oscillates
    # near (s_fast - s_slow) ~ delta_grad / alpha = O(10) rather than
    # accumulating into the full weight magnitude. Required for
    # eventual int8 s_fast storage. Independent of MASS_PRESERVE
    # (which conserves between s_fast <-> v_slow_i8).
    MASS_PRESERVE_CHASE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    # Per-step salt loaded from tensor counter — Dynamo-safe under HOP.
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)

    gW_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + gW_off, mask=nk_mask, other=0.0).to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.float32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.float32)
    inv_exp = mantissa_bias - row_e[:, None] - col_e[None, :]
    scale_inv = tl.exp2(inv_exp)
    delta_grad = -lr * grad_W * scale_inv

    v_prev = (s_fast - s_slow).to(tl.float32)
    delta_t = delta_grad - beta1 * v_prev

    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Chase + leak gated on APPLY_CHASE.
    if APPLY_CHASE:
        gap = (s_fast - s_slow).to(tl.float32)
        delta_slow_f = alpha * gap
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(delta_slow_f)
        frac_s = delta_slow_f - floor_s
        tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow = s_slow + tick_slow
        if MASS_PRESERVE_CHASE:
            # Mass-preserving chase: what flowed into s_slow exits
            # s_fast, so live weight (s_slow + s_fast) is unchanged
            # by the chase step itself. s_fast becomes a true delta.
            s_fast = s_fast - tick_slow

        # v_slow leak (mirrors linear / conv-fused paths).
        if USE_V_SLOW:
            v_slow_old = tl.load(v_slow_ptr + s_off,
                                 mask=nk_mask, other=0).to(tl.int32)
            gap_v_full = (s_fast - v_slow_old * V_SLOW_FACTOR).to(tl.float32)
            delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
            r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
            floor_v = tl.floor(delta_v8)
            frac_v = delta_v8 - floor_v
            tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
            new_v_int32 = v_slow_old + tick_v8
            new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
            if MASS_PRESERVE:
                actual_tick_v8 = new_v_int8 - v_slow_old
                actual_tick_full = actual_tick_v8 * V_SLOW_FACTOR
                s_fast = s_fast - actual_tick_full
            tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8),
                       mask=nk_mask)

        s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
        tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)

    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)

    # Overflow tracking: while the new state is in registers, compute its
    # per-tile max and atomic-max into the global per-row/per-col buffers.
    # Comes ~free since the data is already in registers; replaces the
    # separate "reduce" pass of the old rebalance.
    abs_eff = tl.abs(s_slow + s_fast)
    abs_eff = tl.where(nk_mask, abs_eff, 0)
    tile_row_max = tl.max(abs_eff, axis=1)
    tile_col_max = tl.max(abs_eff, axis=0)
    tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask)
    tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask)


def apply_update_from_grad_W(grad_W, s_slow, s_fast, row_exp, col_exp,
                             row_max, col_max,
                             lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                             v_slow=None, v_slow_factor=128,
                             alpha_v_fast=0.001,
                             apply_chase=True,
                             mass_preserve_chase=False,
                             block_n=64, block_k=64):
    """Apply the concord momentum update given a precomputed grad_W tensor.
    Modifies s_slow, s_fast in place. Also populates row_max, col_max
    (pre-zeroed by caller) via atomic max — used by the immediate
    overflow-correction pass that follows."""
    N, K = grad_W.shape
    step_counter = _get_step_counter(grad_W.device)
    step_counter.add_(1)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow
    if use_v_slow:
        assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8
    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    _apply_update_kernel[grid](
        grad_W, s_slow, s_fast, row_exp, col_exp,
        row_max, col_max, v_slow_ptr,
        N, K,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        step_counter,
        float(alpha_v_fast),
        grad_W.stride(0), grad_W.stride(1),
        s_slow.stride(0), s_slow.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k,
        USE_V_SLOW=use_v_slow, V_SLOW_FACTOR=int(v_slow_factor),
        MASS_PRESERVE=False,
        APPLY_CHASE=bool(apply_chase),
        MASS_PRESERVE_CHASE=bool(mass_preserve_chase),
    )


# ============================================================
# AdamW three-accumulator apply-only kernel: takes a precomputed grad_W
# tensor (from cuBLAS matmul) and runs the SAME drift-cancel + AdamW
# update logic as _fused_grad_W_and_adamw_three_accum_update_kernel but
# WITHOUT the inline M-reduction matmul. Splitting matmul-from-update
# avoids the fused-mega-kernel's register pressure and the resulting
# silent hang on big SDXL FFN shapes (in/out up to 10240).
# ============================================================

@triton.jit
def _apply_adamw_three_accum_kernel(
    grad_W_ptr,      # [N, K] bf16 -- precomputed by cuBLAS matmul
    s_slow_ptr,      # [N, K] int16, mutated
    s_fast_ptr,      # [N, K] int16, mutated
    v_slow_ptr,      # [N, K] int8, mutated
    row_exp_ptr,     # [N] int8
    col_exp_ptr,     # [K] int8
    row_max_ptr,     # [N] int32, pre-zeroed (atomic-max sink)
    col_max_ptr,     # [K] int32, pre-zeroed (atomic-max sink)
    N, K,
    lr, mantissa_bias, alpha, beta1,
    weight_decay, eps, step_cap,
    v_scale, drift_cancel_C, alpha_v_fast,
    wd_sv, wd_sf,
    step_salt_ptr,
    stride_gn, stride_gk,
    stride_sn, stride_sk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    MASS_PRESERVE: tl.constexpr,
    APPLY_CHASE: tl.constexpr,
    # See _apply_update_kernel for MASS_PRESERVE_CHASE semantics.
    MASS_PRESERVE_CHASE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # Load precomputed grad_W from HBM.
    gW_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + gW_off, mask=nk_mask, other=0.0).to(tl.float32)

    # Load state + scale. v_slow_i8 at shifted scale V_SLOW_FACTOR.
    s_off = offs_n[:, None] * stride_sn + offs_k[None, :] * stride_sk
    s_slow = tl.load(s_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    v_slow_i8 = tl.load(v_slow_ptr + s_off, mask=nk_mask, other=0).to(tl.int32)
    v_slow_full = v_slow_i8 * V_SLOW_FACTOR
    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)

    m_eff = s_slow + s_fast + v_slow_full
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_inv = tl.exp2(-total_exp)
    scale_fwd = tl.exp2(total_exp)
    current_weight = m_eff.to(tl.float32) * scale_fwd

    # Drift-cancelled noise residual.
    d_fs = (s_fast - s_slow).to(tl.float32)
    d_sv = (s_slow - v_slow_full).to(tl.float32)
    noise = d_fs - drift_cancel_C * d_sv
    noise_in_w = noise * scale_fwd
    v_proxy = noise_in_w * noise_in_w * v_scale

    # AdamW step.
    step_live = grad_W / tl.sqrt(v_proxy + eps) + weight_decay * current_weight
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv
    delta_t = delta_grad - beta1 * d_fs

    # SR tick s_fast.
    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    if APPLY_CHASE:
        # Chase s_slow toward s_fast.
        gap = (s_fast - s_slow).to(tl.float32)
        delta_slow_f = alpha * gap
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(delta_slow_f)
        frac_s = delta_slow_f - floor_s
        tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow = s_slow + tick_slow
        if MASS_PRESERVE_CHASE:
            s_fast = s_fast - tick_slow

        # v_slow_i8 leak toward s_fast (int8 SR rounding at shifted scale).
        gap_v_full = (s_fast - v_slow_full).to(tl.float32)
        delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
        r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
        floor_v = tl.floor(delta_v8)
        frac_v = delta_v8 - floor_v
        tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
        new_v_int32 = v_slow_i8 + tick_v8
        new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
        if MASS_PRESERVE:
            actual_tick_v8 = new_v_int8 - v_slow_i8
            actual_tick_full = actual_tick_v8 * V_SLOW_FACTOR
            s_fast = s_fast - actual_tick_full

        # Bayesian-anchored weight decay: pull s_slow / s_fast toward
        # the (post-tick) v_slow_full anchor.
        v_slow_full_post = new_v_int8 * V_SLOW_FACTOR
        d_sv_full_post = (s_slow - v_slow_full_post).to(tl.float32)
        wd_sv_delta = lr * wd_sv * d_sv_full_post
        r4 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x66665555)
        floor_wd_sv = tl.floor(wd_sv_delta)
        frac_wd_sv = wd_sv_delta - floor_wd_sv
        tick_wd_sv = (floor_wd_sv + (r4 < frac_wd_sv).to(tl.float32)).to(tl.int32)
        s_slow = s_slow - tick_wd_sv

        d_sf_full_post = (s_fast - v_slow_full_post).to(tl.float32)
        wd_sf_delta = lr * wd_sf * d_sf_full_post
        r5 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x77770000)
        floor_wd_sf = tl.floor(wd_sf_delta)
        frac_wd_sf = wd_sf_delta - floor_wd_sf
        tick_wd_sf = (floor_wd_sf + (r5 < frac_wd_sf).to(tl.float32)).to(tl.int32)
        s_fast = s_fast - tick_wd_sf

        s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
        tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=nk_mask)
        tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8), mask=nk_mask)

    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=nk_mask)

    # Per-tile atomic-max into row_max / col_max for the rebalance pass.
    abs_eff = tl.abs(s_slow + s_fast)
    abs_eff = tl.where(nk_mask, abs_eff, 0)
    tile_row_max = tl.max(abs_eff, axis=1)
    tile_col_max = tl.max(abs_eff, axis=0)
    tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask=n_mask)
    tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask=k_mask)


def apply_adamw_three_accum_from_grad_W(
    grad_W, s_slow, s_fast, v_slow, row_exp, col_exp,
    row_max, col_max,
    lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
    weight_decay=0.0, eps=1.0, step_cap=10.0,
    v_scale=1.0, drift_cancel_C=0.1, alpha_v_fast=0.001,
    v_slow_factor=128, mass_preserve=False,
    wd_sv=0.0, wd_sf=0.0,
    apply_chase=True,
    mass_preserve_chase=False,
    block_n=64, block_k=64,
):
    """Three-accumulator AdamW state update given a precomputed grad_W.
    Apply-only sibling of fused_grad_W_and_adamw_three_accum_update --
    the fused version's M-reduction matmul has been replaced by a cuBLAS
    matmul on the caller side, removing the register-pressure trigger
    that caused silent hangs on big SDXL FFN shapes."""
    N, K = grad_W.shape
    assert v_slow.shape == s_slow.shape and v_slow.dtype == torch.int8
    step_counter = _get_step_counter(grad_W.device)
    step_counter.add_(1)
    grid = (triton.cdiv(N, block_n), triton.cdiv(K, block_k))
    _apply_adamw_three_accum_kernel[grid](
        grad_W, s_slow, s_fast, v_slow, row_exp, col_exp,
        row_max, col_max,
        N, K,
        float(lr), int(mantissa_bias), float(alpha), float(beta1),
        float(weight_decay), float(eps), float(step_cap),
        float(v_scale), float(drift_cancel_C), float(alpha_v_fast),
        float(wd_sv), float(wd_sf),
        step_counter,
        grad_W.stride(0), grad_W.stride(1),
        s_slow.stride(0), s_slow.stride(1),
        BLOCK_N=block_n, BLOCK_K=block_k,
        V_SLOW_FACTOR=int(v_slow_factor),
        MASS_PRESERVE=bool(mass_preserve),
        APPLY_CHASE=bool(apply_chase),
        MASS_PRESERVE_CHASE=bool(mass_preserve_chase),
    )


# ============================================================
# Custom autograd Function for Conv2d that uses cuDNN for the matmul
# and the concord update kernel for the state update.
# The bf16 weight is transient: reconstructed each forward, freed after
# backward. Persistent state remains s_slow + s_fast (32 bits/param).
# ============================================================

# ============================================================
# EMBEDDING kernels: gather indexed rows from concord int state on
# forward, sparse SR-tick + chase on the unique touched rows during
# backward. Designed for nn.Embedding's (V, D) shape, where the
# gradient is sparse — only the rows for tokens that appear in the
# batch get nonzero grad. Same per-row exponent / int16 mantissa
# split as the Linear path; the row_exp vector is naturally indexed
# by token id.
# ============================================================


@triton.jit
def _embedding_forward_kernel(
    s_slow_ptr,        # [V, D] int16
    s_fast_ptr,        # [V, D] int16
    v_slow_ptr,        # [V, D] int8 (gated by USE_V_SLOW)
    row_exp_ptr,       # [V] int8
    col_exp_ptr,       # [D] int8
    input_ids_ptr,     # [M] int32 — flattened token ids
    out_ptr,           # [M, D] bf16
    M, D, mantissa_bias,
    stride_sv, stride_sd,        # s_* strides (V, D)
    stride_om, stride_od,        # output strides (M, D)
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = offs_m < M
    d_mask = offs_d < D
    md_mask = m_mask[:, None] & d_mask[None, :]

    # Load token ids (clamp to a safe value for masked lanes — 0 is fine
    # because we mask the corresponding output).
    ids = tl.load(input_ids_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)

    row_e = tl.load(row_exp_ptr + ids, mask=m_mask, other=0).to(tl.float32)
    col_e = tl.load(col_exp_ptr + offs_d, mask=d_mask, other=0).to(tl.float32)

    s_off = ids[:, None] * stride_sv + offs_d[None, :] * stride_sd
    s_slow = tl.load(s_slow_ptr + s_off, mask=md_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=md_mask, other=0).to(tl.int32)
    m = s_slow + s_fast
    if USE_V_SLOW:
        v_slow = tl.load(v_slow_ptr + s_off, mask=md_mask, other=0).to(tl.int32)
        m = m + v_slow * V_SLOW_FACTOR

    exp = (row_e[:, None] + col_e[None, :] - mantissa_bias)
    scale = tl.exp2(exp)
    out = m.to(tl.float32) * scale

    out_off = offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptr + out_off, out.to(tl.bfloat16), mask=md_mask)


@triton.jit
def _embedding_sparse_update_kernel(
    s_slow_ptr,        # [V, D] int16, modified in place
    s_fast_ptr,        # [V, D] int16, modified in place
    v_slow_ptr,        # [V, D] int8, modified in place (USE_V_SLOW only)
    row_exp_ptr,       # [V] int8
    col_exp_ptr,       # [D] int8
    unique_ids_ptr,    # [U] int32 — unique tokens that received gradient
    grad_accum_ptr,    # [U, D] fp32 — summed grad per unique row
    U, D, mantissa_bias,
    lr, alpha,
    alpha_v_fast,
    step_salt_ptr,
    stride_sv, stride_sd,
    stride_gu, stride_gd,
    BLOCK_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_V_SLOW: tl.constexpr,
    V_SLOW_FACTOR: tl.constexpr,
    # See _fused_grad_W_and_update_kernel docstring for APPLY_CHASE
    # semantics. For embeddings the dynamic is the same — under
    # accumulation, only the touched rows' s_fast tick, and the chase
    # / v_slow leak wait for the K-th call.
    APPLY_CHASE: tl.constexpr,
):
    pid_u = tl.program_id(0)
    pid_d = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)

    offs_u = pid_u * BLOCK_U + tl.arange(0, BLOCK_U)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    u_mask = offs_u < U
    d_mask = offs_d < D
    ud_mask = u_mask[:, None] & d_mask[None, :]

    ids = tl.load(unique_ids_ptr + offs_u, mask=u_mask, other=0).to(tl.int32)
    row_e = tl.load(row_exp_ptr + ids, mask=u_mask, other=0).to(tl.float32)
    col_e = tl.load(col_exp_ptr + offs_d, mask=d_mask, other=0).to(tl.float32)

    g_off = offs_u[:, None] * stride_gu + offs_d[None, :] * stride_gd
    grad_W = tl.load(grad_accum_ptr + g_off, mask=ud_mask, other=0.0)

    s_off = ids[:, None] * stride_sv + offs_d[None, :] * stride_sd
    s_slow = tl.load(s_slow_ptr + s_off, mask=ud_mask, other=0).to(tl.int32)
    s_fast = tl.load(s_fast_ptr + s_off, mask=ud_mask, other=0).to(tl.int32)

    inv_exp = mantissa_bias - row_e[:, None] - col_e[None, :]
    scale_inv = tl.exp2(inv_exp)
    delta_t = -lr * grad_W * scale_inv

    # SR-tick s_fast. pos_hash uses (token_id, dim_idx) so within-tile
    # decorrelation works even when s_fast is uniform (e.g. cold init).
    pos_hash = (ids[:, None] << 16) ^ offs_d[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # Chase + v_slow leak — gated on APPLY_CHASE so K microbatches'
    # ticks can accumulate into s_fast before the chase smooths.
    if APPLY_CHASE:
        gap = (s_fast - s_slow).to(tl.float32)
        delta_slow_f = alpha * gap
        r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
        floor_s = tl.floor(delta_slow_f)
        frac_s = delta_slow_f - floor_s
        tick_slow = (floor_s + (r2 < frac_s).to(tl.float32)).to(tl.int32)
        s_slow = s_slow + tick_slow

        if USE_V_SLOW:
            v_slow_old = tl.load(v_slow_ptr + s_off, mask=ud_mask,
                                   other=0).to(tl.int32)
            gap_v_full = (s_fast - v_slow_old * V_SLOW_FACTOR).to(tl.float32)
            delta_v8 = alpha_v_fast * gap_v_full / V_SLOW_FACTOR
            r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
            floor_v = tl.floor(delta_v8)
            frac_v = delta_v8 - floor_v
            tick_v8 = (floor_v + (r3 < frac_v).to(tl.float32)).to(tl.int32)
            new_v_int32 = v_slow_old + tick_v8
            new_v_int8 = tl.minimum(tl.maximum(new_v_int32, -128), 127)
            tl.store(v_slow_ptr + s_off, new_v_int8.to(tl.int8),
                       mask=ud_mask)

        s_slow = tl.minimum(tl.maximum(s_slow, -32768), 32767)
        tl.store(s_slow_ptr + s_off, s_slow.to(tl.int16), mask=ud_mask)

    s_fast = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    tl.store(s_fast_ptr + s_off, s_fast.to(tl.int16), mask=ud_mask)


def embedding_forward_triton(s_slow, s_fast, row_exp, col_exp,
                                input_ids,
                                v_slow=None, v_slow_factor=128,
                                mantissa_bias=15,
                                out=None,
                                block_m=32, block_d=64):
    """Gather indexed rows from the concord int state into a bf16
    output tensor. input_ids may be any shape; the returned tensor has
    shape (*input_ids.shape, D) and dtype bfloat16."""
    V, D = s_slow.shape
    orig_shape = tuple(input_ids.shape)
    ids_flat = input_ids.reshape(-1).contiguous()
    if ids_flat.dtype != torch.int32:
        ids_flat = ids_flat.to(torch.int32)
    M = ids_flat.numel()

    out_flat_shape = (M, D)
    if out is None:
        out_flat = torch.empty(out_flat_shape, dtype=torch.bfloat16,
                                device=s_slow.device)
    else:
        assert out.shape == out_flat_shape or out.shape == orig_shape + (D,)
        out_flat = out.reshape(out_flat_shape)

    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow

    grid = (triton.cdiv(M, block_m), triton.cdiv(D, block_d))
    with PROFILER.time('embedding_forward'):
        _embedding_forward_kernel[grid](
            s_slow, s_fast, v_slow_ptr, row_exp, col_exp,
            ids_flat, out_flat,
            M, D, int(mantissa_bias),
            s_slow.stride(0), s_slow.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_M=block_m, BLOCK_D=block_d,
            USE_V_SLOW=use_v_slow,
            V_SLOW_FACTOR=int(v_slow_factor),
        )
    return out_flat.reshape(orig_shape + (D,))


def embedding_sparse_update_triton(s_slow, s_fast, row_exp, col_exp,
                                      input_ids, grad_out,
                                      lr, mantissa_bias=15, alpha=0.1,
                                      v_slow=None, v_slow_factor=128,
                                      alpha_v_fast=0.001,
                                      apply_chase=True,
                                      block_u=32, block_d=64):
    """Sparse SR-tick + chase on the unique rows that received gradient.

    input_ids : any shape; flattened internally.
    grad_out  : (*input_ids.shape, D) bf16 or fp32. The per-occurrence
                gradient on the embedding output.

    Mechanics:
      1. Flatten input_ids → (M,) and grad_out → (M, D).
      2. torch.unique(ids, return_inverse=True) → (U,) unique tokens
         and (M,) inverse map (which unique row each occurrence belongs
         to). U <= min(M, vocab).
      3. accum = zeros(U, D, fp32);
         accum.index_add_(0, inverse, grad_out_fp32) — built-in scatter,
         very fast.
      4. Launch the embedding update kernel over (U, D) tiles. Each
         tile loads its block's unique-token ids, gathers the
         corresponding s_slow / s_fast rows, applies SR-tick + chase,
         and writes back."""
    V, D = s_slow.shape
    ids_flat = input_ids.reshape(-1).contiguous()
    if ids_flat.dtype != torch.int64 and ids_flat.dtype != torch.int32:
        ids_flat = ids_flat.to(torch.int64)
    M = ids_flat.numel()
    grad_flat = grad_out.reshape(M, D)
    if grad_flat.dtype != torch.float32:
        grad_flat = grad_flat.float()

    unique_ids, inverse = torch.unique(ids_flat, return_inverse=True)
    U = unique_ids.numel()
    if U == 0:
        return
    accum = torch.zeros((U, D), dtype=torch.float32, device=s_slow.device)
    accum.index_add_(0, inverse, grad_flat)

    if unique_ids.dtype != torch.int32:
        unique_ids = unique_ids.to(torch.int32)

    step_counter = _get_step_counter(s_slow.device)
    step_counter.add_(1)
    use_v_slow = v_slow is not None
    v_slow_ptr = v_slow if use_v_slow else s_slow

    grid = (triton.cdiv(U, block_u), triton.cdiv(D, block_d))
    with PROFILER.time('embedding_sparse_update'):
        _embedding_sparse_update_kernel[grid](
            s_slow, s_fast, v_slow_ptr, row_exp, col_exp,
            unique_ids, accum,
            U, D, int(mantissa_bias),
            float(lr), float(alpha),
            float(alpha_v_fast),
            step_counter,
            s_slow.stride(0), s_slow.stride(1),
            accum.stride(0), accum.stride(1),
            BLOCK_U=block_u, BLOCK_D=block_d,
            USE_V_SLOW=use_v_slow,
            V_SLOW_FACTOR=int(v_slow_factor),
            APPLY_CHASE=bool(apply_chase),
        )


class FusedConcordEmbedding(torch.autograd.Function):
    """Autograd Function wrapping the embedding forward + sparse update.

    forward(ctx, input_ids, s_slow, s_fast, row_exp, col_exp,
            v_slow_i8, lr, alpha, alpha_v_fast,
            v_slow_factor, mantissa_bias):
        Returns bf16 output of shape (*input_ids.shape, D).

    backward(ctx, grad_output):
        Runs the sparse SR-tick + chase update on s_slow / s_fast
        (and optionally v_slow_i8) for each unique token in input_ids,
        accumulating the per-occurrence grad rows.

    Returns no gradient for any input (concord state is updated
    in-place; input_ids is integer)."""

    @staticmethod
    def forward(ctx, input_ids, s_slow, s_fast, row_exp, col_exp,
                v_slow_i8, lr, alpha, alpha_v_fast,
                v_slow_factor, mantissa_bias, grad_anchor,
                apply_chase):
        """`grad_anchor` is a fp32 zero Parameter — its only job is to
        carry `requires_grad=True` so the autograd engine attaches a
        grad_fn to our output and calls our backward (where the actual
        concord update kernel runs). We never read its value and we
        return None for its gradient in backward, so it doesn't move.

        ``apply_chase`` is the Concord-native grad-accumulation gate:
        when False, backward only SR-ticks s_fast; the chase into
        s_slow and the v_slow leak are skipped so K microbatches'
        ticks accumulate before any smoothing fires.
        """
        out = embedding_forward_triton(
            s_slow, s_fast, row_exp, col_exp, input_ids,
            v_slow=v_slow_i8, v_slow_factor=v_slow_factor,
            mantissa_bias=mantissa_bias)
        # Stash everything the backward needs to apply the concord update.
        ctx.s_slow = s_slow
        ctx.s_fast = s_fast
        ctx.row_exp = row_exp
        ctx.col_exp = col_exp
        ctx.v_slow_i8 = v_slow_i8
        ctx.input_ids = input_ids
        ctx.lr = lr
        ctx.alpha = alpha
        ctx.alpha_v_fast = alpha_v_fast
        ctx.v_slow_factor = v_slow_factor
        ctx.mantissa_bias = mantissa_bias
        ctx.apply_chase = apply_chase
        return out

    @staticmethod
    def backward(ctx, grad_output):
        embedding_sparse_update_triton(
            ctx.s_slow, ctx.s_fast, ctx.row_exp, ctx.col_exp,
            ctx.input_ids, grad_output,
            lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
            alpha=ctx.alpha,
            v_slow=ctx.v_slow_i8,
            v_slow_factor=ctx.v_slow_factor,
            alpha_v_fast=ctx.alpha_v_fast,
            apply_chase=ctx.apply_chase,
        )
        # 13 forward args; nothing accepts a gradient.
        return (None,) * 13


class FusedConcordConv2d(torch.autograd.Function):
    """Fully fused conv2d: forward, grad_x, and grad_W+update are all Triton
    kernels with bf16 weight reconstructed inline in registers. No bf16
    weight ever materialized in HBM."""

    @staticmethod
    def forward(ctx, x, s_slow, s_fast, row_exp, col_exp, bias,
                in_channels, out_channels, kh, kw, stride, padding,
                mantissa_bias, lr, alpha, beta1,
                v_slow_i8_buf, v_slow_factor, alpha_v_fast,
                weight_buf, apply_chase,
                grad_W_buf, row_max_buf, col_max_buf):
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        if not x_bf16.is_contiguous():
            x_bf16 = x_bf16.contiguous()
        # Materialise the live bf16 weight to HBM (flat 2D), then reshape
        # to 4D for cuDNN's conv2d. cuDNN conv kernels are years ahead
        # of our Triton conv recon — switching to materialised + cuDNN
        # is ~2× faster than the inline-recon conv kernels. Transient
        # HBM cost = one bf16 weight per layer per step.
        weight_2d = materialize_bf16_weight(
            s_slow, s_fast, row_exp, col_exp,
            mantissa_bias=mantissa_bias,
            v_slow=v_slow_i8_buf, v_slow_factor=v_slow_factor,
            out=weight_buf)
        weight_4d = weight_2d.view(out_channels, in_channels, kh, kw)
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = torch.nn.functional.conv2d(x_bf16, weight_4d, bias=bias_bf16,
                                         stride=stride, padding=padding)
        ctx.save_for_backward(x_bf16, weight_4d)
        ctx.s_slow = s_slow
        ctx.s_fast = s_fast
        ctx.row_exp = row_exp
        ctx.col_exp = col_exp
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.kh, ctx.kw = kh, kw
        ctx.stride = stride
        ctx.padding = padding
        ctx.mantissa_bias = mantissa_bias
        ctx.lr = lr
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.has_bias = bias is not None
        ctx.H_in, ctx.W_in = x_bf16.shape[-2:]
        ctx.v_slow_i8_buf = v_slow_i8_buf
        ctx.v_slow_factor = v_slow_factor
        ctx.alpha_v_fast = alpha_v_fast
        ctx.apply_chase = apply_chase
        # Pre-allocated backward buffers (no per-step allocation).
        ctx.grad_W_buf = grad_W_buf
        ctx.row_max_buf = row_max_buf
        ctx.col_max_buf = col_max_buf
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_bf16, weight_4d = ctx.saved_tensors
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()

        # grad_x via cuDNN's specialised conv2d_input against the
        # materialised bf16 weight. Released after this return.
        grad_x = torch.nn.grad.conv2d_input(
            x_bf16.shape, weight_4d, grad_y,
            stride=ctx.stride, padding=ctx.padding)

        # Always: cuDNN conv2d_weight for grad_W, then apply_update kernel
        # on the bf16 grad_W tile.
        #
        # The OLD path used a size-gated branch: small convs (out_ch*K<2048)
        # via cuDNN, big convs via _fused_conv2d_grad_W_and_update_kernel
        # (grad_W compute + state update in one mega-kernel). The big-conv
        # branch was added on the theory that an HBM round-trip for grad_W
        # was the bottleneck. That theory was wrong: on the SDXL shapes
        # the fused kernel hit 255 registers and spilled 752 bytes per
        # thread, which is local-memory (DRAM) traffic on EVERY iteration
        # of the BLOCK_BHW reduction loop. The "saved bandwidth" was paid
        # back many times over by spill loads/stores. The fused kernel
        # also tipped the Windows CUDA 12.8 ptxas into a stack-buffer
        # overrun (0xC0000409) under concurrent autotune compilation.
        #
        # cuDNN's conv2d_weight is hand-tuned; apply_update_from_grad_W
        # is a small clean kernel (~96 regs, no spill). Cost: one
        # transient bf16 grad_W tensor per backward (freed immediately).
        # Largest SDXL conv (1280x1280x3x3) = ~30 MB transient peak.
        K_w = ctx.in_channels * ctx.kh * ctx.kw
        with PROFILER.time(f'conv_gW_{ctx.in_channels}x{ctx.out_channels}'):
            # grad_W via cuDNN. The 4D output is allocated fresh by
            # cuDNN and freed after this backward returns; under CUDA
            # graph capture it goes into the graph pool and is reused
            # across replays. No persistent buffer needed.
            grad_W_4d = torch.nn.grad.conv2d_weight(
                x_bf16,
                (ctx.out_channels, ctx.in_channels, ctx.kh, ctx.kw),
                grad_y,
                stride=ctx.stride, padding=ctx.padding,
            )
            grad_W_2d = grad_W_4d.reshape(ctx.out_channels, -1).contiguous()
            if grad_W_2d.dtype != torch.bfloat16:
                grad_W_2d = grad_W_2d.to(torch.bfloat16)
            # row/col max are small persistent buffers, zeroed in place.
            ctx.row_max_buf.zero_()
            ctx.col_max_buf.zero_()
            apply_update_from_grad_W(
                grad_W_2d, ctx.s_slow, ctx.s_fast,
                ctx.row_exp, ctx.col_exp,
                ctx.row_max_buf, ctx.col_max_buf,
                lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                alpha=ctx.alpha, beta1=ctx.beta1,
                v_slow=ctx.v_slow_i8_buf,
                v_slow_factor=ctx.v_slow_factor,
                alpha_v_fast=ctx.alpha_v_fast,
                apply_chase=ctx.apply_chase,
                mass_preserve_chase=_GLOBAL_MASS_PRESERVE_CHASE)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.sum(dim=(0, 2, 3))

        # 24 forward args; only x and bias have grads.
        return (grad_x, None, None, None, None, grad_bias,
                None, None, None, None, None, None,
                None, None, None, None,
                None, None, None,
                None, None,
                None, None, None)


class FusedConcordLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s_slow, s_fast, row_exp, col_exp, bias,
                mantissa_bias, lr, alpha, beta1,
                optimizer_kind,
                weight_decay, eps, step_cap,
                optimizer_v_kind, v_row, v_col, g2_row, g2_col,
                v_beta2, v_step,
                v_diag, v_diag_beta2,
                v_slow_buf, v_scale, drift_cancel_C, alpha_v_fast,
                v_slow_i8_buf, v_slow_factor,
                weight_buf, wd_sv, wd_sf, apply_chase,
                grad_W_buf, row_max_buf, col_max_buf):
        """Forward arg list.

        optimizer_kind   : 'sgd' | 'adamw'
        optimizer_v_kind : (AdamW only) 'three_accum' | 'v_rank1'
        weight_decay     : decoupled wd coefficient
        eps              : denominator floor in the AdamW step
        v_diag           : optional fp32 (out, in) buffer EMA-tracking g²
                           for diagnostic comparison.
        v_diag_beta2     : EMA decay for v_diag (ignored if v_diag None).
        """
        # Materialise the live bf16 weight to HBM and use cuBLAS for the
        # matmul. ~2× faster than the inline-recon fused kernel at small
        # batch sizes (cuBLAS is years ahead of Triton for matmul). The
        # transient weight stays alive until backward, then is freed.
        # The grad_W+update kernel still computes grad_W inline from
        # grad_y and x — no HBM grad_W tensor.
        weight = materialize_bf16_weight(
            s_slow, s_fast, row_exp, col_exp,
            mantissa_bias=mantissa_bias,
            v_slow=v_slow_i8_buf, v_slow_factor=v_slow_factor,
            out=weight_buf)
        # Bias is kept fp32 on the layer (aux optimiser likes the
        # precision) — cast to bf16 for the cuBLAS call. Small tensor,
        # negligible cost.
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = torch.nn.functional.linear(x, weight, bias_bf16)
        # Save the materialised weight for grad_x, plus x for grad_W.
        ctx.save_for_backward(x, weight)
        ctx.s_slow = s_slow
        ctx.s_fast = s_fast
        ctx.row_exp = row_exp
        ctx.col_exp = col_exp
        ctx.mantissa_bias = mantissa_bias
        ctx.lr = lr
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.has_bias = bias is not None
        ctx.optimizer_kind = optimizer_kind
        ctx.weight_decay = weight_decay
        ctx.eps = eps
        ctx.step_cap = step_cap
        ctx.optimizer_v_kind = optimizer_v_kind
        ctx.v_row = v_row
        ctx.v_col = v_col
        ctx.g2_row = g2_row
        ctx.g2_col = g2_col
        ctx.v_beta2 = v_beta2
        ctx.v_step = v_step
        ctx.v_diag = v_diag
        ctx.v_diag_beta2 = v_diag_beta2
        ctx.v_slow_buf = v_slow_buf
        ctx.v_scale = v_scale
        ctx.drift_cancel_C = drift_cancel_C
        ctx.alpha_v_fast = alpha_v_fast
        ctx.v_slow_i8_buf = v_slow_i8_buf
        ctx.v_slow_factor = v_slow_factor
        ctx.wd_sv = wd_sv
        ctx.wd_sf = wd_sf
        ctx.apply_chase = apply_chase
        # Pre-allocated backward buffers.
        ctx.grad_W_buf = grad_W_buf
        ctx.row_max_buf = row_max_buf
        ctx.col_max_buf = col_max_buf
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        s_slow, s_fast = ctx.s_slow, ctx.s_fast
        row_exp, col_exp = ctx.row_exp, ctx.col_exp

        # Ensure grad_y is bf16 contiguous
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()

        # grad_x via cuBLAS against the materialised bf16 weight that
        # forward saved. The weight already reflects v_slow_i8's
        # contribution at shifted scale, so no special handling needed
        # here. The materialised weight lives only across the
        # forward→backward window; freed after this return.
        grad_x = torch.matmul(grad_y, weight)

        # Optional per-element β2-EMA diagnostic. Compute grad_W via
        # torch (fp32 matmul) and EMA-update v_diag. The production
        # update kernels still consume their own in-register grad_W;
        # this is a parallel observation channel only, paid for when
        # the diagnostic is explicitly enabled on the layer.
        if ctx.v_diag is not None:
            with torch.no_grad():
                grad_W_torch = (grad_y.float().T @ x.float())  # (N, K) fp32
                beta2 = ctx.v_diag_beta2
                ctx.v_diag.mul_(beta2).add_(grad_W_torch * grad_W_torch,
                                              alpha=(1.0 - beta2))

        # grad_W computed inline + concord update applied in place.
        # SGD path: classic SR-tick + chase. AdamW path: either
        # three-accumulator drift-cancelled noise residual (default,
        # 40 bits/param, best accuracy) or v_rank1 Adafactor rank-1
        # EMA (32 bits/param + O(N+K) per layer, cheaper memory).
        if ctx.optimizer_kind == 'adamw':
            if ctx.optimizer_v_kind == 'v_rank1':
                # NB: v_rank1 path doesn't gate on apply_chase yet —
                # its update is a different kernel that we haven't
                # extended. Treated as always-chase for now (so
                # Concord-native grad accumulation in v_rank1 mode
                # falls back to standard PyTorch-style per-microbatch
                # steps).
                fused_grad_W_and_adamw_v_rank1_update(
                    grad_y, x, s_slow, s_fast, row_exp, col_exp,
                    ctx.v_row, ctx.v_col, ctx.g2_row, ctx.g2_col,
                    lr=ctx.lr, beta2=ctx.v_beta2, v_step=ctx.v_step,
                    mantissa_bias=ctx.mantissa_bias,
                    alpha=ctx.alpha, beta1=ctx.beta1,
                    weight_decay=ctx.weight_decay, eps=ctx.eps,
                    step_cap=ctx.step_cap,
                )
            else:
                # 'three_accum' (default). cuBLAS matmul for grad_W,
                # then apply_adamw_three_accum_from_grad_W. grad_W is
                # allocated fresh (small per-layer transient, freed
                # after this backward; under graph capture it lives
                # in the graph pool and is reused across replays).
                grad_W = torch.matmul(grad_y.transpose(-1, -2), x)
                if grad_W.dtype != torch.bfloat16:
                    grad_W = grad_W.to(torch.bfloat16)
                if not grad_W.is_contiguous():
                    grad_W = grad_W.contiguous()
                ctx.row_max_buf.zero_()
                ctx.col_max_buf.zero_()
                apply_adamw_three_accum_from_grad_W(
                    grad_W, s_slow, s_fast, ctx.v_slow_i8_buf,
                    row_exp, col_exp,
                    ctx.row_max_buf, ctx.col_max_buf,
                    lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                    alpha=ctx.alpha, beta1=ctx.beta1,
                    weight_decay=ctx.weight_decay, eps=ctx.eps,
                    step_cap=ctx.step_cap,
                    v_scale=getattr(ctx, 'v_scale', 1.0),
                    drift_cancel_C=getattr(ctx, 'drift_cancel_C', 0.1),
                    alpha_v_fast=getattr(ctx, 'alpha_v_fast', 0.001),
                    v_slow_factor=ctx.v_slow_factor,
                    wd_sv=getattr(ctx, 'wd_sv', 0.0),
                    wd_sf=getattr(ctx, 'wd_sf', 0.0),
                    apply_chase=ctx.apply_chase,
                    mass_preserve_chase=_GLOBAL_MASS_PRESERVE_CHASE,
                )
        else:
            # SGD path. cuBLAS matmul for grad_W, then apply_update.
            # grad_W is allocated fresh per backward (transient; under
            # graph capture it goes into the graph pool and is reused).
            # Shapes: grad_y (M, N), x (M, K). grad_W = grad_y.T @ x
            # has shape (N, K) == s_slow.shape.
            grad_W = torch.matmul(grad_y.transpose(-1, -2), x)
            if grad_W.dtype != torch.bfloat16:
                grad_W = grad_W.to(torch.bfloat16)
            if not grad_W.is_contiguous():
                grad_W = grad_W.contiguous()
            ctx.row_max_buf.zero_()
            ctx.col_max_buf.zero_()
            apply_update_from_grad_W(
                grad_W, s_slow, s_fast, row_exp, col_exp,
                ctx.row_max_buf, ctx.col_max_buf,
                lr=ctx.lr, mantissa_bias=ctx.mantissa_bias,
                alpha=ctx.alpha, beta1=ctx.beta1,
                v_slow=ctx.v_slow_i8_buf,
                v_slow_factor=ctx.v_slow_factor,
                alpha_v_fast=ctx.alpha_v_fast,
                apply_chase=ctx.apply_chase,
                mass_preserve_chase=_GLOBAL_MASS_PRESERVE_CHASE)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y.sum(dim=0)

        # Forward args (36 total: original 33 + 3 buffer refs).
        # Return None for each non-tensor / non-grad-receiving arg.
        return (grad_x, None, None, None, None, grad_bias,
                None, None, None, None,
                None,
                None, None, None,
                None, None, None, None, None,
                None, None,
                None, None,
                None, None, None, None,
                None, None,
                None, None, None, None,
                None, None, None)
