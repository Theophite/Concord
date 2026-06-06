"""Optimize the fused dequant-matmul: factor the separable powers-of-two OUT of the
dot. 2^(row_e[n]+col_e[k]-bias) = 2^(row_e[n]-bias) * 2^col_e[k], both exact, so:
  - fold 2^col_e into x (per-K, once per tile),
  - dot on the raw integer mantissa m_eff (no exp2 per weight element),
  - fold 2^(row_e-bias) into the output (per-N, once).
The N*K exp2 calls in the hot loop collapse to N+K. Must stay BIT-IDENTICAL to the
original fused kernel (power-of-two factoring is exact through fp32 accumulation).
Run from OneTrainer-clean dir.
"""
import sys
sys.path.insert(0, r"C:/fisher/OneTrainer-clean")
import torch, triton, triton.language as tl, torch.nn.functional as F
from modules.util.optimizer.concord.prototype_packed_b import (
    materialize_packed_bf16, _fused_packed_linear_kernel, fused_packed_linear)
DEV = "cuda"


# ---- OPTIMIZED forward: powers-of-two factored out of the dot ----
@triton.jit
def _fused_opt_linear_kernel(
    x_ptr, packed_ptr, row_exp_ptr, col_exp_ptr, bias_ptr, y_ptr,
    M, N, K, mantissa_bias,
    stride_xm, stride_xk, stride_pn, stride_pk, stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_e = tl.load(row_exp_ptr + offs_n, mask=offs_n < N, other=0).to(tl.int32)
    row_scale = tl.exp2((row_e - mantissa_bias).to(tl.float32))          # [N] once
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
        col_scale = tl.exp2(col_e.to(tl.float32))                       # [K] once per K-tile
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0).to(tl.float32)
        x = (x * col_scale[None, :]).to(tl.bfloat16)                    # fold col into x (exact)
        packed = tl.load(packed_ptr + offs_k[:, None] * stride_pk + offs_n[None, :] * stride_pn,
                         mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0).to(tl.int32)
        s_fast = packed >> 16
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        m_eff = (s_slow * 128 + s_fast + v_slow * 128).to(tl.bfloat16)  # raw mantissa, no exp2
        acc += tl.dot(x, m_eff)
    acc = acc * row_scale[None, :]                                      # fold row into output
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_opt_linear(x, packed_w, row_exp, col_exp, bias=None, mantissa_bias=15):
    *lead, K = x.shape
    N = packed_w.shape[0]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BM, BN, BK = 64, 64, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fused_opt_linear_kernel[grid](
        x2d, packed_w, row_exp, col_exp, bias if bias is not None else x2d, y,
        M, N, K, int(mantissa_bias),
        x2d.stride(0), x2d.stride(1), packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return y.reshape(*lead, N)


# ---- OPTIMIZED v2: factored + COALESCED packed load (load [N,K] contiguous, tl.trans) ----
@triton.jit
def _fused_coal_linear_kernel(
    x_ptr, packed_ptr, row_exp_ptr, col_exp_ptr, bias_ptr, y_ptr,
    M, N, K, mantissa_bias,
    stride_xm, stride_xk, stride_pn, stride_pk, stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_e = tl.load(row_exp_ptr + offs_n, mask=offs_n < N, other=0).to(tl.int32)
    row_scale = tl.exp2((row_e - mantissa_bias).to(tl.float32))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
        col_scale = tl.exp2(col_e.to(tl.float32))
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0).to(tl.float32)
        x = (x * col_scale[None, :]).to(tl.bfloat16)
        # COALESCED: [N,K] tile, inner dim K is contiguous (stride_pk=1)
        packed = tl.load(packed_ptr + offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk,
                         mask=(offs_n[:, None] < N) & (k_mask[None, :]), other=0).to(tl.int32)
        s_fast = packed >> 16
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        m_eff = (s_slow * 128 + s_fast + v_slow * 128).to(tl.bfloat16)   # [N, K]
        acc += tl.dot(x, tl.trans(m_eff))                                # x[M,K] @ [K,N]
    acc = acc * row_scale[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_coal_linear(x, packed_w, row_exp, col_exp, bias=None, mantissa_bias=15):
    *lead, K = x.shape
    N = packed_w.shape[0]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BM, BN, BK = 64, 64, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fused_coal_linear_kernel[grid](
        x2d, packed_w, row_exp, col_exp, bias if bias is not None else x2d, y,
        M, N, K, int(mantissa_bias),
        x2d.stride(0), x2d.stride(1), packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return y.reshape(*lead, N)


def bench(fn, *a, iters=50):
    for _ in range(10):
        fn(*a)
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    torch.manual_seed(0)
    for (M, N, K) in [(4096, 1280, 1280), (4096, 5120, 1280), (1024, 1280, 5120)]:
        packed_w = torch.randint(-(2**31), 2**31 - 1, (N, K), dtype=torch.int32, device=DEV)
        row_exp = torch.randint(-3, 3, (N,), dtype=torch.int8, device=DEV)
        col_exp = torch.randint(-3, 3, (K,), dtype=torch.int8, device=DEV)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        bias = torch.randn(N, dtype=torch.bfloat16, device=DEV)

        orig = fused_packed_linear(x, packed_w, row_exp, col_exp, bias, 15)
        opt = fused_opt_linear(x, packed_w, row_exp, col_exp, bias, 15)
        coal = fused_coal_linear(x, packed_w, row_exp, col_exp, bias, 15)
        bi_opt = torch.equal(orig, opt)
        bi_coal = torch.equal(orig, coal)
        t_orig = bench(lambda: fused_packed_linear(x, packed_w, row_exp, col_exp, bias, 15))
        t_opt = bench(lambda: fused_opt_linear(x, packed_w, row_exp, col_exp, bias, 15))
        t_coal = bench(lambda: fused_coal_linear(x, packed_w, row_exp, col_exp, bias, 15))
        print(f"M={M} N={N} K={K}: bit[factored={bi_opt},coal={bi_coal}] | "
              f"orig={t_orig:.0f}us  factored={t_opt:.0f}us ({t_orig/t_opt:.2f}x)  "
              f"coalesced={t_coal:.0f}us ({t_orig/t_coal:.2f}x)")


if __name__ == "__main__":
    main()
