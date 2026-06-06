"""Standalone validation of a fused dequant-matmul for Concord's packed_w format.

Goal: compute  y = x @ dequant(packed_w)^T  WITHOUT ever materializing the full
bf16 weight buffer (the ~5 GB cache we want to eliminate). Dequant happens inside
the matmul, one K x N tile at a time.

Validate against the reference: materialize_packed_bf16(packed_w) then F.linear.
Run from the OneTrainer-clean dir (for the module import).
"""
import sys, os
sys.path.insert(0, r"C:/fisher/OneTrainer-clean")
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from modules.util.optimizer.concord.prototype_packed_b import materialize_packed_bf16

DEV = "cuda"


@triton.jit
def _fused_packed_linear_kernel(
    x_ptr, packed_ptr, row_exp_ptr, col_exp_ptr, bias_ptr, y_ptr,
    M, N, K, mantissa_bias,
    stride_xm, stride_xk,
    stride_pn, stride_pk,          # packed_w is [N, K]
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_e = tl.load(row_exp_ptr + offs_n, mask=offs_n < N, other=0).to(tl.int32)  # [N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # x tile [M, K]
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0).to(tl.bfloat16)
        # packed W tile arranged as [K, N]: element [ik, in] = packed_w[offs_n[in], offs_k[ik]]
        packed = tl.load(packed_ptr + offs_k[:, None] * stride_pk + offs_n[None, :] * stride_pn,
                         mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0).to(tl.int32)
        s_fast = packed >> 16
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        m_eff = s_slow * 128 + s_fast + v_slow * 128                 # [K, N] int32
        col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)  # [K]
        exp = (col_e[:, None] + row_e[None, :] - mantissa_bias).to(tl.float32)    # [K, N]
        w = (m_eff.to(tl.float32) * tl.exp2(exp)).to(tl.bfloat16)                 # [K, N]
        acc += tl.dot(x, w)                                          # [M,K]@[K,N] -> [M,N]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_packed_linear(x, packed_w, row_exp, col_exp, bias=None, mantissa_bias=15):
    M, K = x.shape
    N, K2 = packed_w.shape
    assert K == K2
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_packed_linear_kernel[grid](
        x, packed_w, row_exp, col_exp, bias if bias is not None else x, y,
        M, N, K, int(mantissa_bias),
        x.stride(0), x.stride(1), packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


@triton.jit
def _fused_packed_gradx_kernel(
    gy_ptr, packed_ptr, row_exp_ptr, col_exp_ptr, gx_ptr,
    M, N, K, mantissa_bias,
    stride_gym, stride_gyn,
    stride_pn, stride_pk,          # packed_w is [N, K]
    stride_gxm, stride_gxk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grad_x[M,K] = grad_y[M,N] @ W[N,K], W dequantized on the fly. Contract over N.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    col_e = tl.load(col_exp_ptr + offs_k, mask=offs_k < K, other=0).to(tl.int32)  # [K]
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        gy = tl.load(gy_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn,
                     mask=(offs_m[:, None] < M) & (n_mask[None, :]), other=0.0).to(tl.bfloat16)  # [M,N]
        packed = tl.load(packed_ptr + offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk,
                         mask=(n_mask[:, None]) & (offs_k[None, :] < K), other=0).to(tl.int32)    # [N,K]
        s_fast = packed >> 16
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        m_eff = s_slow * 128 + s_fast + v_slow * 128
        row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)  # [N]
        exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)    # [N,K]
        w = (m_eff.to(tl.float32) * tl.exp2(exp)).to(tl.bfloat16)                 # [N,K]
        acc += tl.dot(gy, w)                                                      # [M,N]@[N,K] -> [M,K]
    tl.store(gx_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


def fused_packed_gradx(grad_y, packed_w, row_exp, col_exp, mantissa_bias=15):
    M, N = grad_y.shape
    N2, K = packed_w.shape
    assert N == N2
    gx = torch.empty((M, K), dtype=torch.bfloat16, device=grad_y.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _fused_packed_gradx_kernel[grid](
        grad_y, packed_w, row_exp, col_exp, gx,
        M, N, K, int(mantissa_bias),
        grad_y.stride(0), grad_y.stride(1), packed_w.stride(0), packed_w.stride(1),
        gx.stride(0), gx.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return gx


def main():
    torch.manual_seed(0)
    for (M, N, K) in [(64, 128, 256), (320, 640, 1280), (77, 1280, 768)]:
        packed_w = torch.randint(-(2**31), 2**31 - 1, (N, K), dtype=torch.int32, device=DEV)
        row_exp = torch.randint(-4, 4, (N,), dtype=torch.int8, device=DEV)
        col_exp = torch.randint(-4, 4, (K,), dtype=torch.int8, device=DEV)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        bias = torch.randn(N, dtype=torch.bfloat16, device=DEV)

        wbuf = torch.empty((N, K), dtype=torch.bfloat16, device=DEV)
        materialize_packed_bf16(packed_w, row_exp, col_exp, out=wbuf, mantissa_bias=15)
        # fp32 matmul of the SAME bf16 weight = the "true" result both bf16 paths approximate
        ref_fp32 = (x.float() @ wbuf.float().T) + bias.float()
        base = F.linear(x, wbuf, bias)                                   # cuBLAS bf16 matmul
        fused = fused_packed_linear(x, packed_w, row_exp, col_exp, bias, mantissa_bias=15)

        scale = ref_fp32.abs().max().clamp_min(1e-6)
        base_err = (base.float() - ref_fp32).abs().max() / scale
        fused_err = (fused.float() - ref_fp32).abs().max() / scale
        fb_diff = (fused.float() - base.float()).abs().max() / scale
        verdict = "OK (fused ~= cuBLAS vs fp32)" if fused_err <= base_err * 3 + 1e-4 else "BUG?"
        print(f"[fwd] M={M} N={N} K={K}: cuBLAS-vs-fp32={base_err.item():.5f}  "
              f"fused-vs-fp32={fused_err.item():.5f}  fused-vs-cuBLAS={fb_diff.item():.5f}  -> {verdict}")

        # ---- backward grad_x = grad_y @ W ----
        grad_y = torch.randn(M, N, dtype=torch.bfloat16, device=DEV)
        gx_ref_fp32 = grad_y.float() @ wbuf.float()
        gx_base = grad_y @ wbuf                                          # cuBLAS
        gx_fused = fused_packed_gradx(grad_y, packed_w, row_exp, col_exp, mantissa_bias=15)
        gscale = gx_ref_fp32.abs().max().clamp_min(1e-6)
        gbase_err = (gx_base.float() - gx_ref_fp32).abs().max() / gscale
        gfused_err = (gx_fused.float() - gx_ref_fp32).abs().max() / gscale
        gverdict = "OK" if gfused_err <= gbase_err * 3 + 1e-4 else "BUG?"
        print(f"[bwd] M={M} N={N} K={K}: cuBLAS-vs-fp32={gbase_err.item():.5f}  "
              f"fused-vs-fp32={gfused_err.item():.5f}  -> {gverdict}")


if __name__ == "__main__":
    main()
