"""Autotune the fused factored forward kernel: let Triton pick block sizes / warps /
stages per shape, vs the module's fixed 64x64x64. Measure the gain on real SDXL
Linear shapes. (Must still be bit-identical -- block choice doesn't change the math.)
"""
import sys
sys.path.insert(0, r"C:/fisher/OneTrainer-clean")
import torch, triton, triton.language as tl
from modules.util.optimizer.concord.prototype_packed_b import fused_packed_linear  # fixed-64 factored
DEV = "cuda"

CONFIGS = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _at_linear_kernel(
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
        packed = tl.load(packed_ptr + offs_k[:, None] * stride_pk + offs_n[None, :] * stride_pn,
                         mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0).to(tl.int32)
        s_fast = packed >> 16
        s_slow = (packed << 16) >> 24
        v_slow = (packed << 24) >> 24
        m_eff = (s_slow * 128 + s_fast + v_slow * 128).to(tl.bfloat16)
        acc += tl.dot(x, m_eff)
    acc = acc * row_scale[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def at_linear(x, packed_w, row_exp, col_exp, bias=None, mantissa_bias=15):
    *lead, K = x.shape
    N = packed_w.shape[0]
    x2d = x.reshape(-1, K)
    M = x2d.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _at_linear_kernel[grid](
        x2d, packed_w, row_exp, col_exp, bias if bias is not None else x2d, y,
        M, N, K, int(mantissa_bias),
        x2d.stride(0), x2d.stride(1), packed_w.stride(0), packed_w.stride(1),
        y.stride(0), y.stride(1), HAS_BIAS=bias is not None)
    return y.reshape(*lead, N)


def bench(fn, iters=60):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    torch.manual_seed(0)
    # SDXL-ish Linear shapes: (M=tokens, N=out, K=in). attn qkv/out + ff.
    shapes = [(4096, 1280, 1280), (4096, 5120, 1280), (4096, 1280, 5120), (1024, 1280, 1280), (9216, 640, 640)]
    for (M, N, K) in shapes:
        pw = torch.randint(-(2**31), 2**31 - 1, (N, K), dtype=torch.int32, device=DEV)
        re = torch.randint(-3, 3, (N,), dtype=torch.int8, device=DEV)
        ce = torch.randint(-3, 3, (K,), dtype=torch.int8, device=DEV)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        b = torch.randn(N, dtype=torch.bfloat16, device=DEV)
        fix = fused_packed_linear(x, pw, re, ce, b, 15)
        at = at_linear(x, pw, re, ce, b, 15)
        bit = torch.equal(fix, at)
        t_fix = bench(lambda: fused_packed_linear(x, pw, re, ce, b, 15))
        t_at = bench(lambda: at_linear(x, pw, re, ce, b, 15))
        best = _at_linear_kernel.best_config
        print(f"M={M} N={N} K={K}: bit={bit} | fixed64={t_fix:.0f}us autotuned={t_at:.0f}us "
              f"({t_fix/t_at:.2f}x) best={best}")


if __name__ == "__main__":
    main()
