"""Validate the memory-free accumulation design BEFORE touching production.

Design: s_fast IS the gradient accumulator. Gate consolidation (evap + chase +
leak + materialize + rebalance) on a `should_consolidate` device flag:
  - tick-only  (flag=0): SR-tick s_fast from this micro-batch's grad only; freeze
                         weight_buf, no chase/leak.
  - consolidate(flag=1): full apply on the accumulated s_fast.

Two tests:
  T1 (flag=1 == today): the gated kernel with flag=1 must be BIT-EXACT vs the
     ungated active apply (so accum==1 is byte-for-byte the current path).
  T2 (accumulation works): N tick-only micro-steps (grad_i = grad_full_i / N)
     then 1 consolidate must match a SINGLE apply on the summed gradient
     (the true-accumulation result) — in expectation (SR makes it not bit-exact,
     so compare mean/relerr, not equality).
"""
import importlib.util
import sys

import torch
import triton
import triton.language as tl

_FAP = r"C:\Concord\src\fused_apply_proto.py"
_spec = importlib.util.spec_from_file_location("fap", _FAP)
fap = importlib.util.module_from_spec(_spec); sys.modules["fap"] = fap
_spec.loader.exec_module(fap)
ref = fap.ref


@triton.jit
def _hash_uniform(x, pos, salt):
    h = x ^ salt ^ pos
    h = h ^ (h << 13); h = h ^ (h >> 17); h = h ^ (h << 5); h = h ^ (h >> 7)
    return (h & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _apply_gated_kernel(
    packed_ptr, grad_W_ptr, weight_buf_ptr, row_exp_ptr, col_exp_ptr,
    row_max_ptr, col_max_ptr, v_row_ptr, v_col_ptr, sum_v_inv_ptr,
    consolidate_ptr,                                    # [1] int32: 0=tick-only, 1=consolidate
    N, K, lr_ptr, mantissa_bias, alpha, beta1, drift_cancel_C, alpha_v_fast,
    eps_ptr, step_cap, v_scale, precond_p, gf_consol, gf_trust_delta_sq, gate_gain,
    wd_sv, wd_sf, chase_floor_ptr, leak_floor_ptr, v_bc_ptr, step_salt_ptr,
    stride_pn, stride_pk, stride_gn, stride_gk, stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    lr = tl.load(lr_ptr).to(tl.float32)
    eps = tl.load(eps_ptr).to(tl.float32)
    chase_floor = tl.load(chase_floor_ptr).to(tl.float32)
    leak_floor = tl.load(leak_floor_ptr).to(tl.float32)
    cons = tl.load(consolidate_ptr).to(tl.int32)       # gate
    consf = cons.to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N; k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = packed >> 16
    s_slow_i8 = (packed << 16) >> 24
    v_slow_i8 = (packed << 24) >> 24
    s_slow_full = s_slow_i8 * 128
    v_slow_full = v_slow_i8 * 128

    g_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + g_off, mask=nk_mask, other=0.0).to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_fwd = tl.exp2(total_exp); scale_inv = tl.exp2(-total_exp)

    d_fs = s_fast.to(tl.float32)
    d_sv = (s_slow_full - v_slow_full).to(tl.float32)
    noise = d_fs - drift_cancel_C * d_sv
    noise_in_w = noise * scale_fwd
    v_proxy = noise_in_w * noise_in_w * v_scale
    v_row_tile = tl.load(v_row_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    v_col_tile = tl.load(v_col_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
    sum_v_inv = tl.load(sum_v_inv_ptr).to(tl.float32)
    v_hat = v_row_tile[:, None] * v_col_tile[None, :] * sum_v_inv * tl.load(v_bc_ptr).to(tl.float32)
    v_proxy = v_proxy + gf_trust_delta_sq * v_hat
    sig_w = drift_cancel_C * d_sv * scale_fwd
    sig2 = sig_w * sig_w
    coh = sig2 / (sig2 + noise_in_w * noise_in_w + 1e-30)
    coh = tl.minimum(tl.maximum(coh, 0.0), 1.0)

    denom_p = tl.exp2(precond_p * tl.log2(v_proxy + eps))
    step_live = grad_W / denom_p
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    delta_grad = -lr * step_live * scale_inv               # GRADIENT tick: ALWAYS
    # consolidation terms (momentum + evaporation): ONLY when consolidating
    evap = lr * gf_consol * (1.0 - coh) * d_fs
    delta_t = delta_grad + consf * (beta1 * coh * d_fs - evap)

    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t); frac_t = delta_t - floor_t
    s_fast = s_fast + (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)

    # APPLY_CHASE — gated: chase_mantissa * consf -> 0 ticks when tick-only
    gate = chase_floor + (1.0 - chase_floor) * coh
    chase_int8_f = (alpha * gate * gate_gain * s_fast.to(tl.float32)) / 128.0 * consf
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_s = tl.floor(chase_int8_f)
    tick_slow_i8 = (floor_s + (r2 < (chase_int8_f - floor_s)).to(tl.float32)).to(tl.int32)
    s_slow_i8 = s_slow_i8 + tick_slow_i8
    s_fast = s_fast - tick_slow_i8 * 128

    # leak — gated
    gap_v_full = (s_slow_i8 * 128 - v_slow_full).to(tl.float32)
    delta_v8 = alpha_v_fast * gap_v_full / 128.0 * (leak_floor + (1.0 - leak_floor) * coh) * consf
    r3 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x33335555)
    floor_v = tl.floor(delta_v8)
    tick_v8 = (floor_v + (r3 < (delta_v8 - floor_v)).to(tl.float32)).to(tl.int32)
    new_v_int8 = tl.minimum(tl.maximum(v_slow_i8 + tick_v8, -128), 127)
    s_slow_i8 = s_slow_i8 - (new_v_int8 - v_slow_i8)
    # wd_sv/wd_sf = 0 in recipe -> omitted (no-ops)

    s_fast_c = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    s_slow_c = tl.minimum(tl.maximum(s_slow_i8, -128), 127)
    packed_new = ((s_fast_c & 0xFFFF) << 16) | ((s_slow_c & 0xFF) << 8) | (new_v_int8 & 0xFF)
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)      # ALWAYS store packed (s_fast accumulates)

    # materialize weight_buf — ONLY on consolidate (freeze during accumulation)
    new_m_eff = s_slow_c * 128 + s_fast_c + new_v_int8 * 128
    new_weight = new_m_eff.to(tl.float32) * scale_fwd
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_buf_ptr + w_off, new_weight.to(tl.bfloat16), mask=nk_mask & (cons == 1))

    # rebalance atomics — ONLY on consolidate
    abs_eff = tl.where(nk_mask, tl.abs(new_m_eff), 0)
    tl.atomic_max(row_max_ptr + offs_n, tl.max(abs_eff, axis=1), mask=n_mask & (cons == 1))
    tl.atomic_max(col_max_ptr + offs_k, tl.max(abs_eff, axis=0), mask=k_mask & (cons == 1))


def apply_gated(packed_w, grad_W, weight_buf, row_exp, col_exp, row_max, col_max,
                v_row, v_col, sum_v_inv, consolidate, *, lr, alpha, beta1, drift_cancel_C,
                alpha_v_fast, eps, step_cap, v_scale, precond_p, gf_consol,
                gf_trust_delta_sq, wd_sv, wd_sf, mantissa_bias, step_salt,
                chase_floor, leak_floor, gate_gain=1.0, v_bc=1.0):
    N, K = packed_w.shape; dev = packed_w.device
    t = lambda v, dt=torch.float32: torch.tensor([v], dtype=dt, device=dev)
    cons_t = consolidate if isinstance(consolidate, torch.Tensor) else torch.tensor([int(consolidate)], dtype=torch.int32, device=dev)
    BN, BK = 32, 64
    grid = (triton.cdiv(N, BN), triton.cdiv(K, BK))
    _apply_gated_kernel[grid](
        packed_w, grad_W, weight_buf, row_exp, col_exp, row_max, col_max,
        v_row, v_col, sum_v_inv, cons_t, N, K, t(lr), int(mantissa_bias), float(alpha),
        float(beta1), float(drift_cancel_C), float(alpha_v_fast), t(eps), float(step_cap),
        float(v_scale), float(precond_p), float(gf_consol), float(gf_trust_delta_sq),
        float(gate_gain), float(wd_sv), float(wd_sf), t(chase_floor), t(leak_floor),
        t(v_bc), t(step_salt, torch.int32),
        packed_w.stride(0), packed_w.stride(1), grad_W.stride(0), grad_W.stride(1),
        weight_buf.stride(0), weight_buf.stride(1), BLOCK_N=BN, BLOCK_K=BK)


COMMON = dict(lr=1e-3, alpha=0.1, beta1=0.0, drift_cancel_C=ref.compute_drift_cancel_C(0.1, 0.001),
              alpha_v_fast=0.001, eps=1e-10, step_cap=10.0, v_scale=0.0, precond_p=0.5,
              gf_consol=50.0, gf_trust_delta_sq=1.0, wd_sv=0.0, wd_sf=0.0, mantissa_bias=15,
              chase_floor=0.9, leak_floor=0.999)


def t1_flag1_matches_today(out_f=1280, in_f=1280):
    print("=== T1: gated(flag=1) vs ungated active apply (must be bit-exact) ===")
    gW = (torch.randn(out_f, in_f, device='cuda') * 0.02).to(torch.bfloat16)
    salt = 7777
    A = fap.build_layer(out_f, in_f, seed=0)   # ungated reference
    B = fap.build_layer(out_f, in_f, seed=0)   # gated, flag=1
    wbA = torch.zeros_like(gW); wbB = torch.zeros_like(gW)
    z = lambda n: torch.zeros(n, dtype=torch.int32, device='cuda')
    rmA, cmA, rmB, cmB = z(out_f), z(in_f), z(out_f), z(in_f)
    fap.apply_active(A.packed_w, gW, wbA, A.row_exp, A.col_exp, rmA, cmA, A.v_row, A.v_col,
                     A._sum_v_inv, step_salt=salt, **COMMON)
    apply_gated(B.packed_w, gW, wbB, B.row_exp, B.col_exp, rmB, cmB, B.v_row, B.v_col,
                B._sum_v_inv, 1, step_salt=salt, **COMMON)
    torch.cuda.synchronize()
    pk = (A.packed_w == B.packed_w).float().mean().item()
    wb = (wbA.float() - wbB.float()).abs().max().item()
    print(f"   packed agreement: {pk:.6f}   weight_buf max|d|: {wb:.2e}   -> {'PASS' if pk==1.0 and wb==0 else 'FAIL'}")


def t2_accumulation_equivalence(out_f=1280, in_f=1280, N=2):
    print(f"\n=== T2: {N}-step accum (tick*{N-1}+consolidate) vs single apply on summed grad ===")
    # micro-batch grads already 1/N-scaled (mimics loss/=N). Their sum = the averaged full grad.
    torch.manual_seed(3)
    gmicro = [(torch.randn(out_f, in_f, device='cuda') * 0.02 / N).to(torch.bfloat16) for _ in range(N)]
    gsum = torch.stack([g.float() for g in gmicro]).sum(0).to(torch.bfloat16)
    salt = 5555
    R = fap.build_layer(out_f, in_f, seed=0)   # reference: single apply on gsum
    Acc = fap.build_layer(out_f, in_f, seed=0) # accumulation path
    wbR = torch.zeros_like(gsum); wbAcc = torch.zeros_like(gsum)
    z = lambda n: torch.zeros(n, dtype=torch.int32, device='cuda')
    rmR, cmR, rmA, cmA = z(out_f), z(in_f), z(out_f), z(in_f)
    apply_gated(R.packed_w, gsum, wbR, R.row_exp, R.col_exp, rmR, cmR, R.v_row, R.v_col,
                R._sum_v_inv, 1, step_salt=salt, **COMMON)
    for i, g in enumerate(gmicro):
        apply_gated(Acc.packed_w, g, wbAcc, Acc.row_exp, Acc.col_exp, rmA, cmA, Acc.v_row,
                    Acc.v_col, Acc._sum_v_inv, 1 if i == N - 1 else 0, step_salt=salt + i, **COMMON)
    torch.cuda.synchronize()
    dw = wbR.float() - wbAcc.float()
    base = wbR.float().abs()
    print(f"   weight_buf  mean|ref|={base.mean():.4e}  mean|d|={dw.abs().mean():.4e}  "
          f"rel(meand/mean|ref|)={dw.abs().mean()/base.mean().clamp(min=1e-12):.3%}")
    print(f"   bias check  mean(d)={dw.mean():.4e} (should be ~0 -> unbiased)")
    print(f"   correlation ref vs acc: {torch.corrcoef(torch.stack([wbR.float().flatten(), wbAcc.float().flatten()]))[0,1]:.5f}")


if __name__ == "__main__":
    torch.cuda.init()
    t1_flag1_matches_today()
    t2_accumulation_equivalence(N=2)
    t2_accumulation_equivalence(N=4)
