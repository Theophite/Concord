# Source Generated with Decompyle++
# File: prototype_packed_e.cpython-311.pyc (Python 3.11)

'''Prototype C: packed int32 with int16 s_fast + int16 position (byte-concat,
shared sign).

Builds on packed-B but exploits the empirical observation that v_slow and
s_slow are same-sign in practice (their magnitudes differ by <0.1% at
80-epoch equilibrium). Collapses v_slow + s_slow into a single signed
int16 "position":
  - High byte of position = "slow / anchor" role (changes only on carries
    from the low byte, naturally lagging at ~1/256 of low-byte updates)
  - Low byte of position = "recent fine" role (changes per chase tick)
  - Sign is shared (it\'s the int16\'s sign bit)

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
'''
import sys
import time
import torch
from torch.nn import nn

functional
import triton = import torch.nn.functional, nn
from triton.language import language as tl
from triton.language.extra import libdevice
MANTISSA_BIAS = 15
(INT8_MIN, INT8_MAX) = (-128, 127)
(INT16_MIN, INT16_MAX) = (-32768, 32767)
S_SLOW_FACTOR = 128
V_SLOW_FACTOR = 128
_hash_uniform = (lambda x, pos, salt: h = x ^ salt ^ posh = h ^ h << 13h = h ^ h >> 17h = h ^ h << 5h = h ^ h >> 7(h & 16777215).to(tl.float32) * 5.96046e-08)()

def _torch_spread_bits(x):
    '''CPU/torch version of _spread_bits — same Morton encoding, on int32
    tensors. Used by load_weights / get_weight for the bit interleave.'''
    x = x & 255
    x = (x | x << 4) & 3855
    x = (x | x << 2) & 13107
    x = (x | x << 1) & 21845
    return x


def _torch_gather_even_bits(x):
    '''CPU/torch version of _gather_even_bits — inverse of _torch_spread_bits.'''
    x = x & 21845
    x = (x | x >> 1) & 13107
    x = (x | x >> 2) & 3855
    x = (x | x >> 4) & 255
    return x

_spread_bits = (lambda x: x = x & 255x = (x | x << 4) & 3855x = (x | x << 2) & 13107x = (x | x << 1) & 21845x)()
_gather_even_bits = (lambda x: x = x & 21845x = (x | x >> 1) & 13107x = (x | x >> 2) & 3855x = (x | x >> 4) & 255x)()
_interleave = (lambda s_i8, v_i8: spread_s = _spread_bits(s_i8)spread_v = _spread_bits(v_i8)combined_unsigned = spread_s | spread_v << 1combined_unsigned << 16 >> 16)()
_deinterleave = (lambda combined_i32: unsigned16 = combined_i32 & 65535s_unsigned = _gather_even_bits(unsigned16)v_unsigned = _gather_even_bits(unsigned16 >> 1)s_signed = s_unsigned << 24 >> 24v_signed = v_unsigned << 24 >> 24(s_signed, v_signed))()
_materialize_packed_bf16_kernel = (lambda packed_ptr, weight_ptr, row_exp_ptr, col_exp_ptr, step_salt_ptr, N, K, mantissa_bias, stride_pn, stride_pk, stride_wn = triton.jit, stride_wk = triton.jit, BLOCK_N = triton.jit, BLOCK_K = ('BLOCK_N', tl.constexpr, 'BLOCK_K', tl.constexpr): pid_n = tl.program_id(0)pid_k = tl.program_id(1)step_salt = tl.load(step_salt_ptr).to(tl.int32)offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)n_mask = offs_n < Nk_mask = offs_k < Knk_mask = n_mask[(:, None)] & k_mask[(None, :)]p_off = offs_n[(:, None)] * stride_pn + offs_k[(None, :)] * stride_pkpacked = tl.load(packed_ptr + p_off, mask = nk_mask, other = 0).to(tl.int32)s_fast = packed >> 16s_slow_i8 = packed << 16 >> 24v_slow_i8 = packed << 24 >> 24combined = _interleave(s_slow_i8 & 255, v_slow_i8 & 255)m_eff = (combined << 1) + s_fastrow_e = tl.load(row_exp_ptr + offs_n, mask = n_mask, other = 0).to(tl.int32)col_e = tl.load(col_exp_ptr + offs_k, mask = k_mask, other = 0).to(tl.int32)exp = (row_e[(:, None)] + col_e[(None, :)] - mantissa_bias).to(tl.float32)weight_fp32 = m_eff.to(tl.float32) * tl.exp2(exp)fp32_bits = weight_fp32.to(tl.int32, bitcast = True)pos_hash = offs_n[(:, None)] << 16 ^ offs_k[(None, :)]r_u32 = pos_hash ^ step_salt ^ pos_hash << 13 ^ (pos_hash ^ step_salt) >> 17r_u32 = r_u32 ^ r_u32 << 5r_u32 = r_u32 ^ r_u32 >> 7dither = r_u32 & 65535biased_bits = fp32_bits + ditherbf16_bits = biased_bits & -65536weight_sr = bf16_bits.to(tl.float32, bitcast = True).to(tl.bfloat16)w_off = offs_n[(:, None)] * stride_wn + offs_k[(None, :)] * stride_wktl.store(weight_ptr + w_off, weight_sr, mask = nk_mask))()
_STEP_COUNTERS = { }

def _get_step_counter(device):
    key = str(device)
    if key not in _STEP_COUNTERS:
        _STEP_COUNTERS[key] = torch.zeros(1, dtype = torch.int32, device = device)
    return _STEP_COUNTERS[key]


def materialize_packed_bf16(packed_w, row_exp, col_exp, out, mantissa_bias = (15,)):
    (N, K) = packed_w.shape
# WARNING: Decompyle incomplete

_apply_packed_sgd_kernel = (lambda packed_ptr, grad_W_ptr, row_exp_ptr, col_exp_ptr, row_max_ptr, col_max_ptr, N, K, lr, mantissa_bias, alpha, beta1, alpha_v_fast, step_salt_ptr, stride_pn, stride_pk, stride_gn = triton.jit, stride_gk = triton.jit, BLOCK_N = triton.jit, BLOCK_K = ('BLOCK_N', tl.constexpr, 'BLOCK_K', tl.constexpr): pid_n = tl.program_id(0)pid_k = tl.program_id(1)step_salt = tl.load(step_salt_ptr).to(tl.int32)offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)n_mask = offs_n < Nk_mask = offs_k < Knk_mask = n_mask[(:, None)] & k_mask[(None, :)]p_off = offs_n[(:, None)] * stride_pn + offs_k[(None, :)] * stride_pkpacked = tl.load(packed_ptr + p_off, mask = nk_mask, other = 0).to(tl.int32)s_fast = packed >> 16s_slow_i8 = packed << 16 >> 24v_slow_i8 = packed << 24 >> 24combined = _interleave(s_slow_i8 & 255, v_slow_i8 & 255)g_off = offs_n[(:, None)] * stride_gn + offs_k[(None, :)] * stride_gkgrad_W = tl.load(grad_W_ptr + g_off, mask = nk_mask, other = 0).to(tl.float32)row_e = tl.load(row_exp_ptr + offs_n, mask = n_mask, other = 0).to(tl.int32)col_e = tl.load(col_exp_ptr + offs_k, mask = k_mask, other = 0).to(tl.int32)total_exp = (row_e[(:, None)] + col_e[(None, :)] - mantissa_bias).to(tl.float32)scale_inv = tl.exp2(-total_exp)delta_grad = -lr * grad_W * scale_invdelta_t = delta_grad - beta1 * s_fast.to(tl.float32)pos_hash = offs_n[(:, None)] << 16 ^ offs_k[(None, :)]r1 = _hash_uniform(s_fast, pos_hash, step_salt)floor_t = tl.floor(delta_t)frac_t = delta_t - floor_ttick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)s_fast = s_fast + tick_fastchase_mantissa = alpha * s_fast.to(tl.float32)chase_combined_f = chase_mantissa * 0.5r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 1515870810)floor_c = tl.floor(chase_combined_f)frac_c = chase_combined_f - floor_ctick_combined = (floor_c + (r2 < frac_c).to(tl.float32)).to(tl.int32)combined = combined + tick_combineds_fast = s_fast - (tick_combined << 1)s_fast_c = tl.minimum(tl.maximum(s_fast, -32768), 32767)combined_c = tl.minimum(tl.maximum(combined, -32768), 32767)(s_slow_new, v_slow_new) = _deinterleave(combined_c)packed_new = (s_fast_c & 65535) << 16 | (s_slow_new & 255) << 8 | v_slow_new & 255tl.store(packed_ptr + p_off, packed_new, mask = nk_mask)abs_eff = tl.abs((combined_c << 1) + s_fast_c)abs_eff = tl.where(nk_mask, abs_eff, 0)tile_row_max = tl.max(abs_eff, axis = 1)tile_col_max = tl.max(abs_eff, axis = 0)tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask = n_mask)tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask = k_mask))()

def apply_packed_sgd(packed_w, grad_W, row_exp, col_exp, row_max, col_max, lr, mantissa_bias, alpha, beta1, alpha_v_fast = (15, 0.1, 0, 0.001)):
    (N, K) = packed_w.shape
# WARNING: Decompyle incomplete

_apply_packed_adamw_kernel = (lambda packed_ptr, grad_W_ptr, row_exp_ptr, col_exp_ptr, row_max_ptr, col_max_ptr, N, K, lr, mantissa_bias, alpha, beta1, weight_decay, eps, step_cap, v_scale, drift_cancel_C, alpha_v_fast, wd_sv, wd_sf, step_salt_ptr, stride_pn, stride_pk, stride_gn, stride_gk, BLOCK_N = None, BLOCK_K = triton.jit, MASS_PRESERVE = triton.jit, APPLY_CHASE = ('BLOCK_N', tl.constexpr, 'BLOCK_K', tl.constexpr, 'MASS_PRESERVE', tl.constexpr, 'APPLY_CHASE', tl.constexpr): pid_n = tl.program_id(0)pid_k = tl.program_id(1)step_salt = tl.load(step_salt_ptr).to(tl.int32)offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)n_mask = offs_n < Nk_mask = offs_k < Knk_mask = n_mask[(:, None)] & k_mask[(None, :)]p_off = offs_n[(:, None)] * stride_pn + offs_k[(None, :)] * stride_pkpacked = tl.load(packed_ptr + p_off, mask = nk_mask, other = 0).to(tl.int32)s_fast = packed >> 16s_slow_i8 = packed << 16 >> 24v_slow_i8 = packed << 24 >> 24combined = _interleave(s_slow_i8 & 255, v_slow_i8 & 255)position_doubled = combined << 1g_off = offs_n[(:, None)] * stride_gn + offs_k[(None, :)] * stride_gkgrad_W = tl.load(grad_W_ptr + g_off, mask = nk_mask, other = 0).to(tl.float32)row_e = tl.load(row_exp_ptr + offs_n, mask = n_mask, other = 0).to(tl.int32)col_e = tl.load(col_exp_ptr + offs_k, mask = k_mask, other = 0).to(tl.int32)total_exp = (row_e[(:, None)] + col_e[(None, :)] - mantissa_bias).to(tl.float32)scale_fwd = tl.exp2(total_exp)scale_inv = tl.exp2(-total_exp)m_eff = position_doubled + s_fastcurrent_weight = m_eff.to(tl.float32) * scale_fwdd_fs = s_fast.to(tl.float32)d_sv = v_slow_i8.to(tl.float32)noise = d_fs - drift_cancel_C * d_svnoise_in_w = noise * scale_fwdv_proxy = noise_in_w * noise_in_w * v_scalestep_live = grad_W / tl.sqrt(v_proxy + eps) + weight_decay * current_weightstep_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)delta_grad = -lr * step_live * scale_invdelta_t = delta_grad - beta1 * d_fspos_hash = offs_n[(:, None)] << 16 ^ offs_k[(None, :)]r1 = _hash_uniform(s_fast, pos_hash, step_salt)floor_t = tl.floor(delta_t)frac_t = delta_t - floor_ttick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)s_fast = s_fast + tick_fastif APPLY_CHASE:
chase_mantissa = alpha * s_fast.to(tl.float32)chase_combined_f = chase_mantissa * 0.5r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 1515870810)floor_c = tl.floor(chase_combined_f)frac_c = chase_combined_f - floor_ctick_combined = (floor_c + (r2 < frac_c).to(tl.float32)).to(tl.int32)combined = combined + tick_combineds_fast = s_fast - (tick_combined << 1)s_fast_c = tl.minimum(tl.maximum(s_fast, -32768), 32767)combined_c = tl.minimum(tl.maximum(combined, -32768), 32767)(s_slow_new, v_slow_new) = _deinterleave(combined_c)packed_new = (s_fast_c & 65535) << 16 | (s_slow_new & 255) << 8 | v_slow_new & 255tl.store(packed_ptr + p_off, packed_new, mask = nk_mask)abs_eff = tl.abs((combined_c << 1) + s_fast_c)abs_eff = tl.where(nk_mask, abs_eff, 0)tile_row_max = tl.max(abs_eff, axis = 1)tile_col_max = tl.max(abs_eff, axis = 0)tl.atomic_max(row_max_ptr + offs_n, tile_row_max, mask = n_mask)tl.atomic_max(col_max_ptr + offs_k, tile_col_max, mask = k_mask))()

def apply_packed_adamw(packed_w, grad_W, row_exp, col_exp, row_max, col_max, lr, mantissa_bias, alpha, beta1, weight_decay, eps, step_cap, v_scale, drift_cancel_C, alpha_v_fast, wd_sv, wd_sf, mass_preserve, apply_chase = (15, 0.1, 0, 0, 1, 10, 1, 0.1, 0.001, 0, 0, False, True)):
    '''Wrapper for the AdamW three-accumulator packed apply kernel.'''
    (N, K) = packed_w.shape
# WARNING: Decompyle incomplete

_rebalance_packed_decide_kernel = (lambda packed_ptr, row_exp_ptr, col_exp_ptr, row_max_ptr, col_max_ptr, N, K, MAX_M, EXP_MAX, seed, stride_pn = None, stride_pk = None, BLOCK_N = triton.jit, BLOCK_K = ('BLOCK_N', tl.constexpr, 'BLOCK_K', tl.constexpr): pid_n = tl.program_id(0)pid_k = tl.program_id(1)offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)n_mask = offs_n < Nk_mask = offs_k < Knk_mask = n_mask[(:, None)] & k_mask[(None, :)]row_max = tl.load(row_max_ptr + offs_n, mask = n_mask, other = 0)col_max = tl.load(col_max_ptr + offs_k, mask = k_mask, other = 0)row_exp = tl.load(row_exp_ptr + offs_n, mask = n_mask, other = 0)col_exp = tl.load(col_exp_ptr + offs_k, mask = k_mask, other = 0)row_up = (row_max > MAX_M) & (row_exp < EXP_MAX)col_up = (col_max > MAX_M) & (col_exp < EXP_MAX)row_t = row_up.to(tl.int32)col_t = col_up.to(tl.int32)pos = row_t[(:, None)] + col_t[(None, :)]p_off = offs_n[(:, None)] * stride_pn + offs_k[(None, :)] * stride_pkpacked = tl.load(packed_ptr + p_off, mask = nk_mask, other = 0).to(tl.int32)s_fast = packed >> 16s_slow_i8 = packed << 16 >> 24v_slow_i8 = packed << 24 >> 24combined = _interleave(s_slow_i8 & 255, v_slow_i8 & 255)rand_off = offs_n[(:, None)] * K + offs_k[(None, :)]two_pos = tl.exp2(pos.to(tl.float32))q_fast = s_fast >> posrem_fast = (s_fast - (q_fast << pos)).to(tl.float32)up_fast = (tl.rand(seed, rand_off) * two_pos < rem_fast).to(tl.int32)s_fast_new = q_fast + up_fastq_comb = combined >> posrem_comb = (combined - (q_comb << pos)).to(tl.float32)up_comb = (tl.rand(seed, rand_off + N * K) * two_pos < rem_comb).to(tl.int32)combined_new = q_comb + up_combs_fast_c = tl.minimum(tl.maximum(s_fast_new, -32768), 32767)combined_c = tl.minimum(tl.maximum(combined_new, -32768), 32767)(s_slow_new, v_slow_new) = _deinterleave(combined_c)packed_new = (s_fast_c & 65535) << 16 | (s_slow_new & 255) << 8 | v_slow_new & 255tl.store(packed_ptr + p_off, packed_new, mask = nk_mask)if pid_k == 0:
tl.store(row_exp_ptr + offs_n, row_exp + row_t, mask = n_mask)if pid_n == 0:
tl.store(col_exp_ptr + offs_k, col_exp + col_t, mask = k_mask)None)()

def rebalance_packed(packed_w, row_exp, col_exp, row_max, col_max, MAX_M, EXP_MAX, seed = (24000, 7, 0)):
    '''Apply the rebalance ratchet (tick-up exponent if row/col max
    exceeds threshold). Assumes row_max/col_max already populated by a
    prior apply kernel.'''
    (N, K) = packed_w.shape
    (BLOCK_N, BLOCK_K) = (32, 64)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _rebalance_packed_decide_kernel[grid](packed_w, row_exp, col_exp, row_max, col_max, N, K, int(MAX_M), int(EXP_MAX), int(seed), packed_w.stride(0), packed_w.stride(1), BLOCK_N = BLOCK_N, BLOCK_K = BLOCK_K)


class FusedConcordLinearPackedB(torch.autograd.Function):
    '''Dispatches to SGD or AdamW apply kernel based on optimizer_kind.'''
    forward = (lambda ctx, x, packed_w, row_exp, col_exp, bias, lr, alpha, beta1, mantissa_bias, optimizer_kind, weight_decay, eps, step_cap, v_scale, drift_cancel_C, alpha_v_fast, wd_sv, wd_sf, mass_preserve, apply_chase, weight_buf, row_max_buf, col_max_buf: materialize_packed_bf16(packed_w, row_exp, col_exp, out = weight_buf, mantissa_bias = mantissa_bias)# WARNING: Decompyle incomplete
)()
    backward = (lambda ctx, grad_y: (x, weight) = ctx.saved_tensorsif grad_y.dtype != torch.bfloat16:
grad_y = grad_y.to(torch.bfloat16)if not grad_y.is_contiguous():
grad_y = grad_y.contiguous()grad_x = grad_y @ weightgrad_W = grad_y.transpose(-1, -2) @ xif grad_W.dtype != torch.bfloat16:
grad_W = grad_W.to(torch.bfloat16)if not grad_W.is_contiguous():
grad_W = grad_W.contiguous()ctx.row_max_buf.zero_()ctx.col_max_buf.zero_()# WARNING: Decompyle incomplete
)()


class ConcordLinearPackedB(nn.Module):
    pass
# WARNING: Decompyle incomplete


class FusedConcordConv2dPackedB(torch.autograd.Function):
    '''Conv2d that uses cuDNN for forward + grad_x + grad_W, and the
    packed apply kernel for the state update. Materializes a bf16
    weight from packed int32 once per forward, freed after backward.'''
    forward = (lambda ctx, x, packed_w, row_exp, col_exp, bias, in_channels, out_channels, kh, kw, stride, padding, lr, alpha, beta1, mantissa_bias, optimizer_kind, weight_decay, eps, step_cap, v_scale, drift_cancel_C, alpha_v_fast, wd_sv, wd_sf, mass_preserve, apply_chase, weight_buf, row_max_buf, col_max_buf: materialize_packed_bf16(packed_w, row_exp, col_exp, out = weight_buf, mantissa_bias = mantissa_bias)weight_4d = weight_buf.view(out_channels, in_channels, kh, kw)x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else xif not x_bf16.is_contiguous():
x_bf16 = x_bf16.contiguous()# WARNING: Decompyle incomplete
)()
    backward = (lambda ctx, grad_y: (x_bf16, weight_4d) = ctx.saved_tensorsif grad_y.dtype != torch.bfloat16:
grad_y = grad_y.to(torch.bfloat16)if not grad_y.is_contiguous():
grad_y = grad_y.contiguous()grad_x = torch.nn.grad.conv2d_input(x_bf16.shape, weight_4d, grad_y, stride = ctx.stride, padding = ctx.padding)grad_W_4d = torch.nn.grad.conv2d_weight(x_bf16, (ctx.out_channels, ctx.in_channels, ctx.kh, ctx.kw), grad_y, stride = ctx.stride, padding = ctx.padding)grad_W_2d = grad_W_4d.reshape(ctx.out_channels, -1).contiguous()if grad_W_2d.dtype != torch.bfloat16:
grad_W_2d = grad_W_2d.to(torch.bfloat16)ctx.row_max_buf.zero_()ctx.col_max_buf.zero_()# WARNING: Decompyle incomplete
)()


class ConcordConv2dPackedB(ConcordLinearPackedB):
    pass
# WARNING: Decompyle incomplete


def _run_one(model_factory, lr, n_steps, bsz = (500, 32)):
    pass
# WARNING: Decompyle incomplete


def _packed_b_mlp(in_f, hid, out_f, lr, device):
    return nn.Sequential(ConcordLinearPackedB(in_f, hid, device = device, lr = lr, alpha = 0.1), nn.ReLU(), ConcordLinearPackedB(hid, out_f, device = device, lr = lr, alpha = 0.1))


def _packed_b_mlp_adamw(in_f, hid, out_f, lr, device):
    m = nn.Sequential(ConcordLinearPackedB(in_f, hid, device = device, lr = lr, alpha = 0.1), nn.ReLU(), ConcordLinearPackedB(hid, out_f, device = device, lr = lr, alpha = 0.1))
    for layer in m:
        if isinstance(layer, ConcordLinearPackedB):
            layer.set_optimizer_kind('adamw', weight_decay = 0.01, eps = 1, step_cap = 10)
            layer.wd_sv = 1e-05
            layer.wd_sf = 1e-05
        return m


def _torch_mlp_adamw(in_f, hid, out_f, lr, device):
    model = nn.Sequential(nn.Linear(in_f, hid, device = device), nn.ReLU(), nn.Linear(hid, out_f, device = device))
    model._torch_opt = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.01)
    return model


def _torch_mlp(in_f, hid, out_f, lr, device):
    model = nn.Sequential(nn.Linear(in_f, hid, device = device), nn.ReLU(), nn.Linear(hid, out_f, device = device))
    model._torch_opt = torch.optim.SGD(model.parameters(), lr = lr)
    return model


def diagnose():
    '''Trace single-layer + MLP dynamics.'''
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    layer = ConcordLinearPackedB(32, 16, device = device, lr = 0.05, alpha = 0.1)
    w_init = layer.get_weight()
    print(f'''[diag] init weight: |w|max={w_init.abs().max().item():.4f} |w|mean={w_init.float().abs().mean().item():.4f}''')
    (s_fast0, s_slow0, _) = layer.get_state()
    print(f'''[diag] init state: s_fast|max={s_fast0.abs().max().item()} s_slow_i8|max={s_slow0.abs().max().item()}''')
    x = torch.randn(16, 32, device = device, dtype = torch.bfloat16)
    y_target = torch.randn(16, 16, device = device, dtype = torch.bfloat16)
    for step in range(20):
        pred = layer(x)
        loss = F.mse_loss(pred.float(), y_target.float())
        loss.backward()
        (s_fast, s_slow_i8, _) = layer.get_state()
        if step < 5 or step % 5 == 0:
            print(f'''[diag] step {step:2d}: loss={loss.item():.4f}  |s_fast|max={s_fast.abs().max().item():5d}  |s_slow_i8|max={s_slow_i8.abs().max().item():3d}  |w|max={layer.get_weight().abs().max().item():.4f}''')
        return None


def smoke_test():
    print('=== Storage roundtrip ===')
    device = 'cuda'
    torch.manual_seed(0)
    layer = ConcordLinearPackedB(8, 4, device = device)
    w = layer.get_weight()
    print(f'''[OK] get_weight works: shape={tuple(w.shape)} dtype={w.dtype} |w|max={w.abs().max().item():.4f}''')
    (wbuf, _, _) = layer._ensure_buffers()
    materialize_packed_bf16(layer.packed_w, layer.row_exp, layer.col_exp, wbuf, mantissa_bias = layer.MANTISSA_BIAS)
    diff = (w.float() - wbuf.float()).abs().max().item()
    print(f'''[OK] kernel materialize matches CPU recon: max_diff={diff:.6f}''')
    print()
    print('=== Single-layer dynamics ===')
    diagnose()
    print()
    print('=== MLP smoke (sweep over lr) ===')
    results = { }
    for tag, factory, lr in (('packed-B SGD lr=0.1', _packed_b_mlp, 0.1), ('packed-B SGD lr=0.05', _packed_b_mlp, 0.05), ('packed-B SGD lr=0.01', _packed_b_mlp, 0.01), ('packed-B AdamW lr=0.05', _packed_b_mlp_adamw, 0.05), ('packed-B AdamW lr=0.01', _packed_b_mlp_adamw, 0.01), ('packed-B AdamW lr=0.001', _packed_b_mlp_adamw, 0.001), ('SGD baseline lr=0.05', _torch_mlp, 0.05), ('SGD baseline lr=0.01', _torch_mlp, 0.01), ('AdamW baseline lr=0.01', _torch_mlp_adamw, 0.01), ('AdamW baseline lr=0.001', _torch_mlp_adamw, 0.001)):
        L = _run_one(factory, lr = lr)
        nans = (lambda .0: pass# WARNING: Decompyle incomplete
)(L())
        ratio = L[0] / max(L[-1], 1e-30)
        print(f'''[smoke] {tag:>22}  init={L[0]:6.3f}  s100={L[100]:7.3f}  s500={L[-1]:7.3f}  ratio={ratio:6.1f}x''' + f'''  [NaN x{nans}]''' if nans else '')
        results[tag] = (L, ratio, nans)
        print()
        pkdB_sgd = (lambda .0: pass# WARNING: Decompyle incomplete
)(results.items()(), default = 0)
        base_sgd = (lambda .0: pass# WARNING: Decompyle incomplete
)(results.items()(), default = 0)
        pkdB_adam = (lambda .0: pass# WARNING: Decompyle incomplete
)(results.items()(), default = 0)
        base_adam = (lambda .0: pass# WARNING: Decompyle incomplete
)(results.items()(), default = 0)
        print(f'''[smoke] packed-B SGD best:    {pkdB_sgd:.1f}x   (vs torch.SGD baseline {base_sgd:.1f}x)''')
        print(f'''[smoke] packed-B AdamW best:  {pkdB_adam:.1f}x   (vs torch.AdamW baseline {base_adam:.1f}x)''')
        print()
        sgd_ok = pkdB_sgd >= base_sgd * 0.7
        adam_works = pkdB_adam > 1.5
        if sgd_ok and adam_works:
            print('[PASS] packed-B SGD >= 70% of torch.SGD, AdamW path converges')
            return True
        if not max:
            print(f'''[FAIL] packed-B SGD ratio too low ({pkdB_sgd:.1f}x vs baseline {base_sgd:.1f}x)''')
            return False
        None('[FAIL] packed-B AdamW path failed to converge')
        return False


def main():
    if not torch.cuda.is_available():
        print('CUDA not available, skipping')
        return None
    ok = None()
    print()
    if ok:
        print('=== PROTOTYPE B WORKS ===')
        return None
    None('=== PROTOTYPE B FAILED ===')
    sys.exit(1)

if __name__ == '__main__':
    main()
    return None
