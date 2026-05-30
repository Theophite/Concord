"""Prototype: two-accumulator packed int32 (s_fast i16 + s_slow i16).

Storage layout per param (32 bits):
    bits [31:16]  s_fast   int16   — SR-tick accumulator (velocity)
    bits [15:0]   s_slow   int16   — position carrier

Live weight:
    m_eff = s_slow + s_fast
    weight = m_eff × 2^(row_exp + col_exp - MANTISSA_BIAS)

Versus packed-B (s_fast i16 + s_slow_i8 × 128 + v_slow_i8 × 128):
  + s_slow has 1-mantissa resolution (vs 128-mantissa). At typical
    T5 weight scale, each slow tick = ~2e-6 weight change instead
    of ~2.5e-4. ~128× finer — critical for fine-tune where each
    AdamW-equivalent step is ~1e-4 weight change.
  + No v_slow drift issue: there's no third accumulator that
    eventually loses its anchor.
  − No long-time Bayesian anchor: if you need one (e.g. for
    pretrained anchoring with `wd_sv`), use packed-B with
    α_v_fast = 0 instead.

Init pattern: full mantissa lands in s_slow; s_fast starts at 0.
For pretrained weights this is the "finetune steady-state" — d_fs = 0
and the chase starts from rest.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


MANTISSA_BIAS = 15
INT16_MIN, INT16_MAX = -32768, 32767


# ============================================================
# Triton kernels
# ============================================================

@triton.jit
def _hash_uniform(x, pos, salt):
    h = x ^ salt ^ pos
    h = h ^ (h << 13)
    h = h ^ (h >> 17)
    h = h ^ (h << 5)
    h = h ^ (h >> 7)
    return (h & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _materialize_2acc_bf16_kernel(
    packed_ptr, weight_ptr,
    row_exp_ptr, col_exp_ptr,
    N, K, mantissa_bias,
    stride_pn, stride_pk, stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MAG_WEIGHTED: tl.constexpr,
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
    s_fast = packed >> 16             # arith shift sign-extends
    s_slow = (packed << 16) >> 16     # sign-extend low int16

    if MAG_WEIGHTED:
        # Forward weight is a magnitude-weighted blend of s_fast and s_slow:
        #   (|s_fast|·s_fast + |s_slow|·s_slow) / (|s_fast| + |s_slow|)
        # Whichever accumulator has larger magnitude dominates the
        # blend. Stored s_fast + s_slow stays as the true position
        # (chase still mass-preserves m_eff = s_fast + s_slow).
        sf_f = s_fast.to(tl.float32)
        ss_f = s_slow.to(tl.float32)
        abs_sf = tl.abs(sf_f)
        abs_ss = tl.abs(ss_f)
        m_eff_f = (sf_f * abs_sf + ss_f * abs_ss) / (abs_sf + abs_ss + 1.0)
    else:
        m_eff = s_slow + s_fast
        m_eff_f = m_eff.to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    weight_fp32 = m_eff_f * tl.exp2(exp)

    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_ptr + w_off, weight_fp32.to(tl.bfloat16), mask=nk_mask)


def materialize_2acc_bf16(packed_w, row_exp, col_exp, out,
                            mantissa_bias=15, mag_weighted=False):
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.shape == packed_w.shape
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _materialize_2acc_bf16_kernel[grid](
        packed_w, out, row_exp, col_exp, N, K, int(mantissa_bias),
        packed_w.stride(0), packed_w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        MAG_WEIGHTED=bool(mag_weighted),
    )


@triton.jit
def _apply_2acc_kernel(
    packed_ptr,        # [N, K] int32, mutated in place
    grad_W_ptr,        # [N, K] bf16
    weight_buf_ptr,    # [N, K] bf16 — emit updated weight here
    row_exp_ptr,       # [N] int8
    col_exp_ptr,       # [K] int8
    N, K,
    lr_ptr, mantissa_bias, alpha, beta1, weight_decay, step_cap,
    step_salt_ptr,
    stride_pn, stride_pk,
    stride_gn, stride_gk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MAG_WEIGHTED: tl.constexpr,
):
    """Two-accumulator apply: SR-tick into s_fast, mass-preserve chase
    into s_slow, decoupled weight decay in s_fast (chase migrates).

    Both s_fast and s_slow are int16, so the chase has 1-mantissa
    resolution (vs packed-B's 128-mantissa quantization on s_slow).
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    step_salt = tl.load(step_salt_ptr).to(tl.int32)
    lr = tl.load(lr_ptr).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    # ── load packed + unpack ──
    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    s_fast = packed >> 16
    s_slow = (packed << 16) >> 16

    g_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + g_off, mask=nk_mask, other=0.0).to(tl.float32)

    row_e = tl.load(row_exp_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
    col_e = tl.load(col_exp_ptr + offs_k, mask=k_mask, other=0).to(tl.int32)
    total_exp = (row_e[:, None] + col_e[None, :] - mantissa_bias).to(tl.float32)
    scale_fwd = tl.exp2(total_exp)
    scale_inv = tl.exp2(-total_exp)

    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]

    # ── current weight + step ────────────────────────────────
    # Forward weight = f(s_fast, s_slow) * scale_fwd, where
    #   f(a,b) = a + b              (non-mag), or
    #   f(a,b) = (a|a| + b|b|) / (|a| + |b|)  (mag-weighted blend)
    # In MAG_WEIGHTED mode the gradient is projected onto s_fast via
    # the chain-rule Jacobian df_dsf. NOTE: df_dsf is *negative* near
    # the cusp at sf=0 (the mag-weighted blend has a local maximum of
    # f along the sf axis there), and the chase carries that signed
    # step into s_slow. The net dynamic is "follow the local gradient
    # of f", which — for pretrained init with sf≈0 — actively moves
    # the forward weight AWAY from the loss-minimizing direction.
    # In effect: an *unlearning* engine when fed targets pointing at
    # capabilities you want to ablate.
    if MAG_WEIGHTED:
        sf_f = s_fast.to(tl.float32)
        ss_f = s_slow.to(tl.float32)
        abs_sf = tl.abs(sf_f)
        abs_ss = tl.abs(ss_f)
        sign_sf = tl.where(sf_f >= 0, 1.0, -1.0)
        denom = abs_sf + abs_ss + 1.0
        f_val = (sf_f * abs_sf + ss_f * abs_ss) / denom
        df_dsf = (2.0 * abs_sf - f_val * sign_sf) / denom
        current_weight = f_val * scale_fwd
    else:
        f_val = (s_slow + s_fast).to(tl.float32)
        df_dsf = 1.0
        current_weight = f_val * scale_fwd

    # SGD-with-decoupled-wd:
    #   step_live = grad + wd * current_weight  (in weight units)
    step_live = grad_W
    step_live = tl.minimum(tl.maximum(step_live, -step_cap), step_cap)
    step_live = step_live + weight_decay * current_weight
    delta_grad = -lr * step_live * scale_inv * df_dsf  # mantissa units
    delta_t = delta_grad - beta1 * s_fast.to(tl.float32)

    # ── SR-tick s_fast ────────────────────────────────────────
    r1 = _hash_uniform(s_fast, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = (floor_t + (r1 < frac_t).to(tl.float32)).to(tl.int32)
    s_fast = s_fast + tick_fast

    # ── mass-preserve chase: α × s_fast → s_slow (at int16,
    # 1-mantissa resolution — no /128 quantization). ──
    chase_target = alpha * s_fast.to(tl.float32)
    r2 = _hash_uniform(s_fast, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_c = tl.floor(chase_target)
    frac_c = chase_target - floor_c
    tick_slow = (floor_c + (r2 < frac_c).to(tl.float32)).to(tl.int32)
    s_slow = s_slow + tick_slow
    s_fast = s_fast - tick_slow      # mass preserve

    # ── clamp and repack ──
    s_fast_c = tl.minimum(tl.maximum(s_fast, -32768), 32767)
    s_slow_c = tl.minimum(tl.maximum(s_slow, -32768), 32767)
    packed_new = ((s_fast_c & 0xFFFF) << 16) | (s_slow_c & 0xFFFF)
    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)

    # ── materialize-merge: emit the NEW bf16 weight ──
    # Optionally magnitude-weight the s_fast/s_slow blend so the forward
    # sees a value biased toward whichever accumulator has larger
    # magnitude. Stored m_eff = s_slow + s_fast still represents the
    # "true" position (chase mass-preserves it); this only changes what
    # cuBLAS sees in forward / what backward computes grad against.
    if MAG_WEIGHTED:
        sf_f = s_fast_c.to(tl.float32)
        ss_f = s_slow_c.to(tl.float32)
        abs_sf = tl.abs(sf_f)
        abs_ss = tl.abs(ss_f)
        new_m_eff_f = (sf_f * abs_sf + ss_f * abs_ss) / (abs_sf + abs_ss + 1.0)
    else:
        new_m_eff_f = (s_slow_c + s_fast_c).to(tl.float32)
    new_weight = new_m_eff_f * scale_fwd
    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_buf_ptr + w_off,
             new_weight.to(tl.bfloat16), mask=nk_mask)


_STEP_COUNTERS = {}


def _get_step_counter(device):
    key = str(device)
    if key not in _STEP_COUNTERS:
        _STEP_COUNTERS[key] = torch.zeros(1, dtype=torch.int32, device=device)
    return _STEP_COUNTERS[key]


_LR_SCALAR_CACHE = {}


def _ensure_lr_tensor(lr, device):
    if isinstance(lr, torch.Tensor):
        return lr
    key = str(device)
    buf = _LR_SCALAR_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _LR_SCALAR_CACHE[key] = buf
    buf.fill_(float(lr))
    return buf


@torch.no_grad()
def rebalance_2acc(packed_w, row_exp, threshold=24576, exp_max=7):
    """Shift-on-saturation: per row, if max(|s_fast|, |s_slow|) > threshold,
    tick row_exp up by 1 and arithmetic-shift both mantissas right by 1.
    Weight-preserving (mantissa /2 × scale ×2). Caps at exp_max so we
    don't tick past the int8 row_exp range we use elsewhere.

    Graph-capturable (no host-side branches; the per-row decision is a
    tensor mask under torch.where).
    """
    s_fast = packed_w >> 16             # arith shift sign-extends
    s_slow = (packed_w << 16) >> 16
    row_max = torch.maximum(s_fast.abs().max(dim=1).values,
                              s_slow.abs().max(dim=1).values)
    needs_tick = (row_max > threshold) & (row_exp.to(torch.int32) < exp_max)
    row_exp.add_(needs_tick.to(torch.int8))
    halve = needs_tick.unsqueeze(1)
    s_fast_new = torch.where(halve, s_fast >> 1, s_fast)
    s_slow_new = torch.where(halve, s_slow >> 1, s_slow)
    new_packed = ((s_fast_new & 0xFFFF) << 16) | (s_slow_new & 0xFFFF)
    packed_w.copy_(new_packed)


def apply_2acc(packed_w, grad_W, weight_buf, row_exp, col_exp,
                lr, mantissa_bias=15, alpha=0.1, beta1=0.0,
                weight_decay=0.0, step_cap=10.0, mag_weighted=False,
                rebalance=True, rebalance_threshold=24576):
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16
    assert weight_buf.dtype == torch.bfloat16
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    lr_ptr = _ensure_lr_tensor(lr, packed_w.device)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_2acc_kernel[grid](
        packed_w, grad_W, weight_buf, row_exp, col_exp,
        N, K,
        lr_ptr, int(mantissa_bias), float(alpha), float(beta1),
        float(weight_decay), float(step_cap),
        step_counter,
        packed_w.stride(0), packed_w.stride(1),
        grad_W.stride(0), grad_W.stride(1),
        weight_buf.stride(0), weight_buf.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        MAG_WEIGHTED=bool(mag_weighted),
    )
    if rebalance:
        # bf16 weight_buf was just emitted with PRE-rebalance scale; rebalance
        # preserves W exactly (modulo 1 bit/tick), so weight_buf stays
        # numerically valid for the next forward.
        rebalance_2acc(packed_w, row_exp,
                         threshold=int(rebalance_threshold), exp_max=7)


# ============================================================
# Autograd Function + nn.Module
# ============================================================

class FusedConcordLinear2acc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_w, row_exp, col_exp, bias,
                lr_buf, alpha, beta1, mantissa_bias,
                weight_decay, step_cap, mag_weighted,
                weight_buf):
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = F.linear(x, weight_buf, bias_bf16)
        ctx.save_for_backward(x, weight_buf)
        ctx.packed_w = packed_w
        ctx.row_exp = row_exp
        ctx.col_exp = col_exp
        ctx.lr_buf = lr_buf
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.mantissa_bias = mantissa_bias
        ctx.weight_decay = weight_decay
        ctx.step_cap = step_cap
        ctx.mag_weighted = mag_weighted
        ctx.has_bias = bias is not None
        ctx.weight_buf = weight_buf
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()
        grad_x = grad_y @ weight
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        x_flat = x.reshape(-1, in_features)
        grad_y_flat = grad_y.reshape(-1, out_features)
        grad_W = grad_y_flat.transpose(0, 1) @ x_flat
        if grad_W.dtype != torch.bfloat16:
            grad_W = grad_W.to(torch.bfloat16)
        if not grad_W.is_contiguous():
            grad_W = grad_W.contiguous()
        apply_2acc(
            ctx.packed_w, grad_W, ctx.weight_buf,
            ctx.row_exp, ctx.col_exp,
            lr=ctx.lr_buf, mantissa_bias=ctx.mantissa_bias,
            alpha=ctx.alpha, beta1=ctx.beta1,
            weight_decay=ctx.weight_decay,
            step_cap=ctx.step_cap, mag_weighted=ctx.mag_weighted)
        grad_bias = grad_y_flat.sum(0) if ctx.has_bias else None
        # 13 forward args; only x (0) and bias (4) receive grads.
        return (grad_x, None, None, None, grad_bias,
                None, None, None, None,
                None, None, None,
                None)


class ConcordLinear2acc(nn.Module):
    """Two-accumulator packed Linear: int16 s_fast || int16 s_slow."""

    MANTISSA_BIAS = 15
    EXP_MIN = -8
    EXP_MAX = 7

    def __init__(self, in_features, out_features, bias=True,
                 device='cuda', alpha=0.1, beta1=0.0, lr=0.01,
                 weight_decay=0.0, step_cap=10.0,
                 mag_weighted=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta1 = beta1
        self._lr_value = float(lr)
        self.weight_decay = float(weight_decay)
        self.step_cap = float(step_cap)
        self.mag_weighted = bool(mag_weighted)
        self.register_buffer('packed_w',
            torch.zeros(out_features, in_features,
                        dtype=torch.int32, device=device))
        self.register_buffer('row_exp',
            torch.zeros(out_features, dtype=torch.int8, device=device))
        self.register_buffer('col_exp',
            torch.zeros(in_features, dtype=torch.int8, device=device))
        self.register_buffer('_lr_buf',
            torch.full((1,), self._lr_value,
                       dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,
                                                  dtype=torch.bfloat16,
                                                  device=device))
        else:
            self.register_parameter('bias', None)
        self._init_weight()
        self._ensure_buffers()

    @property
    def weight(self):
        return getattr(self, '_bf16_weight_buf', None)

    @property
    def lr(self):
        return self._lr_value

    @lr.setter
    def lr(self, value):
        v = float(value)
        self._lr_value = v
        buf = getattr(self, '_lr_buf', None)
        if buf is not None:
            buf.fill_(v)

    def _init_weight(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        w = torch.randn(self.out_features, self.in_features,
                        device=self.packed_w.device) * std
        self.load_weights(w)

    @torch.no_grad()
    def load_weights(self, W):
        """Init: all mantissa lands in s_slow (the position carrier).
        s_fast = 0 at start. For fine-tune of pretrained weights this
        is the steady-state — d_fs = 0, chase starts from rest. For
        training-from-scratch (random init) it works just as well — the
        chase has nothing to migrate until gradient ticks arrive."""
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
        s_slow = m_total          # full mantissa in s_slow (int16)
        s_fast = torch.zeros_like(s_slow)
        packed = ((s_fast & 0xFFFF) << 16) | (s_slow & 0xFFFF)
        self.packed_w.copy_(packed)
        self._resync_weight_buf()

    @torch.no_grad()
    def _resync_weight_buf(self):
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if wbuf is not None:
            materialize_2acc_bf16(self.packed_w, self.row_exp,
                                    self.col_exp, out=wbuf,
                                    mantissa_bias=self.MANTISSA_BIAS,
                                    mag_weighted=self.mag_weighted)

    @torch.no_grad()
    def get_weight(self):
        s_fast = (self.packed_w >> 16)
        s_slow = ((self.packed_w << 16) >> 16)
        m_eff = s_slow + s_fast
        exp = (self.row_exp[:, None].to(torch.int32)
               + self.col_exp[None, :].to(torch.int32)
               - self.MANTISSA_BIAS).to(torch.float32)
        w_fp32 = m_eff.to(torch.float32) * torch.pow(2.0, exp)
        return w_fp32.to(torch.bfloat16)

    @torch.no_grad()
    def get_state(self):
        s_fast = (self.packed_w >> 16)
        s_slow = ((self.packed_w << 16) >> 16)
        return s_fast, s_slow

    def _ensure_buffers(self):
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if wbuf is None or wbuf.shape != self.packed_w.shape:
            wbuf = torch.empty(self.packed_w.shape, dtype=torch.bfloat16,
                                device=self.packed_w.device)
            self._bf16_weight_buf = wbuf
            materialize_2acc_bf16(self.packed_w, self.row_exp,
                                    self.col_exp, out=wbuf,
                                    mantissa_bias=self.MANTISSA_BIAS)
        return wbuf

    def forward(self, x):
        in_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        wbuf = self._ensure_buffers()
        y = FusedConcordLinear2acc.apply(
            x, self.packed_w, self.row_exp, self.col_exp, self.bias,
            self._lr_buf, self.alpha, self.beta1, self.MANTISSA_BIAS,
            self.weight_decay, self.step_cap, self.mag_weighted,
            wbuf)
        return y.to(in_dtype)


# ============================================================
# Smoke test
# ============================================================

def smoke_test():
    print("=== Two-accumulator smoke ===")
    device = 'cuda'
    torch.manual_seed(0)
    layer = ConcordLinear2acc(64, 32, device=device, lr=0.05,
                                weight_decay=0.001)
    w = layer.get_weight()
    print(f"  init |w|max={w.abs().max().item():.4f}")
    s_fast, s_slow = layer.get_state()
    print(f"  init |s_fast|max={s_fast.abs().max().item()}  "
          f"|s_slow|max={s_slow.abs().max().item()}")
    x = torch.randn(16, 64, device=device, dtype=torch.bfloat16)
    y_target = torch.randn(16, 32, device=device, dtype=torch.bfloat16)
    for step in range(20):
        pred = layer(x)
        loss = F.mse_loss(pred.float(), y_target.float())
        loss.backward()
        if step in (0, 1, 5, 10, 19):
            s_fast, s_slow = layer.get_state()
            print(f"  step {step:2d}: loss={loss.item():.4f}  "
                  f"|s_fast|max={s_fast.abs().max().item():5d}  "
                  f"|s_slow|max={s_slow.abs().max().item():5d}  "
                  f"|w|max={layer.get_weight().abs().max().item():.4f}")
    print("  PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        sys.exit(0)
    smoke_test()
