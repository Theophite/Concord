"""Prototype: packed-bf16 Concord layer.

One persistent int32 buffer per layer. Layout (little-endian):
    bits [31:16]  bf16 weight       — live, materialized to transient bf16 for cuBLAS
    bits [15:8]   s_fast_delta i8   — Kahan-style sub-LSB compensation
    bits [7:0]    v_slow_i8 (unused in this prototype)

Per param: 32 bits total (== fp32 weight budget; the optimizer is "free").

Forward: unpack kernel writes contiguous bf16 → cuBLAS matmul.
  (Strided bf16-view of int32 doesn't work with cuBLAS — the inner
   stride 2 isn't cuBLAS-compatible. So we materialize, but the
   materialize is just "copy upper 16 bits per word" — no arithmetic.)
Backward: cuBLAS grad_W → one Triton kernel that maps int32 → int32
  in place. All optimizer math in registers between the int32 load
  and the int32 store.

Smoke test: tiny linear regression. Random target W, random Gaussian
data, MSE loss. Should converge to near-zero loss in a few hundred
steps if the dynamics are correct.

Run:
    python prototype_packed.py
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Storage roundtrip sanity check (does NOT use Triton).
# ============================================================

def storage_roundtrip_check(device='cuda'):
    """Verify pack/unpack of bf16-in-upper-16 is lossless and that the
    unpacked weight produces identical matmul to the original bf16."""
    torch.manual_seed(0)
    w = torch.randn(8, 4, device=device, dtype=torch.bfloat16)

    # Pack: bf16 bits into upper 16 bits of int32.
    w_bits_i16 = w.view(torch.int16)                      # (8, 4) int16
    w_bits_i32 = w_bits_i16.to(torch.int32) & 0xFFFF      # mask sign-ext
    packed = w_bits_i32 << 16                              # bf16 in [31:16]
    assert packed.dtype == torch.int32

    # Unpack: arith shift right preserves the bf16 bits as int16.
    recovered_bits_i32 = packed >> 16                      # arith shift
    recovered_bits_i16 = recovered_bits_i32.to(torch.int16)
    recovered = recovered_bits_i16.view(torch.bfloat16)
    assert torch.equal(w, recovered), "pack/unpack roundtrip failed"

    # Matmul equivalence.
    x = torch.randn(3, 4, device=device, dtype=torch.bfloat16)
    y_orig = F.linear(x, w)
    y_pack = F.linear(x, recovered)
    assert torch.equal(y_orig, y_pack), "matmul through pack/unpack differs"
    print("[OK] storage roundtrip + matmul equivalence")


# ============================================================
# Triton kernels: unpack + apply (SGD path)
# ============================================================

@triton.jit
def _unpack_packed_to_bf16_kernel(
    packed_ptr,       # [N, K] int32
    weight_ptr,       # [N, K] bf16 (output)
    N, K,
    stride_pn, stride_pk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Materialize bf16 weight from packed int32 storage. The bf16
    bits live in [31:16] of each int32; we arith-shift right by 16 to
    expose them in the low 16 bits of int32, then bitcast int16 →
    bf16. No arithmetic — pure bit extraction."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    nk_mask = n_mask[:, None] & k_mask[None, :]

    p_off = offs_n[:, None] * stride_pn + offs_k[None, :] * stride_pk
    packed = tl.load(packed_ptr + p_off, mask=nk_mask, other=0).to(tl.int32)
    w_bits_i32 = packed >> 16                              # arith shift
    w_bits_i16 = w_bits_i32.to(tl.int16)
    w_bf16 = w_bits_i16.to(tl.bfloat16, bitcast=True)

    w_off = offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    tl.store(weight_ptr + w_off, w_bf16, mask=nk_mask)


def unpack_packed_to_bf16(packed_w, out):
    """Wrapper for _unpack_packed_to_bf16_kernel. ``out`` must be a
    bf16 tensor of the same shape as packed_w."""
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert out.dtype == torch.bfloat16 and out.shape == packed_w.shape
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _unpack_packed_to_bf16_kernel[grid](
        packed_w, out, N, K,
        packed_w.stride(0), packed_w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


# Per-element SR hash — same pattern as concord_triton_fused.py's
# _hash_uniform. Pure bitwise (no big-int multiplies) keeps Triton in
# int32 without uint32 literal headaches.
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
def _apply_packed_sgd_kernel(
    packed_ptr,       # [N, K] int32, mutated in place
    grad_W_ptr,       # [N, K] bf16
    N, K,
    lr, alpha, beta1,
    step_salt_ptr,
    stride_pn, stride_pk,
    stride_gn, stride_gk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Packed apply (SGD path).

    Load int32, unpack (bf16, delta_i8, v_slow_i8) in registers,
    SR-tick the delta with -lr*grad/lsb, mass-preserve chase rolls
    alpha*delta into the bf16 (incrementing the bf16 by N LSBs),
    repack, store int32. v_slow_i8 stays at zero in this prototype.

    All math happens between the single int32 load and the single
    int32 store — no extra HBM traffic for optimizer state."""
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

    # ── unpack ──────────────────────────────────────────────────
    w_bits_i16 = (packed >> 16).to(tl.int16)
    w_bf16 = w_bits_i16.to(tl.bfloat16, bitcast=True)
    w_fp32 = w_bf16.to(tl.float32)
    # Sign-extend bits[15:8] for s_fast_delta.
    delta_i32 = (packed << 16) >> 24  # left shift puts bit 15 at bit 31,
                                       # right arith shift fills with sign
    delta_f = delta_i32.to(tl.float32)

    # ── grad load ───────────────────────────────────────────────
    g_off = offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk
    grad_W = tl.load(grad_W_ptr + g_off, mask=nk_mask, other=0.0).to(tl.float32)

    # ── per-element bf16 LSB ────────────────────────────────────
    # bf16 raw bits: bit 15 = sign, [14:7] = exp (biased 127), [6:0] = mantissa
    # For normals: lsb = 2^(raw_exp - 127 - 7) = 2^(raw_exp - 134)
    w_raw = w_bits_i16.to(tl.uint16, bitcast=True).to(tl.int32)
    raw_exp = (w_raw >> 7) & 0xFF
    # Subnormal floor: clamp raw_exp to >= 1 so lsb >= 2^-133. Even tinier
    # exponents would give absurd grad_step magnitudes that overflow int8.
    raw_exp_clamped = tl.maximum(raw_exp, 1)
    lsb_exp = (raw_exp_clamped - 134).to(tl.float32)
    lsb = tl.exp2(lsb_exp)

    # ── SR-tick the delta ───────────────────────────────────────
    # grad_step in delta-units (1 unit = 1 bf16 LSB at current exponent).
    # step_cap bounds per-step grad-driven tick so that small-weight
    # elements (tiny lsb) don't produce huge delta increments. Each
    # bf16 LSB is roughly a 1/128 relative change; 32 LSBs is ~25%
    # relative change per step's grad contribution, plenty of slack.
    STEP_CAP = 32.0
    grad_step_units = -lr * grad_W / lsb
    grad_step_units = tl.minimum(tl.maximum(grad_step_units, -STEP_CAP), STEP_CAP)
    delta_t = grad_step_units - beta1 * delta_f

    pos_hash = (offs_n[:, None] << 16) ^ offs_k[None, :]
    r1 = _hash_uniform(delta_i32, pos_hash, step_salt)
    floor_t = tl.floor(delta_t)
    frac_t = delta_t - floor_t
    tick_fast = floor_t + (r1 < frac_t).to(tl.float32)
    delta_new_f = delta_f + tick_fast

    # Clamp delta to int8 range BEFORE chase so the chase magnitude
    # is bounded by saturation. Without this, an unclamped delta of
    # -250 produces chase = -25 LSBs per step = 25/128 ≈ 20%
    # relative weight change, which can runaway-double the weight in
    # ~10 steps under sustained gradient pressure.
    delta_new_f = tl.minimum(tl.maximum(delta_new_f, -128.0), 127.0)

    # ── chase + mass-preserve ───────────────────────────────────
    chase_f = alpha * delta_new_f
    r2 = _hash_uniform(delta_i32, pos_hash, step_salt ^ 0x5A5A5A5A)
    floor_s = tl.floor(chase_f)
    frac_s = chase_f - floor_s
    tick_slow = floor_s + (r2 < frac_s).to(tl.float32)
    # bf16 += tick_slow * lsb (in weight-space; bf16 carries its own rounding)
    w_fp32_new = w_fp32 + tick_slow * lsb
    delta_new_f = delta_new_f - tick_slow

    # ── repack ──────────────────────────────────────────────────
    w_bf16_new = w_fp32_new.to(tl.bfloat16)
    w_bits_new = w_bf16_new.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    delta_clamped = tl.minimum(tl.maximum(delta_new_f, -128.0), 127.0).to(tl.int32)
    packed_new = (w_bits_new << 16) | ((delta_clamped & 0xFF) << 8)

    tl.store(packed_ptr + p_off, packed_new, mask=nk_mask)


# Process-global step counter (tensor-backed; ensures different SR
# streams across calls without static-salt bias).
_STEP_COUNTERS = {}


def _get_step_counter(device):
    key = str(device)
    if key not in _STEP_COUNTERS:
        _STEP_COUNTERS[key] = torch.zeros(1, dtype=torch.int32, device=device)
    return _STEP_COUNTERS[key]


def apply_packed_sgd(packed_w, grad_W, lr, alpha=0.1, beta1=0.0):
    """Wrapper for _apply_packed_sgd_kernel."""
    N, K = packed_w.shape
    assert packed_w.dtype == torch.int32
    assert grad_W.dtype == torch.bfloat16 and grad_W.shape == packed_w.shape
    step_counter = _get_step_counter(packed_w.device)
    step_counter.add_(1)
    BLOCK_N, BLOCK_K = 32, 64
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _apply_packed_sgd_kernel[grid](
        packed_w, grad_W, N, K,
        float(lr), float(alpha), float(beta1),
        step_counter,
        packed_w.stride(0), packed_w.stride(1),
        grad_W.stride(0), grad_W.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


# ============================================================
# Autograd Function + nn.Module
# ============================================================

class FusedConcordLinearPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_w, bias, lr, alpha, beta1, weight_buf):
        # Materialize: extract bf16 weight from packed int32.
        unpack_packed_to_bf16(packed_w, out=weight_buf)
        bias_bf16 = (bias.to(torch.bfloat16)
                     if bias is not None and bias.dtype != torch.bfloat16
                     else bias)
        y = F.linear(x, weight_buf, bias_bf16)
        ctx.save_for_backward(x, weight_buf)
        ctx.packed_w = packed_w
        ctx.lr = lr
        ctx.alpha = alpha
        ctx.beta1 = beta1
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)
        if not grad_y.is_contiguous():
            grad_y = grad_y.contiguous()
        grad_x = grad_y @ weight
        grad_W = grad_y.transpose(-1, -2) @ x
        if grad_W.dtype != torch.bfloat16:
            grad_W = grad_W.to(torch.bfloat16)
        if not grad_W.is_contiguous():
            grad_W = grad_W.contiguous()
        apply_packed_sgd(ctx.packed_w, grad_W,
                          ctx.lr, ctx.alpha, ctx.beta1)
        grad_bias = grad_y.sum(0) if ctx.has_bias else None
        # 7 forward args: only x and bias receive grads.
        return grad_x, None, grad_bias, None, None, None, None


class ConcordLinearPacked(nn.Module):
    """Single-buffer Concord Linear. 32 bits/param total.

    Persistent state: ``packed_w`` int32 of shape (out, in). Bias is
    a separate small bf16 Parameter (kept fp32-precision-ish via the
    aux optimizer in real training; here it stays bf16 for the
    prototype).
    """

    def __init__(self, in_features, out_features, bias=True,
                 device='cuda', alpha=0.1, beta1=0.0, lr=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta1 = beta1
        self.lr = lr
        self.register_buffer('packed_w',
            torch.zeros(out_features, in_features,
                        dtype=torch.int32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,
                                                  dtype=torch.bfloat16,
                                                  device=device))
        else:
            self.register_parameter('bias', None)
        self._init_weight()

    def _init_weight(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        w = torch.randn(self.out_features, self.in_features,
                        device=self.packed_w.device) * std
        self.load_weights(w)

    @torch.no_grad()
    def load_weights(self, W):
        """Pack bf16(W) into bits[31:16], zero the lower 16 bits."""
        W_bf16 = W.to(device=self.packed_w.device, dtype=torch.bfloat16)
        w_bits = W_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
        self.packed_w.copy_(w_bits << 16)

    @torch.no_grad()
    def get_weight(self):
        """Return the live bf16 weight (read-only view)."""
        w_bits = (self.packed_w >> 16).to(torch.int16)
        return w_bits.view(torch.bfloat16)

    def forward(self, x):
        in_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if (wbuf is None or wbuf.shape != self.packed_w.shape
                or wbuf.device != self.packed_w.device):
            wbuf = torch.empty(self.packed_w.shape,
                                dtype=torch.bfloat16,
                                device=self.packed_w.device)
            self._bf16_weight_buf = wbuf
        y = FusedConcordLinearPacked.apply(
            x, self.packed_w, self.bias,
            self.lr, self.alpha, self.beta1, wbuf)
        return y.to(in_dtype)


# ============================================================
# Smoke test: tiny linear regression
# ============================================================

def _run_one(model_factory, lr, n_steps=500, bsz=32):
    """Train ``model_factory()`` for ``n_steps`` SGD steps on a fixed
    random regression task. Returns the loss curve.
    """
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
    # Reseed for the training loop's randperms etc.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # If using a torch optimizer (baseline), pull params.
    if hasattr(model, '_torch_opt'):
        opt = model._torch_opt
    else:
        opt = None

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


def _packed_mlp(in_f, hid, out_f, lr, device):
    return nn.Sequential(
        ConcordLinearPacked(in_f, hid, device=device, lr=lr, alpha=0.1),
        nn.ReLU(),
        ConcordLinearPacked(hid, out_f, device=device, lr=lr, alpha=0.1),
    )


def _torch_mlp(in_f, hid, out_f, lr, device):
    """Plain fp32 nn.Linear + torch.optim.SGD baseline. Same shape and
    init as the packed model so the comparison is apples-to-apples
    modulo storage."""
    model = nn.Sequential(
        nn.Linear(in_f, hid, device=device),
        nn.ReLU(),
        nn.Linear(hid, out_f, device=device),
    )
    model._torch_opt = torch.optim.SGD(model.parameters(), lr=lr)
    return model


def diagnose_packed_mlp():
    """Trace the dynamics of a 2-layer MLP at lr=0.001. Identifies
    which step blows up."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    model = nn.Sequential(
        ConcordLinearPacked(32, 64, device=device, lr=0.001, alpha=0.1),
        nn.ReLU(),
        ConcordLinearPacked(64, 16, device=device, lr=0.001, alpha=0.1),
    )
    target_W1 = torch.randn(64, 32, device=device) * 0.3
    target_W2 = torch.randn(16, 64, device=device) * 0.3
    def target(x):
        return F.linear(F.relu(F.linear(x, target_W1)), target_W2)
    x = torch.randn(32, 32, device=device)
    y_target = target(x)
    for step in range(20):
        pred = model(x)
        loss = F.mse_loss(pred.float(), y_target.float())
        if loss.item() != loss.item():
            print(f"[diag-mlp] step {step}: NaN LOSS")
            for i, layer in enumerate(model):
                if hasattr(layer, 'packed_w'):
                    w = layer.get_weight().float()
                    delta = ((layer.packed_w << 16) >> 24).to(torch.float32)
                    print(f"  layer{i}: |w|max={w.abs().max().item():.4g} "
                          f"nan_w={(w!=w).sum().item()} "
                          f"|delta|max={delta.abs().max().item():.1f}")
            return
        loss.backward()
        # Print layer stats
        if step < 5 or step % 5 == 0:
            stats = []
            for i, layer in enumerate(model):
                if hasattr(layer, 'packed_w'):
                    w = layer.get_weight().float()
                    delta = ((layer.packed_w << 16) >> 24).to(torch.float32)
                    stats.append(
                        f"L{i}[|w|max={w.abs().max().item():.3g} "
                        f"|d|max={delta.abs().max().item():.0f}]")
            print(f"[diag-mlp] step {step}: loss={loss.item():.4f} " + " ".join(stats))
    print("[diag-mlp] 20 steps stable")


def diagnose_packed():
    """Trace the dynamics of a single packed layer on a fixed
    gradient. Identifies which step (if any) blows up."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    layer = ConcordLinearPacked(8, 4, device=device, lr=0.001, alpha=0.1)

    print(f"[diag] init weight stats: "
          f"min={layer.get_weight().min().item():.4f} "
          f"max={layer.get_weight().max().item():.4f} "
          f"mean={layer.get_weight().float().abs().mean().item():.4f}")
    x = torch.randn(16, 8, device=device, dtype=torch.bfloat16)
    y_target = torch.randn(16, 4, device=device, dtype=torch.bfloat16)
    for step in range(20):
        pred = layer(x)
        loss = F.mse_loss(pred.float(), y_target.float())
        if loss.item() != loss.item():  # NaN check
            print(f"[diag] step {step}: NaN LOSS")
            w = layer.get_weight().float()
            print(f"  weight: min={w.min().item():.4g} max={w.max().item():.4g} "
                  f"nan_count={(w != w).sum().item()} "
                  f"inf_count={torch.isinf(w).sum().item()}")
            delta = ((layer.packed_w << 16) >> 24).to(torch.float32)
            print(f"  delta:  min={delta.min().item():.4g} max={delta.max().item():.4g}")
            return
        loss.backward()
        w = layer.get_weight().float()
        if (w != w).any():
            print(f"[diag] step {step}: NaN WEIGHT after backward")
            print(f"  nan_count={(w != w).sum().item()}")
            return
        delta = ((layer.packed_w << 16) >> 24).to(torch.float32)
        delta_max_abs = delta.abs().max().item()
        if step < 5 or step % 5 == 0:
            print(f"[diag] step {step}: loss={loss.item():.4f} "
                  f"|w|_mean={w.abs().mean().item():.4f} "
                  f"|w|_max={w.abs().max().item():.4f} "
                  f"|delta|_max={delta_max_abs:.1f}")
    print("[diag] 20 steps stable")


def smoke_test():
    print("[diag] === single-layer dynamics trace ===")
    diagnose_packed()
    print()
    print("[diag] === 2-layer MLP dynamics trace ===")
    diagnose_packed_mlp()
    print()

    results = {}
    for tag, factory, lr in [
        ('packed lr=0.05',    _packed_mlp, 0.05),
        ('packed lr=0.01',    _packed_mlp, 0.01),
        ('packed lr=0.001',   _packed_mlp, 0.001),
        ('baseline lr=0.05',  _torch_mlp,  0.05),
        ('baseline lr=0.01',  _torch_mlp,  0.01),
        ('baseline lr=0.001', _torch_mlp,  0.001),
    ]:
        L = _run_one(factory, lr=lr)
        nans = sum(1 for v in L if v != v)
        ratio = L[0] / max(L[-1], 1e-30)
        print(f"[smoke] {tag:>22}  init={L[0]:6.3f}  "
              f"s100={L[100]:7.3f}  s500={L[-1]:7.3f}  "
              f"ratio={ratio:6.1f}x"
              + (f"  [NaN x{nans}]" if nans else ""))
        results[tag] = (L, ratio, nans)

    # Headline comparison: best packed vs best baseline.
    best_packed = max(r[1] for r in [results[f'packed lr={lr}']
                                       for lr in (0.05, 0.01, 0.001)]
                       if r[2] == 0)
    best_baseline = max(r[1] for r in [results[f'baseline lr={lr}']
                                         for lr in (0.05, 0.01, 0.001)]
                         if r[2] == 0)
    print()
    print(f"[smoke] best packed reduction:   {best_packed:.1f}x")
    print(f"[smoke] best baseline reduction: {best_baseline:.1f}x")
    # Packed should reach at least 70% of baseline's reduction ratio
    # (so a 10x baseline → packed must hit >= 7x). Tighter than this
    # would constrain the bf16 + delta dynamics's intrinsic
    # quantisation noise.
    if best_packed >= best_baseline * 0.7:
        print(f"[PASS] packed within 70% of baseline reduction ratio")
        return True
    else:
        print(f"[FAIL] packed got {best_packed:.1f}x vs baseline "
              f"{best_baseline:.1f}x (expected packed ≥ "
              f"{best_baseline * 0.7:.1f}x)")
        return False


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    print("=== Storage roundtrip check ===")
    storage_roundtrip_check()
    print()
    print("=== Smoke test: 2-layer MLP linear regression ===")
    ok = smoke_test()
    print()
    if ok:
        print("=== PROTOTYPE WORKS ===")
    else:
        print("=== PROTOTYPE FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
