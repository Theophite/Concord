"""ConcordLinearFused — no fp32/bf16 weight cache.

Storage per parameter:
    s_slow: int16 buffer (16 bits)
    s_fast: int16 buffer (16 bits)
    row_exp, col_exp: shared along an axis, amortized ~0

Total: 32 bits per parameter. The int16 state is widened to int32 in
registers inside the Triton kernels for arithmetic; the bf16 weight is
reconstructed there and never materialized.

The gradient update is fused into the backward pass: as grad_W is
accumulated per (N, K) tile, the concord momentum tick is applied
immediately and the int state is written back. There is no separate
optimizer.step() — backward IS the update.

The current learning rate is set via `set_lr` before each forward, so
cosine/etc. schedules can drive the per-step lr.
"""
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from concord_triton import apply_ticks_triton, rebalance_fused_triton
from concord_triton_fused import FusedConcordLinear

torch._dynamo.config.suppress_errors = True




class ConcordLinearFused(nn.Module):
    MANTISSA_BIAS = 15
    INT16_MAX = 32767
    INT16_MIN = -32768
    MAX_M = 24000
    MIN_M = 6000
    # Per-component exponent range, tightened to fit int4 signed [-8, 7].
    # The CLZ-bitcast bf16 emission adds a per-element 'h' contribution
    # (0..16) on top of (r+c-B), so the format's effective dynamic range
    # is r+c+h ∈ [-16, 30] — 46 binary orders of magnitude. Plenty for
    # any neural-net weight regime. row_exp/col_exp are stored as int8
    # but constrained to int4 range (the upper 4 bits per cell are zero);
    # this honors the "half-word storage" rule without paying the kernel
    # complexity tax of true nibble-packing on an O(M+N) buffer.
    EXP_MIN = -8
    EXP_MAX = 7
    _reb_seed = 0   # process-global rebalance counter; keys the tick-up SR

    def __init__(self, in_features, out_features, bias=True,
                 device='cuda', max_iters=2, alpha=0.1, beta1=0.0,
                 lr=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_iters = max_iters
        self.alpha = alpha
        self.beta1 = beta1
        self.lr = lr

        # Optimizer kind. 'sgd' (default) uses the classic SR-tick + chase
        # path via fused_grad_W_and_update. 'adamw' uses the single-kernel
        # path via fused_grad_W_and_adamw_update, which preconditions each
        # step by the per-element CLZ-derived |W| proxy — no v_row/v_col
        # tracking, the weight's bit pattern IS the variance signal. The
        # β2 / temporal-discount role is filled by `discount_t`, a per-
        # layer scalar updated CPU-side on the lazy refit-cadence path
        # from the cascade's |W| trajectory (see update_discount_from_
        # cascade). Switch via set_optimizer_kind.
        self.optimizer_kind = 'sgd'
        self.weight_decay = 0.0
        self.eps = 1e-8
        # Per-row + per-col discount factoring (rank-1 of the per-
        # element precondition). The kernel forms
        #     discount_ij = discount_row[i] · discount_col[j]
        # and applies it as the multiplicative effective-lr modulator
        # alongside the |W|-via-CLZ proxy. At init both vectors are 1.0
        # (so discount_ij = 1 everywhere = pure LARS-style step). The
        # cascade-update rewrites them at refit cadence.
        self.register_buffer('discount_row', None)
        self.register_buffer('discount_col', None)
        # Per-row + per-col init |W| magnitudes (mean |W_ij| over the
        # other axis), captured at AdamW-enable. These normalise the
        # cascade signal so the rank-1 factoring is init-scale invariant.
        self.register_buffer('_W_init_row', None)
        self.register_buffer('_W_init_col', None)
        # power_exp=1 keeps per-element step magnitude roughly constant
        # as |W| grows (the kernel's own 1/|W| term in step = grad/|W|
        # cancels the linear discount growth, so |grad/|W|_init| stays
        # constant). Tune up if you want LR boosting at convergence,
        # down for more damping. min/max are safety clamps.
        self._discount_power_exp = 1.0
        self._discount_min = 0.01
        self._discount_max = 10.0
        # tick-down was removed when the forward kernel switched to CLZ-
        # bitcast bf16 emission; small mantissas now emit at the correct
        # bf16 value via per-element h without any rebalance intervention.
        # Rebalance is tick-up only — see _rebalance_decide_apply_kernel
        # docstring. No parity / dn_axis state needed.

        self.register_buffer('s_slow',
                             torch.zeros(out_features, in_features,
                                         dtype=torch.int16, device=device))
        self.register_buffer('s_fast',
                             torch.zeros(out_features, in_features,
                                         dtype=torch.int16, device=device))
        # row_exp / col_exp at int8 storage: 4x smaller than int32, still
        # comfortably wider than the EXP_MIN/EXP_MAX range we actually use
        # ([-48, 47] vs int8's [-128, 127]). Kernels load these as int8
        # and widen to int32 in registers for arithmetic.
        self.register_buffer('row_exp',
                             torch.zeros(out_features, dtype=torch.int8,
                                         device=device))
        self.register_buffer('col_exp',
                             torch.zeros(in_features, dtype=torch.int8,
                                         device=device))
        # Previous sign of the velocity (s_fast - s_slow), for the
        # oscillation-damping regularizer. Allocated lazily by
        # enable_osc_damp() / the first damp_oscillation() call -- with
        # osc-damp off (the default) it stays None and costs no HBM, so the
        # total persistent state stays 32 bits/param. One bit/param of
        # information when used; a bool buffer, packable to a true bit later.
        self.register_buffer('vsign', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,
                                                  dtype=torch.bfloat16,
                                                  device=device))
        else:
            self.register_parameter('bias', None)

        self._init_concord()

    def _init_concord(self):
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        W = torch.randn(self.out_features, self.in_features,
                        device=self.s_slow.device) * std
        self.load_weights(W)

    @torch.no_grad()
    def load_weights(self, W):
        """From-scratch decomposition. Splits the (out_features,
        in_features) weight matrix 50/50 across s_slow + s_fast,
        leaves v_slow_i8 (if allocated) untouched. Correct init for
        random initialisation: v_slow legitimately starts at zero
        and chases s_fast over the first ~1/alpha_v_fast ≈ 1000
        steps to fill in the long-time average.

        For loading a PRETRAINED weight (fine-tuning), prefer
        `load_weights_finetune()`: it puts the weight in the
        zero-gradient steady-state (1/3-1/3-1/3 split) so the
        drift-cancel noise estimator and the Bayesian-anchored
        wd_sv / wd_sf regularizers are physically meaningful from
        step 1 instead of needing ~1000 steps to warm up.
        """
        W = W.to(device=self.s_slow.device, dtype=torch.float32)
        assert W.shape == (self.out_features, self.in_features), (
            f'{tuple(W.shape)} != {(self.out_features, self.in_features)}')
        max_abs_row = W.abs().max(dim=1).values.clamp(min=1e-30)
        self.row_exp.copy_(
            torch.ceil(torch.log2(max_abs_row) + 1.0)
            .clamp(self.EXP_MIN, self.EXP_MAX).to(torch.int8))
        self.col_exp.zero_()
        exp = (self.row_exp[:, None] + self.col_exp[None, :]
               - self.MANTISSA_BIAS).float()
        scale = torch.pow(2.0, exp)
        m_total = (W / scale).round().to(torch.int32).clamp(
            self.INT16_MIN, self.INT16_MAX)
        half = (m_total / 2).round().to(torch.int32)
        self.s_slow.copy_(half)
        self.s_fast.copy_(m_total - half)
        if self.vsign is not None:
            self.vsign.copy_(self.s_fast >= self.s_slow)

    @torch.no_grad()
    def load_weights_finetune(self, W):
        """Bayesian-prior decomposition for fine-tuning a pretrained
        weight. Splits the mantissa roughly 1/3 into v_slow_i8 (at
        shifted scale V_SLOW_FACTOR) and the remaining 2/3 evenly into
        s_slow and s_fast. This is the *zero-gradient steady state*
        of the three-accumulator dynamic — at this point both
        velocity_short = (s_fast - s_slow) and velocity_long =
        (s_slow - v_slow_full) are zero (up to int8-quantisation
        remainder), so:

          1. The drift-cancel noise estimator
             `noise = (s_fast - s_slow) - C·(s_slow - v_slow_full)`
             starts at ~0 instead of carrying spurious "drift" equal
             to the full pretrained weight (which the from-scratch
             init would produce, since v_slow_full=0 makes
             (s_slow - v_slow_full) = s_slow = W/2).
          2. The Bayesian-anchored weight decay terms `wd_sv` (pulls
             s_slow → v_slow_full) and `wd_sf` (pulls s_fast →
             v_slow_full) become a proper prior regulariser, anchoring
             the live weight at the pretrained value. Under the
             from-scratch init they'd pull toward v_slow_full=0, i.e.
             standard L2-toward-zero — the wrong direction for
             fine-tuning.

        Allocates v_slow_i8 if not already enabled (with default
        V_SLOW_FACTOR=128). int8-quantisation remainder is absorbed
        evenly into s_slow and s_fast so the live weight matches W
        exactly (to within bf16 rounding); the residual
        velocity_long is bounded by ±V_SLOW_FACTOR/2 ≈ 64 mantissa
        units, negligible compared to the typical accumulator
        magnitude.
        """
        # ConcordLinearFused doesn't pre-create the v_slow_i8 attribute
        # (it's lazily set by enable_v_slow_i8 — see __init__). Use
        # getattr so the pre-allocation check works without triggering
        # nn.Module's __getattr__ guard.
        if getattr(self, 'v_slow_i8', None) is None:
            self.enable_v_slow_i8()
        W = W.to(device=self.s_slow.device, dtype=torch.float32)
        assert W.shape == (self.out_features, self.in_features), (
            f'{tuple(W.shape)} != {(self.out_features, self.in_features)}')
        max_abs_row = W.abs().max(dim=1).values.clamp(min=1e-30)
        self.row_exp.copy_(
            torch.ceil(torch.log2(max_abs_row) + 1.0)
            .clamp(self.EXP_MIN, self.EXP_MAX).to(torch.int8))
        self.col_exp.zero_()
        exp = (self.row_exp[:, None] + self.col_exp[None, :]
               - self.MANTISSA_BIAS).float()
        scale = torch.pow(2.0, exp)

        # Total mantissa we want s_slow + s_fast + v_slow_full to sum
        # to. NOT pre-clamped: each individual accumulator only sees
        # ~1/3 of m_total, so the int16 limit on s_slow / s_fast is
        # rarely the binding constraint; v_slow_i8's int8 range
        # (±127·V_SLOW_FACTOR = ±16256) is the tighter one.
        m_total = (W / scale).round().to(torch.int32)

        # Quantise v_slow_i8 to one-third of m_total at shifted scale.
        target_v_full = (m_total.float() / 3.0).round().to(torch.int32)
        v_slow_int = (target_v_full.float() / float(self.v_slow_factor)
                       ).round().to(torch.int32).clamp(-128, 127)
        actual_v_full = v_slow_int * self.v_slow_factor

        # Remaining mantissa goes equally into s_slow and s_fast so
        # velocity_short = s_fast - s_slow = 0 at init. Absorbs both
        # the explicit "1/3 each in s_slow + s_fast" share AND the
        # int8-quantisation remainder of v_slow_i8 (so the live
        # weight = W exactly within bf16 quantisation).
        remaining = m_total - actual_v_full
        half = (remaining / 2).round().to(torch.int32)
        self.s_slow.copy_(half.clamp(self.INT16_MIN, self.INT16_MAX))
        self.s_fast.copy_((remaining - half).clamp(
            self.INT16_MIN, self.INT16_MAX))
        self.v_slow_i8.copy_(v_slow_int.to(torch.int8))
        if self.vsign is not None:
            self.vsign.copy_(self.s_fast >= self.s_slow)

    def set_lr(self, lr):
        self.lr = lr

    def set_optimizer_kind(self, kind, weight_decay=0.0, eps=1e-8):
        """Switch the optimizer kind. Valid values:
            'sgd'   — classic SR-tick + chase (fused_grad_W_and_update).
            'adamw' — single-kernel AdamW (fused_grad_W_and_adamw_update),
                      preconditioned by per-element CLZ-derived |W|.
        For 'adamw', no extra state is allocated — the variance signal
        lives in the weight's own bit pattern. The temporal-discount
        scalar `discount_t` is set on the layer attribute and updated
        on the lazy refit path via update_discount_from_cascade.

        On switch to 'adamw' we capture the current mean |W| as
        `_W_init_mag` to normalise the cascade-derived discount: the
        signal that gets mapped to discount is |W|_current / |W|_init,
        making the formula init-scale-invariant across different layer
        sizes and weight inits."""
        if kind not in ('sgd', 'adamw'):
            raise ValueError(
                f"optimizer kind must be 'sgd' or 'adamw', got {kind!r}")
        self.optimizer_kind = kind
        if kind == 'adamw':
            self.weight_decay = float(weight_decay)
            self.eps = float(eps)
            # Snapshot initial per-row and per-col |W| magnitudes for
            # discount normalisation. Cached as fp32 GPU buffers; both
            # discount_row/discount_col init to ones (= layer-wide
            # scalar discount = 1.0 = pure LARS-style step).
            dev = self.s_slow.device
            with torch.no_grad():
                m_eff = (self.s_slow.to(torch.int32)
                         + self.s_fast.to(torch.int32)).float()
                exp = (self.row_exp[:, None].float()
                       + self.col_exp[None, :].float()
                       - self.MANTISSA_BIAS)
                abs_W_now = (m_eff * torch.exp2(exp)).abs()
                self._W_init_row = abs_W_now.mean(dim=1).clamp(min=1e-30).contiguous()
                self._W_init_col = abs_W_now.mean(dim=0).clamp(min=1e-30).contiguous()
            # AdamW preconditioner state. Two variance sources are
            # supported:
            #   'three_accum' (default) — drift-cancelled noise residual
            #     from s_fast/s_slow/v_slow_i8. 40 bits/param. Highest
            #     accuracy; the empirical CIFAR champion.
            #   'v_rank1' — Adafactor rank-1 EMA: v_row[N] / v_col[K]
            #     carry EMA of per-row / per-col mean(g²); g2_row /
            #     g2_col are scratch the kernel atomic_adds into. 32
            #     bits/param + 4 fp32 vectors per layer. Use for large
            #     layers where the (N,K) v_slow_i8 buffer is too costly.
            # Earlier 'v_from_velocity' and 'W_proxy' variants were
            # ablated as below-baseline and removed (see git history).
            # Note: combining v_rank1 + v_slow_i8 was also ablated as
            # below-baseline — they compete on the same timescale. Pick
            # one variance source per layer.
            self.optimizer_v_kind = getattr(self, 'optimizer_v_kind',
                                             'three_accum')
            self.v_beta2 = float(getattr(self, 'v_beta2', 0.999))
            self.v_step = 0
            self.step_cap = float(getattr(self, 'step_cap', 10.0))
            if self.optimizer_v_kind == 'v_rank1':
                # Initialise to a small positive value so the first-step
                # Adafactor reconstruction (mean(v_row) in the
                # denominator) doesn't divide by zero. Bias correction
                # handles the ratio fairly quickly (a few steps).
                v_init = 1e-6
                self.v_row = torch.full((self.out_features,), v_init,
                                         dtype=torch.float32, device=dev)
                self.v_col = torch.full((self.in_features,), v_init,
                                         dtype=torch.float32, device=dev)
                self.g2_row = torch.zeros(self.out_features,
                                           dtype=torch.float32, device=dev)
                self.g2_col = torch.zeros(self.in_features,
                                           dtype=torch.float32, device=dev)
            else:
                self.v_row = None
                self.v_col = None
                self.g2_row = None
                self.g2_col = None
            if self.optimizer_v_kind == 'three_accum':
                # v_slow_i8 — auto-allocate at init=0 if the caller
                # didn't call enable_v_slow_i8 first.
                if getattr(self, 'v_slow_i8', None) is None:
                    self.enable_v_slow_i8(
                        factor=int(getattr(self, 'v_slow_factor', 128)),
                        alpha_v_fast=float(getattr(self, 'alpha_v_fast',
                                                     0.001)))
            # Tunable knobs on the layer (passed through to the kernel
            # by forward()).
            self.v_scale = float(getattr(self, 'v_scale', 1.0))
            self.drift_cancel_C = float(
                getattr(self, 'drift_cancel_C', 0.1))
            self.alpha_v_fast = float(getattr(self, 'alpha_v_fast', 0.001))
            self.alpha_v_slow = float(getattr(self, 'alpha_v_slow', 0.01))
            # Bayesian-anchored decay: small pull of s_fast and s_slow
            # toward v_slow_full each step. Damps the "unconfirmed"
            # transient (the part of the weight not yet supported by
            # the long-time gradient average). 1e-5 was the empirically
            # -best magnitude on bigger CIFAR.
            self.wd_sv = float(getattr(self, 'wd_sv', 1e-5))
            self.wd_sf = float(getattr(self, 'wd_sf', 1e-5))
            # The legacy int16 self.v_slow buffer is no longer used —
            # the three-accumulator path consumes self.v_slow_i8.
            if not hasattr(self, 'v_slow'):
                self.v_slow = None

    # Int8 rebalance threshold. v_slow_i8 saturates at ±127. We trigger
    # an exp tick-up once a row/col's max |v_slow_i8| exceeds this — the
    # int8 spacing is much coarser than the int16 mantissa space (each
    # v_slow_i8 unit = v_slow_factor mantissa units, default 128), so
    # leaving more headroom is wise: if a v_slow_i8 hits 127 the next
    # leak tick saturates and the mass is silently lost. 96 = 75% of
    # int8 max gives ~5 binades of leak headroom before saturation.
    V_SLOW_I8_MAX = 96

    @torch.no_grad()
    def _v_slow_i8_rebalance(self):
        """Tick row/col_exp up when v_slow_i8 magnitudes approach int8
        saturation, then SR-right-shift all three buffers (s_slow,
        s_fast, v_slow_i8) in lock-step so the live weight is preserved.

        The int16 rebalance (rebalance_fused_triton) only watches
        |s_slow + s_fast|. v_slow_i8 has a coarser quantum and a much
        smaller dynamic range (±127) than the int16 buffers (±32k), so
        it can saturate before s_slow+s_fast triggers a rebalance. This
        pass is a defensive check after every rebalance: cheap when
        nothing's saturating (one .amax + comparison), correct when it
        is."""
        v8 = getattr(self, 'v_slow_i8', None)
        if v8 is None:
            return
        v8_abs = v8.abs().to(torch.int32)
        row_max_v8 = v8_abs.amax(dim=1)
        col_max_v8 = v8_abs.amax(dim=0)
        row_tick = ((row_max_v8 > self.V_SLOW_I8_MAX)
                    & (self.row_exp < self.EXP_MAX)).to(torch.int32)
        col_tick = ((col_max_v8 > self.V_SLOW_I8_MAX)
                    & (self.col_exp < self.EXP_MAX)).to(torch.int32)
        if not (bool(row_tick.any()) or bool(col_tick.any())):
            return
        # Per-element shift d = row_tick[i] + col_tick[j] ∈ {0, 1, 2}.
        d = row_tick[:, None] + col_tick[None, :]
        # SR right-shift s_slow / s_fast (independent rounding).
        for buf in (self.s_slow, self.s_fast):
            s = buf.to(torch.int32)
            q = s >> d
            rem = (s - (q << d)).to(torch.float32)
            two_pow_d = torch.pow(2.0, d.to(torch.float32))
            u = torch.rand(d.shape, device=d.device, dtype=torch.float32)
            up = (u * two_pow_d < rem).to(torch.int32)
            new = (q + up).clamp(self.INT16_MIN, self.INT16_MAX).to(torch.int16)
            buf.copy_(new)
        # SR right-shift v_slow_i8 in its int8 space.
        v_fp = v8.to(torch.float32)
        scaled = v_fp / torch.pow(2.0, d.to(torch.float32))
        floor = torch.floor(scaled)
        frac = scaled - floor
        u = torch.rand(d.shape, device=d.device, dtype=torch.float32)
        new_v = (floor + (u < frac).to(torch.float32)).clamp(-128, 127).to(torch.int8)
        v8.copy_(new_v)
        # Commit the exp ticks.
        self.row_exp.add_(row_tick.to(self.row_exp.dtype))
        self.col_exp.add_(col_tick.to(self.col_exp.dtype))

    @torch.no_grad()
    def _v_slow_apply_exp_shift(self, d_row, d_col):
        """Rescale v_slow_i8 to preserve its live-weight contribution
        when row_exp / col_exp tick by d_row / d_col.

        Live-weight contribution from v_slow is
            v_slow * factor * 2^(row_exp + col_exp - mantissa_bias).
        If row_exp ticks up by d_row[i] and col_exp by d_col[j], the
        exponent multiplier grows by 2^(d_row[i] + d_col[j]); to keep
        the live contribution invariant we must right-shift v_slow_i8
        by the same per-element amount via stochastic rounding.

        Called by rebalance() and refit_envelope() — without it, v_slow
        silently doubles every time an exponent ticks up. d_row / d_col
        are int32 tensors of the deltas already committed to row_exp /
        col_exp (so passing the post-minus-pre snapshot is correct)."""
        v_slow_i8 = getattr(self, 'v_slow_i8', None)
        if v_slow_i8 is None:
            return
        d = d_row[:, None].to(torch.int32) + d_col[None, :].to(torch.int32)
        if not bool(d.any()):
            return
        # SR right-shift / left-shift. Positive d => divide by 2^d
        # (right-shift; common case under tick-up rebalance). Negative d
        # would multiply (left-shift) — supported but rare; on overflow
        # to outside [-128, 127] the int8 clamp absorbs the loss. v_slow
        # is the SLOW accumulator so this lossy clamp on the rare
        # tick-down path is acceptable.
        v_fp = v_slow_i8.to(torch.float32)
        scale = torch.pow(2.0, -d.to(torch.float32))
        scaled = v_fp * scale
        floor = torch.floor(scaled)
        frac = scaled - floor
        u = torch.rand(d.shape, device=d.device, dtype=torch.float32)
        new_v_int32 = (floor + (u < frac).to(torch.float32)).to(torch.int32)
        new_v_int8 = new_v_int32.clamp(-128, 127).to(torch.int8)
        v_slow_i8.copy_(new_v_int8)

    @torch.no_grad()
    def enable_v_slow_i8(self, factor=128, alpha_v_fast=0.001):
        """Allocate the int8 v_slow accumulator for the
        three-accumulator path. Stored at int8 with shifted scale
        ``factor`` (default 128 = 2^7): each int8 unit of v_slow
        represents ``factor`` mantissa units of s_slow, and v_slow
        contributes additively to the live weight via
        ``w = (s_slow + s_fast + v_slow*factor) * scale``.

        The leak ``v_slow ← s_fast`` is non-mass-preserving (a "second
        chase"): the live weight grows by what v_slow gained, mirroring
        the existing s_fast/s_slow chase that is also non-mass-
        preserving and which the CIFAR headline depends on. The
        earlier mass-preserving variant was ablated as below-baseline.
        Idempotent."""
        if getattr(self, 'v_slow_i8', None) is None:
            self.v_slow_i8 = torch.zeros_like(self.s_slow, dtype=torch.int8)
        self.v_slow_factor = int(factor)
        self.alpha_v_fast = float(alpha_v_fast)

    def enable_v_diagnostic(self, beta2=0.999):
        """Allocate a per-element fp32 v_diag buffer and an EMA β2. While
        v_diag exists, the autograd Function's backward will EMA-update
        it with a torch-computed g² each step (cost: one fp32 matmul per
        backward; only the diagnostic-run pays it). Use compare_discount_
        to_real_v() to read out how the cascade-derived per-layer
        discount_t compares to what a per-element β2 EMA would prescribe."""
        dev = self.s_slow.device
        self.v_diag = torch.zeros(self.out_features, self.in_features,
                                   dtype=torch.float32, device=dev)
        self.v_diag_beta2 = float(beta2)
        self.v_diag_steps = 0

    def disable_v_diagnostic(self):
        """Free the v_diag buffer and disable EMA tracking."""
        self.v_diag = None
        self.v_diag_steps = 0

    def compare_discount_to_real_v(self):
        """Return comparison stats between our scalar discount_t and the
        per-element β2-EMA-derived preconditioner. Returns None if
        enable_v_diagnostic hasn't been called or hasn't accumulated yet.

        The 'effective discount per element' is |W|_ij / sqrt(v_ij);
        if this ratio is roughly constant across elements within the
        layer, a scalar discount_t (our choice) captures the same shape
        as a true per-element EMA would. The std tells us how lossy the
        scalar approximation is."""
        v_diag = getattr(self, 'v_diag', None)
        if v_diag is None or self.v_diag_steps == 0:
            return None
        beta2 = self.v_diag_beta2
        bc2 = 1.0 - beta2 ** self.v_diag_steps
        v_hat = v_diag / max(bc2, 1e-30)
        sqrt_v = v_hat.clamp(min=0).sqrt() + 1e-8
        exp = (self.row_exp[:, None].float() + self.col_exp[None, :].float()
               - self.MANTISSA_BIAS)
        abs_W = ((self.s_slow.to(torch.int32) + self.s_fast.to(torch.int32))
                 .float().abs() * torch.exp2(exp))
        # Per-element implied discount = |W| / sqrt(v).
        implied = abs_W / sqrt_v
        # Our rank-1 prediction: discount_row[i] * discount_col[j].
        d_row = (self.discount_row if self.discount_row is not None
                 else torch.ones(self.out_features, device=abs_W.device))
        d_col = (self.discount_col if self.discount_col is not None
                 else torch.ones(self.in_features, device=abs_W.device))
        predicted = d_row[:, None] * d_col[None, :]
        residual = (predicted - implied)
        return {
            'discount_row_mean': float(d_row.mean().item()),
            'discount_col_mean': float(d_col.mean().item()),
            'predicted_mean': predicted.mean().item(),
            'implied_mean': implied.mean().item(),
            'implied_std': implied.std().item(),
            'implied_p50': implied.median().item(),
            'residual_mean': residual.abs().mean().item(),
            'residual_rel': (residual.abs() / implied.clamp(min=1e-30)
                              ).mean().item(),
            'v_diag_steps': self.v_diag_steps,
            'beta2': beta2,
        }

    def update_discount_from_cascade(self, cascade, layer_slice):
        """Lazy-path discount_t update. Fits the |W| trajectory in the
        cascade ring (this layer's slice) and emits a scalar discount_t
        that plays the β2-like temporal-discount role for this layer.

        Called by the optimizer wrapper at refit cadence, NOT per-step.
        Cheap CPU op: K snapshots × layer_size fp16 reads + a few diffs.

        cascade      — the BoxVelocityMean (or compatible) instance.
        layer_slice  — Python slice into cascade's flat n_fol_total
                       vector identifying this layer's elements.
        """
        K = getattr(cascade, "filled_count", 0)
        if K < 2 or self.discount_row is None or self._W_init_row is None:
            return
        # Reshape this layer's slice of the cascade ring back to (K, N, K_in).
        ring_flat = cascade.ring[:K, layer_slice].float()
        layer_size = ring_flat.shape[1]
        if layer_size != self.out_features * self.in_features:
            return
        ring_2d = ring_flat.view(K, self.out_features, self.in_features)
        # Per-row and per-col mean |W| from the most-recent cascade slot.
        abs_W_recent = ring_2d[-1].abs()  # (N, K_in) on CPU fp32
        cur_row = abs_W_recent.mean(dim=1).clamp(min=1e-30)
        cur_col = abs_W_recent.mean(dim=0).clamp(min=1e-30)
        # Init refs were captured at set_optimizer_kind on GPU; bring
        # to CPU for the ratio op then push the result back.
        init_row = self._W_init_row.detach().to(cur_row.device,
                                                  dtype=torch.float32)
        init_col = self._W_init_col.detach().to(cur_col.device,
                                                  dtype=torch.float32)
        ratio_row = cur_row / init_row.clamp(min=1e-30)
        ratio_col = cur_col / init_col.clamp(min=1e-30)
        # Power-law mapping; each axis gets the same exponent so the
        # rank-1 product covers the joint scaling. power_exp=1 keeps
        # per-element step magnitude constant as |W| grows (kernel's
        # 1/|W| cancels the linear discount growth).
        p = self._discount_power_exp
        d_row = ratio_row.pow(p).clamp(min=self._discount_min,
                                        max=self._discount_max)
        d_col = ratio_col.pow(p).clamp(min=self._discount_min,
                                        max=self._discount_max)
        self.discount_row.copy_(d_row.to(self.discount_row.device))
        self.discount_col.copy_(d_col.to(self.discount_col.device))

    def _ensure_backward_buffers(self):
        """Lazily allocate the small persistent buffers used by the
        backward path: row_max (out_features,) int32, col_max
        (in_features,) int32. These receive atomic_max writes from
        apply_update_*_from_grad_W; caller zeros them before each use.

        Originally also pre-allocated a bf16 (out, in) grad_W buffer,
        but that was ~7 GB persistent across SDXL's 1059 layers --
        enough to push VRAM over the 24 GB ceiling. grad_W is now
        allocated fresh inside backward (`torch.matmul(grad_y.T, x)`
        without out=). Inside a captured graph, that allocation goes
        into the graph pool, and since only one layer's grad_W is
        live at any moment in the autograd traversal, the pool's
        peak grad_W footprint is just the largest single layer
        (~30 MB), not the sum (~7 GB). Net save ~7 GB at the cost of
        an extra ~10-30 us alloc per layer per backward in the
        dynamic path (negligible vs the per-microbatch wall time).

        Returns (row_max_buf, col_max_buf). The grad_W slot in the
        return is None for compatibility with call sites that haven't
        been updated yet."""
        s_shape = self.s_slow.shape
        device = self.s_slow.device
        rm = getattr(self, '_row_max_buf', None)
        if (rm is None or rm.shape[0] != s_shape[0]
                or rm.device != device):
            self._row_max_buf = torch.empty(
                s_shape[0], dtype=torch.int32, device=device)
        cm = getattr(self, '_col_max_buf', None)
        if (cm is None or cm.shape[0] != s_shape[1]
                or cm.device != device):
            self._col_max_buf = torch.empty(
                s_shape[1], dtype=torch.int32, device=device)
        # Keep the 3-tuple return shape so call sites don't break;
        # the first slot is now None (grad_W is allocated in backward).
        return None, self._row_max_buf, self._col_max_buf

    def forward(self, x):
        in_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        orig_shape = None
        if x.dim() > 2:
            orig_shape = x.shape
            x = x.view(-1, self.in_features).contiguous()
        else:
            x = x.contiguous()

        # Optional per-element β2 EMA diagnostic (set up via
        # enable_v_diagnostic). When active, the autograd Function
        # backward will EMA-update v_diag with a torch-computed grad_W².
        v_diag = getattr(self, 'v_diag', None)
        v_diag_beta2 = float(getattr(self, 'v_diag_beta2', 0.999))
        if v_diag is not None:
            # Bump step counter; backward sees the updated value via the
            # layer ref captured below (we stash a weakref-style accessor
            # on ctx so the Function can also bump v_diag_steps after the
            # EMA update). Simpler: bump here under the assumption that
            # forward+backward run in lock-step.
            self.v_diag_steps = int(getattr(self, 'v_diag_steps', 0)) + 1

        # Two AdamW variance sources are supported: 'three_accum'
        # (drift-cancelled noise residual, 40 bits/param) and 'v_rank1'
        # (Adafactor rank-1 EMA, 32 bits/param + O(N+K) per layer).
        # When v_rank1 is active, close out the EMA from the PRIOR
        # step's g² accumulation now (v = β2·v + g2; g2 = 0) via one
        # fused Triton kernel per buffer, then bump v_step so the bias
        # correction in the kernel is current.
        optimizer_v_kind = getattr(self, 'optimizer_v_kind', 'three_accum')
        v_row = getattr(self, 'v_row', None)
        v_col = getattr(self, 'v_col', None)
        g2_row = getattr(self, 'g2_row', None)
        g2_col = getattr(self, 'g2_col', None)
        v_beta2 = float(getattr(self, 'v_beta2', 0.999))
        if (self.optimizer_kind == 'adamw'
                and optimizer_v_kind == 'v_rank1'
                and v_row is not None):
            from concord_triton_fused import v_ema_close
            with torch.no_grad():
                v_ema_close(v_row, g2_row, v_beta2)
                v_ema_close(v_col, g2_col, v_beta2)
            self.v_step = int(getattr(self, 'v_step', 0)) + 1
        v_step = int(getattr(self, 'v_step', 0))

        v_slow_buf = getattr(self, 'v_slow', None)
        v_scale = float(getattr(self, 'v_scale', 1.0))
        drift_cancel_C = float(getattr(self, 'drift_cancel_C', 0.1))
        alpha_v_fast = float(getattr(self, 'alpha_v_fast', 0.001))
        # Int8 v_slow accumulator for the SGD three-accumulator path.
        # Allocated lazily via enable_v_slow_i8(); when unset, the kernel
        # behaviour is identical to the classic two-accumulator chase.
        v_slow_i8_buf = getattr(self, 'v_slow_i8', None)
        v_slow_factor = int(getattr(self, 'v_slow_factor', 128))
        wd_sv = float(getattr(self, 'wd_sv', 0.0))
        wd_sf = float(getattr(self, 'wd_sf', 0.0))
        # Persistent bf16 weight buffer reused across steps. This is
        # restored from the pre-refactor baseline -- the briefly-tried
        # fresh-alloc variant cost 1059 torch.empty calls per forward
        # (doubled by grad-ckpt's recompute = 2118/microbatch) at
        # ~10-30 us each, ~30 ms / microbatch of pure dispatch
        # overhead. With grad ckpt enabled and _grad_W_buf removed,
        # the ~7 GB persistent cost fits comfortably.
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if (wbuf is None or wbuf.shape != self.s_slow.shape
                or wbuf.device != self.s_slow.device):
            wbuf = torch.empty(self.s_slow.shape, dtype=torch.bfloat16,
                                device=self.s_slow.device)
            self._bf16_weight_buf = wbuf
        grad_W_buf, row_max_buf, col_max_buf = self._ensure_backward_buffers()
        y = FusedConcordLinear.apply(
            x, self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            self.bias, self.MANTISSA_BIAS, self.lr, self.alpha, self.beta1,
            self.optimizer_kind,
            self.weight_decay, self.eps, float(getattr(self, 'step_cap', 10.0)),
            optimizer_v_kind, v_row, v_col, g2_row, g2_col,
            v_beta2, v_step,
            v_diag, v_diag_beta2,
            v_slow_buf, v_scale, drift_cancel_C, alpha_v_fast,
            v_slow_i8_buf, v_slow_factor,
            wbuf, wd_sv, wd_sf,
            bool(getattr(self, '_apply_chase', True)),
            grad_W_buf, row_max_buf, col_max_buf,
        )

        if orig_shape is not None:
            y = y.view(*orig_shape[:-1], self.out_features)
        return y.to(in_dtype)

    @torch.no_grad()
    def sync_weights(self):
        # No-op: there is no cached weight to sync. Provided for API
        # compatibility with ConcordConvNet / qtridiag training loops.
        pass

    @torch.no_grad()
    def rebalance(self):
        """Tick-up-only fused rebalance. When a row's or col's max-magnitude
        exceeds MAX_M, the corresponding exponent is bumped up and the
        row/col's mantissas right-shifted (stochastically rounded). Tick-
        down was removed when forward emission moved to CLZ-bitcast — see
        _rebalance_decide_apply_kernel in concord_triton.py for the
        reasoning."""
        # bump the process-global counter so every (layer, call) keys a
        # distinct tick-up SR stream — no cross-layer rounding correlation.
        ConcordLinearFused._reb_seed += 1
        # Snapshot exponents so we can SR-rescale v_slow_i8 to match
        # whatever row_exp/col_exp tick-ups the rebalance commits.
        v_slow_i8_present = getattr(self, 'v_slow_i8', None) is not None
        if v_slow_i8_present:
            row_exp_pre = self.row_exp.clone()
            col_exp_pre = self.col_exp.clone()
        rebalance_fused_triton(
            self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            MAX_M=self.MAX_M, EXP_MAX=self.EXP_MAX,
            max_iters=self.max_iters,
            seed=ConcordLinearFused._reb_seed)
        self.s_slow.clamp_(self.INT16_MIN, self.INT16_MAX)
        self.s_fast.clamp_(self.INT16_MIN, self.INT16_MAX)
        if v_slow_i8_present:
            d_row = self.row_exp.to(torch.int32) - row_exp_pre.to(torch.int32)
            d_col = self.col_exp.to(torch.int32) - col_exp_pre.to(torch.int32)
            self._v_slow_apply_exp_shift(d_row, d_col)
            # Defensive int8 pass: catches v_slow_i8 saturation that the
            # int16 rebalance didn't trigger because s_slow+s_fast was
            # still in range.
            self._v_slow_i8_rebalance()
        # Three-accumulator: per-rebalance v_slow leak toward s_slow at
        # alpha_v_slow (default 0.01). Stochastic-rounded into int16 so it
        # stays in the same storage class. The complementary per-step
        # v_slow <- s_fast leak lives inside the AdamW kernel.
        if getattr(self, 'optimizer_v_kind', None) == 'three_accum' \
                and self.v_slow is not None:
            alpha_v_slow = float(getattr(self, 'alpha_v_slow', 0.01))
            gap = (self.s_slow.to(torch.float32)
                   - self.v_slow.to(torch.float32))
            delta_f = alpha_v_slow * gap
            floor = torch.floor(delta_f)
            frac = delta_f - floor
            # Per-element SR threshold from a uniform RNG. Same pattern
            # the kernel uses for s_slow / v_slow ticks.
            u = torch.rand(gap.shape, device=gap.device, dtype=torch.float32)
            tick = (floor + (u < frac).to(torch.float32)).to(torch.int32)
            new_v = (self.v_slow.to(torch.int32) + tick).clamp(
                self.INT16_MIN, self.INT16_MAX).to(torch.int16)
            self.v_slow.copy_(new_v)

        # SGD int8 three-accumulator: per-rebalance v_slow ← s_slow
        # leak. Leak operates in s_slow's mantissa space; ticks quantise
        # into v_slow's int8 space (1 unit = v_slow_factor mantissa
        # units). When mass_preserve=True (Option A), s_slow loses
        # exactly what v_slow gains (in mantissa units) so the live
        # weight is unchanged. When mass_preserve=False (Option B), the
        # leak is a second chase — the live weight grows by what v_slow
        # gained, mirroring the existing s_fast/s_slow chase.
        v_slow_i8 = getattr(self, 'v_slow_i8', None)
        if v_slow_i8 is not None:
            alpha_v_slow = float(getattr(self, 'alpha_v_slow', 0.01))
            factor = int(getattr(self, 'v_slow_factor', 128))
            mass_preserve = bool(getattr(self, 'v_slow_mass_preserve', True))
            v_eff = v_slow_i8.to(torch.int32) * factor
            gap_full = (self.s_slow.to(torch.int32) - v_eff).to(torch.float32)
            delta_v8 = alpha_v_slow * gap_full / factor
            floor = torch.floor(delta_v8)
            frac = delta_v8 - floor
            u = torch.rand(gap_full.shape, device=gap_full.device,
                           dtype=torch.float32)
            tick_v8 = (floor + (u < frac).to(torch.float32)).to(torch.int32)
            new_v_int32 = v_slow_i8.to(torch.int32) + tick_v8
            new_v_int8 = new_v_int32.clamp(-128, 127).to(torch.int8)
            # Compute the actual committed tick (post-clamp) BEFORE
            # overwriting v_slow_i8, in case the mass-preserving branch
            # needs it.
            actual_tick_v8 = (new_v_int8.to(torch.int32)
                              - v_slow_i8.to(torch.int32))
            v_slow_i8.copy_(new_v_int8)
            if mass_preserve:
                actual_tick_full = actual_tick_v8 * factor
                new_slow = (self.s_slow.to(torch.int32) - actual_tick_full).clamp(
                    self.INT16_MIN, self.INT16_MAX).to(torch.int16)
                self.s_slow.copy_(new_slow)

    @torch.no_grad()
    def refit_envelope(self, target_M=16384, n_iter=4, decay=1.0):
        """Periodic two-axis envelope re-fit. Alternately anchor every row-max
        and every column-max of |s_slow+s_fast| to target_M, value-preservingly
        (each exponent tick paired with the inverse mantissa shift) -- a
        max-plus Sinkhorn on the exponents. It pulls the separable (rank-1)
        row+column magnitude structure into row_exp/col_exp, so no row or
        column is left systematically denormal; the non-separable residual
        stays scattered in the mantissa. Clock-triggered and value-targeted, so
        unlike reactive column tick-down it does not feed the qtridiag
        counter-tick loop.

        decay < 1.0 multiplies the mantissas first -- this is how weight decay
        rides the re-fit. The per-step factor (1-lambda) is sub-integer and
        rounds away on the int mantissa; the accumulated (1-lambda)^N over the
        re-fit interval is supra-integer and survives. The exponent re-fit that
        follows is value-preserving, so it re-envelopes the decayed weights
        without undoing the decay."""
        if decay < 1.0:
            for buf in (self.s_slow, self.s_fast):
                scaled = (buf.float() * decay).round().clamp_(
                    self.INT16_MIN, self.INT16_MAX)
                buf.copy_(scaled.to(torch.int16))
        zero_row = torch.zeros(self.out_features, dtype=torch.int32,
                               device=self.s_slow.device)
        zero_col = torch.zeros(self.in_features, dtype=torch.int32,
                               device=self.s_slow.device)
        v_slow_i8_present = getattr(self, 'v_slow_i8', None) is not None
        for _ in range(n_iter):
            mag = (self.s_slow.to(torch.int32)
                   + self.s_fast.to(torch.int32)).abs()
            row_max = mag.amax(dim=1).clamp(min=1).float()
            d = torch.round(torch.log2(row_max / target_M)).to(torch.int32)
            d = (self.row_exp + d).clamp(self.EXP_MIN, self.EXP_MAX) - self.row_exp
            apply_ticks_triton(self.s_slow, d, zero_col)
            apply_ticks_triton(self.s_fast, d, zero_col)
            self.row_exp.add_(d)
            self.s_slow.clamp_(self.INT16_MIN, self.INT16_MAX)
            self.s_fast.clamp_(self.INT16_MIN, self.INT16_MAX)
            if v_slow_i8_present:
                self._v_slow_apply_exp_shift(
                    d, torch.zeros_like(self.col_exp, dtype=torch.int32))
            mag = (self.s_slow.to(torch.int32)
                   + self.s_fast.to(torch.int32)).abs()
            col_max = mag.amax(dim=0).clamp(min=1).float()
            d = torch.round(torch.log2(col_max / target_M)).to(torch.int32)
            d = (self.col_exp + d).clamp(self.EXP_MIN, self.EXP_MAX) - self.col_exp
            apply_ticks_triton(self.s_slow, zero_row, d)
            apply_ticks_triton(self.s_fast, zero_row, d)
            self.col_exp.add_(d)
            self.s_slow.clamp_(self.INT16_MIN, self.INT16_MAX)
            self.s_fast.clamp_(self.INT16_MIN, self.INT16_MAX)
            if v_slow_i8_present:
                self._v_slow_apply_exp_shift(
                    torch.zeros_like(self.row_exp, dtype=torch.int32), d)
        # row_exp / col_exp just changed; in the |W|-via-CLZ AdamW path
        # the per-element variance proxy is recomputed from the new
        # row/col_exp on every step, so no v-state reset is needed.
        # discount_t is independent of the storage scale, refreshed
        # separately via update_discount_from_cascade on a cadence.

    @torch.no_grad()
    def enable_osc_damp(self):
        """Allocate the per-weight velocity-sign buffer `vsign` that
        damp_oscillation() needs. Costs one bool/param of HBM, so it is
        opt-in: a layer that never damps never pays for it. Idempotent;
        primes `vsign` to the current velocity sign so the first
        damp_oscillation() call registers no flips."""
        if self.vsign is None:
            self.vsign = self.s_fast >= self.s_slow

    @torch.no_grad()
    def damp_oscillation(self, damp):
        """Oscillation-damping regularizer. `vsign` holds the previous sign of
        the velocity (s_fast - s_slow). A weight whose velocity sign has
        flipped since the last call is oscillating with respect to its slow
        accumulator -- uncommitted -- so s_fast is biased a fraction `damp`
        toward s_slow (damp=1 snaps it, killing the velocity; damp<1 bleeds it
        geometrically per flip). Every flipped weight is damped; monotone,
        committed weights never flip. An importance-weighted regularizer.

        A one-way gate -- apply the bias only where it shrinks |w| -- was tried
        and underperformed: that condition selects by weight *size* (it fires
        where s_fast dominates), not by oscillation, so it damps large weights
        and skips the small jittery ones. Damping every flipped weight wins.

        Returns the number of weights damped this call."""
        self.enable_osc_damp()        # lazily allocate vsign on first use
        s_slow = self.s_slow.to(torch.int32)
        s_fast = self.s_fast.to(torch.int32)
        ge = s_fast >= s_slow
        flipped = ge != self.vsign
        delta = (damp * (s_fast - s_slow).float()).round().to(torch.int32)
        new_fast = torch.where(flipped, s_fast - delta, s_fast)
        new_fast.clamp_(self.INT16_MIN, self.INT16_MAX)
        moved = int((new_fast != s_fast).sum().item())
        self.s_fast.copy_(new_fast.to(torch.int16))
        self.vsign.copy_(new_fast >= s_slow)
        return moved   # weights actually damped (a flip with |delta| >= 1)

    # API shims for code expecting an nn.Linear-like interface.
    @property
    def weight(self):
        """Live bf16 weight tensor with the right shape, dtype, and
        values. Used by anything that introspects `.weight.dtype` /
        `.shape` / `.device` (HuggingFace T5, OneTrainer's param-group
        machinery, debugging hooks, etc.), plus anything that actually
        reads `.weight.data` (BMA materialize, ad-hoc checkpoint code).

        Re-reconstructed on each access via the same Triton recon
        kernel that the forward path uses, into a lazy-cached layer
        buffer. Cost per access: one ~30µs Triton kernel launch. Per-
        step access is not the expected use case — the autograd
        Function already does its own recon inside `forward()` and
        saves the buffer for backward.

        ConcordConv2dFused overrides this to return the 4D view of the
        same buffer.
        """
        from concord_triton_fused import materialize_bf16_weight
        cache = getattr(self, '_bf16_weight_buf', None)
        if (cache is None
                or cache.shape != self.s_slow.shape
                or cache.device != self.s_slow.device):
            cache = torch.empty(self.s_slow.shape, dtype=torch.bfloat16,
                                 device=self.s_slow.device)
            self._bf16_weight_buf = cache
        materialize_bf16_weight(
            self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            mantissa_bias=self.MANTISSA_BIAS,
            v_slow=getattr(self, 'v_slow_i8', None),
            v_slow_factor=int(getattr(self, 'v_slow_factor', 128)),
            out=cache)
        return cache

    # qtridiag reads .mantissa and .m
    @property
    def mantissa(self):
        return self.s_slow

    @property
    def m(self):
        return self.s_fast


class ConcordConv2dFused(ConcordLinearFused):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, device='cuda', max_iters=2,
                 alpha=0.1, beta1=0.0, lr=0.05):
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = kh, kw
        self.stride = stride
        self.padding = padding
        super().__init__(in_features=in_channels * kh * kw,
                         out_features=out_channels,
                         bias=bias, device=device, max_iters=max_iters,
                         alpha=alpha, beta1=beta1, lr=lr)

    @property
    def weight(self):
        """Live bf16 weight reshaped to (C_out, C_in, KH, KW). Same
        recon-on-access path as ConcordLinearFused.weight, viewed as
        4D so anything that introspects a Conv2d weight (shape,
        dtype, slicing) sees the right tensor layout."""
        w2d = ConcordLinearFused.weight.fget(self)  # the recon
        return w2d.view(self.out_channels, self.in_channels,
                         self.kh, self.kw)

    def forward(self, x):
        # Materialised bf16 weight + cuDNN conv. Persistent weight buf
        # reused across steps -- see ConcordLinearFused.forward.
        from concord_triton_fused import FusedConcordConv2d
        in_dtype = x.dtype
        v_slow_i8_buf = getattr(self, 'v_slow_i8', None)
        v_slow_factor = int(getattr(self, 'v_slow_factor', 128))
        alpha_v_fast = float(getattr(self, 'alpha_v_fast', 0.001))
        wbuf = getattr(self, '_bf16_weight_buf', None)
        if (wbuf is None or wbuf.shape != self.s_slow.shape
                or wbuf.device != self.s_slow.device):
            wbuf = torch.empty(self.s_slow.shape, dtype=torch.bfloat16,
                                device=self.s_slow.device)
            self._bf16_weight_buf = wbuf
        grad_W_buf, row_max_buf, col_max_buf = self._ensure_backward_buffers()
        y = FusedConcordConv2d.apply(
            x, self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            self.bias,
            self.in_channels, self.out_channels, self.kh, self.kw,
            self.stride, self.padding,
            self.MANTISSA_BIAS, self.lr, self.alpha, self.beta1,
            v_slow_i8_buf, v_slow_factor, alpha_v_fast,
            wbuf, bool(getattr(self, '_apply_chase', True)),
            grad_W_buf, row_max_buf, col_max_buf)
        return y.to(in_dtype)



# ============================================================
# ConcordEmbeddingFused: int-storage replacement for nn.Embedding.
# Same per-row exponent scheme as ConcordLinearFused, just keyed by
# token id instead of output-feature index. Forward gathers indexed
# rows from int state into bf16; backward applies sparse SR-tick +
# chase to the unique rows that received gradient.
# ============================================================


class ConcordEmbeddingFused(nn.Module):
    """Drop-in replacement for `nn.Embedding` with concord int storage.

    Shape contract: `(num_embeddings, embedding_dim)` — matches
    `nn.Embedding.weight.shape`. Forward takes integer `input_ids` of
    any shape and returns bf16 of shape `(*input_ids.shape, embedding_dim)`.

    Storage (per element): 32 bits (s_slow int16 + s_fast int16) vs
    `nn.Embedding`'s 32-bit fp32 weight + AdamW's 64-bit m+v = 96 bits
    total. Optional `enable_v_slow_i8()` adds an int8 v_slow buffer
    for the three-accumulator path (40 bits/param total, same as the
    Linear three_accum variant).

    No support for `padding_idx`, `max_norm`, `norm_type`,
    `scale_grad_by_freq`, `sparse` — those nn.Embedding knobs are not
    wired here. The most common SDXL/CLIP usage doesn't need any of
    them.
    """
    # Inherit the format constants from ConcordLinearFused — same int
    # range, same exponent envelope, same MANTISSA_BIAS.
    MANTISSA_BIAS = ConcordLinearFused.MANTISSA_BIAS
    INT16_MIN = ConcordLinearFused.INT16_MIN
    INT16_MAX = ConcordLinearFused.INT16_MAX
    EXP_MIN = ConcordLinearFused.EXP_MIN
    EXP_MAX = ConcordLinearFused.EXP_MAX

    def __init__(self, num_embeddings, embedding_dim,
                 device='cuda', alpha=0.1, lr=1e-3):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.lr = lr
        self.register_buffer('s_slow',
            torch.zeros(num_embeddings, embedding_dim,
                        dtype=torch.int16, device=device))
        self.register_buffer('s_fast',
            torch.zeros(num_embeddings, embedding_dim,
                        dtype=torch.int16, device=device))
        self.register_buffer('row_exp',
            torch.zeros(num_embeddings, dtype=torch.int8, device=device))
        self.register_buffer('col_exp',
            torch.zeros(embedding_dim, dtype=torch.int8, device=device))
        # Lazily allocated by enable_v_slow_i8(); None = two-accumulator.
        self.v_slow_i8 = None
        self.v_slow_factor = 128
        self.alpha_v_fast = 0.001
        # The autograd graph needs SOMETHING with requires_grad=True on
        # the forward call so backward runs (and our sparse concord
        # update kernel fires). The Linear/Conv2d paths get this from
        # the bias Parameter; nn.Embedding has no bias. We add a single
        # zero-valued fp32 Parameter that the autograd Function takes
        # as one of its inputs — its grad is always None (we return
        # None for it in backward), so it never moves. Costs 4 bytes.
        # Excluded from state_dict by the patch in
        # onetrainer_concord_patch._patch_concord_state_dict.
        self._grad_anchor = nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device=device))
        # Random init that mirrors nn.Embedding's default (N(0, 1)).
        self._init_concord()

    @torch.no_grad()
    def _init_concord(self):
        W = torch.randn(self.num_embeddings, self.embedding_dim,
                        device=self.s_slow.device)
        self.load_weights(W)

    @torch.no_grad()
    def load_weights(self, W):
        """Decompose `W` (shape `(num_embeddings, embedding_dim)`,
        any float dtype) into the concord int state. Same row/col
        exponent recipe as the Linear path."""
        W = W.to(device=self.s_slow.device, dtype=torch.float32)
        assert W.shape == (self.num_embeddings, self.embedding_dim), (
            f'load_weights expected {(self.num_embeddings, self.embedding_dim)}, '
            f'got {tuple(W.shape)}')
        max_abs_row = W.abs().max(dim=1).values.clamp(min=1e-30)
        self.row_exp.copy_(
            torch.ceil(torch.log2(max_abs_row) + 1.0)
            .clamp(self.EXP_MIN, self.EXP_MAX).to(torch.int8))
        self.col_exp.zero_()
        exp = (self.row_exp[:, None] + self.col_exp[None, :]
               - self.MANTISSA_BIAS).float()
        scale = torch.pow(2.0, exp)
        m_total = (W / scale).round().to(torch.int32).clamp(
            self.INT16_MIN, self.INT16_MAX)
        half = (m_total / 2).round().to(torch.int32)
        self.s_slow.copy_(half)
        self.s_fast.copy_(m_total - half)
        if self.v_slow_i8 is not None:
            self.v_slow_i8.zero_()

    @torch.no_grad()
    def load_weights_finetune(self, W):
        """Bayesian-prior decomposition for fine-tuning a pretrained
        embedding. Same 1/3-1/3-1/3 split as
        `ConcordLinearFused.load_weights_finetune` — see its docstring
        for the steady-state / noise-residual rationale. Allocates
        v_slow_i8 if not already enabled.
        """
        if self.v_slow_i8 is None:
            self.enable_v_slow_i8()
        W = W.to(device=self.s_slow.device, dtype=torch.float32)
        assert W.shape == (self.num_embeddings, self.embedding_dim), (
            f'load_weights_finetune expected '
            f'{(self.num_embeddings, self.embedding_dim)}, '
            f'got {tuple(W.shape)}')
        max_abs_row = W.abs().max(dim=1).values.clamp(min=1e-30)
        self.row_exp.copy_(
            torch.ceil(torch.log2(max_abs_row) + 1.0)
            .clamp(self.EXP_MIN, self.EXP_MAX).to(torch.int8))
        self.col_exp.zero_()
        exp = (self.row_exp[:, None] + self.col_exp[None, :]
               - self.MANTISSA_BIAS).float()
        scale = torch.pow(2.0, exp)
        m_total = (W / scale).round().to(torch.int32)
        target_v_full = (m_total.float() / 3.0).round().to(torch.int32)
        v_slow_int = (target_v_full.float() / float(self.v_slow_factor)
                       ).round().to(torch.int32).clamp(-128, 127)
        actual_v_full = v_slow_int * self.v_slow_factor
        remaining = m_total - actual_v_full
        half = (remaining / 2).round().to(torch.int32)
        self.s_slow.copy_(half.clamp(self.INT16_MIN, self.INT16_MAX))
        self.s_fast.copy_((remaining - half).clamp(
            self.INT16_MIN, self.INT16_MAX))
        self.v_slow_i8.copy_(v_slow_int.to(torch.int8))

    @torch.no_grad()
    def enable_v_slow_i8(self, factor=128, alpha_v_fast=0.001):
        """Allocate the int8 v_slow accumulator for the three-
        accumulator path. Same semantics as ConcordLinearFused —
        non-mass-preserving leak from s_fast to v_slow. Idempotent."""
        if self.v_slow_i8 is None:
            self.v_slow_i8 = torch.zeros_like(self.s_slow, dtype=torch.int8)
        self.v_slow_factor = int(factor)
        self.alpha_v_fast = float(alpha_v_fast)

    def set_lr(self, lr):
        self.lr = lr

    @property
    def weight(self):
        """Live bf16 weight reconstructed from int state on access.
        Same recon-on-access semantics as ConcordLinearFused.weight,
        so anything inspecting `.weight.shape` / `.dtype` / `.data`
        sees a real Embedding-shaped tensor."""
        from concord_triton_fused import materialize_bf16_weight
        return materialize_bf16_weight(
            self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            mantissa_bias=self.MANTISSA_BIAS,
            v_slow=self.v_slow_i8, v_slow_factor=self.v_slow_factor,
        )

    def forward(self, input_ids):
        from concord_triton_fused import FusedConcordEmbedding
        return FusedConcordEmbedding.apply(
            input_ids,
            self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            self.v_slow_i8,
            self.lr, self.alpha, self.alpha_v_fast,
            self.v_slow_factor, self.MANTISSA_BIAS,
            self._grad_anchor,
            bool(getattr(self, '_apply_chase', True)),
        )

    def extra_repr(self):
        return (f'num_embeddings={self.num_embeddings}, '
                f'embedding_dim={self.embedding_dim}, '
                f'v_slow_i8={"on" if self.v_slow_i8 is not None else "off"}')
