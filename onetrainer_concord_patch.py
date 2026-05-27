"""OneTrainer patch for Concord SGD.

Concord SGD is an int16-storage optimizer that reaches AdamW-tied accuracy
on CIFAR-10 / tiny-shakespeare / SST-2 at one-third the per-parameter state.
The optimizer doesn't have a separate `step()` for the layers it manages --
the weight update is fused into the backward pass via Triton kernels.

The core modules (concord_linear_fused, concord_triton_fused,
concord_triton, fused_profiler) live alongside this file in the
concord_onetrainer package; install() ensures the package directory is
on sys.path so they resolve regardless of CWD.

To wire this into OneTrainer, three things have to happen:

  1. The concord_onetrainer package directory must be on sys.path so
     concord_linear_fused, concord_triton_fused etc. are importable.
     install() handles this.

  2. The trainer must cache the loaded model on this module so that the
     CONCORD_SGD case in modules/util/create.py can find which nn.Linears
     to swap. GenericTrainer.py sets `_cached_model` after the loader
     returns; install() doesn't have to do anything for this beyond defining
     the slot.

  3. The TrainConfig schema and create_optimizer dispatch must know about
     the CONCORD_SGD enum value and its hyperparameters. Those live in
     modules/util/{enum/Optimizer.py, optimizer_util.py, create.py} and
     modules/util/config/TrainConfig.py.

Usage:
    1. Copy this file into your OneTrainer installation directory along
       with the rest of the concord_*.py modules.
    2. Add to the top of scripts/train.py (and scripts/train_ui.py):
         import onetrainer_concord_patch
         onetrainer_concord_patch.install()
    3. In your TrainConfig (or the GUI), select optimizer = CONCORD_SGD.
       The standard fields are read from `config.optimizer.concord_*`:

         concord_alpha           slow-chase rate                (default 0.1)
         concord_beta1           velocity feedback              (default 0.0)
         concord_aux_lr          AdamW lr for biases/norms       (default 1e-4)
         concord_rebalance_every steps between rebalance calls  (default 8)
         concord_refit_every     steps between refit_envelope   (default 250)
         concord_refit_target    int mantissa anchor             (default 16384)
         concord_tickdown        'off' | 'row' | 'alt'           (default 'row')
         concord_lr_flat_after   pin LR at step K (0 = off)     (default 0)
         concord_lr_flat_frac    LR fraction in flat region     (default 0.0)
         concord_qtridiag        Q-aware tridiag coupling        (default False)
         concord_qt_refresh      steps between Q refresh          (default 3000)
         concord_qtridiag_pairs  regex restricting discovered     (default None
                                  boundaries (e.g. 'fc1->fc2')      = all)
         concord_bma_obs_every   BMA centroid cadence (0 = off) (default 0)
         concord_polyak_bias     Polyak-leak hypothesis selector (default TRUE
                                  -- the v3_polyak config that beat the README
                                  headline; acceptance gate makes it safe to
                                  default on.)
         concord_polyak_observe_every  cascade observe cadence    (default 8)
         concord_polyak_leak     EMA rate Polyak -> H            (default 0.05)
         concord_polyak_commit   commit strength on accept       (default 0.1)
         concord_polyak_probe_every  steps between probes        (default 200)
         concord_polyak_level    cascade level driving Polyak    (default 1)
         concord_polyak_warmup   min cascade fill before activate (default 2)
         concord_polyak_temperature  MH temperature scale T_0;   (default 0.0)
                                  effective T = T_0/n_blocks,     (= greedy)
                                  anneals as cascade fills.
         concord_target_modules  regex of module paths to wrap (default '.*')
         concord_aux_optimizer   'adamw' | 'sgd' for the aux set    (default 'adamw';
                                  'sgd' matches train_cifar_fused.py's bias optimizer
                                  -- use for CIFAR-style nets, leave AdamW for SDXL)

Trainer integration for Polyak: at the start of training, call
opt.set_polyak_probe(x, y, criterion) once with a fixed probe minibatch
and the loss function. The wrapper then auto-fires the probe + commit at
the configured cadence; no per-step trainer code needed. (Alternatively
call opt.polyak.probe_and_commit(...) manually for variable probe batches.)

    This wrapper supports the classic 32-bit concord format
    (ConcordLinearFused + ConcordConv2dFused) only, with the optional
    int8 v_slow three-accumulator AdamW path documented in CONCORD_README.md.
    Experimental flags from the upstream research (dual-fast 24/16-bit,
    gauge_anneal, hull_clamp, async_refit, osc_damp) are NOT wired here
    -- they were verified unsupported, broken, or unhelpful for the
    OneTrainer (image-generation) use case; see the
    concord_optimizer.py module docstring "Scope" section for the audit.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stash slots: the trainer fills these in before create_optimizer runs.
# ---------------------------------------------------------------------------

_cached_train_config = None
_cached_model = None


def cache_model(model):
    """Called by GenericTrainer after the model loader returns. Lets the
    CONCORD_SGD dispatch in create.py find the model to wrap."""
    global _cached_model
    _cached_model = model


def cache_train_config(cfg):
    """Optional twin of cache_model -- not strictly required (create.py
    receives the config directly) but useful if a downstream caller
    wants to read the train config without re-plumbing it."""
    global _cached_train_config
    _cached_train_config = cfg


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

_CONCORD_SRC = str(Path(__file__).resolve().parent)


def _patch_concord_conv2d_init():
    """Make ConcordConv2dFused tolerate the tuple stride/padding/kernel_size
    that PyTorch's nn.Conv2d hands out.

    The upstream class stores `stride` and `padding` verbatim, and the Triton
    kernel uses them as scalars (e.g. ``H_in + 2 * padding - kh``). A bare
    ``nn.Conv2d(..., padding=1)`` round-trips its constructor args through
    ``_pair`` and ends up with ``self.padding == (1, 1)``; passing that
    straight into ``ConcordConv2dFused`` crashes the kernel with
    ``int + tuple``.

    This patch wraps ``__init__`` so that any symmetric 2-tuple becomes the
    scalar the kernel expects, and asymmetric pairs raise a clear message
    pointing at ``concord_target_modules``. It is idempotent and a no-op
    if the user constructs ``ConcordConv2dFused`` directly with int args
    (the original behaviour).
    """
    try:
        import concord_linear_fused as _fl
    except Exception:                              # caller will surface this
        return
    Cls = _fl.ConcordConv2dFused
    if getattr(Cls, "_ot_inttuple_patched", False):
        return

    def _collapse(v, attr):
        if isinstance(v, int):
            return v
        if isinstance(v, tuple) and len(v) == 2 and v[0] == v[1]:
            return int(v[0])
        raise ValueError(
            f"ConcordConv2dFused requires symmetric {attr} (same value on "
            f"both H and W); got {attr}={v}. The Triton kernel uses the "
            f"value as a scalar. Narrow concord_target_modules to exclude "
            f"this layer or rebuild the conv with a symmetric {attr}.")

    _orig_init = Cls.__init__

    def _patched_init(self, in_channels, out_channels, kernel_size,
                       stride=1, padding=0, *args, **kw):
        stride = _collapse(stride, "stride")
        padding = _collapse(padding, "padding")
        # kernel_size: pass int through, collapse symmetric tuple to int,
        # leave asymmetric tuple alone -- the kernel handles kh/kw separately.
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2 \
                and kernel_size[0] == kernel_size[1]:
            kernel_size = int(kernel_size[0])
        _orig_init(self, in_channels, out_channels, kernel_size,
                   stride=stride, padding=padding, *args, **kw)

    Cls.__init__ = _patched_init
    Cls._ot_inttuple_patched = True
    logger.info("Patched ConcordConv2dFused.__init__ for tuple "
                "stride/padding (idempotent).")


def _patch_concord_linear_forward():
    """Make ConcordLinearFused.forward tolerate non-contiguous inputs whose
    stride pattern .view() refuses (SDXL's UNet hands proj_in a strided
    tensor that spans two contiguous subspaces, where .view() raises
    "view size is not compatible ... Use .reshape(...) instead.").

    Switches to .reshape, which falls back to a copy when needed. Same fix
    on ConcordConv2dFused via inheritance (its forward calls a different
    Triton path but doesn't view).

    NOTE: this used to also wrap forward with torch.compiler.disable to
    handle gradient-checkpointing's Dynamo-compile region. That made the
    problem worse -- Dynamo treats torch.compiler.disable as a hard error
    inside compile regions ("Skip inlining torch.compiler.disable()'d
    function") rather than as a graph break. The Dynamo side of the issue
    is instead handled by _patch_dynamo_suppress_errors() below, which
    sets the global config to fall back to eager on Unsupported ops.

    Idempotent.
    """
    try:
        import concord_linear_fused as _fl
    except Exception:
        return
    Cls = _fl.ConcordLinearFused
    if getattr(Cls, "_ot_view_reshape_patched", False):
        return
    import torch
    from concord_triton_fused import FusedConcordLinear

    def _patched_forward(self, x):
        in_dtype = x.dtype
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        orig_shape = None
        if x.dim() > 2:
            orig_shape = x.shape
            # .reshape instead of .view: handles non-contiguous strided
            # inputs (SDXL proj_in case) by copying when necessary.
            x = x.reshape(-1, self.in_features).contiguous()
        else:
            x = x.contiguous()
        # Optimizer-kind routing. SGD: classic SR + chase. AdamW: either
        # 'three_accum' (40 bits/param, drift-cancelled noise residual)
        # or 'v_rank1' (32 bits/param + O(N+K) Adafactor rank-1).
        kind = getattr(self, "optimizer_kind", "sgd")
        weight_decay = float(getattr(self, "weight_decay", 0.0))
        eps = float(getattr(self, "eps", 1e-8))
        v_diag = getattr(self, "v_diag", None)
        v_diag_beta2 = float(getattr(self, "v_diag_beta2", 0.999))
        if v_diag is not None:
            self.v_diag_steps = int(getattr(self, "v_diag_steps", 0)) + 1
        # v_rank1 close-out + state pass-through (mirrors
        # ConcordLinearFused.forward).
        optimizer_v_kind = getattr(self, "optimizer_v_kind", "three_accum")
        v_row = getattr(self, "v_row", None)
        v_col = getattr(self, "v_col", None)
        g2_row = getattr(self, "g2_row", None)
        g2_col = getattr(self, "g2_col", None)
        v_beta2 = float(getattr(self, "v_beta2", 0.999))
        if (kind == 'adamw' and optimizer_v_kind == 'v_rank1'
                and v_row is not None):
            from concord_triton_fused import v_ema_close
            with torch.no_grad():
                v_ema_close(v_row, g2_row, v_beta2)
                v_ema_close(v_col, g2_col, v_beta2)
            self.v_step = int(getattr(self, "v_step", 0)) + 1
        v_step = int(getattr(self, "v_step", 0))
        v_slow_buf = getattr(self, "v_slow", None)
        v_scale = float(getattr(self, "v_scale", 1.0))
        drift_cancel_C = float(getattr(self, "drift_cancel_C", 0.1))
        alpha_v_fast = float(getattr(self, "alpha_v_fast", 0.001))
        # Int8 v_slow accumulator + knobs; None when enable_v_slow_i8
        # hasn't been called on this layer.
        v_slow_i8_buf = getattr(self, "v_slow_i8", None)
        v_slow_factor = int(getattr(self, "v_slow_factor", 128))
        # Persistent bf16 weight buffer (lazy-alloc) -- restored from
        # the pre-refactor baseline. See ConcordLinearFused.forward.
        wbuf = getattr(self, "_bf16_weight_buf", None)
        if (wbuf is None or wbuf.shape != self.s_slow.shape
                or wbuf.device != self.s_slow.device):
            wbuf = torch.empty(self.s_slow.shape, dtype=torch.bfloat16,
                                device=self.s_slow.device)
            self._bf16_weight_buf = wbuf
        # Pre-allocated backward-path buffers (Phase 1-3 of the
        # launch-overhead refactor). Required for CUDA graph capture
        # and a perf win on its own. The _ensure_backward_buffers
        # helper lives on ConcordLinearFused; we call it through the
        # patched instance.
        from concord_linear_fused import ConcordLinearFused
        grad_W_buf, row_max_buf, col_max_buf = \
            ConcordLinearFused._ensure_backward_buffers(self)
        y = FusedConcordLinear.apply(
            x, self.s_slow, self.s_fast, self.row_exp, self.col_exp,
            self.bias, self.MANTISSA_BIAS, self.lr, self.alpha, self.beta1,
            kind, weight_decay, eps,
            float(getattr(self, "step_cap", 10.0)),
            optimizer_v_kind, v_row, v_col, g2_row, g2_col,
            v_beta2, v_step,
            v_diag, v_diag_beta2,
            v_slow_buf, v_scale, drift_cancel_C, alpha_v_fast,
            v_slow_i8_buf, v_slow_factor,
            wbuf,
            float(getattr(self, "wd_sv", 0.0)),
            float(getattr(self, "wd_sf", 0.0)),
            bool(getattr(self, "_apply_chase", True)),
            grad_W_buf, row_max_buf, col_max_buf,
        )
        if orig_shape is not None:
            y = y.reshape(*orig_shape[:-1], self.out_features)
        return y.to(in_dtype)

    Cls.forward = _patched_forward
    Cls._ot_view_reshape_patched = True
    logger.info("Patched ConcordLinearFused.forward to use .reshape "
                "instead of .view (handles SDXL non-contiguous inputs).")


def _patch_step_counter_pure():
    """Retired: the upstream linear/conv SGD launchers and the
    apply_update launcher now use a tensor-backed step counter
    (`_get_step_counter` in concord_triton_fused.py). The launcher
    increments the counter via in-place ``add_`` on a 1-elem int32 GPU
    tensor and passes the tensor's pointer to the kernel, which reads
    the value with ``tl.load``. Tensor in-place ops are Dynamo-
    traceable under HOP gradient checkpointing, so the upstream
    launchers are already HOP-safe; the previous static-salt override
    (0xC0FFEE) is no longer needed and actively harmful (a static salt
    biases per-element SR rounding because the per-step error doesn't
    average out).

    Pre-warms the cuda:0 counter so the first traced call doesn't have
    to write to the module-level dict under HOP. Idempotent.
    """
    try:
        import concord_triton_fused as _ftf
        import torch
        if torch.cuda.is_available():
            _ftf._get_step_counter(torch.device("cuda:0"))
        logger.info("Step-counter patch retired (upstream launchers now "
                    "use a tensor-backed counter; cuda:0 pre-warmed).")
    except Exception as e:
        logger.warning(f"step-counter pre-warm skipped: {e!r}")


def _patch_checkpoint_use_reentrant_true():
    """Force torch.utils.checkpoint.checkpoint to use the legacy
    (reentrant, non-HOP) backend whenever concord layers are inside
    a checkpointed block.

    Why this matters: diffusers' ``unet.enable_gradient_checkpointing()``
    wraps each transformer block in
    ``torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)``.
    The ``use_reentrant=False`` path routes through TorchDynamo's
    HigherOrderOperator (HOP). Dynamo then tries to trace the concord
    Triton autograd.Function calls through the HOP. It can't:

      - The Triton kernels are opaque to Dynamo.
      - The kernels mutate int16 buffers (s_slow / s_fast) during
        backward; Dynamo treats buffer mutation as a hard error inside
        HOPs (no graph breaks allowed).
      - autograd.Function.apply() inside a HOP isn't reliably traceable
        when the apply body is non-Pythonic.

    Net result: Dynamo's guards fail every step, the HOP cache evicts,
    Dynamo recompiles. We measured ~200s per step at SDXL scale with
    use_reentrant=False -- about 100x slower than the actual compute.

    The legacy (reentrant) path uses saved-tensor hooks and the
    standard autograd-recompute mechanism. It bypasses Dynamo entirely:
    the forward runs in eager mode, gets re-run during backward, and
    the concord autograd.Function.apply runs normally both times.

    Caveats of use_reentrant=True (all OK for SDXL UNet):
      - All inputs must be Tensors (no Python ints/bools). diffusers'
        create_custom_forward closure absorbs the non-Tensor kwargs
        before calling checkpoint, so checkpoint() only sees Tensors.
      - Cannot reuse the same Tensor object as two different args. The
        UNet blocks pass hidden_states / temb / encoder_hidden_states
        as distinct Tensors.
      - Doesn't support keyword arguments. Same closure caveat.

    Idempotent. The original checkpoint function is preserved so other
    code that genuinely needs use_reentrant=False keeps working --
    we only force True when the caller passes use_reentrant=False or
    omits it; an explicit use_reentrant=True passes through unchanged.

    To disable this patch (e.g. for a non-concord optimizer test in
    the same process), set env var OT_CONCORD_NO_CHECKPOINT_PATCH=1
    before install() runs.
    """
    import os
    if os.environ.get("OT_CONCORD_NO_CHECKPOINT_PATCH"):
        logger.info("Skipping checkpoint use_reentrant patch "
                    "(OT_CONCORD_NO_CHECKPOINT_PATCH set).")
        return
    import torch.utils.checkpoint as _ckpt
    if getattr(_ckpt, "_ot_force_reentrant", False):
        return
    _orig_checkpoint = _ckpt.checkpoint

    def _force_reentrant_checkpoint(function, *args,
                                     use_reentrant=None, **kwargs):
        # Force True unless the caller explicitly asked for True.
        # (False or None both get coerced to True.)
        if use_reentrant is None or use_reentrant is False:
            use_reentrant = True
        return _orig_checkpoint(function, *args,
                                 use_reentrant=use_reentrant, **kwargs)

    _ckpt.checkpoint = _force_reentrant_checkpoint
    _ckpt._ot_force_reentrant = True
    _ckpt._ot_orig_checkpoint = _orig_checkpoint
    logger.info("Patched torch.utils.checkpoint to force "
                "use_reentrant=True (bypasses Dynamo HOP -- gradient "
                "checkpointing now uses legacy saved-tensor-hooks path, "
                "eliminating ~200s/step recompile penalty on concord "
                "modules in HOP-wrapped blocks).")


def _patch_concord_state_dict():
    """Make ConcordLinearFused / ConcordConv2dFused emit standard
    ``weight`` (and ``bias``) state_dict keys instead of the internal
    int16 buffers (``s_slow``, ``s_fast``, ``row_exp``, ``col_exp``,
    ``vsign``).

    Without this, ``model.unet.state_dict()`` on an SDXL run with a
    concord UNet returns keys like ``conv_in.s_slow`` -- and the SDXL
    checkpoint converter (``modules/util/convert/convert_sdxl_diffusers_to_ckpt.py``)
    crashes with ``KeyError: 'conv_in.weight'`` because every ``map_wb``
    in the converter explicitly indexes by ``<layer>.weight``. Same for
    diffusers ``save_pretrained`` (which writes state_dict keys directly).

    The reconstruction is the same one used by the BMA centroid path:
    ``W = (s_slow + s_fast) * 2^(row_exp + col_exp - MANTISSA_BIAS)``.
    For conv layers we reshape from the (out, in*kh*kw) internal layout
    back to (out, in, kh, kw) for nn.Conv2d compatibility.

    We also override ``_load_from_state_dict`` so a concord module can
    re-ingest its own saved checkpoint (consume the ``weight`` key by
    routing it through ``load_weights()``, then copy bias). This lets
    OneTrainer's "resume from checkpoint" flow round-trip a concord
    training run.

    Idempotent.
    """
    try:
        import concord_linear_fused as _fl
    except Exception:
        return
    Lin = _fl.ConcordLinearFused
    Conv = _fl.ConcordConv2dFused
    Emb = getattr(_fl, "ConcordEmbeddingFused", None)
    if getattr(Lin, "_ot_state_dict_patched", False):
        return
    import torch

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Reconstruct fp32 weight from int mantissas + shared exponents.
        # MUST include v_slow_i8's contribution at shifted scale
        # V_SLOW_FACTOR when that buffer is allocated — the live
        # forward path (materialize_bf16_weight) adds it, so the
        # saved weight has to match. Without this term, three-
        # accumulator runs (SDXL TE training with the Bayesian-prior
        # init, or any CIFAR run that enabled enable_v_slow_i8) save
        # checkpoints that silently drop the v_slow contribution and
        # the resulting safetensors no longer matches the model that
        # produced it.
        with torch.no_grad():
            exp = (self.row_exp[:, None] + self.col_exp[None, :]
                   - self.MANTISSA_BIAS).float()
            m = (self.s_slow.to(torch.int32)
                 + self.s_fast.to(torch.int32))
            v_slow_i8 = getattr(self, 'v_slow_i8', None)
            if v_slow_i8 is not None:
                factor = int(getattr(self, 'v_slow_factor', 128))
                m = m + v_slow_i8.to(torch.int32) * factor
            W = m.float() * torch.exp2(exp)
            if isinstance(self, Conv):
                W = W.reshape(self.out_channels, self.in_channels,
                              self.kh, self.kw)
        # The saver downcasts to the user-selected dtype (typically fp16)
        # via DtypeModelSaverMixin._convert_state_dict_dtype; emitting
        # fp32 here is correct and lossless.
        destination[prefix + "weight"] = W if keep_vars else W.detach()
        bias = getattr(self, "bias", None)
        if bias is not None:
            destination[prefix + "bias"] = bias if keep_vars else bias.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys,
                               error_msgs):
        wkey = prefix + "weight"
        bkey = prefix + "bias"
        if wkey in state_dict:
            W = state_dict.pop(wkey)
            try:
                if isinstance(self, Conv):
                    W2d = W.reshape(self.out_channels,
                                    self.in_channels * self.kh * self.kw)
                else:
                    W2d = W
                self.load_weights(W2d)
                # load_weights re-stuffs the entire W into
                # s_slow + s_fast (50/50) but does NOT touch
                # v_slow_i8. Since the saved W now includes the
                # v_slow contribution, we have to zero v_slow_i8
                # explicitly here so the live weight after load
                # equals the saved W rather than W + v_slow_full.
                # The long-time v_slow signal is lost on resume;
                # training continues from zero v_slow and warms up
                # over ~1/alpha_v_fast steps. (To preserve v_slow
                # across save/load, we'd need a separate raw-int
                # state file — not what the SDXL checkpoint format
                # supports.)
                v_slow_i8 = getattr(self, 'v_slow_i8', None)
                if v_slow_i8 is not None:
                    v_slow_i8.zero_()
            except Exception as e:
                error_msgs.append(
                    f"While loading {wkey} into concord module: {e}")
        bias = getattr(self, "bias", None)
        if bias is not None and bkey in state_dict:
            b = state_dict.pop(bkey)
            try:
                bias.data.copy_(b.to(bias.dtype))
            except Exception as e:
                error_msgs.append(
                    f"While loading {bkey} into concord module: {e}")
        # Strip any leftover concord buffer keys (s_slow etc.) silently
        # in case a checkpoint was saved BEFORE this patch was installed.
        for suffix in ("s_slow", "s_fast", "row_exp", "col_exp", "vsign"):
            state_dict.pop(prefix + suffix, None)
        # We have intentionally NOT consumed the buffers via the default
        # path; remove them from the "missing" tracking PyTorch wants to
        # populate. We do this by injecting placeholder no-op buffers --
        # but the cleaner approach is to mark them via the
        # _non_persistent_buffers_set, which is set up per-instance in
        # __init__. We don't touch that here; instead we explicitly clear
        # the relevant missing-key entries after the default
        # super-class call would have ADDED them. Easiest: don't call
        # super (we've already consumed weight+bias above), and report
        # nothing as missing for the concord buffers because their
        # values are reconstructed from weight on load.

    Lin._save_to_state_dict = _save_to_state_dict
    Lin._load_from_state_dict = _load_from_state_dict
    Lin._ot_state_dict_patched = True
    # ConcordEmbeddingFused is NOT a subclass of ConcordLinearFused
    # (different __init__ contract, no bias, no in_features /
    # out_features) so we attach the same patches to it explicitly.
    # The reconstruction is identical: W = (s_slow + s_fast) * 2^exp,
    # already in the right (vocab, dim) shape for nn.Embedding.
    if Emb is not None:
        Emb._save_to_state_dict = _save_to_state_dict
        Emb._load_from_state_dict = _load_from_state_dict
        Emb._ot_state_dict_patched = True
    logger.info("Patched ConcordLinearFused/Conv2dFused state_dict: "
                "emit standard weight/bias keys for SDXL checkpoint "
                "converter compatibility.")


def _patch_dynamo_suppress_errors():
    """Configure Dynamo to be as permissive as possible with the concord
    kernels' execution.

      1. suppress_errors = True: fall back to eager when Dynamo hits the
         concord kernels' side effects (module-level _step_counter
         mutation -- now removed by _patch_step_counter_pure -- and
         int-buffer in-place writes). Belt-and-suspenders with the
         purified salt.

      2. cache_size_limit raised from default 8 to 256: gradient
         checkpointing's HOP wrapper retraces the concord forward+backward
         for each unique input shape; SDXL UNet has many shape variants
         (different attention layer widths and seq lengths). At the
         default cache size of 8, Dynamo exhausts the cache after a few
         dozen unique shapes and then RECOMPILES every step, costing
         ~15-30 sec per recompile and 3-4 min per training step. Raising
         the limit lets Dynamo cache all variants once and stabilize.

      3. accumulated_recompile_limit raised similarly so the cache stays
         valid across many steps.

      4. automatic_dynamic_shapes = True (default but set explicitly):
         after the first recompile due to a shape change, Dynamo treats
         that dim as symbolic, further reducing future recompiles.

      5. TORCHDYNAMO_SUPPRESS_ERRORS env var for child processes.
    """
    import os
    import torch
    try:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        # accumulated_recompile_limit exists in PyTorch 2.4+; guard for older.
        if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
            torch._dynamo.config.accumulated_recompile_limit = 1024
        if hasattr(torch._dynamo.config, "automatic_dynamic_shapes"):
            torch._dynamo.config.automatic_dynamic_shapes = True
    except Exception as e:
        logger.warning(f"Could not set Dynamo config: {e}")
    os.environ.setdefault("TORCHDYNAMO_SUPPRESS_ERRORS", "1")
    logger.info("Dynamo config: suppress_errors=True, cache_size_limit=256, "
                "accumulated_recompile_limit=1024, automatic_dynamic_shapes="
                "True (lets gradient-checkpointing recompiles stabilize "
                "instead of recompiling every step).")


def install():
    """Make Concord SGD importable from anywhere in the OneTrainer process.

    Idempotent. Safe to call multiple times (it just no-ops on the second
    call). Side effects:

      1. The directory containing this file is added to sys.path (once)
         so the concord_* modules are importable from anywhere in the
         OneTrainer process.
      2. ``concord_linear_fused`` is smoke-imported so a broken Triton /
         CUDA install surfaces here rather than at first training step.
      3. ``ConcordConv2dFused.__init__`` is monkeypatched to accept the
         tuple ``stride``/``padding`` that PyTorch's nn.Conv2d hands out;
         see ``_patch_concord_conv2d_init`` for the rationale.
      4. ``ConcordLinearFused.forward`` is monkeypatched to use ``.reshape``
         instead of ``.view`` so non-contiguous inputs (e.g. SDXL proj_in
         after an attention reshape) don't trigger ``view`` errors.
    """
    if _CONCORD_SRC not in sys.path:
        sys.path.insert(0, _CONCORD_SRC)

    # Smoke-import: catch a missing-Triton or missing-CUDA install early
    # rather than at first training step.
    try:
        import concord_linear_fused  # noqa: F401
        logger.info(
            "Concord SGD patch installed (sys.path += %s)", _CONCORD_SRC)
    except Exception as e:
        # Don't raise -- the user may have selected a non-concord
        # optimizer, in which case the concord modules are never used.
        # The CONCORD_SGD dispatch case will raise a clearer error if it
        # tries to use them.
        logger.warning(
            "Concord SGD modules not importable yet (will be required "
            "only if you select CONCORD_SGD): %s", e)
        return

    _patch_concord_conv2d_init()
    _patch_concord_linear_forward()
    _patch_step_counter_pure()
    _patch_concord_state_dict()
    _patch_checkpoint_use_reentrant_true()
    _patch_dynamo_suppress_errors()


if __name__ == "__main__":
    print("OneTrainer Concord SGD Patch")
    print("=============================")
    print()
    print("Usage:")
    print("  1. Copy this file into your OneTrainer directory.")
    print("  2. Add to the top of scripts/train.py and scripts/train_ui.py,")
    print("     before the trainer is constructed:")
    print("       import onetrainer_concord_patch")
    print("       onetrainer_concord_patch.install()")
    print("  3. Select optimizer = CONCORD_SGD in your config.")
    install()
