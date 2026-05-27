"""OneTrainer wrapper around Concord SGD.

Concord SGD does not have a separate `optimizer.step()` for the layers it
manages: the weight update is fused into the backward pass via Triton kernels,
and the int16 mantissa state is written back as a side effect of the autograd
Function's backward. Everything that PyTorch would expect to live on
nn.Parameter (the bf16 weight) is reconstructed in registers and never sits in
HBM.

To plug this into OneTrainer (whose training loop is just
zero_grad -> backward -> step), this module:

  1. Swaps every nn.Linear and nn.Conv2d in the configured target_modules
     regex for a ConcordLinearFused / ConcordConv2dFused (classic 32
     bits/param: int16 s_slow + int16 s_fast), initialised from the
     pretrained weight via load_weights(). After this, the concord layers'
     int state is registered as buffers (not Parameters), so OneTrainer's
     parameter-group collection naturally excludes them; any surviving bias
     stays an nn.Parameter and feeds the aux set.

  2. Wraps a plain torch.optim.AdamW (the "aux" optimizer for non-concord
     parameters: biases, norm gains, embeddings) in a ConcordSGD class
     that exposes the OneTrainer-standard surface (param_groups, step,
     zero_grad, state_dict, load_state_dict). The wrapper:

     - in zero_grad(): propagates the current per-group LR to every concord
       layer via m.set_lr() so the upcoming backward picks it up. The
       concord LR follows the aux LR by a fixed ratio captured at
       construction (config.learning_rate / concord_aux_lr).
     - in step(): calls aux.step() to update aux params, then fires the
       periodic concord callbacks at their configured cadences:
       rebalance_every, refit_every, qtridiag (with optional Q refresh),
       BMA centroid accumulation.

Scope -- what's supported here:

  Format         classic (32 bits/param) only. Dual-fast (24/16 bits) is
                 dropped because it has no Conv2d variant upstream, which
                 makes it dead-on-arrival for image-generation use cases
                 (SDXL UNet, Flux DiT, etc).

  Mechanisms     rebalance, refit_envelope, qtridiag (generalized to conv
                 chains via consecutive-layer divisibility), LR cliff
                 (lr_flat_after / lr_flat_frac), BMA centroid (with
                 Conv2d materialisation).

  Dropped        gauge_anneal / hull_clamp (T5-residual-stream specific,
                 destroys training on non-T5 architectures), async_refit,
                 osc_damp (transformer-only findings, untested
                 elsewhere).

Every retained knob is exposed via TrainOptimizerConfig fields prefixed
`concord_`. Defaults match the original scripts' CIFAR-applicable
settings.
"""

import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

# The concord_sgd core modules (concord_linear_fused, concord_triton_fused,
# concord_triton, fused_profiler) live alongside this file in
# C:\concord_onetrainer. Make sure our own directory is on sys.path so
# `import concord_linear_fused` resolves locally regardless of CWD.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Module-wrapping helpers
# ---------------------------------------------------------------------------


def _resolve_target_regex(pat):
    """Compile the concord_target_modules string into a regex matcher.
    Empty / None / '.*' all mean "every nn.Linear"."""
    if pat is None or pat == "" or pat == ".*":
        return re.compile(r".*")
    return re.compile(pat)


def _named_modules_safe(model):
    """Like nn.Module.named_modules(), but works on OneTrainer-style
    container models (which aren't nn.Modules themselves but hold nn.Modules
    as attributes: model.unet, model.text_encoder_1, etc.). Yields
    (full_dotted_name, module) for every nn.Module reachable from `model`."""
    if isinstance(model, nn.Module):
        yield from model.named_modules()
        return
    # Container: prefix the attribute name onto each submodule's path.
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(model, attr_name)
        except Exception:
            continue
        if isinstance(val, nn.Module):
            yield attr_name, val
            for sub_name, sub_mod in val.named_modules():
                if sub_name:
                    yield f"{attr_name}.{sub_name}", sub_mod


def _resolve_wrap_roots(model, parameter_dicts=None):
    """Return [(root_prefix, root_nn_module), ...] -- the set of top-level
    nn.Module subtrees we are allowed to walk into and wrap.

    Three cases:

      1. `model` IS an nn.Module (the CIFAR / standalone test case): return
         [("", model)] so we walk it directly.

      2. `model` is an OneTrainer-style container with nn.Module attributes
         (StableDiffusionXLModel / FluxModel / etc., which holds `unet`,
         `text_encoder_1`, `text_encoder_2`, `vae`, ...), AND the caller
         provided parameter_dicts: filter to those nn.Module attributes
         whose name matches a parameter group's `name` field. This restricts
         wrapping to what OneTrainer's parameter groups asked to train --
         a UNet-only fine-tune doesn't accidentally wrap the frozen text
         encoders.

      3. Same container model but no parameter_dicts hint: walk every
         nn.Module attribute we find (last-resort fallback).
    """
    if isinstance(model, nn.Module):
        return [("", model)]
    # Container: collect nn.Module attributes.
    candidates = []
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(model, attr_name)
        except Exception:
            continue
        if isinstance(val, nn.Module):
            candidates.append((attr_name, val))
    # Filter by parameter group names if supplied.
    if parameter_dicts:
        wanted = {g.get("name") for g in parameter_dicts if g.get("name")}
        filtered = [(n, m) for n, m in candidates if n in wanted]
        if filtered:
            return filtered
    return candidates


def _collect_wrappable(model, target_re, parameter_dicts=None,
                          wrap_embeddings=False):
    """Return [(kind, full_name, parent, attr, child), ...] for every
    candidate module whose dotted path matches target_re.

    `kind` is 'linear' for nn.Linear, 'conv2d' for nn.Conv2d, and
    'embedding' for nn.Embedding (only when wrap_embeddings=True —
    nn.Embedding has a sparse-gradient forward that needs a separate
    update kernel, and most non-LLM users don't want token-embedding
    matrices on int storage by default).

    Skip conditions:
      - attr name is 'lm_head' or 'shared' (common HuggingFace tied-weight
        modules).
      - Conv2d with groups != 1 / dilation != 1 / padding_mode != 'zeros':
        ConcordConv2dFused only implements plain stride+padding convs.
        These layers are left untouched (they remain in the aux set).
      - nn.Embedding with padding_idx / max_norm / scale_grad_by_freq /
        sparse: ConcordEmbeddingFused doesn't honor those knobs and
        would silently change behaviour. Left in the aux set.

    Handles two model shapes: an nn.Module directly (CIFAR test), or an
    OneTrainer container (StableDiffusionXLModel etc.) whose nn.Module
    attributes we walk individually. See _resolve_wrap_roots.
    """
    out = []
    skip_attrs = {"lm_head", "shared"}
    for root_prefix, root in _resolve_wrap_roots(model, parameter_dicts):
        for pname, parent in root.named_modules():
            for attr, child in parent.named_children():
                if attr in skip_attrs:
                    continue
                if pname:
                    sub = f"{pname}.{attr}"
                else:
                    sub = attr
                full = f"{root_prefix}.{sub}" if root_prefix else sub
                if not target_re.search(full):
                    continue
                if isinstance(child, nn.Linear):
                    out.append(("linear", full, parent, attr, child))
                elif isinstance(child, nn.Conv2d):
                    # Concord conv kernel does not implement groups,
                    # dilation, or non-zero padding modes. Leave any non-
                    # plain conv in the aux set rather than silently
                    # miscompiling.
                    dilation = (child.dilation
                                if isinstance(child.dilation, tuple)
                                else (child.dilation, child.dilation))
                    if (child.groups == 1
                            and dilation == (1, 1)
                            and child.padding_mode == "zeros"):
                        out.append(("conv2d", full, parent, attr, child))
                elif wrap_embeddings and isinstance(child, nn.Embedding):
                    # ConcordEmbeddingFused only implements the plain
                    # lookup → bf16 path. Skip embeddings that use any
                    # of the optional nn.Embedding behaviours.
                    if (child.padding_idx is None
                            and child.max_norm is None
                            and not child.scale_grad_by_freq
                            and not child.sparse):
                        out.append(("embedding", full, parent, attr, child))
    return out


# Backward-compatibility alias: earlier code paths only handled nn.Linear.
def _collect_linears(model, target_re):
    return [(name, parent, attr, child)
            for kind, name, parent, attr, child
            in _collect_wrappable(model, target_re)
            if kind == "linear"]


def _backfill_concord_from_foliated(optimizer_config):
    """OneTrainer-fork backward-compat shim. The upstream OneTrainer
    TrainConfig schema still declares fields named foliated_* (alpha,
    beta1, rebalance_every, …); once the OneTrainer refactor lands the
    schema will switch to concord_* and this shim becomes a no-op.
    Until then, any foliated_X attribute is mirrored to concord_X if
    the latter isn't already explicitly set.

    Safe to call repeatedly; idempotent. Best-effort: silently ignores
    attributes that can't be set (e.g. read-only config wrappers)."""
    for attr in list(dir(optimizer_config)):
        if not attr.startswith('foliated_'):
            continue
        new_name = 'concord_' + attr[len('foliated_'):]
        if getattr(optimizer_config, new_name, None) is None:
            try:
                setattr(optimizer_config, new_name,
                        getattr(optimizer_config, attr))
            except (AttributeError, TypeError):
                pass


def wrap_model(model, optimizer_config, device=None, parameter_dicts=None):
    """Replace nn.Linear and nn.Conv2d modules in `model` (in-place) with
    ConcordLinearFused / ConcordConv2dFused, initialised from the
    pretrained weights.

    The original nn.Parameter (`.weight`) of each swapped module becomes
    unreferenced after `setattr(parent, attr, concord)`; PyTorch will GC it.
    A surviving bias nn.Parameter is preserved on the concord module so the
    aux optimizer can still train biases as before.

    `parameter_dicts`, when given, restricts wrapping to nn.Module
    attributes of `model` whose attribute name matches a parameter group's
    `name` field. For OneTrainer-style training (e.g. a UNet-only fine-tune
    where parameter_dicts == [{'name': 'unet', ...}]) this means only
    model.unet is wrapped, not text_encoder / vae. See
    _resolve_wrap_roots for the full dispatch.

    Returns the list of concord module instances that now live in the
    model -- the caller passes this list to ConcordSGD so the periodic
    rebalance/refit callbacks can iterate them.
    """
    _backfill_concord_from_foliated(optimizer_config)
    alpha = (optimizer_config.concord_alpha
             if optimizer_config.concord_alpha is not None else 0.1)
    beta1 = (optimizer_config.concord_beta1
             if optimizer_config.concord_beta1 is not None else 0.0)
    # concord_tickdown is now a no-op — rebalance is tick-up only since
    # the forward kernel emits bf16 via CLZ-bitcast (small mantissas are
    # absorbed by per-element h, no tick-down needed). The config field
    # is read here for backward compat with existing presets but ignored.
    _ = optimizer_config.concord_tickdown
    refit_target = (optimizer_config.concord_refit_target
                    if optimizer_config.concord_refit_target is not None
                    else 16384)
    target_re = _resolve_target_regex(
        optimizer_config.concord_target_modules)

    if device is None:
        # Don't try to read model.parameters() -- OneTrainer overwrites that
        # attribute with a NamedParameterGroupCollection in
        # init_model_parameters before the optimizer is built. The concord
        # Triton kernels require CUDA anyway; fall back to CPU only if CUDA
        # is genuinely unavailable (smoke test on CPU-only box).
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    from concord_linear_fused import (ConcordConv2dFused,
                                        ConcordEmbeddingFused,
                                        ConcordLinearFused)

    wrap_embeddings = bool(
        getattr(optimizer_config, "concord_wrap_embeddings", False))
    # Bayesian-prior fine-tune init: split the pretrained weight 1/3
    # each into s_slow / s_fast / v_slow at wrap time. See
    # ConcordLinearFused.load_weights_finetune for the steady-state /
    # noise-residual rationale. Default off so existing
    # from-random-init runs (e.g. the CIFAR 90.91 reproduction) get
    # the same load_weights path as before.
    finetune_init = bool(
        getattr(optimizer_config, "concord_finetune_init", False))
    candidates = _collect_wrappable(
        model, target_re,
        parameter_dicts=parameter_dicts,
        wrap_embeddings=wrap_embeddings,
    )

    def _init_concord_layer(fol, W):
        """Pick the right init path based on the finetune flag. Same
        method on Linear / Conv2d / Embedding — ConcordEmbeddingFused
        has its own implementation; ConcordConv2dFused inherits from
        ConcordLinearFused so it gets it via MRO."""
        if finetune_init:
            fol.load_weights_finetune(W)
        else:
            fol.load_weights(W)

    concord = []
    n_conv = 0
    n_lin = 0
    n_emb = 0
    # Per-layer progress logging. Each Triton kernel compiles + autotunes
    # on first call for each new shape (refit_envelope is invoked
    # per-layer below). On SDXL UNet (~800 layers) that compilation cost
    # is concentrated in the first few layers, then is cached; the rest
    # are fast. Without logging, the first invocation looks like a hang
    # for several minutes. Print every N layers so it's visible.
    n_total = len(candidates)
    progress_every = max(1, n_total // 20)
    import time as _time
    _t_wrap_start = _time.time()
    if n_total > 50:
        print(f"[Concord] wrapping {n_total} candidate modules "
              f"(this can take a few minutes on first run as Triton "
              f"compiles + autotunes the kernels)...", flush=True)

    for i_layer, (kind, name, parent, attr, lin) in enumerate(candidates):
        if kind == "embedding":
            # nn.Embedding → ConcordEmbeddingFused. Separate path because
            # the forward is a lookup (not a matmul) and the backward
            # update is sparse — only the touched rows tick.
            num_emb, emb_dim = lin.weight.shape
            fol = ConcordEmbeddingFused(num_emb, emb_dim,
                                          device=device, alpha=alpha,
                                          lr=0.0)
            _init_concord_layer(fol, lin.weight.data)
            setattr(parent, attr, fol)
            concord.append(fol)
            n_emb += 1
            if n_total > 50 and ((i_layer + 1) % progress_every == 0
                                  or i_layer == n_total - 1):
                dt = _time.time() - _t_wrap_start
                print(f"[Concord]   {i_layer+1}/{n_total} concord  "
                      f"({dt:.1f}s elapsed)", flush=True)
            continue

        if kind == "linear":
            out_f, in_f = lin.weight.shape
            has_bias = lin.bias is not None
            fol = ConcordLinearFused(in_f, out_f, bias=has_bias,
                                       device=device, alpha=alpha,
                                       beta1=beta1, lr=0.0)
            W2d = lin.weight.data
            n_lin += 1
        else:  # conv2d
            out_ch = lin.out_channels
            in_ch = lin.in_channels
            ks = (lin.kernel_size if isinstance(lin.kernel_size, tuple)
                  else (lin.kernel_size, lin.kernel_size))
            kh, kw = ks
            has_bias = lin.bias is not None
            # `lin.stride` and `lin.padding` are 2-tuples coming out of
            # nn.Conv2d; ConcordConv2dFused's Triton kernel uses both as
            # scalars. The tuple -> scalar normalisation is handled by the
            # monkeypatch installed by onetrainer_concord_patch.install(),
            # which is called before any concord optimizer runs. We pass
            # the tuples through verbatim; the patch collapses symmetric
            # pairs and raises on asymmetric ones.
            fol = ConcordConv2dFused(in_ch, out_ch, ks, stride=lin.stride,
                                       padding=lin.padding, bias=has_bias,
                                       device=device, alpha=alpha,
                                       beta1=beta1, lr=0.0)
            # The concord conv stores weights as a 2D (out_ch, in_ch*kh*kw)
            # matrix internally; load_weights expects that flattened form.
            W2d = lin.weight.data.reshape(out_ch, in_ch * kh * kw)
            n_conv += 1

        _init_concord_layer(fol, W2d)

        # load_weights() copied the data into the int16 concord state -- an
        # independent copy. The original lin.weight Parameter is held by
        # OneTrainer's NamedParameterGroupCollection (cached before
        # wrapping), so we can't just let setattr replace the module and
        # expect GC -- the Parameter survives and its storage stays pinned
        # (~5 GB on SDXL).
        #
        # Resist the urge to mutate lin.weight.data to an empty tensor:
        # this was tried and broke training -- with the storage cleared,
        # OneTrainer's EMA or autograd-graph setup downstream loses the
        # gradient chain for the entire model. Until we understand exactly
        # WHICH downstream consumer is sensitive to it, keep the original
        # storage alive. The ~5 GB is paid for now; a follow-up can
        # surgically remove the Parameter from OneTrainer's collection
        # AFTER all the downstream init paths have run.
        # (See conversation log in this directory for the failure mode:
        #  "element 0 of tensors does not require grad and does not have
        #   a grad_fn" during loss.backward().)

        if has_bias:
            # Transplant the ORIGINAL bias Parameter (preserve id, dtype,
            # requires_grad) instead of overwriting the concord layer's
            # freshly-allocated bias. The original is what OneTrainer's
            # parameter group references; keeping the same Python object
            # means the aux optimizer continues to see it after the swap.
            fol.bias = lin.bias

        # tickdown removed — rebalance is tick-up only post-CLZ.
        fol.refit_envelope(refit_target)

        setattr(parent, attr, fol)
        concord.append(fol)

        if n_total > 50 and ((i_layer + 1) % progress_every == 0
                              or i_layer == n_total - 1):
            dt = _time.time() - _t_wrap_start
            print(f"[Concord]   {i_layer+1}/{n_total} concord  "
                  f"({dt:.1f}s elapsed)", flush=True)

    if concord:
        dt = _time.time() - _t_wrap_start
        emb_str = f" + {n_emb} nn.Embedding" if n_emb else ""
        print(f"[Concord] swapped {n_lin} nn.Linear + {n_conv} nn.Conv2d{emb_str} "
              f"modules (classic 32-bit, {dt:.1f}s total)", flush=True)

    return concord


def filter_param_groups_to_live(parameter_dicts, model):
    """Given an OneTrainer-style list of parameter group dicts
    ({'name': str, 'params': [Parameter,...], 'lr': float, 'initial_lr': float}),
    drop any Parameter that is no longer reachable from the model (i.e. the
    one we just detached by swapping its containing nn.Linear). Returns a new
    list of dicts; empty groups are kept so OneTrainer's per-group LR
    scheduling stays index-stable, but their `params` list may be empty.

    `model` can be either an nn.Module (CIFAR case) or an OneTrainer
    container holding nn.Module attributes (StableDiffusionXLModel etc.).
    In the container case nn.Module.parameters() doesn't exist (and in
    OneTrainer's flow `model.parameters` was overwritten with a
    NamedParameterGroupCollection by init_model_parameters anyway), so we
    walk via _named_modules_safe and union the parameter ids from each
    reachable nn.Module.
    """
    live_ids = set()
    for _name, sub in _named_modules_safe(model):
        for p in sub.parameters(recurse=False):
            live_ids.add(id(p))
    new_groups = []
    for g in parameter_dicts:
        kept = [p for p in g["params"] if id(p) in live_ids]
        new_g = dict(g)
        new_g["params"] = kept
        new_groups.append(new_g)
    return new_groups


# ---------------------------------------------------------------------------
# The optimizer wrapper
# ---------------------------------------------------------------------------


class ConcordSGD(Optimizer):
    """Composite optimizer: an inner AdamW for the non-concord (aux) params
    plus a list of ConcordLinearFused / ConcordConv2dFused modules whose
    state is updated by their own Triton kernels during backward().

    Surfaces a torch.optim.Optimizer-compatible API (param_groups, step,
    zero_grad, state_dict, load_state_dict) so OneTrainer's training loop,
    LR scheduler, gradient scaler, and grad clipping all work unchanged.

    The concord layers' LR follows the aux LR by a fixed ratio captured at
    construction: ratio = concord_lr / aux_lr (e.g. 0.1 / 1e-4 = 1000).
    OneTrainer's LR scheduler operates on param_groups[i]['lr'] (which is
    aux LR); each zero_grad() reads it back and pushes ratio * aux_lr down
    to every concord layer via set_lr().

    Concord-native gradient accumulation: call ``set_accum_steps(K)``
    once at training start to make every backward only SR-tick s_fast
    for K-1 consecutive microbatches and run the full tick + chase +
    leak on the K-th. ``step()`` advances the cycle counter, runs the
    aux AdamW step, and fires the rebalance / refit cadences once per
    effective step (every K backwards). Default K=1 is the standard
    per-step behaviour. See the README for the noise-preservation
    property.
    """

    def __init__(self, aux_optimizer, concord_layers, optimizer_config,
                 base_concord_lr, base_aux_lr):
        self._aux = aux_optimizer
        self._concord = list(concord_layers)
        self._cfg = optimizer_config
        self._step_count = 0

        # Fixed LR ratio so the concord LR follows aux LR through the
        # OneTrainer scheduler. Guard against base_aux_lr == 0.
        self._fol_to_aux_ratio = (base_concord_lr / max(base_aux_lr, 1e-30)
                                  if base_aux_lr else 1.0)
        self._base_concord_lr = base_concord_lr
        self._base_aux_lr = base_aux_lr

        # Gradient-accumulation cycle. Set by the trainer to match
        # config.gradient_accumulation_steps. accum_pos tracks the
        # microbatch position within the current K-cycle; backward
        # uses (accum_pos == accum_steps - 1) to decide whether to
        # run the chase + v_slow leak (the K-th microbatch) or just
        # SR-tick s_fast (microbatches 0..K-2). Default K=1
        # reproduces the per-step (no-accum) behaviour.
        self._accum_steps = 1
        self._accum_pos = 0
        self._sync_concord_apply_chase()

        # Cadences (all step counts).
        self._rebalance_every = (optimizer_config.concord_rebalance_every
                                 if optimizer_config.concord_rebalance_every
                                 is not None else 8)
        self._refit_every = (optimizer_config.concord_refit_every
                             if optimizer_config.concord_refit_every
                             is not None else 250)
        self._refit_target = (optimizer_config.concord_refit_target
                              if optimizer_config.concord_refit_target
                              is not None else 16384)

        # LR-tail pinning (lr_flat_after / lr_flat_frac in finetune_t5_sst2):
        # cosine until step K, then constant lr_flat_frac * base_lr -- the
        # SWA-friendly schedule that lets BMA centroid average a converged
        # tail.
        self._lr_flat_after = int(optimizer_config.concord_lr_flat_after or 0)
        self._lr_flat_frac = float(optimizer_config.concord_lr_flat_frac
                                    or 0.0)

        # qtridiag and BMA: lazy init on first qualifying step() call.
        self._qtridiag_on = bool(optimizer_config.concord_qtridiag)
        # qt_refresh default raised from 1000 to 3000 per the CIFAR study --
        # 1000 was too aggressive (Schur decomposition built from too few
        # accumulated boundary samples, Q quality lower). 3000 ~ 2 epochs
        # at b=32 / batch=128 typical, matches train_cifar_fused.py's
        # refresh_every=2 epochs.
        self._qt_refresh = (optimizer_config.concord_qt_refresh
                            if optimizer_config.concord_qt_refresh
                            is not None else 3000)
        # Optional regex restricting which discovered boundaries actually get
        # qtridiag wired. Default (None or '') = every consecutive concord
        # pair where R.col_exp.numel() is divisible by L.row_exp.numel().
        # Useful when generalised discovery picks up boundaries the original
        # config opted out of (the README CIFAR headline runs only the
        # fc1->fc2 boundary -- set concord_qtridiag_pairs='fc1->fc2' to
        # reproduce that).
        qt_pairs_pat = getattr(
            optimizer_config, "concord_qtridiag_pairs", None)
        if qt_pairs_pat:
            self._qt_pairs_re = re.compile(qt_pairs_pat)
        else:
            self._qt_pairs_re = None
        self._bma_every = int(optimizer_config.concord_bma_obs_every or 0)

        # Polyak-leak hypothesis selector (see concord_polyak.PolyakHypothesis).
        # DEFAULT OFF.
        #
        # Polyak buys ~+0.5pp on CIFAR-scale (188k concord params) at the
        # cost of +32 bits/concord-param of H state on GPU AND a
        # BoxVelocityMean ring of K (default 5) fp16 snapshots on CPU.
        # At SDXL UNet scale (~2.5B concord params), that is:
        #   - +5 GB GPU just for H (bf16)
        #   - ~25 GB CPU RAM for the ring (K * 2.5B * 2 bytes at K=5)
        # Acceptable on a typical image-gen training box if you have RAM
        # to spare; off by default so the wrapper is zero-cost when unused.
        #
        # The trainer drives the probe cadence by either:
        #   (a) calling opt.polyak.probe_and_commit(model, x, y, criterion)
        #       directly, OR
        #   (b) calling opt.set_polyak_probe(x, y, criterion) ONCE with a
        #       fixed probe batch -- the wrapper then fires probes on its
        #       own at concord_polyak_probe_every cadence.
        # Mode (b) is the simpler default; (a) is for callers who want to
        # vary the probe batch over training.
        self._polyak_on = bool(
            getattr(optimizer_config, "concord_polyak_bias", False))
        self._polyak_observe_every = int(
            getattr(optimizer_config, "concord_polyak_observe_every", None)
            or 8)
        self._polyak = None             # lazy-built PolyakHypothesis
        self._polyak_probe_x = None     # set via set_polyak_probe()
        self._polyak_probe_y = None
        self._polyak_probe_criterion = None

        self._qt_boundaries = []
        self._bma_sums = {}
        self._bma_count = 0
        self._bma_warmup = (self._lr_flat_after if self._lr_flat_after > 0
                            else 0)

        # Pretend to be an Optimizer for type-checks; the param_groups
        # property proxies to the aux optimizer so OneTrainer's lr_scheduler
        # / grad scaler / grad clipping all act on the right list.
        self.defaults = getattr(aux_optimizer, "defaults", {})
        self.state = aux_optimizer.state

    # ------------------------------------------------------------------ #
    # torch.optim.Optimizer surface
    # ------------------------------------------------------------------ #

    @property
    def param_groups(self):
        return self._aux.param_groups

    @param_groups.setter
    def param_groups(self, v):
        self._aux.param_groups = v

    def add_param_group(self, group):
        return self._aux.add_param_group(group)

    # ------------------------------------------------------------------ #
    # Concord-native gradient accumulation
    # ------------------------------------------------------------------ #

    def set_accum_steps(self, k: int):
        """Configure K-microbatch gradient accumulation. Each Concord
        layer's backward SR-ticks s_fast for K-1 microbatches and runs
        the full chase + leak on the K-th. Step() advances the cycle
        and runs the aux AdamW step + rebalance/refit on a per-
        effective-step cadence. Idempotent. K=1 reproduces standard
        per-step behaviour.

        Usage pattern (trainer side)::

            opt.set_accum_steps(K)            # once at training start
            for effective_step in range(N):
                opt.zero_grad()
                for k in range(K):
                    loss = compute(microbatch) / K
                    loss.backward()
                    opt.advance_accum()        # between microbatches
                opt.step()                     # aux + rebalance + cycle reset
        """
        k = max(1, int(k))
        if k == self._accum_steps:
            return
        self._accum_steps = k
        # Reset cycle position so the chase fires exactly on the K-th
        # backward after this call.
        self._accum_pos = 0
        self._sync_concord_apply_chase()

    def advance_accum(self):
        """Bump the microbatch position inside the current
        accumulation cycle. Call after each ``loss.backward()`` inside
        a K-microbatch accumulation. After K calls the cycle wraps to
        0 and the next backward becomes tick-only again (unless K=1,
        in which case every backward is full)."""
        self._accum_pos = (self._accum_pos + 1) % self._accum_steps
        self._sync_concord_apply_chase()

    def _sync_concord_apply_chase(self):
        """Push the current (accum_pos, accum_steps) decision into every
        Concord layer's `_apply_chase` flag. Called by set_accum_steps
        on configuration, advance_accum() between microbatches, and
        zero_grad() at the start of each effective step."""
        apply_chase = (self._accum_pos == self._accum_steps - 1)
        for m in self._concord:
            m._apply_chase = apply_chase

    def state_dict(self):
        sd = self._aux.state_dict()
        sd["_concord_step_count"] = self._step_count
        sd["_concord_bma_count"] = self._bma_count
        # The concord int buffers live on each module as register_buffer'd
        # tensors -- they're already part of the model state_dict, so we
        # don't double-save them here.
        return sd

    def load_state_dict(self, state_dict):
        self._step_count = state_dict.pop("_concord_step_count", 0)
        self._bma_count = state_dict.pop("_concord_bma_count", 0)
        self._aux.load_state_dict(state_dict)

    def zero_grad(self, set_to_none=True):
        self._aux.zero_grad(set_to_none=set_to_none)
        # Reset the gradient-accumulation cycle. zero_grad is called
        # once per effective step (between K-microbatch cycles), so
        # this puts every Concord layer back into tick-only mode for
        # the first microbatch of the next cycle. K=1 stays in
        # full-step mode trivially (apply_chase always True).
        self._accum_pos = 0
        self._sync_concord_apply_chase()
        # Push the current (scheduler-set) aux LR down to every concord
        # layer, scaled by the concord/aux ratio so backward() reads it.
        if self._aux.param_groups:
            cur_aux_lr = float(self._aux.param_groups[0]["lr"])
        else:
            cur_aux_lr = self._base_aux_lr
        fol_lr = cur_aux_lr * self._fol_to_aux_ratio

        # Optional LR-tail pinning: after step K, freeze concord lr at
        # frac * base_concord_lr (or hold whatever LR was at step K if
        # frac == 0). This is the SWA-friendly schedule from
        # finetune_t5_sst2.py: cosine until lr_flat_after, then constant.
        if self._lr_flat_after > 0 and self._step_count > self._lr_flat_after:
            if self._lr_flat_frac > 0.0:
                fol_lr = self._base_concord_lr * self._lr_flat_frac

        for m in self._concord:
            m.set_lr(fol_lr)

    @torch.no_grad()
    def step(self, closure=None):
        # Optional per-step wall-clock timing for debugging
        # gradient-checkpointing / Dynamo recompile pathology.
        # Activate with env OT_CONCORD_TIMING=1. Prints the first 6
        # step times (covers warmup + autotune + steady-state),
        # then every 50th step.
        import os
        _timing = os.environ.get("OT_CONCORD_TIMING")
        if _timing and not hasattr(self, "_t_prev"):
            import time as _time
            self._t_prev = _time.perf_counter()
            self._t_first = self._t_prev
        loss = self._aux.step(closure)
        self._step_count += 1
        s = self._step_count
        if _timing:
            import time as _time
            t_now = _time.perf_counter()
            dt = t_now - self._t_prev
            self._t_prev = t_now
            if s <= 6 or s % 50 == 0:
                tot = t_now - self._t_first
                print(f"[Concord timing] step {s}: dt={dt:.2f}s  "
                      f"total={tot:.1f}s", flush=True)

        # --- Periodic concord housekeeping ---
        if s % self._rebalance_every == 0:
            self._maybe_init_qtridiag()
            if self._qt_boundaries:
                self._rebalance_with_qtridiag()
            else:
                for m in self._concord:
                    m.rebalance()

        if self._refit_every > 0 and s % self._refit_every == 0:
            for m in self._concord:
                m.refit_envelope(self._refit_target)
            # Lazy-path discount update: any layer in AdamW mode gets its
            # discount_t refreshed from the cascade's |W| trajectory at
            # refit cadence. Cheap CPU op per layer; runs only if the
            # polyak cascade was initialised. Falls back to discount_t=1.0
            # if the cascade can't provide enough history yet.
            if self._polyak is not None:
                cascade = getattr(self._polyak, 'cascade', None)
                offsets = getattr(cascade, 'offsets', None)
                if cascade is not None and offsets is not None:
                    for i, m in enumerate(self._concord):
                        if getattr(m, 'optimizer_kind', 'sgd') != 'adamw':
                            continue
                        if i + 1 >= len(offsets):
                            continue
                        layer_slice = slice(offsets[i], offsets[i + 1])
                        m.update_discount_from_cascade(cascade, layer_slice)

        # qtridiag Q refresh (driven from the lag-1 cross-correlation R
        # accumulated by qtridiag_boundary on each rebalance).
        if self._qt_boundaries and s % self._qt_refresh == 0:
            self._refresh_qtridiag_Q()

        # BMA centroid accumulator (CPU running sum of reconstructed fp32
        # concord weights). End-of-training the caller can divide by
        # _bma_count and swap modules for nn.Linear / nn.Conv2d with the
        # centroid via materialise_bma_centroid().
        if self._bma_every > 0 and s > self._bma_warmup and s % self._bma_every == 0:
            self._bma_observe()

        # Polyak-leak: lazy init on first qualifying step, then per-step
        # observe (every K) + update H. The probe + commit step fires here
        # too IF the trainer cached a fixed probe batch via
        # set_polyak_probe(); otherwise the trainer is responsible for
        # calling opt.polyak.probe_and_commit(...) on its own cadence.
        if self._polyak_on:
            self._maybe_init_polyak()
            if self._polyak is not None:
                if s % self._polyak_observe_every == 0:
                    self._polyak.observe()
                self._polyak.update_H()
                # Auto-probe if a fixed probe is cached.
                if (self._polyak_probe_x is not None
                        and s % self._polyak.probe_every == 0):
                    try:
                        import onetrainer_concord_patch as _patch
                        model = getattr(_patch, "_cached_model", None)
                    except ImportError:
                        model = None
                    if model is not None:
                        # LR-gated commit scale: suppress polyak leaks while
                        # LR is at the head of its schedule (trajectory is
                        # still descending fast, the box-mean lag drags W
                        # backward). Goes from 0 at the start to ~1 in the
                        # tail. Falls back to 1.0 if initial_lr is missing.
                        pg = self.param_groups[0]
                        init_lr = float(pg.get("initial_lr", pg["lr"]))
                        cur_lr = float(pg["lr"])
                        if init_lr > 0:
                            commit_scale = max(0.0, 1.0 - cur_lr / init_lr)
                        else:
                            commit_scale = 1.0
                        self._polyak.probe_and_commit(
                            model, self._polyak_probe_x,
                            self._polyak_probe_y,
                            self._polyak_probe_criterion,
                            commit_scale=commit_scale)

        return loss

    # ------------------------------------------------------------------ #
    # qtridiag (Q-aware tridiagonal coupling at MLP wi->wo boundaries)
    # ------------------------------------------------------------------ #

    def _maybe_init_qtridiag(self):
        if not self._qtridiag_on or self._qt_boundaries:
            return
        # Discover boundaries by walking the concord layers in module-
        # traversal order (which matches forward execution order for
        # sequential nets like BaselineConvNet and T5's per-block FFs).
        # At each consecutive pair (L, R), check whether R's flattened
        # input (col_exp) can be partitioned into `group = R.col_exp /
        # L.row_exp` equal-sized blocks per left channel.
        #
        # This generalizes the 1:1 MLP case (fc1->fc2, wi->wo, group=1) to
        # the 1:G conv cases (c1->c2 with 3x3 kernel: group=9; c3->fc1
        # with 64 channels feeding 64*4*4=1024 columns: group=16). The
        # upstream qtridiag_boundary function takes `group` natively;
        # all four boundary types share the same C-channel coordination
        # space.
        try:
            import onetrainer_concord_patch as _patch
            model = getattr(_patch, "_cached_model", None)
        except ImportError:
            model = None
        if model is None:
            return

        from train_cifar_qtridiag import build_Q_from_R   # noqa: F401
        layer_names = {}
        for name, mod in _named_modules_safe(model):
            if mod in self._concord:
                layer_names[id(mod)] = name or "<root>"

        ordered = list(self._concord)
        skipped = []
        for L, R in zip(ordered[:-1], ordered[1:]):
            C = L.row_exp.shape[0]
            K = R.col_exp.numel()
            if C == 0 or K == 0 or K % C != 0:
                # Incompatible: R's input shape doesn't factor into L's
                # output channels. Skip silently -- the user got the
                # boundary structure they actually have.
                continue
            group = K // C
            ln = layer_names.get(id(L), f"layer{id(L)}")
            rn = layer_names.get(id(R), f"layer{id(R)}")
            tag = f"{ln}->{rn}" if group == 1 else f"{ln}->{rn}[1:{group}]"
            # Honour the user-supplied regex restriction (e.g. fc1->fc2 only).
            if (self._qt_pairs_re is not None
                    and not self._qt_pairs_re.search(tag)):
                skipped.append(tag)
                continue
            dev = L.row_exp.device
            self._qt_boundaries.append({
                "name": tag,
                "up": L, "down": R, "group": int(group),
                "Q": torch.eye(C, device=dev, dtype=torch.float32),
                "Rlag": torch.zeros(C, C, dtype=torch.float64, device=dev),
                "prev_left": torch.zeros(C, dtype=torch.int32, device=dev),
            })
        if self._qt_boundaries or skipped:
            msg = (f"[Concord] qtridiag discovered "
                   f"{len(self._qt_boundaries)} boundaries: "
                   f"{', '.join(b['name'] for b in self._qt_boundaries)}")
            if skipped:
                msg += (f"  (skipped by concord_qtridiag_pairs regex: "
                        f"{', '.join(skipped)})")
            print(msg, flush=True)

    def _rebalance_with_qtridiag(self):
        from train_cifar_qtridiag import qtridiag_boundary
        pre = [(b["up"].row_exp.clone(),
                b["down"].col_exp.clone()) for b in self._qt_boundaries]
        for m in self._concord:
            m.rebalance()
        counts = {}
        for b, (up_row, dn_col) in zip(self._qt_boundaries, pre):
            cl = qtridiag_boundary(b["up"], b["down"], b["group"],
                                    up_row, dn_col,
                                    b["Q"], None, b["Rlag"], b["prev_left"],
                                    counts, b["name"])
            b["prev_left"].copy_(cl)

    def _refresh_qtridiag_Q(self):
        from train_cifar_qtridiag import build_Q_from_R
        for b in self._qt_boundaries:
            Qn, _, _ = build_Q_from_R(b["Rlag"].cpu())
            b["Q"] = Qn.to(b["Q"].device)
            b["Rlag"].zero_()

    # ------------------------------------------------------------------ #
    # BMA centroid accumulator
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _bma_observe(self):
        self._bma_count += 1
        for i, m in enumerate(self._concord):
            if hasattr(m, "reconstruct_W"):
                W = m.reconstruct_W(dtype=torch.float32)
            else:
                exp = (m.row_exp[:, None] + m.col_exp[None, :]
                       - m.MANTISSA_BIAS).float()
                W = ((m.s_slow.to(torch.int32) + m.s_fast.to(torch.int32))
                     .float() * torch.exp2(exp))
            Wcpu = W.cpu()
            if i not in self._bma_sums:
                self._bma_sums[i] = torch.zeros_like(Wcpu)
            self._bma_sums[i].add_(Wcpu)

    @torch.no_grad()
    def materialise_bma_centroid(self, model):
        """End-of-training: replace each concord layer with an fp32
        nn.Linear (or nn.Conv2d) carrying the trajectory centroid
        (sum / count). Returns the number of layers replaced. No-op if no
        BMA snapshots were observed.

        Conv-concord layers store their centroid as a 2D (out_ch,
        in_ch*kh*kw) matrix (matching the concord internal layout);
        rebuilding requires reshaping it back to nn.Conv2d's 4D
        (out_ch, in_ch, kh, kw) weight and carrying stride/padding
        through. nn.Linear-concord layers use the existing 2D copy.
        """
        if self._bma_count == 0:
            return 0
        from concord_linear_fused import ConcordConv2dFused
        # Don't read model.parameters() -- OneTrainer overwrites that
        # attribute. Use any concord layer's device instead.
        device = self._concord[0].s_slow.device
        loc = {}
        fol_set = {id(m) for m in self._concord}
        for pname, parent in _named_modules_safe(model):
            for attr, child in parent.named_children():
                if id(child) in fol_set:
                    loc[id(child)] = (parent, attr)
        n = 0
        for i, m in enumerate(self._concord):
            if i not in self._bma_sums:
                continue
            parent, attr = loc[id(m)]
            centroid = (self._bma_sums[i] / self._bma_count).to(device)
            has_bias = m.bias is not None
            if isinstance(m, ConcordConv2dFused):
                W4d = centroid.reshape(m.out_channels, m.in_channels,
                                        m.kh, m.kw)
                new_mod = nn.Conv2d(
                    m.in_channels, m.out_channels, (m.kh, m.kw),
                    stride=m.stride, padding=m.padding, bias=has_bias,
                    device=device, dtype=torch.float32)
                new_mod.weight.data.copy_(W4d)
            else:
                new_mod = nn.Linear(m.in_features, m.out_features,
                                     bias=has_bias, device=device,
                                     dtype=torch.float32)
                new_mod.weight.data.copy_(centroid)
            if has_bias:
                new_mod.bias.data.copy_(m.bias.data.float())
            setattr(parent, attr, new_mod)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # Polyak-leak hypothesis selector
    # ------------------------------------------------------------------ #

    def _maybe_init_polyak(self):
        if self._polyak is not None:
            return
        try:
            from concord_polyak import PolyakHypothesis, BoxVelocityMean
        except ImportError as e:
            print(f"[Concord] polyak_bias requested but imports failed: "
                  f"{e}; disabling.", flush=True)
            self._polyak_on = False
            return
        n_fol_total = sum(m.out_features * m.in_features
                          for m in self._concord)
        cfg = self._cfg
        # Polyak backend: BoxVelocityMean. K fp16 snapshots in a ring
        # plus a per-layer velocity (s_fast - s_slow) * scale anchor
        # that projects the lagged box mean forward to the present step.
        # Memory: K * n_fol * 2 bytes CPU (~25 GB at SDXL for K=5).
        # Tuning: concord_polyak_box_K (default 5), concord_polyak_box_extrap
        # (default K/2, the natural lag of a uniform K-sample mean).
        K = int(getattr(cfg, "concord_polyak_box_K", None) or 5)
        extrap_cfg = getattr(cfg, "concord_polyak_box_extrap", None)
        cascade = BoxVelocityMean(
            self._concord, K=K,
            extrap_steps=(float(extrap_cfg)
                          if extrap_cfg is not None else None),
        )
        self._polyak = PolyakHypothesis(
            self._concord, cascade,
            polyak_leak=float(
                getattr(cfg, "concord_polyak_leak", None) or 0.05),
            commit_strength=float(
                getattr(cfg, "concord_polyak_commit", None) or 0.1),
            probe_every=int(
                getattr(cfg, "concord_polyak_probe_every", None) or 200),
            # BoxVelocityMean is single-level; level is always 0.
            polyak_level=0,
            polyak_warmup=int(
                getattr(cfg, "concord_polyak_warmup", None) or 2),
            temperature=float(
                getattr(cfg, "concord_polyak_temperature", None) or 0.0),
            seed=0,
        )
        print(f"[Concord] polyak_bias enabled (box_velocity K={K}): leak="
              f"{self._polyak.polyak_leak} commit="
              f"{self._polyak.commit_strength} probe_every="
              f"{self._polyak.probe_every} warmup="
              f"{self._polyak.polyak_warmup} "
              f"T_0={self._polyak.temperature} "
              f"observe/{self._polyak_observe_every}  "
              f"(n_fol_total={n_fol_total})", flush=True)

    @property
    def polyak(self):
        """The PolyakHypothesis instance, or None if disabled / not yet
        initialised. Caller can drive probes manually via
        opt.polyak.probe_and_commit(model, probe_x, probe_y, criterion);
        alternatively use opt.set_polyak_probe(...) to cache a fixed probe
        and let the wrapper auto-fire at the configured cadence."""
        return self._polyak

    def set_polyak_probe(self, probe_x, probe_y, criterion):
        """Cache a fixed probe minibatch + criterion. After this call, the
        wrapper auto-fires PolyakHypothesis.probe_and_commit every
        concord_polyak_probe_every steps using the cached batch.

        This is the convenience path for OneTrainer-style training loops
        that don't want to manage probe cadence themselves: the trainer
        picks one held-out batch at the start and the wrapper does the
        rest. Pass None to disable auto-probing (manual mode)."""
        self._polyak_probe_x = probe_x
        self._polyak_probe_y = probe_y
        self._polyak_probe_criterion = criterion


# ---------------------------------------------------------------------------
# Top-level constructor: called from create.py's CONCORD_SGD case
# ---------------------------------------------------------------------------


def create_concord_optimizer(parameter_dicts, config, optimizer_config):
    """Build the ConcordSGD wrapper, doing module wrapping in place along
    the way.

    Inputs:
        parameter_dicts -- the list of dicts returned by
            NamedParameterGroupCollection.parameters_for_optimizer(config).
            We rebuild this list to drop weight Parameters that were
            detached by wrapping; the surviving Parameters (biases, norms,
            embeddings) feed the aux AdamW.
        config -- the full TrainConfig.
        optimizer_config -- config.optimizer (the TrainOptimizerConfig).

    Returns:
        A ConcordSGD instance.
    """
    # OneTrainer-fork backward-compat: mirror foliated_X -> concord_X
    # if the OneTrainer schema hasn't been migrated yet. Removed once
    # the upstream TrainConfig.py renames its fields.
    _backfill_concord_from_foliated(optimizer_config)

    import onetrainer_concord_patch as _patch
    model = getattr(_patch, "_cached_model", None)
    if model is None:
        raise RuntimeError(
            "ConcordSGD requires the trainer to have cached the model on "
            "onetrainer_concord_patch._cached_model. The OneTrainer "
            "GenericTrainer should set this after model load. Check that "
            "onetrainer_concord_patch.install() has been called and the "
            "GenericTrainer hook is in place.")

    # 1. Wrap candidate modules in place.
    concord_layers = wrap_model(model, optimizer_config,
                                      parameter_dicts=parameter_dicts)
    if not concord_layers:
        raise RuntimeError(
            "ConcordSGD: wrap_model found no nn.Linear / nn.Conv2d modules to wrap "
            "(target regex did not match). Set concord_target_modules to a "
            "regex that matches the modules you want concord, or '.*' for "
            "every nn.Linear.")

    # 2. Drop detached weight Parameters from the aux groups.
    surviving = filter_param_groups_to_live(parameter_dicts, model)

    # Empty-group cleanup: if EVERY group ended up empty, AdamW would error.
    # In that case we still need an optimizer object (OneTrainer expects
    # one), so we synthesize a dummy group with a single trivial parameter
    # so AdamW does not crash. This is the deepest-coverage edge case --
    # a model where every trainable nn.Parameter was inside a concord
    # linear (e.g. a bias-less, embedding-less corner).
    any_params = any(g["params"] for g in surviving)
    if not any_params:
        dummy = torch.nn.Parameter(torch.zeros(1, device=next(
            iter(concord_layers)).s_slow.device, dtype=torch.float32),
            requires_grad=False)
        surviving = [{"name": "_concord_dummy_aux",
                       "params": [dummy],
                       "lr": 1e-30,
                       "initial_lr": 1e-30}]

    # 3. Build the aux AdamW. The concord_aux_lr field overrides the
    # per-group LR (which OneTrainer set from config.learning_rate); if
    # the user did not set it explicitly, the per-group LR survives.
    aux_lr_override = optimizer_config.concord_aux_lr
    if aux_lr_override is not None and aux_lr_override > 0:
        for g in surviving:
            g["lr"] = aux_lr_override
            g["initial_lr"] = aux_lr_override
        base_aux_lr = aux_lr_override
    else:
        base_aux_lr = surviving[0]["initial_lr"]

    weight_decay = (optimizer_config.weight_decay
                    if optimizer_config.weight_decay is not None else 0.0)
    beta1_aux = (optimizer_config.beta1
                 if optimizer_config.beta1 is not None else 0.9)
    beta2_aux = (optimizer_config.beta2
                 if optimizer_config.beta2 is not None else 0.999)
    eps = (optimizer_config.eps
           if optimizer_config.eps is not None else 1e-8)

    # Aux optimizer choice. Default 'adamw' matches the wrapper's prior
    # behaviour and is the right pick for OneTrainer's image-generation
    # use case (norms / embeddings / time-embeddings benefit from adaptive
    # second-moment normalisation). 'sgd' (no momentum, no weight decay)
    # matches train_cifar_fused.py's bias optimizer -- useful when
    # reproducing the CIFAR headline or any setting where SGD-class
    # behaviour is what the original recipe assumes.
    aux_optim_name = (getattr(optimizer_config, "concord_aux_optimizer",
                                None) or "adamw").lower()
    if aux_optim_name == "sgd":
        aux = torch.optim.SGD(
            surviving,
            lr=base_aux_lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )
    elif aux_optim_name == "adamw":
        aux = torch.optim.AdamW(
            surviving,
            lr=base_aux_lr,
            betas=(beta1_aux, beta2_aux),
            weight_decay=weight_decay,
            eps=eps,
        )
    else:
        raise ValueError(
            f"concord_aux_optimizer must be 'adamw' or 'sgd', got "
            f"{aux_optim_name!r}")

    base_concord_lr = config.learning_rate

    wrapper = ConcordSGD(
        aux_optimizer=aux,
        concord_layers=concord_layers,
        optimizer_config=optimizer_config,
        base_concord_lr=base_concord_lr,
        base_aux_lr=base_aux_lr,
    )

    # Stash for diagnostics; OneTrainer's `model._concord_layers` lets
    # downstream code (saver, sampler, BMA centroid materialisation) find
    # the concord set without going through the optimizer.
    setattr(model, "_concord_layers", concord_layers)

    print(f"[Concord] wrapped {len(concord_layers)} concord modules in "
          f"ConcordSGD (classic 32-bit); "
          f"aux {aux_optim_name.upper()} on "
          f"{sum(len(g['params']) for g in surviving)} live "
          f"aux Parameters @ lr={base_aux_lr:.3g}; "
          f"concord lr={base_concord_lr:.3g} "
          f"(ratio {wrapper._fol_to_aux_ratio:.2g}); "
          f"rebalance/{wrapper._rebalance_every} "
          f"refit/{wrapper._refit_every}", flush=True)

    return wrapper
