from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer, is_concord_family
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class StableDiffusionXLFineTuneSetup(
    BaseStableDiffusionXLSetup,
):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        # ANY Concord-trained TE (winner-recipe default OR frozen-anchor opt-in) self-steps in the
        # captured backward (like the packed embeddings below), so its params must NOT go to the SGD
        # aux optimizer -- its weight_decay would decay the one live Parameter (the bias) off the
        # kernel's discipline. Skip on Concord + TE-train, regardless of the anchor-mode flag (the
        # flag now selects the MODE, not whether the TE is Concord-trained). Non-Concord: unchanged.
        _concord = is_concord_family(config.optimizer.optimizer)
        if not (_concord and config.text_encoder.train):
            self._create_model_part_parameters(parameter_group_collection, "text_encoder_1", model.text_encoder_1, config.text_encoder)
        if not (_concord and config.text_encoder_2.train):
            self._create_model_part_parameters(parameter_group_collection, "text_encoder_2", model.text_encoder_2, config.text_encoder_2)

        # Concord packed embeddings self-step inside the backward (no optimizer.step), so they
        # must NOT be handed to the SGD optimizer -- skip the embedding param groups entirely.
        from modules.util.optimizer.concord_ot import packed_embeddings_active
        packed_embeddings = packed_embeddings_active(config)
        if (config.train_any_embedding() or config.train_any_output_embedding()) and not packed_embeddings:
            if config.text_encoder.train_embedding:
                self._add_embedding_param_groups(
                    model.all_text_encoder_1_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_1"
                )

            if config.text_encoder_2.train_embedding:
                self._add_embedding_param_groups(
                    model.all_text_encoder_2_embeddings(), parameter_group_collection, config.embedding_learning_rate,
                    "embeddings_2"
                )

        self._create_model_part_parameters(parameter_group_collection, "unet", model.unet, config.unet,
                                           freeze=ModuleFilter.create(config), debug=config.debug_mode)

        return parameter_group_collection

    def __setup_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        self._setup_embeddings_requires_grad(model, config)

        self._setup_model_part_requires_grad("text_encoder_1", model.text_encoder_1, config.text_encoder, model.train_progress)
        self._setup_model_part_requires_grad("text_encoder_2", model.text_encoder_2, config.text_encoder_2, model.train_progress)
        self._setup_model_part_requires_grad("unet", model.unet, config.unet, model.train_progress)

        # Concord packed embeddings: the TE freezes above turn off the control-plane
        # trainable's dummy _grad_anchor (it lives under text_encoder_*); re-enable it so the
        # self-step autograd Function still fires. Idempotent; runs at setup and every step.
        from modules.util.optimizer.concord_ot import reenable_packed_embedding_grad
        reenable_packed_embedding_grad(model)

        model.vae.requires_grad_(False)

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if config.train_any_embedding():
            model.text_encoder_1.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())
            model.text_encoder_2.get_input_embeddings().to(dtype=config.embedding_weight_dtype.torch_dtype())

        if config.rescale_noise_scheduler_to_zero_terminal_snr:
            model.rescale_noise_scheduler_to_zero_terminal_snr()
            model.force_v_prediction()
        elif config.force_v_prediction:
            model.force_v_prediction()
        elif config.force_epsilon_prediction:
            model.force_epsilon_prediction()

        self._remove_added_embeddings_from_tokenizer(model.tokenizer_1)
        self._remove_added_embeddings_from_tokenizer(model.tokenizer_2)
        self._setup_embeddings(model, config)
        # Concord packed embeddings REPLACE token_embedding outright (see the packed-embedding
        # block below); the plain-SGD AdditionalEmbeddingWrapper hook must NOT be installed --
        # it monkeypatches the very module the control plane wraps as its frozen base, which
        # would make the control plane's base lookup recurse into the wrapper.
        from modules.util.optimizer.concord_ot import packed_embeddings_active
        if not packed_embeddings_active(config):
            self._setup_embedding_wrapper(model, config)

        # Fused dequant-matmul flag -- MUST be set before the Concord swap below. The swapped
        # layers' __init__ -> _ensure_buffers() reads prototype_packed_b._FUSED_MATMUL at
        # CONSTRUCTION time to choose shared scratch (fused) vs a per-layer bf16 weight cache
        # (~5 GB). Setting it AFTER the swap leaves the cache already allocated (no saving), so
        # it must precede the swap. On by default (config field default True); also settable via
        # the CONCORD_FUSED_MATMUL env var (kept so the standalone harnesses still work). Fused
        # is honored exactly as selected, including with gradient accumulation: accumulation is
        # driven by the apply kernel's per-device consolidate gate, not by the bf16 cache, so it
        # is orthogonal to fused vs cached (fused just dequants packed_w in the forward instead of
        # reading a mid-cycle-frozen weight_buf).
        # CONCORD_STEPLESS: bind the cloned kernel lineage BEFORE any
        # prototype_packed_b import in this process. kernel_select installs the
        # single stepless module object under BOTH import spellings, so every
        # existing import site (bare or package) resolves to the clone; the
        # CONCORD default path never calls this and keeps its exact topology.
        if config.optimizer.optimizer == Optimizer.CONCORD_STEPLESS:
            from modules.util.optimizer.concord.kernel_select import bind_kernel
            bind_kernel("stepless")

        if is_concord_family(config.optimizer.optimizer):
            import os
            import sys as _sys
            from modules.util.optimizer.concord import prototype_packed_b as _ppb
            want_fused = bool(getattr(config, "concord_fused_matmul", False)) or _ppb._FUSED_MATMUL
            # The Concord internals import prototype_packed_b by its BARE name during the swap
            # (the concord dir is on sys.path) -- a module object DISTINCT from the package-
            # qualified import above (verified: `bare is pkg` -> False). The swapped layers read
            # _FUSED_MATMUL from the bare copy, and _ensure_buffers() reads it at CONSTRUCTION
            # time, so setting only the package copy here is a no-op (the bug this fixes). Cover
            # every copy: (1) the env var, so the bare copy -- imported LATER, during the swap --
            # reads the right value at its import; (2) the attribute on every already-loaded copy
            # (handles the GUI's long-lived process where the bare copy persists across runs).
            if want_fused:
                os.environ["CONCORD_FUSED_MATMUL"] = "1"
            else:
                os.environ.pop("CONCORD_FUSED_MATMUL", None)
            for _m in list(_sys.modules.values()):
                if getattr(_m, "__name__", "").rsplit(".", 1)[-1] in ("prototype_packed_b", "prototype_packed_stepless"):
                    try:
                        _m._FUSED_MATMUL = want_fused
                    except Exception:
                        pass

        # Concord: swap the UNet's Linear/Conv2d for packed self-stepping layers BEFORE
        # collecting parameters, so create_parameters() naturally hands the optimizer only
        # the non-swapped (aux) params -- the Concord layers carry no nn.Parameter weight
        # and self-step in backward. The controller carries the schedule + rebalance.
        if is_concord_family(config.optimizer.optimizer):
            from modules.util.optimizer.concord_ot import ConcordController
            # Pass the SAME layer_filter the param path uses (the GUI "Layer Filter" dropdown),
            # so the Concord swap only packs the SELECTED layers -- e.g. preset "attn-mlp"
            # (["attentions"]) trains attn+MLP and leaves the conv resnets frozen, dropping
            # their packed state. Empty filter (preset "full") swaps everything as before.
            # TE training under Concord (whenever text_encoder.train): WINNER recipe by DEFAULT
            # (train like the UNet), frozen ANCHOR opt-in via concord_te_anchor / concord_te2_anchor
            # (default False = winner). The flag now selects MODE, not whether the TE trains. lr from
            # the text_encoder LR field (controller falls back to the UNet lr if unset).
            te_train = config.text_encoder.train
            te2_train = config.text_encoder_2.train
            te_use_anchor = getattr(config, "concord_te_anchor", False)
            te2_use_anchor = getattr(config, "concord_te2_anchor", False)
            te_lr = config.text_encoder.learning_rate if te_train else None
            te2_lr = config.text_encoder_2.learning_rate if te2_train else None
            # UNet per-component LR override, parity with the TE LR fields above: the GUI's
            # "UNet Learning Rate" maps to config.unet.learning_rate. Concord bypasses torch
            # param groups, so unless it is threaded in explicitly the field is dead and the UNet
            # always rides the base lr. Fall back to the base lr when the override is unset.
            unet_lr = config.unet.learning_rate or config.learning_rate
            # Mirror the TE: only swap (and self-step) the UNet when config.unet.train. With
            # unet.train=False the UNet stays a frozen standard module -> genuine embeddings-only /
            # TE-only training (the backward still flows through it to reach the embeddings).
            unet_train = config.unet.train
            # Held-out router accum guard MUST run BEFORE the controller construction:
            # the controller applies the module flag and runs the 2-fast self-test (the
            # first kernel launches) during __init__, so clearing the flag after it is
            # too late for the self-test and any capture that follows.
            # Contrastive arms drives BOTH arms itself inside one graph step()
            # (full->L, dropped->H as two replays), so it is the update at
            # accum==1 and does NOT need the accumulation cycle to supply the
            # arm split. Exempt it from the router-off forcing.
            if bool(getattr(config.optimizer, "heldout_router", False)) \
                    and int(config.gradient_accumulation_steps) < 2 \
                    and not bool(getattr(config, "concord_contrast_arms", False)):
                from modules.util.optimizer.concord.prototype_packed_b import (
                    set_heldout_router, set_router_noise)
                set_heldout_router(False)
                set_router_noise(False)
                config.optimizer.heldout_router = False
                config.optimizer.router_coh_noise = False
                print("[concord] HELD-OUT ROUTER FORCED OFF: gradient_accumulation_steps="
                      f"{config.gradient_accumulation_steps} < 2 -- the router's data split is "
                      "the accumulation cycle; set accumulation >= 2 to enable.", flush=True)
            model.concord_controller = ConcordController(
                (model.unet if unet_train else None), self.train_device, unet_lr, total_steps=1,
                optimizer_config=config.optimizer,
                module_filters=ModuleFilter.create(config),
                text_encoder=(model.text_encoder_1 if te_train else None),
                text_encoder_2=(model.text_encoder_2 if te2_train else None),
                te_lr=te_lr, te2_lr=te2_lr,
                te_use_anchor=te_use_anchor, te2_use_anchor=te2_use_anchor,
                te_wd_anchor=getattr(config, "concord_te_wd_anchor", 0.5),
                te_chase_alpha=getattr(config, "concord_te_chase_alpha", 0.1),
                workspace_dir=getattr(config, "workspace_dir", None))
            # RESUME: __load_internal rebuilt a STANDARD UNet, so the saved packed_w buffers were
            # dropped and the swap above just packed RANDOM weights. Re-load the backup's packed
            # UNet state into the now-swapped layers to restore the exact Concord state (packed_w
            # + s_fast/s_slow/v_slow); without this, continue silently resumes from ~random.
            if config.continue_last_backup:
                if unet_train:
                    self.__restore_concord_unet(model, config)
                else:
                    # unet.train=False -> UNet NOT swapped. A PACKED backup (a trained UNet) can't be
                    # restored into the unswapped UNet (__load_internal drops packed_w -> RANDOM
                    # weights -> pure noise). CONSOLIDATE it to deploy weights and load into the FROZEN
                    # standard UNet -> "switch to embeddings-only from a trained checkpoint" works. A
                    # STANDARD backup needs nothing (already loaded). Raises if consolidation fails.
                    self.__restore_frozen_unet_from_packed(model, config)
                self.__restore_concord_te(model, config)
        else:
            model.concord_controller = None

        # Independent control plane: zero specified single-token vocab words (sanitize),
        # so the saved model embeds them to ~nothing. Works with any optimizer.
        if config.concord_sanitize_tokens.strip():
            from modules.util.optimizer.concord_ot import SanitizePlane
            model.concord_sanitize = SanitizePlane(model, config.concord_sanitize_tokens)
        else:
            model.concord_sanitize = None

        # Concord: route the trainable new-token embeddings through the norm-preserving packed
        # self-stepping core (ControlPlaneEmbedding + ConcordPackedEmbedding) instead of plain
        # SGD -- pins each token's deploy norm to the vocab median (the anti-overfit property
        # the SGD path lacked). Must follow _setup_embeddings (restored .vector + tokenizer
        # placeholder ids) and the fused-matmul flag block above (the packed core reads
        # _FUSED_MATMUL at construction). Save/restore is bridged in the embedding saver.
        if packed_embeddings_active(config):
            from modules.util.optimizer.concord_ot import setup_packed_embeddings
            setup_packed_embeddings(model, config)
        else:
            model.concord_control_planes = None

        # (concord_fused_matmul flag is set ABOVE, before the Concord swap -- see the note
        # there. It must precede layer construction, so it cannot live here.)

        # Concord gradient accumulation requires fast_gain == 1.0: the freeze-during-
        # accumulation semantics rely on the forward reading the cached weight_buf
        # (FusedConcordLinearPackedB.forward), but fast_gain < 1.0 makes the forward
        # re-materialize from the live packed_w (which keeps accumulating s_fast mid-
        # cycle), breaking the frozen-weight invariant. fast_gain is 1.0 by default and
        # nothing schedules it lower today; this guards a future schedule from silently
        # corrupting accumulated training rather than failing loudly.
        if is_concord_family(config.optimizer.optimizer) \
                and int(config.gradient_accumulation_steps) > 1:
            bad = [n for n, m in model.unet.named_modules()
                   if float(getattr(m, "fast_gain", 1.0)) != 1.0]
            if bad:
                raise ValueError(
                    f"Concord gradient accumulation (accum="
                    f"{config.gradient_accumulation_steps}) requires fast_gain == 1.0 on all "
                    f"swapped layers so the forward reads the frozen cached weight; "
                    f"{len(bad)} layer(s) violate this (e.g. {bad[:3]}). Disable the fast-gain "
                    f"schedule or set gradient_accumulation_steps = 1.")


        # The evaporation term u <- u - lr*gf_consol*(1-coh)*u diverges past
        # lr*gf_consol ~= 2 (CPU-verified bracketing: 1.5 trains, 2.5 NaNs; on GPU the
        # int16 clamp saturates instead of NaN-ing, but training is equally dead). The
        # shipped configs sit two orders inside the bound — guard it anyway.
        if is_concord_family(config.optimizer.optimizer):
            gf = float(getattr(model.concord_controller.config, "gf_consol", 0.0))
            if config.learning_rate * gf >= 2.0:
                raise ValueError(
                    f"lr*gf_consol = {config.learning_rate * gf:.2f} >= 2: the "
                    f"dissipation term is linearly unstable (u <- u - lr*k*(1-coh)*u). "
                    f"Lower the learning rate or gf_consol.")

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

        # Stage 3 v2: build the manual UNet fwd+bwd graph manager when the gate allows
        # (concord_cuda_graph). The trainer routes the step through it on the gated path;
        # the default path is untouched. After the optimizer so requires_grad is final.
        model.concord_graph_v2 = None
        if is_concord_family(config.optimizer.optimizer):
            from modules.util.optimizer.concord_graph import should_graph, should_graph_te, ManualUNetGraph
            if should_graph(config):
                aux = [p for p in model.unet.parameters() if p.requires_grad]
                model.concord_graph_v2 = ManualUNetGraph(
                    self, aux, config.train_dtype.torch_dtype(), graph_te=should_graph_te(config),
                    accum=config.gradient_accumulation_steps)

    def __restore_frozen_unet_from_packed(self, model, config):
        """SWITCH-to-embeddings-only from a trained checkpoint. unet.train=False leaves the UNet
        UNSWAPPED, so a PACKED backup (packed_w) can't be restored into it -- __load_internal drops
        those buffers, leaving RANDOM weights (pure noise). Instead CONSOLIDATE each packed layer to
        its DEPLOY weight ((s_slow+v_slow)*128 * 2^(row_exp+col_exp-15); the validated deploy formula)
        and copy it into the frozen standard UNet, so the frozen UNet carries the TRAINED state. A
        STANDARD backup has no packed_w -> nothing to do (already loaded by __load_internal). Fail
        loudly if ANY packed layer can't be reconstructed rather than train on a partly-random UNet."""
        import glob as _glob, os as _os
        import torch as _t
        from safetensors.torch import load_file as _load_file
        backup = config.get_last_backup_path()
        files = sorted(_glob.glob(_os.path.join(backup, "unet", "*.safetensors"))) if backup else []
        sd = {}
        for f in files:
            try:
                sd.update(_load_file(f))
            except Exception:
                pass
        keys = [k for k in sd if k.endswith(".packed_w")]
        if not keys:
            return                        # STANDARD backup -> already loaded by __load_internal
        # setup_train_device.to_empty (which materialized the meta swapped-weights just BEFORE this
        # runs) ALSO re-allocated every NON-packed UNet param uninitialized -- GroupNorm/LayerNorm
        # affine weight+bias, swapped-layer .bias, persistent buffers -- clobbering what __load_internal
        # had loaded correctly. The per-layer decode below only rewrites packed layers' .weight, so
        # restore the rest FIRST (mirrors __restore_concord_unet's load_state_dict for the trained
        # path). strict=False: packed_w/row_exp/col_exp are unexpected on the standard UNet (ignored),
        # and the swapped .weight is missing from the backup (filled by the decode loop below). Without
        # this, switch-to-frozen left the norms/biases as to_empty GARBAGE -> the UNet predicts
        # ~unconditional -> eps-MSE pinned ~1.0, deploy==live (frozen UNet has no s_fast plane). This
        # was the dead-frozen-UNet bug: the docstring above wrongly assumed the non-packed params were
        # "already loaded by __load_internal" -- true at load time, but to_empty clobbers them first.
        _incompat = model.unet.load_state_dict(sd, strict=False)
        _decoded = {k[:-len(".packed_w")] + ".weight" for k in keys}
        _pnames = {nm for nm, _ in model.unet.named_parameters()}
        _orphans = [k for k in _incompat.missing_keys if k in _pnames and k not in _decoded]
        if _orphans:
            raise ValueError(
                f"unet.train=False consolidate: {len(_orphans)} UNet params are neither in the backup "
                f"nor decoded from packed_w (e.g. {_orphans[:4]}) -- uninitialized garbage left by the "
                f"pre-restore to_empty. Refusing to train on a partly-random UNet.")
        n = bad = 0
        with _t.no_grad():
            for k in keys:
                mod = k[:-len(".packed_w")]
                re_ = sd.get(mod + ".row_exp"); ce_ = sd.get(mod + ".col_exp")
                try:
                    tgt = model.unet.get_submodule(mod)
                    w = tgt.weight
                except AttributeError:
                    w = None
                if re_ is None or ce_ is None or w is None:
                    bad += 1; continue
                p = sd[k].to(_t.int32)
                s = (p >> 8) & 0xFF; s = _t.where(s >= 128, s - 256, s)   # s_slow_i8 (signed)
                v = p & 0xFF;        v = _t.where(v >= 128, v - 256, v)   # v_slow_i8 (signed)
                m = (s + v).to(_t.float32) * 128.0                        # S_SLOW/V_SLOW_FACTOR
                exp = (re_.to(_t.int32)[:, None] + ce_.to(_t.int32)[None, :] - 15).to(_t.float32)
                deploy = m * _t.pow(_t.tensor(2.0), exp)                  # [out, K] fp32
                if deploy.numel() != w.numel() or not bool(_t.isfinite(deploy).all()):
                    bad += 1; continue
                w.data.copy_(deploy.reshape(w.shape).to(device=w.device, dtype=w.dtype))
                n += 1
        if bad:
            raise ValueError(
                f"unet.train=False consolidate: reconstructed {n} packed UNet layers to frozen deploy "
                f"weights but {bad} FAILED (missing row/col_exp, shape/size mismatch, or non-finite). "
                f"Refusing to train on a partly-random UNet. Export this model to a standard checkpoint "
                f"and start fresh from it instead.")
        print(f"[concord] resume(frozen UNet): load_state_dict restored the non-packed params "
              f"(norms/biases/buffers), consolidated {n} packed layers -> deploy weights; "
              f"UNet is now FROZEN at the trained state (switch-to-embeddings-only).", flush=True)

    def __restore_concord_unet(self, model, config):
        # The INTERNAL backup dumped the swapped UNet's full state_dict (packed_w + s_fast/
        # s_slow/v_slow) under <backup>/unet/*.safetensors, but __load_internal rebuilt a
        # STANDARD UNet (those keys discarded) and the swap then packed random weights. Merge
        # the backup shards and load them into the now-swapped layers (strict=False: packed_w
        # matches the buffers; non-swapped weights match too) to restore the exact state.
        import glob
        import os

        from safetensors.torch import load_file

        if os.environ.get("CONCORD_NO_RESTORE"):     # A/B switch: simulate the unfixed bug
            print("[concord] resume: CONCORD_NO_RESTORE set -> NOT restoring (random-swap baseline)")
            return
        backup = config.get_last_backup_path()
        files = sorted(glob.glob(os.path.join(backup, "unet", "*.safetensors"))) if backup else []
        if not files:
            print("[concord] resume: no backup UNet state found; continuing from loaded weights")
            return
        sd = {}
        for f in files:
            sd.update(load_file(f))
        model.unet.load_state_dict(sd, strict=False)
        # The forward reads a CACHED bf16 buffer, _bf16_weight_buf, which is a PLAIN
        # attribute -- NOT a registered buffer -- so it is absent from the backup and the
        # load_state_dict above updated only packed_w, leaving the cache holding the random
        # post-swap weights. _ensure_buffers() materializes that cache exactly once (when
        # it's None) and never again, so the next forward returns the stale cache -> garbage
        # output. Re-materialize it from the just-restored packed_w on every swapped layer.
        # Without this, the plain continue_last_backup / GUI resume produces mud; the
        # checkpoint-restart wrapper only dodged it by skipping the resumed-step sample
        # (a training step's apply kernel happens to rewrite the cache first).
        n_resync = 0
        for m in model.unet.modules():
            if hasattr(m, "_resync_weight_buf"):
                m._resync_weight_buf()
                n_resync += 1
        n_packed = sum(1 for k in sd if k.endswith("packed_w"))
        print(f"[concord] resume: restored UNet Concord state from backup "
              f"({n_packed} packed layers, {len(sd)} tensors); "
              f"re-materialized {n_resync} weight buffers from restored packed_w")

    def __restore_concord_te(self, model, config):
        # Mirror of __restore_concord_unet for the frozen-anchor TEs. The INTERNAL backup
        # (diffusers layout) dumped each swapped TE's packed state (packed_w + s_fast/s_slow/
        # v_slow) under <backup>/<subdir>/*.safetensors, but __load_internal rebuilt standard
        # CLIPTextModels (those keys discarded -> meta; setup_train_device to_empty'd them to
        # garbage) and the swap then packed garbage. Reload each anchored encoder's packed_w
        # (incl. the ORIGINAL v_slow anchor) so resume continues from the exact pre-backup state
        # -- a naive re-pack would re-anchor to current weights and lose the anti-drift. Covers
        # BOTH CLIP-L (text_encoder) and CLIP-G (text_encoder_2): a TE2-only omission here left
        # CLIP-G as to_empty garbage on resume.
        import glob
        import os

        from safetensors.torch import load_file

        ctrl = getattr(model, "concord_controller", None)
        if ctrl is None or not getattr(ctrl, "te_layers", None):
            return                                  # gated by the actual swap, not the flag
        if os.environ.get("CONCORD_NO_RESTORE"):
            print("[concord] resume: CONCORD_NO_RESTORE set -> NOT restoring TE")
            return
        backup = config.get_last_backup_path()
        if not backup:
            print("[concord] resume: no backup path; continuing TE from loaded weights")
            return
        # Restore each encoder that the controller actually anchored (same gating as the
        # constructor swap + the setup_train_device to_empty), from its diffusers subdir.
        encoders = []
        if config.text_encoder.train:                       # any Concord-trained TE (winner or anchor)
            encoders.append((model.text_encoder_1, "text_encoder"))
        if config.text_encoder_2.train:
            encoders.append((model.text_encoder_2, "text_encoder_2"))
        for encoder, subdir in encoders:
            files = sorted(glob.glob(os.path.join(backup, subdir, "*.safetensors")))
            if not files:
                print(f"[concord] resume: no backup {subdir} state found; "
                      f"continuing from loaded weights")
                continue
            sd = {}
            for f in files:
                sd.update(load_file(f))
            encoder.load_state_dict(sd, strict=False)
            n_resync = 0
            n_posid = 0
            n_resplit = 0
            for m in encoder.modules():
                if hasattr(m, "_resync_weight_buf"):
                    m._resync_weight_buf()
                    n_resync += 1
                    # DECONTAMINATE a frozen-era anchor backup resumed under the CREEP default.
                    # The restore above faithfully reloads the backup's packed state -- which, for
                    # a backup written in frozen-anchor mode (load_weights_anchor: whole weight in
                    # v_slow, s_slow=0), means d_sv = (s_slow - v_slow) ~= -W. Frozen (alpha_v=0)
                    # that is pinned and harmless; but under the creep default (alpha_v_fast>0) the
                    # live leak drags that fixed -W through the coherence gate as fake "signal" and
                    # drains the anchor -> TE deploy-norm washout (decode_dsv.py: cos(d_sv,deploy)
                    # ~= -0.7 on both TEs, vs +0.01 on the cleanly even-split UNet). Re-split the
                    # slow channel to even (d_sv~=0), mass-preserving: deploy + s_fast are exact,
                    # only s_slow<->v_slow moves. No-op on a clean creep state (||d_sv||/||deploy||
                    # small) and never invoked on a frozen anchor (gated on alpha_v_fast>0).
                    if getattr(m, "alpha_v_fast", 0.0) > 0.0 \
                            and hasattr(m, "resplit_anchor_to_even") \
                            and not os.environ.get("CONCORD_NO_RESPLIT"):
                        if m.resplit_anchor_to_even():
                            n_resplit += 1
                # THE RESUME BUG. setup_train_device.to_empty() re-allocated EVERY buffer as
                # uninitialized garbage -- including CLIP's position_ids, which transformers
                # registers persistent=False (modeling_clip.py:224). Non-persistent => absent from
                # the saved state_dict => the load_state_dict above could NOT restore it, so it kept
                # to_empty's garbage. Garbage position_ids -> garbage positional embeddings -> the TE
                # emits noise -> the UNet runs effectively unconditional -> brown/flat samples on
                # resume. (UNet-only training never hits this: the TE is never swapped/to_empty'd, so
                # its position_ids stays the arange from __init__. The packed weights restore fine --
                # this is a separate buffer the packed proofs never touched.) Re-seed it to arange,
                # exactly what transformers' __init__ set.
                _pid = getattr(m, "position_ids", None)
                if _pid is not None and torch.is_tensor(_pid) and _pid.dim() == 2 \
                        and not os.environ.get("CONCORD_NO_POSID_RESEED"):  # A/B kill-switch: prove the re-seed is load-bearing
                    with torch.no_grad():
                        _pid.copy_(torch.arange(_pid.shape[-1], device=_pid.device,
                                                dtype=_pid.dtype).unsqueeze(0))
                    n_posid += 1
            # FORMAT-CHANGE GUARD (single-fast <-> 2-fast TE row-servo transition). If the winner TE
            # word format changed since the backup -- e.g. first enable: single-fast backup (packed_w,
            # NO arm_row_exp) resumed into a 2-fast layer (arm_row_exp is a registered buffer) -- the
            # load_state_dict above put the backup's int16 s_fast bits into the 2-fast arm bytes (or
            # vice versa on disable), i.e. GARBAGE velocity/arms. The COARSE fields (s_slow bits 8-15 /
            # v_slow 0-7 / row_exp / col_exp) share bit positions across both layouts, so
            # consolidated_weight() is the CORRECT restored deploy; re-even-split from it to rebuild a
            # clean state in the CURRENT layout (deploy exact, velocity/arms clean). Fires ONLY on the
            # one-time transition: a matched format (2-fast backup HAS arm_row_exp in sd; single-fast
            # backup+single-fast layer has none on either side) leaves _layer_2fast == _backup_2fast
            # and skips. Frozen anchors are always single-fast on both sides -> never re-split here.
            n_reformat = 0
            for _name, m in encoder.named_modules():
                if not (hasattr(m, "load_weights") and hasattr(m, "consolidated_weight")
                        and f"{_name}.packed_w" in sd):
                    continue
                _layer_2fast = hasattr(m, "arm_row_exp")
                _backup_2fast = (f"{_name}.arm_row_exp" in sd)
                if _layer_2fast != _backup_2fast:
                    with torch.no_grad():
                        m.load_weights(m.consolidated_weight().float())   # clean even-split of restored deploy
                        if hasattr(m, "_resync_weight_buf"):
                            m._resync_weight_buf()
                    n_reformat += 1
            n_packed = sum(1 for k in sd if k.endswith("packed_w"))
            print(f"[concord] resume: restored {subdir} Concord state from backup "
                  f"({n_packed} packed layers, {len(sd)} tensors); "
                  f"re-materialized {n_resync} weight buffers from restored packed_w; "
                  f"re-split {n_resplit} live-leak TE layers v_slow->even (frozen-era -W d_sv decontamination); "
                  f"re-formatted {n_reformat} TE layers on a single-fast<->2-fast format change "
                  f"(clean even-split of restored deploy); "
                  f"re-seeded {n_posid} position_ids (to_empty garbages this persistent=False buffer)")

    @staticmethod
    def _consolidate_packed_te(encoder, backup_path, subdir):
        """Fill a meta-loaded text encoder's Linear weights with the DEPLOY
        weights decoded from the backup's packed tensors (winner word:
        deploy = (s_slow + v_slow) * 128 * 2^(row_exp + col_exp - 15); the
        2-fast word decodes identically -- deploy drops the fine plane
        either way). Returns the number of layers consolidated."""
        import os                                  # was missing -> NameError on this (untested) path
        import glob as _glob
        import torch as _torch
        from safetensors.torch import load_file as _load_file
        sd = {}
        for f in sorted(_glob.glob(os.path.join(backup_path, subdir, "*.safetensors"))):
            sd.update(_load_file(f))
        n = 0
        for name, module in encoder.named_modules():
            w = module._parameters.get("weight") if hasattr(module, "_parameters") else None
            if w is None or not w.is_meta:
                continue
            pk = sd.get(f"{name}.packed_w")
            re_ = sd.get(f"{name}.row_exp")
            ce = sd.get(f"{name}.col_exp")
            if pk is None or re_ is None or ce is None:
                continue
            coarse = (((pk << 16) >> 24) + ((pk << 24) >> 24)).to(_torch.float32) * 128.0
            scale = _torch.exp2((re_.to(_torch.int32)[:, None] + ce.to(_torch.int32)[None, :]
                                 - 15).to(_torch.float32))
            module._parameters["weight"] = _torch.nn.Parameter(
                (coarse * scale).to(w.dtype), requires_grad=False)
            n += 1
        return n

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        # RESUME (Concord): __load_internal rebuilt a STANDARD UNet, so the packed layers'
        # 'weight' keys are missing -> from_pretrained left them as META tensors, and the
        # move-to-device below crashes ("Cannot copy out of meta tensor"). Materialize the UNet
        # (garbage) here so the move works; setup_model's __restore_concord_unet then reloads the
        # real packed state over it (it carries the full state_dict, swapped + non-swapped).
        if config.continue_last_backup and is_concord_family(config.optimizer.optimizer) \
                and any(p.is_meta for p in model.unet.parameters()):
            model.unet.to_empty(device=self.train_device)
        # Same for a Concord-trained TE: its swapped Linears' 'weight' keys are missing from the
        # backup (only packed_w), so the loader (init_empty_weights + fill-present-keys) left them
        # META. The heal is gated on WHAT WAS LOADED (meta present), NOT on the current train
        # flags: the config may legitimately differ from the backup's (TE training toggled off
        # between segments), and the old train-flag gate skipped the heal while the embedding/
        # caption paths still moved the encoder -> "Cannot copy out of meta tensor" on resume.
        # Two cases per encoder:
        #   - TE trains this run: to_empty (garbage) is fine -- the swap + __restore_concord_te
        #     reload the real packed state in setup_model.
        #   - TE does NOT train this run: nothing will swap or restore it, so to_empty garbage
        #     would run a NOISE text encoder silently (mud). Consolidate the DEPLOY weights
        #     directly from the backup's packed tensors instead (drop s_fast, coarse * 2^exp --
        #     the same deploy the sampler used).
        for _enc, _sub, _trains in ((model.text_encoder_1, "text_encoder", config.text_encoder.train),
                                    (model.text_encoder_2, "text_encoder_2", config.text_encoder_2.train)):
            if not (config.continue_last_backup and is_concord_family(config.optimizer.optimizer)
                    and _enc is not None and any(p.is_meta for p in _enc.parameters())):
                continue
            if _trains:
                _enc.to_empty(device=self.train_device)
            else:
                _bk = config.get_last_backup_path()
                _n = self._consolidate_packed_te(_enc, _bk, _sub) if _bk else 0
                _left = [n for n, p in _enc.named_parameters() if p.is_meta]
                if _left:
                    raise RuntimeError(
                        f"Resume: {_sub} was saved PACKED (Concord-trained) but is not being "
                        f"trained this run, and {len(_left)} weights could not be consolidated "
                        f"from the backup (e.g. {_left[:2]}). Re-enable its training for this "
                        f"resume, or resume from a final save instead of a backup.")
                # Re-seed CLIP position_ids. It is a persistent=False buffer -> absent from the
                # backup state_dict -> META after __load_internal, then garbage after the device
                # move; named_parameters() (the _left guard above) does NOT see it. The TRAINED-TE
                # restore path re-seeds it to arange (that is why it works there); the consolidate
                # branch never did, so a NON-trained TE ran on garbage positional embeddings ->
                # noise conditioning -> loss~1.0 on resume. Replace with a fresh arange (meta-safe).
                _npid = 0
                for _m in _enc.modules():
                    _pid = getattr(_m, "position_ids", None)
                    if _pid is not None and torch.is_tensor(_pid) and _pid.dim() == 2:
                        _m.position_ids = torch.arange(_pid.shape[-1], dtype=_pid.dtype).unsqueeze(0)
                        _npid += 1
                print(f"[concord] resume: {_sub} not trained this run -> consolidated "
                      f"{_n} packed layers from the backup into standard deploy weights "
                      f"(re-seeded {_npid} position_ids)", flush=True)

        vae_on_train_device = not config.latent_caching
        text_encoder_1_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or config.train_caption_vocab() \
            or not config.latent_caching

        text_encoder_2_on_train_device = \
            config.text_encoder_2.train \
            or config.train_any_embedding() \
            or config.train_caption_vocab() \
            or not config.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if config.text_encoder.train:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if config.text_encoder_2.train:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.train()

        if config.unet.train:
            model.unet.train()
        else:
            model.unet.eval()

    def before_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        # Concord: advance the winner lr/sigma/floor schedule before the fused backward.
        if getattr(model, "concord_controller", None) is not None:
            model.concord_controller.before_step()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
            train_progress: TrainProgress
    ):
        if config.preserve_embedding_norm:
            self._normalize_output_embeddings(model.all_text_encoder_1_embeddings())
            self._normalize_output_embeddings(model.all_text_encoder_2_embeddings())
            # Packed embeddings self-pin their deploy norm in backward; their wrapper refs are
            # None (the plain-SGD wrapper path is bypassed), so guard the wrapper normalize.
            if model.embedding_wrapper_1 is not None:
                model.embedding_wrapper_1.normalize_embeddings()
            if model.embedding_wrapper_2 is not None:
                model.embedding_wrapper_2.normalize_embeddings()
        self.__setup_requires_grad(model, config)
        # Concord: gated rebalance (skips the no-op launches) + advance the step index.
        if getattr(model, "concord_controller", None) is not None:
            model.concord_controller.after_step()
        # control plane: keep sanitized rows at zero (in case training perturbed them).
        if getattr(model, "concord_sanitize", None) is not None:
            model.concord_sanitize.reapply(model)

factory.register(BaseModelSetup, StableDiffusionXLFineTuneSetup, ModelType.STABLE_DIFFUSION_XL_10_BASE, TrainingMethod.FINE_TUNE)
factory.register(BaseModelSetup, StableDiffusionXLFineTuneSetup, ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING, TrainingMethod.FINE_TUNE)
