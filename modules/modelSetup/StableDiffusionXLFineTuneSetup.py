from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
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

        self._create_model_part_parameters(parameter_group_collection, "text_encoder_1", model.text_encoder_1, config.text_encoder)
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
        if config.optimizer.optimizer == Optimizer.CONCORD:
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
                if getattr(_m, "__name__", "").rsplit(".", 1)[-1] == "prototype_packed_b":
                    try:
                        _m._FUSED_MATMUL = want_fused
                    except Exception:
                        pass

        # Concord: swap the UNet's Linear/Conv2d for packed self-stepping layers BEFORE
        # collecting parameters, so create_parameters() naturally hands the optimizer only
        # the non-swapped (aux) params -- the Concord layers carry no nn.Parameter weight
        # and self-step in backward. The controller carries the schedule + rebalance.
        if config.optimizer.optimizer == Optimizer.CONCORD:
            from modules.util.optimizer.concord_ot import ConcordController
            # Pass the SAME layer_filter the param path uses (the GUI "Layer Filter" dropdown),
            # so the Concord swap only packs the SELECTED layers -- e.g. preset "attn-mlp"
            # (["attentions"]) trains attn+MLP and leaves the conv resnets frozen, dropping
            # their packed state. Empty filter (preset "full") swaps everything as before.
            te_anchor = getattr(config, "concord_te_anchor", False)
            te_lr = ((getattr(config.text_encoder, "learning_rate", None) or config.learning_rate)
                     if te_anchor else None)
            model.concord_controller = ConcordController(
                model.unet, self.train_device, config.learning_rate, total_steps=1,
                optimizer_config=config.optimizer,
                module_filters=ModuleFilter.create(config),
                text_encoder=(model.text_encoder if te_anchor else None),
                te_lr=te_lr, te_wd_anchor=getattr(config, "concord_te_wd_anchor", 0.5))
            # RESUME: __load_internal rebuilt a STANDARD UNet, so the saved packed_w buffers were
            # dropped and the swap above just packed RANDOM weights. Re-load the backup's packed
            # UNet state into the now-swapped layers to restore the exact Concord state (packed_w
            # + s_fast/s_slow/v_slow); without this, continue silently resumes from ~random.
            if config.continue_last_backup:
                self.__restore_concord_unet(model, config)
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
        if config.optimizer.optimizer == Optimizer.CONCORD \
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

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

        # Stage 3 v2: build the manual UNet fwd+bwd graph manager when the gate allows
        # (concord_cuda_graph). The trainer routes the step through it on the gated path;
        # the default path is untouched. After the optimizer so requires_grad is final.
        model.concord_graph_v2 = None
        if config.optimizer.optimizer == Optimizer.CONCORD:
            from modules.util.optimizer.concord_graph import should_graph, should_graph_te, ManualUNetGraph
            if should_graph(config):
                aux = [p for p in model.unet.parameters() if p.requires_grad]
                model.concord_graph_v2 = ManualUNetGraph(
                    self, aux, config.train_dtype.torch_dtype(), graph_te=should_graph_te(config),
                    accum=config.gradient_accumulation_steps)

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
        if config.continue_last_backup and config.optimizer.optimizer == Optimizer.CONCORD \
                and any(p.is_meta for p in model.unet.parameters()):
            model.unet.to_empty(device=self.train_device)

        vae_on_train_device = not config.latent_caching
        text_encoder_1_on_train_device = \
            config.text_encoder.train \
            or config.train_any_embedding() \
            or not config.latent_caching

        text_encoder_2_on_train_device = \
            config.text_encoder_2.train \
            or config.train_any_embedding() \
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
