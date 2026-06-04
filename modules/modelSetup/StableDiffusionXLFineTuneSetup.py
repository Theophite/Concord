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

        if config.train_any_embedding() or config.train_any_output_embedding():
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
        self._setup_embedding_wrapper(model, config)

        # Concord: swap the UNet's Linear/Conv2d for packed self-stepping layers BEFORE
        # collecting parameters, so create_parameters() naturally hands the optimizer only
        # the non-swapped (aux) params -- the Concord layers carry no nn.Parameter weight
        # and self-step in backward. The controller carries the schedule + rebalance.
        if config.optimizer.optimizer == Optimizer.CONCORD:
            from modules.util.optimizer.concord_ot import ConcordController
            model.concord_controller = ConcordController(
                model.unet, self.train_device, config.learning_rate, total_steps=1,
                optimizer_config=config.optimizer)
        else:
            model.concord_controller = None

        # Independent control plane: zero specified single-token vocab words (sanitize),
        # so the saved model embeds them to ~nothing. Works with any optimizer.
        if config.concord_sanitize_tokens.strip():
            from modules.util.optimizer.concord_ot import SanitizePlane
            model.concord_sanitize = SanitizePlane(model, config.concord_sanitize_tokens)
        else:
            model.concord_sanitize = None

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

        # Stage 3 v2: build the manual UNet fwd+bwd graph manager when the gate allows
        # (concord_cuda_graph). The trainer routes the step through it on the gated path;
        # the default path is untouched. After the optimizer so requires_grad is final.
        model.concord_graph_v2 = None
        if config.optimizer.optimizer == Optimizer.CONCORD:
            from modules.util.optimizer.concord_graph import should_graph, ManualUNetGraph
            if should_graph(config):
                aux = [p for p in model.unet.parameters() if p.requires_grad]
                model.concord_graph_v2 = ManualUNetGraph(
                    self, aux, config.train_dtype.torch_dtype())

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
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
            model.embedding_wrapper_1.normalize_embeddings()
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
