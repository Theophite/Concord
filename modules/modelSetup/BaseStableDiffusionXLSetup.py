from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
import modules.util.spike_log as spike_log
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionMixin import ModelSetupDiffusionMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_basic_transformer_blocks,
    enable_checkpointing_for_clip_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseStableDiffusionXLSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupDiffusionMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["attentions"],
        "attn-only": ["attn"],
        "cross-attn": ["attn2"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=False)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, config)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.unet)
            if model.unet_lora is not None:
                apply_circular_padding_to_conv2d(model.unet_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().unet,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().text_encoder_2,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.vae_autocast_context, model.vae_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().vae,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder_1, self.train_device, model.train_dtype, config)
        quantize_layers(model.text_encoder_2, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.vae_train_dtype, config)
        quantize_layers(model.unet, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                embedding_state_1 = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer_1,
                    model.text_encoder_1,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[0][0][1:],
                )

                embedding_state_2 = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer_2,
                    model.text_encoder_2,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[1][0][1:],
                )
            else:
                embedding_state_1 = embedding_state.get("clip_l_out", embedding_state.get("clip_l", None))
                embedding_state_2 = embedding_state.get("clip_g_out", embedding_state.get("clip_g", None))

            embedding_state_1 = embedding_state_1.to(
                dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding_state_2 = embedding_state_2.to(
                dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = StableDiffusionXLModelEmbedding(
                embedding_config.uuid,
                embedding_state_1,
                embedding_state_2,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        self._add_embeddings_to_tokenizer(model.tokenizer_1, model.all_text_encoder_1_embeddings())
        self._add_embeddings_to_tokenizer(model.tokenizer_2, model.all_text_encoder_2_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_1,
            orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_1_embeddings(),
        )
        model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_2,
            orig_module=model.text_encoder_2.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_2_embeddings(),
        )

        model.embedding_wrapper_1.hook_to_module()
        model.embedding_wrapper_2.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_text_encoder_1_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding_1 = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding_1)

        for embedding, embedding_config in zip(model.all_text_encoder_2_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding_2 = \
                embedding_config.train \
                and config.text_encoder_2.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding_2)

    @staticmethod
    def _concord_token_only_dropout(model, config, batch, generator):
        """With probability p (config.concord_token_only_dropout), rewrite an
        eligible example's caption to ONLY its trainable tokens. Gated to fire
        only AFTER the embedding divot releases (controller.step_idx past the
        delay) and only for examples that actually contain a trainable token
        (else token-only = empty caption, which is not the intent). The
        trainable id set is read per-TE from the control plane's routing
        (kind == 2). No-op unless Concord packed embeddings are active."""
        p = float(getattr(config, "concord_token_only_dropout", 0.0) or 0.0)
        if p <= 0.0:
            return batch
        ctrl = getattr(model, "concord_controller", None)
        planes = getattr(model, "concord_control_planes", None)
        if ctrl is None or not getattr(ctrl, "emb_cores", None) or not planes:
            return batch
        if int(getattr(ctrl, "step_idx", 0)) < int(getattr(ctrl, "emb_delay_steps", 0)):
            return batch                              # still in the divot; tokens frozen
        t1, t2 = batch.get("tokens_1"), batch.get("tokens_2")
        if t1 is None or t2 is None:
            return batch
        from modules.util.optimizer.concord.token_dropout import token_only_keep

        def _train_ids(te_idx):
            for pl in planes:
                if pl.get("te_idx") == te_idx and pl.get("cp") is not None:
                    return (pl["cp"].kind == 2).nonzero(as_tuple=True)[0]
            return None
        ti1, ti2 = _train_ids(1), _train_ids(2)
        elig = torch.zeros(t1.shape[0], dtype=torch.bool, device=t1.device)
        if ti1 is not None and ti1.numel():
            elig |= torch.isin(t1, ti1.to(t1.device)).any(dim=1)
        if ti2 is not None and ti2.numel():
            elig |= torch.isin(t2, ti2.to(t2.device)).any(dim=1)
        r = torch.rand(t1.shape[0], generator=generator, device=generator.device).to(t1.device)
        drop = (r < p) & elig
        if not bool(drop.any()):
            return batch
        b1, e1 = model.tokenizer_1.bos_token_id, model.tokenizer_1.eos_token_id
        b2, e2 = model.tokenizer_2.bos_token_id, model.tokenizer_2.eos_token_id
        batch["tokens_1"] = token_only_keep(t1, ti1, b1, e1, drop)
        batch["tokens_2"] = token_only_keep(t2, ti2, b2, e2, drop)
        return batch

    def predict(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
            return_unet_inputs: bool = False,
            return_raw_inputs: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            vae_scaling_factor = model.vae.config['scaling_factor']

            # Same-example antithetic (opt-in): duplicate the first half of the batch so example
            # i and i+B/2 are identical -> the antithetic timestep/noise pairing runs each example
            # through twice at antithetic (t, eps). Done before the text encode so tokens duplicate
            # too. Default off (cross-example pairing, which tested better).
            if config.concord_antithetic_same_example and config.concord_antithetic_timesteps \
                    and not deterministic:
                batch = self._concord_duplicate_first_half(batch)

            # Concord token-only caption dropout: after the embedding divot releases, with
            # probability p replace an eligible example's caption with ONLY its trainable tokens
            # (drop the context words) so the token must carry the concept. Mutates batch
            # tokens_1/tokens_2 BEFORE either the eager TE (below) or the graphed TE
            # (return_raw_inputs) reads them; shape is preserved (graph-safe). Never on
            # validation/sampling (deterministic).
            if not deterministic:
                batch = self._concord_token_only_dropout(model, config, batch, generator)

            # Stage 3 v2 TE-graph (return_raw_inputs): the text encoder is captured INSIDE
            # the CUDA graph from the raw tokens, so skip the eager TE forward here.
            text_encoder_output = pooled_text_encoder_2_output = None
            if not return_raw_inputs:
                text_encoder_output, pooled_text_encoder_2_output = model.combine_text_encoder_output(*model.encode_text(
                    train_device=self.train_device,
                    batch_size=batch['latent_image'].shape[0],
                    rand=rand,
                    tokens_1=batch['tokens_1'],
                    tokens_2=batch['tokens_2'],
                    text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                    text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                    text_encoder_1_output=batch[
                        'text_encoder_1_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                    text_encoder_2_output=batch[
                        'text_encoder_2_hidden_state'] if not config.train_text_encoder_2_or_embedding() else None,
                    pooled_text_encoder_2_output=batch[
                        'text_encoder_2_pooled_state'] if not config.train_text_encoder_2_or_embedding() else None,
                    text_encoder_1_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
                    text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability if not deterministic else None,
                ))

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            scaled_latent_conditioning_image = None
            if config.model_type.has_conditioning_image_input():
                scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                snr_weight_ctx=(
                    model.noise_scheduler.alphas_cumprod,
                    config.loss_weight_strength,
                    model.noise_scheduler.config.prediction_type == 'v_prediction',
                ) if (config.concord_antithetic_timesteps
                      and config.loss_weight_fn == LossWeight.MIN_SNR_GAMMA
                      and not config.resolution_aware_loss_weight) else None,
            )

            latent_noise = self._create_noise(
                scaled_latent_image,
                config,
                generator,
                timestep,
                model.noise_scheduler.betas,
            )

            scaled_noisy_latent_image = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.betas,
            )

            # original size of the image
            original_height = batch['original_resolution'][0]
            original_width = batch['original_resolution'][1]
            crops_coords_top = batch['crop_offset'][0]
            crops_coords_left = batch['crop_offset'][1]
            target_height = batch['crop_resolution'][0]
            target_width = batch['crop_resolution'][1]

            add_time_ids = torch.stack([
                original_height,
                original_width,
                crops_coords_top,
                crops_coords_left,
                target_height,
                target_width
            ], dim=1)

            add_time_ids = add_time_ids.to(
                dtype=scaled_noisy_latent_image.dtype,
                device=scaled_noisy_latent_image.device,
            )

            if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                latent_input = torch.concat(
                    [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                )
            else:
                latent_input = scaled_noisy_latent_image

            # Stage 3 v2 TE-graph: hand back RAW tokens + the diffusion inputs/target, so the
            # caller captures encode_text->UNet->loss->backward in ONE graph (the backward
            # reaches the embeddings inside the capture -> no eager bridge needed).
            if return_raw_inputs:
                if model.noise_scheduler.config.prediction_type == 'v_prediction':
                    target = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
                else:
                    target = latent_noise
                return {
                    'tokens_1': batch['tokens_1'],
                    'tokens_2': batch['tokens_2'],
                    'latent_input': latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    'timestep': timestep,
                    'time_ids': add_time_ids,
                    'target': target,
                    'loss_type': 'target',
                }

            added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}

            # Stage 3 v2 (Concord CUDA graph): hand the eager prep's UNet inputs + target
            # back to the caller, which captures UNet->loss->backward in a graph. Additive;
            # default path (return_unet_inputs=False) is unchanged.
            if return_unet_inputs:
                if model.noise_scheduler.config.prediction_type == 'v_prediction':
                    target = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
                else:
                    target = latent_noise
                return {
                    'latent_input': latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    'timestep': timestep,
                    'encoder_hidden_states': text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    'added_cond_kwargs': added_cond_kwargs,
                    'target': target,
                    'loss_type': 'target',
                }

            predicted_latent_noise = model.unet(
                sample=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            model_output_data = {}

            if model.noise_scheduler.config.prediction_type == 'epsilon':
                model_output_data = {
                    'loss_type': 'target',
                    'timestep': timestep,
                    'predicted': predicted_latent_noise,
                    'target': latent_noise,
                }
            elif model.noise_scheduler.config.prediction_type == 'v_prediction':
                target_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)
                model_output_data = {
                    'loss_type': 'target',
                    'timestep': timestep,
                    'predicted': predicted_latent_noise,
                    'target': target_velocity,
                }

            if config.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens_1'], model.tokenizer_1),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    # noise
                    self._save_image(
                        self._project_latent_to_image_sdxl(latent_noise),
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step,
                        True
                    )

                    # predicted noise
                    self._save_image(
                        self._project_latent_to_image_sdxl(predicted_latent_noise),
                        config.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step,
                        True
                    )

                    # noisy image
                    self._save_image(
                        self._project_latent_to_image_sdxl(scaled_noisy_latent_image),
                        config.debug_dir + "/training_batches",
                        "3-noisy_image",
                        train_progress.global_step,
                        True
                    )

                    # predicted image
                    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(config.train_device)
                    sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                        / sqrt_alpha_prod
                    self._save_image(
                        self._project_latent_to_image_sdxl(scaled_predicted_latent_image),
                        config.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step,
                        True
                    )

                    # image
                    self._save_image(
                        self._project_latent_to_image_sdxl(scaled_latent_image),
                        config.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step,
                        True
                    )

        model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type
        return model_output_data

    def calculate_loss(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = self._diffusion_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            betas=model.noise_scheduler.betas,
        )
        spike_log.SPIKE_LOG.log(losses, data, batch)   # no-op unless CONCORD_SPIKE_LOG is set
        return losses.mean()

    def prepare_text_caching(self, model: StableDiffusionXLModel, config: TrainConfig):
        model.to(self.temp_device)

        if not config.train_text_encoder_or_embedding():
            model.text_encoder_to(self.train_device)

        if not config.train_text_encoder_2_or_embedding():
            model.text_encoder_2_to(self.train_device)

        model.eval()
        torch_gc()
