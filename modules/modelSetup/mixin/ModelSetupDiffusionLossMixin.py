import os
from abc import ABCMeta
from collections.abc import Callable

from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses, masked_losses_with_prior
from modules.util.loss.vb_loss import vb_losses

import torch
import torch.nn.functional as F
from torch import Tensor


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    __coefficients: DiffusionScheduleCoefficients | None
    __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
    __sigmas: Tensor | None

    def __init__(self):
        super().__init__()
        self.__coefficients = None
        self.__alphas_cumprod_fun = None
        self.__sigmas = None

    def __log_cosh_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> Tensor:
        diff = pred - target
        loss = diff + torch.nn.functional.softplus(-2.0*diff) - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
        return loss

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = 0

        mean_dim = list(range(1, data['predicted'].ndim))

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                prior_losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none'
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                prior_losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none'
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += masked_losses_with_prior(
                losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ),
                prior_losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32)
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.log_cosh_strength

        # Huber Loss
        if config.huber_strength != 0:
            losses += masked_losses_with_prior(
                losses=F.huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none',
                    delta=config.huber_delta,
                ),
                prior_losses=F.huber_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['prior_target'].to(dtype=torch.float32),
                    reduction='none',
                    delta=config.huber_delta,
                ) if 'prior_target' in data else None,
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
                masked_prior_preservation_weight=config.masked_prior_preservation_weight,
            ).mean(mean_dim) * config.huber_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=data['predicted'].to(dtype=torch.float32),
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean(mean_dim) * config.vb_loss_strength

        return losses

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = 0

        mean_dim = list(range(1, data['predicted'].ndim))

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += F.mse_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean(mean_dim) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += F.l1_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean(mean_dim) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ).mean(mean_dim) * config.log_cosh_strength

        # Huber Loss
        if config.huber_strength != 0:
            losses += F.huber_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none',
                delta=config.huber_delta,
            ).mean(mean_dim) * config.huber_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.__coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=data['predicted'].to(dtype=torch.float32),
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean(mean_dim) * config.vb_loss_strength

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch['latent_mask'], config.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(mean_dim)
            losses /= mask_mean

        return losses

    def __snr(self, timesteps: Tensor, device: torch.device) -> Tensor:
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod /
                       self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
            snr = alphas_cumprod / (1.0 - alphas_cumprod)

        return snr

    def __min_snr_weight(
            self,
            timesteps: Tensor,
            gamma: float,
            v_prediction: bool,
            device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        # gamma may be a scalar or a per-image [batch] cap (resolution-aware min-SNR;
        # see __resolution_capped_gamma).
        gamma_t = gamma.to(snr.device) if torch.is_tensor(gamma) else torch.full_like(snr, gamma)
        min_snr_gamma = torch.minimum(snr, gamma_t)
        # Denominator of the snr_weight increased by 1 if v-prediction is being used.
        if v_prediction:
            snr += 1.0
        snr_weight = (min_snr_gamma / snr).to(device)
        return snr_weight

    def __resolution_capped_gamma(
            self,
            batch: dict,
            gamma: float,
            device: torch.device,
    ) -> Tensor:
        """Spectrum-principled per-image min-SNR cap for upscaled images. A natural image has a
        ~1/f^2 power spectrum, so an image upscaled by linear factor f (= sqrt(crop/orig area))
        carries no real signal beyond its native Nyquist, and its representable SNR is ~ gamma/f^2.
        Capping the min-SNR gamma there stops the model from being rewarded for reconstructing the
        fake interpolated high-freq at the low-noise (high-SNR) timesteps -- the upscaling-artifact
        source. 1/f^2 = original_area / crop_area; native/downscaled images (ratio >= 1) keep the
        full gamma, so this is a no-op without aspect-ratio bucketing.
        """
        orig_area = batch['original_resolution'][0].float() * batch['original_resolution'][1].float()
        crop_area = (batch['crop_resolution'][0].float() * batch['crop_resolution'][1].float()).clamp(min=1.0)
        ratio = (orig_area / crop_area).clamp(max=1.0)              # < 1 = upscaled
        capped = (gamma * ratio).to(device)                        # [batch] per-image gamma cap
        if os.environ.get("CONCORD_RESAWARE_DEBUG"):
            oh, ow = batch['original_resolution'][0].tolist(), batch['original_resolution'][1].tolist()
            ch, cw = batch['crop_resolution'][0].tolist(), batch['crop_resolution'][1].tolist()
            seen = {(int(a), int(b), int(c), int(d)): r
                    for a, b, c, d, r in zip(oh, ow, ch, cw, ratio.tolist())}
            msg = " | ".join(f"{a}x{b}->{c}x{d} r={r:.3f} g_eff={gamma * r:.2f}"
                             for (a, b, c, d), r in seen.items())
            print(f"[resaware] gamma={gamma} {msg}", flush=True)
        return capped

    def __debiased_estimation_weight(
        self,
        timesteps: Tensor,
        v_prediction: bool,
        device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        weight = snr
        # The line below is a departure from the original paper.
        # This is to match the Kohya implementation, see: https://github.com/kohya-ss/sd-scripts/pull/889
        # In addition, it helps avoid numerical instability.
        torch.clip(weight, max=1.0e3, out=weight)
        if v_prediction:
            weight += 1.0
        torch.rsqrt(weight, out=weight)
        return weight

    def __p2_loss_weight(
        self,
        timesteps: Tensor,
        gamma: float,
        v_prediction: bool,
        device: torch.device,
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        if v_prediction:
            snr += 1.0
        return (1.0 + snr) ** -gamma

    def __sigma_loss_weight(
        self,
        timesteps: Tensor,
        device: torch.device,
    ) -> Tensor:
        return self.__sigmas[timesteps].to(device=device)

    def _diffusion_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            betas: Tensor | None = None,
            alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas.to(train_device))

        self.__alphas_cumprod_fun = alphas_cumprod_fun

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(batch_size=config.batch_size, accumulation_steps=config.gradient_accumulation_steps)

        losses *= loss_weight

        # Apply timestep based loss weighting.
        if 'timestep' in data:
            v_pred = data.get('prediction_type', '') == 'v_prediction'
            match config.loss_weight_fn:
                case LossWeight.CONSTANT:
                    pass
                case LossWeight.MIN_SNR_GAMMA:
                    gamma = config.loss_weight_strength
                    if getattr(config, "resolution_aware_loss_weight", False) \
                            and 'original_resolution' in batch and 'crop_resolution' in batch:
                        gamma = self.__resolution_capped_gamma(batch, gamma, losses.device)
                    if getattr(config, "concord_antithetic_timesteps", False) \
                            and not getattr(config, "resolution_aware_loss_weight", False):
                        # min-SNR weight already folded into the timestep sampler (importance
                        # sampling); apply only the scalar mean-weight rescale so the expected
                        # gradient AND the effective LR are unchanged (a pure variance reduction).
                        losses = losses * getattr(self, "_tw_mean_w", 1.0)
                    else:
                        losses *= self.__min_snr_weight(data['timestep'], gamma, v_pred, losses.device)
                case LossWeight.DEBIASED_ESTIMATION:
                    losses *= self.__debiased_estimation_weight(data['timestep'], v_pred, losses.device)
                case LossWeight.P2:
                    losses *= self.__p2_loss_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for diffusion models")

        return losses

    def _flow_matching_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            sigmas: Tensor | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        if self.__sigmas is None and sigmas is not None:
            num_timesteps = sigmas.shape[0]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=train_device)
            self.__sigmas = all_timesteps / num_timesteps

        if data['loss_type'] == 'target':
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        losses *= loss_weight

        # Apply timestep based loss weighting.
        if 'timestep' in data:
            match config.loss_weight_fn:
                case LossWeight.CONSTANT:
                    pass
                case LossWeight.SIGMA:
                    losses *= self.__sigma_loss_weight(data['timestep'], losses.device)
                case _:
                    raise NotImplementedError(f"Loss weight function {config.loss_weight_fn} not implemented for flow matching models")

        return losses
