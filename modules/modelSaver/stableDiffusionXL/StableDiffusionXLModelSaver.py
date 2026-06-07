import copy
import os.path
from pathlib import Path

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_sdxl_diffusers_to_ckpt import convert_sdxl_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat

import torch

import yaml
from safetensors.torch import save_file


class StableDiffusionXLModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")

        if dtype is not None:
            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)
        else:
            save_pipeline = pipeline

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: StableDiffusionXLModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_sdxl_diffusers_to_ckpt(
            model.vae.state_dict(),
            model.unet.state_dict(),
            model.text_encoder_1.state_dict(),
            model.text_encoder_2.state_dict(),
            model.noise_scheduler
        )
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

        yaml_name = os.path.splitext(destination)[0] + '.yaml'
        with open(yaml_name, 'w', encoding='utf8') as f:
            yaml.dump(model.sd_config, f, default_flow_style=False, allow_unicode=True)

    def __save_internal(
            self,
            model: StableDiffusionXLModel,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: StableDiffusionXLModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        # Concord: the UNet's Linear/Conv2d were swapped for packed self-stepping layers
        # (their state_dict has 'packed_w', not 'weight'). For the DEPLOYABLE formats,
        # consolidate them back to standard nn.Linear/nn.Conv2d holding the deployable
        # weights so the checkpoint loads as ordinary SDXL. The INTERNAL/backup format
        # keeps the packed state (raw dump). Destructive -- final-save only (do not pair
        # with mid-training safetensors saves while Concord-training).
        if getattr(model, "concord_controller", None) is not None \
                and output_model_format in (ModelFormat.SAFETENSORS, ModelFormat.DIFFUSERS):
            model.concord_controller.consolidate_into_unet(model.unet)

        # Concord packed embeddings: the control plane replaced each TE's token_embedding, so
        # its packed buffers would pollute the saved TE state_dict (a standard CLIPTextModel
        # load on resume expects token_embedding.weight). Materialize the trained tokens into
        # the embedding .vector (the embedding saver then writes them as the usual
        # clip_l/clip_g safetensors) and temporarily restore the original token_embedding for
        # the duration of the save; reinstall the control planes in finally (so a mid-training
        # save never leaves training in a broken state).
        _packed_planes = getattr(model, "concord_control_planes", None)
        if _packed_planes:
            from modules.util.optimizer.concord_ot import (
                deactivate_packed_embeddings,
                materialize_packed_embeddings_to_vectors,
            )
            materialize_packed_embeddings_to_vectors(model)
            deactivate_packed_embeddings(model)
        try:
            match output_model_format:
                case ModelFormat.DIFFUSERS:
                    self.__save_diffusers(model, output_model_destination, dtype)
                case ModelFormat.SAFETENSORS:
                    self.__save_safetensors(model, output_model_destination, dtype)
                case ModelFormat.INTERNAL:
                    self.__save_internal(model, output_model_destination)
        finally:
            if _packed_planes:
                from modules.util.optimizer.concord_ot import reactivate_packed_embeddings
                reactivate_packed_embeddings(model)
