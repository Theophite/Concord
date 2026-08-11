import contextlib
import copy
import json
import math
import os
import shutil
import traceback
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.profiling_util import TorchMemoryRecorder, TorchProfiler
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
from requests.exceptions import ConnectionError
from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

    def start(self):
        if multi.is_master():
            self.__save_config_to_workspace()

            if self.config.clear_cache_before_training and self.config.latent_caching:
                self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(
                    token = self.config.secrets.huggingface_token,
                    new_session = False,
                )

        self.callbacks.on_update_status("loading the model")

        if self.config.quantization.cache_dir is None:
            self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
            quantization=self.config.quantization,
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()

        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.save_filename_prefix}{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text']):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def _cuda_census(self, tag, threshold_gb=2.0):
        """WHO holds CUDA memory: when allocations stay high where they should be ~0 (e.g.
        after the pre-sample offload), walk gc for live CUDA tensors and report the top
        groups by (dtype, ndim). Signatures: int32 2D = packed_w storages (pinned by a stale
        reference if the module's buffers already moved); bfloat16 2D = _bf16_weight_buf
        plain-attr caches (model.to() can NEVER move those). Diagnostic only, runs once per
        sample event and only above threshold."""
        try:
            a = torch.cuda.memory_allocated() / 2 ** 30
            if a < threshold_gb:
                return
            import gc as _gc
            from collections import Counter
            by = Counter()
            seen = set()
            for o in _gc.get_objects():
                try:
                    if torch.is_tensor(o) and o.is_cuda and o.storage().data_ptr() not in seen:
                        seen.add(o.storage().data_ptr())
                        by[(str(o.dtype).replace('torch.', ''), o.dim())] += o.numel() * o.element_size()
                except Exception:
                    continue
            top = by.most_common(8)
            print(f"[cuda-census] {tag}: alloc={a:.2f}G; top holders by (dtype,ndim):", flush=True)
            for (dt, nd), b in top:
                print(f"   {dt} {nd}D: {b / 2 ** 30:.2f}G", flush=True)
        except Exception as e:
            print(f"[cuda-census] failed: {e}", flush=True)

    def _graphmem(self, tag):
        # Memory probe -- ON BY DEFAULT (disable with CONCORD_GRAPHMEM=0). torch_alloc/reserved =
        # PyTorch's view; device_committed/free (mem_get_info) = the driver's dedicated-VRAM view.
        # Used to see whether ManualUNetGraph.release() actually returns the graph's private pool
        # before sampling (committed should drop), or leaves it resident -> over-commit -> WDDM spill.
        if os.environ.get("CONCORD_GRAPHMEM", "1").strip().lower() in ("", "0", "false", "no", "off"):
            return
        try:
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            free, total = torch.cuda.mem_get_info()
            print(f"[graphmem] {tag}: torch_alloc={a:.2f}G torch_reserved={r:.2f}G "
                  f"device_committed={(total - free) / 1e9:.2f}G device_free={free / 1e9:.2f}G",
                  flush=True)
        except Exception as e:
            print(f"[graphmem] {tag}: <error {e}>", flush=True)

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            ema_applied: bool,
            folder_postfix: str = "",
            is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=not self.config.samples_to_tensorboard and not ema_applied
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}"
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}", pil_to_tensor(sampler_output.data),  # noqa: B023
                            train_progress.global_step
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                self.model.to(self.temp_device)
                self.model.eval()
                self._graphmem(f"image {i} entry")
                if i == 0:
                    self._cuda_census("image 0 post-offload")

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                print("Error during sampling, proceeding without sampling")
                # A mid-sample exception aborts __sample_base BEFORE its inline device
                # housekeeping (text_encoder_to(temp_device) etc.), stranding the TEs +
                # pipeline residue on the train device -- repeated failures then bind
                # gigabytes into the post-sample/post-restore window (observed: 27 aborted
                # images -> post-sample 10.77G vs the healthy ~0.1G, device_free 1.71G).
                # Best-effort offload on the failure path; same crash-unwind class as the
                # restore_unet_deploy interrupt fix.
                try:
                    self.model.to(self.temp_device)
                except Exception as _e:
                    print(f"[concord] post-failure offload failed too ({type(_e).__name__}: {_e})")

            torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        # Concord v2: release the captured CUDA graph BEFORE sampling. Sampling's
        # torch_gc()/empty_cache() disturbs the graph's private memory pool, so the next
        # replay would read freed memory and crash on "coming back" from a sample. Dropping
        # it here frees the pool for sampling; the next training step transparently recaptures.
        _v2 = getattr(self.model, "concord_graph_v2", None)
        self._graphmem("pre-release")
        if _v2 is not None:
            _v2.release()
            # The captured graph (which forced zero_grad(set_to_none=False) each step so the backward
            # writes static .grad addresses) is now gone -> those grads are dead weight for the whole
            # sample loop. FREE them so the sampler gets that VRAM (~1x the eager TE/embedding grad set;
            # the big resident block release() doesn't touch). The next training step's recapture warmup
            # (fresh fwd+bwd, concord_graph.py _warmup_and_capture) re-establishes every .grad before it
            # re-captures, so this is a true-recapture-safe reclaim, not a replay hazard.
            self.model.optimizer.zero_grad(set_to_none=True)
        self._graphmem("post-release")
        torch_gc()
        self._graphmem("post-gc")

        # Concord: present the DEPLOY weight (s_fast dropped) to the sampler —
        # samples come from the same object the deployed-sv metric and the final
        # save use, not the live training weight. Restored below before training
        # resumes; if sampling raises, the un-restored state degrades to the
        # correct-but-slower cached path (the apply kernel rewrites the buffer
        # on the next step), never to corruption.
        # (step_idx > 0: at step 0 load_weights has the full mantissa in s_fast
        # and the consolidated weight is ~zero — the chase fills s_slow within
        # ~1/alpha steps. The pre-training baseline sample is the live model.)
        _ctrl = getattr(self.model, "concord_controller", None)
        # PRE-SAMPLE backup (restart-wrapper runs): the boundary's
        # checkpoint+exit(42) fires AFTER the full sample loop, which is
        # the segment's most fragile phase (fragmentation / WDDM spill) --
        # a process death mid-sampling used to rewind the relaunch to the
        # PREVIOUS backup, silently discarding the whole trained segment
        # (observed 2026-07-04: ~330 steps lost to a churned sampling
        # pass). Checkpoint BEFORE sampling too, so a mid-sampling death
        # costs only the sampling attempt; the post-sample backup + prune
        # then replaces this one on the happy path (disk stays flat).
        # Default ON under CONCORD_RESTART_ON_SAMPLE; opt out with
        # CONCORD_BACKUP_BEFORE_SAMPLE=0.
        if (getattr(self.model, "concord_graph_v2", None) is not None
                and os.environ.get("CONCORD_RESTART_ON_SAMPLE")
                and os.environ.get("CONCORD_BACKUP_BEFORE_SAMPLE", "1") != "0"
                and getattr(_ctrl, "step_idx", 0) > 0):
            print("[concord-restart] pre-sample checkpoint (segment is safe "
                  "even if sampling dies)", flush=True)
            self.__backup(train_progress, True, print)
        _deploy_stash = (_ctrl.materialize_unet_deploy()
                         if _ctrl is not None
                         and getattr(self.config, "concord_sample_deploy", True)
                         and getattr(_ctrl, "step_idx", 0) > 0
                         else None)

        # The deploy window MUST be unwound even if sampling raises: the
        # in-place s_fast mask means an un-restored exception path would
        # permanently lose the fast field (the stash is a local).
        try:
            self.callbacks.on_update_status("Sampling ...")

            is_custom_sample = False
            if sample_params_list:
                is_custom_sample = True
            elif self.config.samples is not None:
                sample_params_list = self.config.samples
            else:
                try:
                    with open(self.config.sample_definition_file_name, 'r') as f:
                        samples = json.load(f)
                        for i in range(len(samples)):
                            samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                        sample_params_list = samples
                # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
                except Exception:
                    traceback.print_exc()
                    print("Error during loading the sample definition file, proceeding without sampling")
                    sample_params_list = []

            if self.model.ema:
                #the EMA model only exists in the master process, so EMA sampling is done on one GPU only
                #non-EMA sampling is done on all GPUs
                assert multi.is_master() and self.config.ema != EMAMode.OFF
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                is_custom_sample=is_custom_sample,
                ema_applied = self.config.ema != EMAMode.OFF
            )

            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

            # ema-less sampling, if ema is enabled:
            if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
                self.__sample_loop(
                    train_progress=train_progress,
                    train_device=train_device,
                    sample_config_list=sample_params_list,
                    folder_postfix=" - no-ema",
                    ema_applied = False,
                )


        finally:
            if _deploy_stash is not None:
                _ctrl.restore_unet_deploy(_deploy_stash)

        # [main patch 0003] Return the sampler's dead heap to the driver BEFORE
        # setup_train_device recommits the UNet to dedicated VRAM. With the gc
        # after the move-back (the old order), the driver briefly saw sampler
        # leftovers + the returning UNet, tipped past the VRAM ceiling by a
        # sliver, and WDDM demoted the last fraction of the recommit to shared
        # memory -- silently, stickily (no promotion path), so training crawled
        # afterward.
        self._graphmem("post-sample")
        torch_gc()
        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()
        self._graphmem("post-restore")

        # Concord v2 checkpoint-restart (Windows): the post-sample graph recapture wedges because
        # sampling fragments the VRAM heap irreversibly -- empty_cache/reset/defrag cannot reclaim
        # the fragmented-but-not-live reserved memory, so within a few samples the recapture's
        # warmup thrashes at the 24GB ceiling. The recapture itself is sound (proven: 6 forced
        # recaptures with NO sampling never hung). Fix: checkpoint here and exit(42); the
        # scripts/concord_train_restart.py wrapper relaunches a FRESH process (clean allocator)
        # that resumes from this backup and recaptures cleanly. Opt-in via CONCORD_RESTART_ON_SAMPLE
        # (the wrapper sets it); plain `python train.py` runs are unaffected.
        _v2 = getattr(self.model, "concord_graph_v2", None)
        if _v2 is not None and os.environ.get("CONCORD_RESTART_ON_SAMPLE"):
            import sys
            print("[concord-restart] sample done -> checkpoint + exit(42) for fresh-process relaunch",
                  flush=True)
            self.__backup(train_progress, True, print)
            self.__prune_backups(1)   # resume only ever needs the latest -> keep exactly one, no pile-up
            self._kill_tensorboard_for_restart()
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(42)

    def _kill_tensorboard_for_restart(self):
        """The exit-42 relaunch is NOT a clean exit, so _stop_tensorboard never runs and the TB
        subprocess is orphaned -- it keeps port 6006 and tails the live event file, and its
        no-TensorFlow fallback CRC (tensorflow_stub) occasionally access-violates on a half-written
        record. Kill it before relaunching so each segment owns at most one TB instead of piling up
        orphans (the "could not bind to 6006" errors + the periodic access-violation dumps). Guarded:
        TB may not be running (tensorboard off, tensorboard_always_on skips _start_tensorboard, or it
        already crashed)."""
        proc = getattr(self, "tensorboard_subprocess", None)
        if proc is not None and proc.poll() is None:
            try:
                self._stop_tensorboard()
            except Exception:
                pass

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("Calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                                            average_loss,
                                            train_progress.global_step)

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar("loss/validation_step/total_average",
                                            total_average_loss,
                                            train_progress.global_step)

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def __backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print,
                 restart_after: bool = False):
        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        # Concord v2: release the captured CUDA graph before the backup. The save moves the model
        # to CPU and torch_gc()s, invalidating the graph's private pool -> the next replay would
        # segfault coming back from the backup. Recaptured transparently on the next step.
        # FULL release, always: holding the pool through the save (keep_pool) needs an accurate
        # pool-size estimate to know it's affordable, and the reserved-minus-allocated proxy
        # under-reads on a fresh process (first boundary: guard passed silently, save ran on top
        # of the held pool, committed 25.76G / free 0.00G -- the WDDM demotion tax again). The
        # 1x full-release path is ratchet-free since the post-recapture cleanup landed; backups
        # take it unconditionally. keep_pool remains for bucket flips, where release and
        # recapture are back-to-back with no intervening churn.
        _v2 = getattr(self.model, "concord_graph_v2", None)
        if _v2 is not None:
            _v2.release()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)

            # Concord controller clock: persist the TRUE update count with the
            # backup. The resume seed otherwise derives it as global_step //
            # accum, which breaks when gradient_accumulation_steps changes
            # between segments (2026-06-12: an accum 4->8 resume seeded 236
            # instead of 472 -- the divot re-froze for half an epoch and every
            # controller-clock consumer ran half-rewound).
            _ctrl = getattr(self.model, "concord_controller", None)
            if _ctrl is not None:
                try:
                    with open(os.path.join(backup_path, "concord_clock.json"),
                              "w", encoding="utf-8") as f:
                        # Loss smoothers ride the clock: without them every
                        # resume reseeds the EMAs from the first post-restart
                        # batch (which logs ~0.05 low), stamping a fake dip
                        # (median -0.0055) on the smooth tags at all 49
                        # boundaries -- two "loss cliffs" were called on that
                        # artifact before the 2026-07-12 telemetry audit
                        # caught it (docs/TELEMETRY_AUDIT_2026-07-12.md).
                        _es = getattr(self, "_ema_state", None) or {}
                        json.dump({
                            "update_steps": int(_ctrl.step_idx),
                            "global_step": int(train_progress.global_step),
                            "accum": int(max(1, self.config.gradient_accumulation_steps)),
                            "ema_loss": _es.get("ema_loss"),
                            "ema_deploy": _es.get("ema_deploy"),
                            "ema_loss_steps": _es.get("ema_loss_steps"),
                        }, f)
                    # Per-backup metrics for the across-backup trajectory console
                    # (meter-only): per-layer velocity + state-economy, appended to
                    # a JSONL in workspace_dir so the layer/econ panels accumulate
                    # as the run backs up. Own try; never aborts a backup.
                    try:
                        _ctrl.log_console_snapshot(int(train_progress.global_step),
                                                   int(_ctrl.step_idx))
                    except Exception:
                        pass
                    # Kalman loss meter: PER-RUN state, backup-scoped like the clock.
                    # (Living in the generic workspace_dir let its frozen baseline
                    # survive config changes and score against a regime the run no
                    # longer trains -- stale skill/trend. Own try; never abort backup.)
                    try:
                        if getattr(self, "_kloss", None) is not None:
                            self._kloss.save(os.path.join(backup_path, "concord_kloss.json"))
                    except Exception as _ke:
                        print(f"[concord] could not write kloss sidecar ({_ke})", flush=True)
                except OSError as e:
                    print(f"[concord] could not write backup clock ({e}); a "
                          f"resume with a different accum will mis-seed", flush=True)
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        # Restart-per-segment (concord_train_restart wrapper): the backup is
        # written WITH the controller clock; skip the in-process recommit +
        # graph recapture below and exit(42) instead. On a near-full card that
        # recommit is exactly where the boundary overflows and WDDM-demotes --
        # sticky and compounding (1.08 -> 2.60 s/it observed), because no
        # in-process release reclaims fragmented-but-committed VRAM. The wrapper
        # relaunches a FRESH process (clean allocator) that resumes from this
        # backup; the clock file (exact update-step) + drive sidecar make the
        # resume bit-faithful and the latent cache is reused (CONCORD_RESUMING),
        # so the cost is ~1-2 min of model reload per segment. Exit BEFORE the
        # recommit (the model is already on temp_device here) -- a coincident
        # mid-run save is deferred to the next segment (the backup holds full
        # state); saves are end-of-run in the standard config.
        if restart_after:
            import sys
            print("[concord-restart] backup written -> exit(42) for fresh-process "
                  "relaunch (skipping the in-process recommit)", flush=True)
            self._kill_tensorboard_for_restart()
            sys.stdout.flush()
            sys.stderr.flush()
            sys.exit(42)

        # [main patch 0003] Same recommit-ordering discipline as the sample
        # path: empty the heap before setup_train_device recommits the UNet, or
        # WDDM can demote the tail to shared memory.
        torch_gc()
        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()
        self._graphmem("post-backup-restore")

    def __save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __concord_batch_emb_ids(self, batch):
        """Per-example TRAINABLE-EMBEDDING TOKEN IDS (ids past the tokenizer's base vocab --
        added tokens are appended there, so the threshold needs no model introspection).
        The examples' identity key for hard-negative telemetry: numeric, always present in
        the batch, and the mining unit anyway (the packed embedding rows) -- concept-name
        strings stay out of logs and sidecars. None when tokens are absent."""
        _tok = batch.get("tokens_1", None) if isinstance(batch, dict) else None
        if _tok is None or not torch.is_tensor(_tok):
            return None
        _base = int(getattr(getattr(self.model, "tokenizer_1", None), "vocab_size", 0) or 0)
        if _base <= 0:
            return None
        _tb = _tok.reshape(_tok.shape[0], -1)
        return [sorted(set(int(t) for t in row[row >= _base].tolist())) for row in _tb]

    def __concord_hardneg_audit(self, batch, train_progress):
        """Gap-guarded hard-negative mining TELEMETRY (meter only -- zero sampler effect).
        Re-runs the CURRENT batch's forward under three weight views (deploy / arm-L / arm-H,
        deterministic noise so the views differ ONLY in weights) and logs per-example deploy
        loss, branch disagreement, and the would-be-mined set. The mining rule this meters:
        mined = high loss among examples whose branch views AGREE (low |loss_L - loss_H|) --
        agreement is what separates hard-informative from suspect (a corrupted/unique example
        is corroborated by only one data half). CPU receipts: guarded mining strictly
        dominates the router+floor best (+1.1pp gen / -1.3pp mem, exp48 [epic-williamson];
        exps 33/34 [mechanics]); UNGUARDED mining self-destructs (selects ~77% corruption ->
        the duplication attack). The mined-set purity statistic this logs is the
        safety-critical number to validate on GPU BEFORE any sampler actuation."""
        _ctrl = getattr(self.model, "concord_controller", None)
        # One-time AUDIT CACHE capture (needs no weight views, so it works under fused matmul
        # too): the exact replayable UNet inputs + target for this batch, for the OFFLINE
        # auditor (scripts/concord_hardneg_audit.py), which materializes the branch views from
        # a BACKUP's packed state instead of the live process -- no fused/capture/OOM
        # constraints, and a FIXED audit set measured against every backup gives longitudinal
        # per-example curves the streaming meter cannot.
        _cache_path = os.path.join(self.config.workspace_dir, "concord_hardneg_audit.pt")
        if not os.path.exists(_cache_path):
            with torch.no_grad():
                _cap = self.model_setup.predict(self.model, batch, self.config, train_progress,
                                                deterministic=True, return_unet_inputs=True)
            torch.save({
                "latent_input": _cap["latent_input"].detach().cpu(),
                "timestep": _cap["timestep"].detach().cpu(),
                "encoder_hidden_states": _cap["encoder_hidden_states"].detach().cpu(),
                "added_cond_kwargs": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                      for k, v in (_cap.get("added_cond_kwargs") or {}).items()},
                "target": _cap["target"].detach().cpu(),
                "emb_ids": self.__concord_batch_emb_ids(batch),
                "captured_step": int(train_progress.global_step),
            }, _cache_path)
            print(f"[concord-hardneg] audit cache captured -> {_cache_path} "
                  f"(B={int(_cap['target'].shape[0])}, step {train_progress.global_step}); replay it "
                  "against any backup with scripts/concord_hardneg_audit.py", flush=True)
        per_view = {}
        with torch.no_grad():
            for _which in ("deploy", "L", "H"):
                # set_branch_view INSIDE the try: an OOM mid-swap must still hit the
                # finally, or some layers keep serving a branch view to training
                try:
                    _n = _ctrl.set_branch_view(_which)
                    if _n == 0:
                        self._hardneg_dead = True
                        print(f"[concord-hardneg] no viewable layers ({_ctrl.branch_view_diag()}) -> "
                              "live meter disabled for this run. fused_matmul=True is the usual cause "
                              "(inline dequant leaves no per-layer cache to swap). The audit cache "
                              "above still works: run scripts/concord_hardneg_audit.py against each "
                              "backup for the same telemetry offline.", flush=True)
                        return
                    _out = self.model_setup.predict(
                        self.model, batch, self.config, train_progress, deterministic=True)
                    _pred, _tgt = _out.get("predicted"), _out.get("target")
                    if _pred is None or _tgt is None:
                        self._hardneg_dead = True
                        print("[concord-hardneg] predict() returned no predicted/target -> "
                              "meter disabled for this run", flush=True)
                        return
                    _d = (_pred.float() - _tgt.float()).pow(2)
                    per_view[_which] = _d.reshape(_d.shape[0], -1).mean(dim=1)
                finally:
                    _ctrl.restore_branch_views()
        _ld = per_view["deploy"]
        _dis = (per_view["L"] - per_view["H"]).abs()
        _B = int(_ld.numel())
        _q75 = _dis.quantile(0.75)
        _trusted = _dis <= _q75            # branch-agreement guard (no label-agreement analog here)
        _thr = _ld[_trusted].quantile(2.0 / 3.0) if int(_trusted.sum()) > 0 else _ld.quantile(2.0 / 3.0)
        _mined = _trusted & (_ld >= _thr)  # high AGREED loss: hard AND corroborated by both halves
        _eids = self.__concord_batch_emb_ids(batch)
        _mtag = ""
        if _eids is not None:
            try:
                _picked = [str(_eids[i]) for i in _mined.nonzero().flatten().tolist()[:6]]
                _mtag = " mined_emb_ids=" + ";".join(_picked)
            except Exception:
                pass
        _rel = float((_dis / _ld.clamp_min(1e-12)).median())
        print(f"[concord-hardneg] step {train_progress.global_step} B={_B} "
              f"loss(dep) p50={float(_ld.median()):.4f} p90={float(_ld.quantile(0.9)):.4f} | "
              f"branch-disagree p50={float(_dis.median()):.2e} p75={float(_q75):.2e} "
              f"rel(p50)={_rel:.3f} | trusted={int(_trusted.sum())}/{_B} "
              f"mined={int(_mined.sum())}{_mtag}", flush=True)
        if self.tensorboard is not None:
            self.tensorboard.add_scalar("hardneg/loss_deploy_p50", float(_ld.median()),
                                        train_progress.global_step)
            self.tensorboard.add_scalar("hardneg/branch_disagree_rel_p50", _rel,
                                        train_progress.global_step)
            self.tensorboard.add_scalar("hardneg/trusted_frac", float(_trusted.float().mean()),
                                        train_progress.global_step)
            self.tensorboard.add_scalar("hardneg/mined_frac", float(_mined.float().mean()),
                                        train_progress.global_step)

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        fused_optimizer_step = self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=__optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                __optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)


    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def _concord_setup_capture(self):
        """Capture the run's console + a structured per-step telemetry stream.

        Two artifacts under workspace_dir, both meter-only (no training effect):
          concord_console_<ts>.log  -- a tee of stdout+stderr: the [concord]
              banners, warnings, tracebacks and the [loss] health lines, so a
              post-mortem does not depend on terminal scrollback surviving.
              tqdm bars animate on the REAL console unchanged (isatty/fileno/
              encoding delegate through); the file keeps only the final
              \\r-collapsed segment of each line (no progress-bar spam).
          concord_telemetry_<ts>.jsonl -- one JSON record per health-line print
              (step/loss/smooth/deploy/gap/boil/waste/boil_cf/armgap): diffable,
              plottable, immune to the console log's reformatting.
        Timestamped so each process (incl. resumes) gets its own pair, never
        clobbering. Master-rank only -- a shared file across ranks would
        interleave. Default on; CONCORD_CAPTURE=0 disables. FAIL-SAFE: any error
        here (or in the tee) leaves training untouched; logging must never crash
        the run or corrupt the console (the real stream always gets every write
        verbatim -- the file side is best-effort)."""
        import os
        import sys
        self._telemetry_fh = None                    # read by the health-line emit
        if getattr(self, "_capture_installed", False):
            return
        self._capture_installed = True
        if os.environ.get("CONCORD_CAPTURE", "1") == "0":
            return
        try:
            if not multi.is_master():
                return
        except Exception:
            pass
        try:
            ts = get_string_timestamp()
            d = self.config.workspace_dir
            os.makedirs(d, exist_ok=True)
            con_path = os.path.join(d, f"concord_console_{ts}.log")
            tel_path = os.path.join(d, f"concord_telemetry_{ts}.jsonl")
            con_fh = open(con_path, "a", encoding="utf-8", errors="replace")
            self._telemetry_fh = open(tel_path, "a", encoding="utf-8", errors="replace")

            class _Tee:
                # Mirror a console stream to a file. Console side is untouched
                # (every write passes straight through; isatty/fileno/encoding
                # delegate, so tqdm still animates). File side buffers to a
                # newline and collapses \r refreshes to the final segment so the
                # log is not progress-bar spam. NEVER raises -- a file error
                # drops the file side and keeps the console.
                def __init__(self, stream, fh):
                    self._stream = stream
                    self._fh = fh
                    self._buf = ""

                def write(self, s):
                    try:
                        n = self._stream.write(s)
                    except Exception:
                        n = len(s)
                    try:
                        self._buf += s
                        while True:
                            i = self._buf.find("\n")
                            if i < 0:
                                break
                            line, self._buf = self._buf[:i + 1], self._buf[i + 1:]
                            if "\r" in line:
                                line = line.rsplit("\r", 1)[-1]
                                if not line.endswith("\n"):
                                    line += "\n"
                            self._fh.write(line)
                        self._fh.flush()
                    except Exception:
                        pass
                    return n

                def flush(self):
                    for t in (self._stream, self._fh):
                        try:
                            t.flush()
                        except Exception:
                            pass

                def __getattr__(self, name):
                    return getattr(self._stream, name)

            sys.stdout = _Tee(sys.stdout, con_fh)
            sys.stderr = _Tee(sys.stderr, con_fh)
            print(f"[capture] console   -> {con_path}", flush=True)
            print(f"[capture] telemetry -> {tel_path}", flush=True)
        except Exception as e:
            try:
                print(f"[capture] disabled (setup failed: {e})", flush=True)
            except Exception:
                pass
            self._telemetry_fh = None

    def train(self):
        self._concord_setup_capture()
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching")
                for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                    self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        ema_loss = None
        ema_deploy = None
        ema_loss_steps = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        for _epoch in tqdm(epochs, desc="epoch") if multi.is_master() else epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            # Epoch-boundary hygiene (ALWAYS, regardless of the release flag):
            # the previous epoch's loop locals live in THIS scope through the
            # cache stage and pin GPU memory: `batch` holds the last
            # micro-batch's tensors, `model_output_data` the predicted/target
            # latents (eager path), and in graph mode `loss`/`detached_loss`
            # reference cap_loss INSIDE the graph's private pool — one live
            # tensor keeps its whole allocator segment resident. Unbind them
            # (plain assignment: safe whether or not the names are bound yet
            # on the first epoch) and gc.
            batch = None
            model_output_data = None
            prior_model_output_data = None
            prior_model_prediction = None
            loss = None
            detached_loss = None
            accumulated_loss = accumulated_loss.item() \
                if torch.is_tensor(accumulated_loss) else accumulated_loss
            torch_gc()
            # Graph release (concord_epoch_cache_release, default ON): drop the
            # captured graph + its multi-GB private pool so the cache stage has
            # maximum headroom; the graph recaptures transparently on the next
            # training step. TRADE-OFF, observed live: the recapture re-allocates
            # eager warmup activations + a fresh pool on a heavily-exercised
            # allocator and can OOM where the train-start capture succeeded
            # (fragmentation) — even with the post-cache gc that already runs
            # below. Set FALSE for keep-graph mode: the capture stays alive
            # across the boundary (no recapture, no OOM risk); the cache runs
            # with less headroom, which may slow it on tight VRAM — a soft
            # failure instead of a hard one. With the locals unbound above, the
            # pool may coexist with the cache workspace where it previously
            # could not.
            if getattr(self.config, "concord_epoch_cache_release", True):
                _v2e = getattr(self.model, "concord_graph_v2", None)
                if _v2e is not None:
                    _v2e.release()
                    torch_gc()

            #call start_next_epoch with only one process at first, because it might write to the cache. All subsequent processes can read in parallel:
            for _ in multi.master_first():
                if self.config.latent_caching:
                    self.data_loader.get_data_set().start_next_epoch()
                    self.model_setup.setup_train_device(self.model, self.config)
                else:
                    self.model_setup.setup_train_device(self.model, self.config)
                    self.data_loader.get_data_set().start_next_epoch()

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

                # give the Concord controller the same total-update horizon the scheduler
                # uses. approximate_length() is already BATCHES, not samples (it scales
                # inversely with batch_size: len~1885 @ bs4 vs len~3773 @ bs2, same
                # dataset; the per-epoch step bar equals it), so divide by accumulation
                # ONLY -- dividing by batch_size too halved the horizon on every run
                # before 2026-06-11: cosine ended mid-run, divot released mid-epoch-1.
                if getattr(self.model, "concord_controller", None) is not None:
                    # Contrast pairing = 2 arm-ticks per step (both arms consolidate on
                    # the same image). With a fraction f of steps paired, the effective
                    # tick rate is (1+f)x the loader-batch rate, and the clock advances
                    # by the tick count per step (concord_ot.after_step). Scale the
                    # horizon by the SAME (1+f) so the cosine ends at the true last epoch
                    # and steps_per_epoch (= total_steps/epochs, below) keeps chase / leak
                    # / beta2 / divot / lam_lo calibrated to ticks, not batches.
                    _tick_mult = 1.0
                    if bool(getattr(self.config, "concord_contrast_arms", False)):
                        _cf = max(0.0, min(1.0, float(getattr(self.config, "concord_contrast_fraction", 1.0) or 1.0)))
                        _accum_h = int(self.config.gradient_accumulation_steps)
                        if _accum_h == 1:
                            # accum==1: the pair IS the update (1 update = 2 ticks) -> (1+f) scaling.
                            _tick_mult = 1.0 + _cf
                        elif _accum_h % 2 == 0:
                            # EVEN accum (pair-as-2-microsteps balance): a contrast cycle spends
                            # accum-1 batches (pair = 1 batch counted as 2 slots + accum-2 ordinary)
                            # but is a FULL optimizer step, so an epoch has B/(accum-f) steps, not
                            # B/accum. Scale by accum/(accum-f) so total_steps = epochs*B/(accum-f)
                            # and the cosine + epoch-window timescales stay calibrated (step_idx
                            # lands exactly at total_steps). accum==2 -> 2/(2-f); accum==4 -> 4/(4-f).
                            _tick_mult = _accum_h / (_accum_h - _cf)
                        # odd accum>=3 (old pair-as-micro-0 path, unbalanced 2:1): standard horizon.
                    self.model.concord_controller.total_steps = max(1, int(
                        self.config.epochs * self.data_loader.get_data_set().approximate_length()
                        * _tick_mult / max(1, self.config.gradient_accumulation_steps)))
                    print(f"[concord] schedule horizon = {self.model.concord_controller.total_steps} "
                          f"steps ({self.config.epochs} epochs, "
                          f"len~{self.data_loader.get_data_set().approximate_length()} batches, "
                          f"accum={self.config.gradient_accumulation_steps}"
                          + (f", contrast tick x{_tick_mult:g}" if _tick_mult != 1.0 else "") + ")")
                    # Telescope epoch window (exp-20 freshness law): pin the
                    # anchor's integration window to the dataset revisit period
                    # now that the horizon (and so steps-per-epoch) is known.
                    self.model.concord_controller.apply_epoch_window(
                        self.model.concord_controller.total_steps / max(1, self.config.epochs))
                    # Resume-aware controller clock: step_idx restarts at 0 each
                    # process, but the fill ramp / autotune probe / watchdog arm
                    # delay are calendar mechanisms — on a resumed run (backup
                    # continue, or the restart-on-sample wrapper's per-segment
                    # relaunches) a zeroed clock would re-ramp the friction from
                    # 0 after EVERY sample and re-run the probe each segment.
                    # Prefer the clock persisted in the backup (exact update
                    # count, accum-change-proof); fall back to deriving from
                    # micro-steps, which silently assumes accum never changed.
                    _resumed_updates = None
                    # Only adopt a persisted controller clock when this process is
                    # actually resuming (backup-continue or the restart-on-sample
                    # wrapper, both of which set continue_last_backup). On a fresh
                    # start get_last_backup_path() still returns the most recent
                    # backup on disk -- without this gate a clean run reads that
                    # stale clock (global_step 32) against its own train_progress=0,
                    # prints the spurious "resume is at 0" fallback. Gate it.
                    _bk = (self.config.get_last_backup_path()
                           if (getattr(self.config, "continue_last_backup", False)
                               and hasattr(self.config, "get_last_backup_path"))
                           else None)
                    if _bk:
                        try:
                            with open(os.path.join(_bk, "concord_clock.json"),
                                      encoding="utf-8") as f:
                                _clk = json.load(f)
                            if (int(_clk.get("global_step", -1))
                                    == int(getattr(train_progress, "global_step", 0))):
                                _resumed_updates = int(_clk["update_steps"])
                                print(f"[concord] controller clock restored from the "
                                      f"backup: update-step {_resumed_updates} "
                                      f"(accum-change-proof)", flush=True)
                                # restore the loss smoothers so the smooth tags
                                # continue instead of reseeding from the first
                                # post-restart batch (the fake-dip artifact;
                                # see the backup-write comment). Old backups
                                # lack these keys -- skip gracefully.
                                if _clk.get("ema_loss") is not None:
                                    ema_loss = float(_clk["ema_loss"])
                                    ema_loss_steps = int(_clk.get("ema_loss_steps")
                                                         or 100)
                                    if _clk.get("ema_deploy") is not None:
                                        ema_deploy = float(_clk["ema_deploy"])
                                    print("[concord] loss smoothers restored from "
                                          "the backup clock (no EMA reseed divot)",
                                          flush=True)
                            else:
                                print(f"[concord] backup clock is for global_step "
                                      f"{_clk.get('global_step')} but resume is at "
                                      f"{getattr(train_progress, 'global_step', 0)}; "
                                      f"falling back to micro-step derivation", flush=True)
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                        _kp = os.path.join(_bk, "concord_kloss.json")
                        if os.path.exists(_kp):
                            self._kloss_resume_path = _kp
                            print("[concord] kloss sidecar found in backup; meter "
                                  "restores on the first update", flush=True)
                    if _resumed_updates is None:
                        _resumed_updates = int(getattr(train_progress, "global_step", 0)
                                               // max(1, self.config.gradient_accumulation_steps))
                    if _resumed_updates > self.model.concord_controller.step_idx:
                        self.model.concord_controller.step_idx = _resumed_updates
                        print(f"[concord] controller clock seeded at update-step "
                              f"{_resumed_updates} (resumed run): fill ramp / probe / "
                              f"watchdog continue instead of restarting", flush=True)

            current_epoch_length = self.data_loader.get_data_set().approximate_length()

            if multi.is_master():
                batches = step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step", total=current_epoch_length,
                                 initial=train_progress.epoch_step)
            else:
                batches = self.data_loader.get_data_loader()
            for batch in batches:
                if os.environ.get("CONCORD_MEMLOG") and train_progress.epoch_step == 0:
                    _a = torch.cuda.memory_allocated() / 1e9
                    _r = torch.cuda.memory_reserved() / 1e9
                    _pr = torch.cuda.max_memory_reserved() / 1e9
                    print(f"[memlog] epoch {train_progress.epoch}: allocated={_a:.2f}G "
                          f"reserved={_r:.2f}G gap_frag={_r - _a:.2f}G peak_reserved={_pr:.2f}G",
                          flush=True)
                    # reset so the NEXT epoch's peak_reserved isolates the training peak
                    # (graph capture + step activations + any bf16 weight cache) from one-time
                    # setup/model-load transients.
                    torch.cuda.reset_peak_memory_stats()
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                _concord_resumed_step = bool(os.environ.pop("CONCORD_RESUMING", None))
                if _concord_resumed_step:
                    # Resumed right after a Concord checkpoint-restart. Two things at THIS restored
                    # step already happened in the prior process and must NOT be redone, or the
                    # segment exit(42)s before any training step runs and livelocks at the boundary:
                    #   (1) the sample at this step already ran; the "already sampled" flag doesn't
                    #       survive a restart, so re-sampling would re-checkpoint + re-exit.
                    #   (2) under CONCORD_RESTART_ON_BACKUP the per-epoch backup we resumed FROM is
                    #       this exact (epoch, epoch_step==0) boundary. __needs_backup is STATELESS
                    #       for the EPOCH unit (fires whenever epoch_step==0 and epoch>0), so it
                    #       re-fires here and re-exits(42) at the same step forever -- every backup
                    #       pinned to <step>-<epoch>-0, zero steps trained.
                    # Skip BOTH once (one-shot via the env pop): the training step still runs, so
                    # epoch_step advances past 0 and the next backup fires at the NEXT epoch boundary.
                    print("[concord-restart] resumed -> skipping the already-done sample + redundant "
                          "boundary backup at this step", flush=True)
                elif (not self.commands.get_stop_command() and self.__needs_sample(train_progress)) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress) and not _concord_resumed_step:
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        # Graph-mode hard-neg audit: eager forwards are only safe here, where
                        # the graph is about to be released for the boundary work anyway. The
                        # release frees the private pool (idempotent -- __backup/__save's own
                        # release no-ops after this) and the model is still on the train
                        # device; under restart-on-backup the process exits after the backup,
                        # so the recapture this forces costs nothing.
                        _hn_v2 = getattr(self.model, "concord_graph_v2", None)
                        if (_hn_v2 is not None
                                and not getattr(self, "_hardneg_dead", False)
                                and bool(getattr(self.config, "concord_hardneg_meter", False))
                                and getattr(self, "_hn_batch", None) is not None
                                and getattr(self.model, "concord_controller", None) is not None):
                            _hn_v2.release()
                            try:
                                self.__concord_hardneg_audit(self._hn_batch, train_progress)
                            except torch.cuda.OutOfMemoryError:
                                self._hardneg_dead = True
                                torch.cuda.empty_cache()
                                print("[concord-hardneg] boundary audit OOM -> meter disabled "
                                      "for this run", flush=True)
                        self.model.to(self.temp_device)
                        if backup:
                            # restart-per-segment: when the wrapper set
                            # CONCORD_RESTART_ON_BACKUP, this backup exits(42)
                            # before its in-process recommit (samples are off in
                            # the standard config, so the epoch backup is the
                            # only boundary -- the sample trigger never fires).
                            _rob = bool(getattr(self.model, "concord_graph_v2", None)
                                        and os.environ.get("CONCORD_RESTART_ON_BACKUP"))
                            self.__backup(train_progress, True, step_tqdm.write,
                                          restart_after=_rob)
                        if save:
                            self.__save(train_progress, True, step_tqdm.write)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                with (
                    TorchMemoryRecorder(enabled=False, filename=f"memory-step{train_progress.global_step}.pickle"),
                    TorchProfiler      (enabled=False, filename=f"profile-step{train_progress.global_step}.json"),
                ):
                    step_seed = train_progress.global_step
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)

                    # per-step pre-forward hook (Concord: advance the winner lr/sigma/floor
                    # schedule onto the layer device tensors the fused backward reads)
                    self.model_setup.before_step(self.model, self.config, train_progress)

                    # Concord gradient accumulation: the fused step lives in the backward, so
                    # tell it whether THIS micro-step consolidates (full apply) or only ticks
                    # the gradient into s_fast (weights frozen). __is_update_step is True only
                    # on the cycle's last micro-step — the same condition that gates
                    # optimizer.step() below. accum==1 -> always True -> unchanged behavior.
                    # Contrast whole-step routing (accum==2 only): when the pair fires it IS the
                    # whole 2-tick cycle -- arm L (tick) + arm H (tick + CONSOLIDATE) in one
                    # _contrast_step call -- so force the update here and count it as 2 micros
                    # (global_step += 2 at end of iter). _contrast_fire is a pure hash of
                    # global_step, so this matches step()'s own decision exactly. INERT unless
                    # accum==2 with contrast on in bridge mode -> the accum=3 run is untouched
                    # until the config is switched. See ManualUNetGraph._contrast_step.
                    # EVEN accum: count the pair as 2 microsteps so the arms fill 1:1. At accum==2
                    # the pair IS the whole cycle (it consolidates -> forced update); at accum>=4 it
                    # is micros 0-1 (tick-only, both arms) and the cycle's last ordinary micro
                    # consolidates as usual. INERT at odd accum (old unbalanced pair-micro-0 path).
                    _contrast_pair = False
                    _v2c = getattr(self.model, "concord_graph_v2", None)
                    _accum_c = int(self.config.gradient_accumulation_steps)
                    if (_v2c is not None and getattr(self.config, "concord_contrast_arms", False)
                            and _accum_c % 2 == 0
                            and not getattr(_v2c, "graph_te", False)):
                        _contrast_pair = bool(_v2c._contrast_fire(train_progress))
                    _contrast_whole = _contrast_pair and _accum_c == 2   # pair == whole cycle only at accum==2
                    if getattr(self.model, "concord_controller", None) is not None:
                        from modules.util.optimizer.concord.prototype_packed_b import set_consolidate, set_arm_sel
                        # A contrast whole-step consolidates on the pair's arm H, so force the
                        # consolidate flag on even though this micro is the cycle's first.
                        set_consolidate(train_device, self.__is_update_step(train_progress) or _contrast_whole)
                        # Held-out arm router selector: alternate every micro (parity of the
                        # 0-indexed micro counter; next_step() runs later in this iteration).
                        # Same in-place-fill discipline as consolidate -> propagates into a
                        # captured graph's replays. Read only when the router flag is on.
                        # (On a contrast whole-step the pair overrides arm_sel internally.)
                        set_arm_sel(train_device, (train_progress.global_step & 1) == 0)

                    # NOTE (2026-07-12): loader-level contrastive-arms (yield each
                    # batch twice + per-micro caption/seed) is INCOMPATIBLE with
                    # the CUDA graph -- re-running the same batch through the
                    # eager-prep -> graph pipeline twice disturbs the graph's
                    # memory pool and segfaults (0xC0000005) at capture. Reverted
                    # to inert. The graph-compatible design pairs INSIDE the graph
                    # step() as a second replay with the dropped ehs (the
                    # uncond_pass pattern, concord_graph.py:510) -- deferred to a
                    # GPU-validated build; concord_contrast_arms is a no-op until.

                    _v2 = getattr(self.model, "concord_graph_v2", None)
                    if _v2 is not None:
                        # Stage 3 v2: the manual graph does predict-prep (eager) + a captured
                        # UNet -> loss -> backward replay; loss.backward() happens inside.
                        loss = _v2.step(self.model, batch, self.config, train_progress)
                        # timestep-stratified loss recorder (opt-in; sentinel CONCORD_LOSS_TS.on):
                        # the graph stashed THIS step's timesteps in static["timestep"] -> pair with
                        # the returned loss. Gated + buffered in the controller; no-op otherwise.
                        _lt_ctrl = getattr(self.model, "concord_controller", None)
                        if _lt_ctrl is not None and getattr(_lt_ctrl, "_loss_ts_on", False):
                            _lt_ts = (getattr(_v2, "static", None) or {}).get("timestep")
                            if _lt_ts is not None:
                                _lt_ctrl._maybe_log_loss_ts(_lt_ts, loss)
                    else:
                        prior_pred_indices = [i for i in range(self.config.batch_size)
                                              if ConceptType(batch['concept_type'][i]) == ConceptType.PRIOR_PREDICTION]
                        if len(prior_pred_indices) > 0 \
                                or (self.config.masked_training
                                    and self.config.masked_prior_preservation_weight > 0
                                    and self.config.training_method == TrainingMethod.LORA):
                            with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                                #do NOT create a subbatch using the indices, even though it would be more efficient:
                                #different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                                prior_model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                            prior_model_prediction = prior_model_output_data['predicted'].to(dtype=model_output_data['target'].dtype)
                            model_output_data['target'][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                            model_output_data['prior_target'] = prior_model_prediction
                        else:
                            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                        # gamma-SNR dissipation modulation (eager path; the graph path has
                        # its own hook): timesteps are sampled by predict() above, the fused
                        # backward below reads the kappa buffers.
                        _ctrl = getattr(self.model, "concord_controller", None)
                        if _ctrl is not None and "timestep" in model_output_data:
                            _ac = getattr(self.model.noise_scheduler, "alphas_cumprod", None)
                            if _ac is not None:
                                _ctrl.on_timesteps(model_output_data["timestep"], _ac)

                        loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

                        loss = loss / self.config.gradient_accumulation_steps
                        if scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    has_gradient = True
                    detached_loss = loss.detach()
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss

                    if self.__is_update_step(train_progress) or _contrast_whole:
                        if self.config.fused_gradient_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        # Stage 3 v2: keep aux .grad buffers static (zero in place) so the
                        # captured backward writes to the same memory each replay.
                        self.model.optimizer.zero_grad(
                            set_to_none=getattr(self.model, "concord_graph_v2", None) is None)
                        has_gradient = False

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            accumulated_loss_cpu = accumulated_loss.item()
                            if math.isnan(accumulated_loss_cpu):
                                raise RuntimeError("Training loss became NaN. This may be due to invalid parameters, precision issues, or a bug in the loss computation.")

                            self.tensorboard.add_scalar("loss/train_step",accumulated_loss_cpu , train_progress.global_step)
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix({
                                'loss': accumulated_loss_cpu,
                                'smooth loss': ema_loss,
                            })
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                            # Concord memorization-gap meter: first-order (L_deploy - L_live)
                            # accumulated in the fused backward (sum grad*s_fast -- free).
                            # deploy_est = loss + gap is the number comparable across
                            # friction / gamma-SNR regimes; the live loss alone is deflated
                            # by the batch-fitted transient riding in s_fast.
                            _ctrl = getattr(self.model, "concord_controller", None)
                            _gap = None
                            _boil = _waste = _boil_prot = _armgap = None
                            if _ctrl is not None:
                                _gap = _ctrl.read_memorization_gap()
                                _deploy = accumulated_loss_cpu + _gap
                                # the gap flickers at high lam (transient ~ noise), so the
                                # readable deploy number is the EMA (same decay as smooth loss)
                                ema_deploy = _deploy if ema_deploy is None else \
                                    (ema_deploy * ema_loss_decay + _deploy * (1 - ema_loss_decay))
                                self.tensorboard.add_scalar(
                                    "loss/concord_gap", _gap, train_progress.global_step)
                                self.tensorboard.add_scalar(
                                    "loss/deploy_est", _deploy, train_progress.global_step)
                                self.tensorboard.add_scalar(
                                    "smooth_loss/deploy_est", ema_deploy, train_progress.global_step)
                            # stash for the backup clock (smoothers survive restarts)
                            self._ema_state = {"ema_loss": ema_loss,
                                               "ema_deploy": ema_deploy,
                                               "ema_loss_steps": ema_loss_steps}
                            # Plain stdout loss line: the tqdm postfix is transient
                            # (overwritten in place, lost on redirect); this survives in
                            # console scrollback and piped logs. tqdm.write keeps the
                            # progress bars intact.
                            # spike-logger step attribution (no-op when the
                            # logger is disabled; see modules/util/spike_log.py)
                            from modules.util import spike_log as _spike_log
                            _spike_log.SPIKE_LOG.set_step(train_progress.global_step)
                            # Kalman loss meter: timestep-conditional baseline + [skill, trend]
                            # filter -> a READABLE convergence number with a confidence interval
                            # (see modules/util/kalman_loss.py). Timesteps come from the concord
                            # controller's on_timesteps stash; None degrades to unconditioned mode.
                            if not hasattr(self, "_kloss"):
                                from modules.util.kalman_loss import KalmanLossMeter
                                _spe = float(getattr(_ctrl, "steps_per_epoch", 0) or 0) if _ctrl else 0.0
                                _krp = getattr(self, "_kloss_resume_path", None)
                                # backup-scoped: restore from the resumed backup, else fresh.
                                self._kloss = (KalmanLossMeter.load(_krp, _spe if _spe > 0 else 514.0)
                                               if _krp else KalmanLossMeter(_spe if _spe > 0 else 514.0))
                            _kt = getattr(_ctrl, "_last_timesteps", None) if _ctrl else None
                            _kline = self._kloss.update(
                                accumulated_loss_cpu,
                                _kt.tolist() if _kt is not None else None)
                            _msg = (f"[loss] step {train_progress.global_step}"
                                    f"  loss={accumulated_loss_cpu:.5f}"
                                    f"  smooth={ema_loss:.5f}")
                            if _kline:
                                _msg += f"  | {_kline}"
                            if _gap is not None:
                                _msg += (f"  gap={_gap:+.2e}"
                                         f"  deploy_smooth={ema_deploy:.5f}")
                                _boil, _waste = _ctrl.read_flow_audit()
                                if _boil is not None:
                                    self.tensorboard.add_scalar(
                                        "loss/concord_boil", _boil, train_progress.global_step)
                                    _msg += f"  boil={_boil:.3f}"
                                if _waste is not None:
                                    self.tensorboard.add_scalar(
                                        "loss/concord_waste", _waste, train_progress.global_step)
                                    _msg += f"  waste={_waste:.3f}"
                                _boil_prot = getattr(_ctrl, "_last_boil_protected", None)
                                if _boil_prot is not None:
                                    self.tensorboard.add_scalar(
                                        "loss/concord_boil_protected", _boil_prot, train_progress.global_step)
                                    _msg += f"  boil_cf={_boil_prot:.3f}"
                                # Direct contrast meter: L_dropped - L_full on the
                                # SAME latent/timestep/noise (arms differ by caption
                                # alone). Positive => caption context lowers the loss
                                # = the arms are differentiating. Stashed by
                                # concord_graph._contrast_step; None when contrast off.
                                _cf = getattr(_ctrl, "_last_contrast_full", None)
                                _cd = getattr(_ctrl, "_last_contrast_drop", None)
                                if _cf is not None and _cd is not None:
                                    try:
                                        import torch as _tt2
                                        _fv = _cf.item() if _tt2.is_tensor(_cf) else float(_cf)
                                        _dv = _cd.item() if _tt2.is_tensor(_cd) else float(_cd)
                                        _armgap = _dv - _fv
                                        self.tensorboard.add_scalar(
                                            "loss/concord_arm_gap", _armgap, train_progress.global_step)
                                        _msg += f"  armgap={_armgap:+.2e}"
                                    except Exception:
                                        pass
                                _m6a = getattr(_ctrl, "_last_m6a", None)
                                if _m6a is not None and getattr(self.config, "concord_m6a_meter", False):
                                    self.tensorboard.add_scalar(
                                        "loss/concord_m6a", _m6a, train_progress.global_step)
                                    _msg += f"  m6a={_m6a:.3e}"
                                _csnr = getattr(_ctrl, "_csnr_curve", None)
                                if _csnr is not None and getattr(self, "_kloss", None) is not None:
                                    # CSNR loss-prediction readout (the meter's acceptance gate): does the
                                    # recovered per-timestep gradient-SNR curve predict the t-conditional
                                    # loss baseline (_kloss.f)? Rank-corr over seen loss buckets.
                                    try:
                                        import torch as _tt
                                        from modules.util.kalman_loss import N_BUCKETS as _NB, T_MAX as _TM
                                        _cv = _csnr[0].float().cpu().flatten(); _tn = _cv.numel()
                                        _bi = (_tt.arange(_tn).float() / _TM * _NB).clamp(max=_NB - 1).long()
                                        _snrb = (_tt.zeros(_NB).scatter_add_(0, _bi, _cv)
                                                 / _tt.zeros(_NB).scatter_add_(0, _bi, _tt.ones(_tn)).clamp_min(1.0))
                                        _fb = _tt.tensor(self._kloss.f, dtype=_tt.float32)
                                        _seen = _tt.tensor(self._kloss.f_seen) >= 5
                                        if int(_seen.sum()) >= 4:
                                            _rx = _snrb[_seen].argsort().argsort().float()
                                            _ry = _fb[_seen].argsort().argsort().float()
                                            _rx -= _rx.mean(); _ry -= _ry.mean()
                                            _rho = float((_rx @ _ry) / (_rx.norm() * _ry.norm() + 1e-30))
                                            _msg += f"  csnr~loss={_rho:+.2f}(g{_csnr[1]})"
                                            self.tensorboard.add_scalar("loss/concord_csnr_vs_loss",
                                                                        _rho, train_progress.global_step)
                                    except Exception:
                                        pass
                            step_tqdm.write(_msg)
                            # Structured per-line mirror (meter only; see
                            # _concord_setup_capture). One JSON record per health
                            # line -> diffable/plottable, immune to console
                            # reformatting. Fail-safe: a logging error never
                            # touches training.
                            _tfh = getattr(self, "_telemetry_fh", None)
                            if _tfh is not None:
                                try:
                                    import time as _time
                                    _rec = {"t": _time.time(),
                                            "step": train_progress.global_step,
                                            "loss": accumulated_loss_cpu,
                                            "smooth": ema_loss}
                                    if _gap is not None:
                                        _rec["gap"] = _gap
                                        _rec["deploy_smooth"] = ema_deploy
                                    if _boil is not None:
                                        _rec["boil"] = _boil
                                    if _waste is not None:
                                        _rec["waste"] = _waste
                                    if _boil_prot is not None:
                                        _rec["boil_cf"] = _boil_prot
                                    if _armgap is not None:
                                        _rec["armgap"] = _armgap
                                    _tfh.write(json.dumps(_rec) + "\n")
                                    _tfh.flush()
                                except Exception:
                                    pass
                                # Drain buffered contrast pairs (armgap x SNR meter): one JSON
                                # line per pair carrying its timestep vector + arm losses, so the
                                # offline harness can bin armgap by schedule / live-gradient SNR.
                                # Independent try so a drain error never blocks the _rec write.
                                try:
                                    _pend = getattr(_ctrl, "_contrast_pending", None)
                                    if _pend:
                                        _drained = _pend
                                        _ctrl._contrast_pending = []
                                        import torch as _tt3, time as _time2
                                        for _pts, _pf, _pd in _drained:
                                            _pfv = _pf.item() if _tt3.is_tensor(_pf) else float(_pf)
                                            _pdv = _pd.item() if _tt3.is_tensor(_pd) else float(_pd)
                                            _tfh.write(json.dumps({
                                                "t": _time2.time(),
                                                "step": train_progress.global_step,
                                                "contrast_full": _pfv, "contrast_drop": _pdv,
                                                "armgap": _pdv - _pfv,
                                                "contrast_ts": [int(x) for x in
                                                                _pts.detach().flatten().cpu().tolist()],
                                            }) + "\n")
                                        _tfh.flush()
                                except Exception:
                                    pass

                            # Concord hard-negative mining telemetry (meter only; see
                            # __concord_hardneg_audit). Runs at an UPDATE boundary by
                            # construction (this block is update-gated), which the branch-view
                            # swap requires. Self-disables on OOM or missing prerequisites.
                            if _ctrl is not None and not getattr(self, "_hardneg_dead", False) \
                                    and bool(getattr(self.config, "concord_hardneg_meter", False)):
                                # keep a reference to the freshest batch for the BOUNDARY audit
                                # (the graph-mode path: eager forwards are only safe there)
                                self._hn_batch = batch
                                _hn_every = max(1, int(getattr(self.config,
                                                               "concord_hardneg_every", 128) or 128))
                                self._hn_updates = getattr(self, "_hn_updates", 0) + 1
                                if self._hn_updates % _hn_every == 0 \
                                        and getattr(self.model, "concord_graph_v2", None) is None:
                                    # cadence audits are EAGER-mode only: under the graph the
                                    # pool holds the VRAM an eager forward needs; the audit
                                    # instead fires once per backup boundary (below), where the
                                    # release frees the pool -- and under restart-on-backup the
                                    # recapture cost is zero (the process exits right after).
                                    try:
                                        self.__concord_hardneg_audit(batch, train_progress)
                                    except torch.cuda.OutOfMemoryError:
                                        self._hardneg_dead = True
                                        torch.cuda.empty_cache()
                                        print("[concord-hardneg] audit OOM -> meter disabled for "
                                              "this run (raise concord_hardneg_every or lower "
                                              "batch)", flush=True)

                        accumulated_loss = 0.0
                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True

                if self.config.validation and multi.is_master():
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                if _contrast_pair:
                    # The pair does BOTH arms in one call, so count it as 2 microsteps: advance
                    # global_step by one more. Keeps gs % accum cycle-aligned and gs // accum =
                    # update count correct (how the router parity and resume both read it), and
                    # makes the ordinary-micro count per cycle EVEN so the arms fill 1:1 (accum==2:
                    # 0 ordinary; accum==4: 2 ordinary). Raw bump: no extra batch/sample consumed
                    # (the pair reused one image across both arms) -> epoch_step stays on 1 batch.
                    train_progress.global_step += 1
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            self.model.to(self.temp_device)

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)
                if os.path.isdir(self.config.output_model_destination) and self.config.output_model_format.is_single_file():
                    save_path = os.path.join(
                        self.config.output_model_destination,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}"
                    )
                else:
                    save_path = self.config.output_model_destination
                print("Saving " + save_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype()
                )

        if self.model is not None:
            self.model.to(self.temp_device)

        if multi.is_master():
            self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
