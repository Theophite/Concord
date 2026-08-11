import json
import os
import uuid
from copy import deepcopy
from typing import Any

from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.CloudConfig import CloudConfig
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.GradientReducePrecision import GradientReducePrecision
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.Optimizer import Optimizer, is_concord_family
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.ModelNames import EmbeddingName, ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.torch_util import default_device


class TrainOptimizerConfig(BaseConfig):
    optimizer: Optimizer
    adam_w_mode: bool
    alpha: float
    amsgrad: bool
    beta1: float
    beta2: float
    beta3: float
    bias_correction: bool
    block_wise: bool
    capturable: bool
    centered: bool
    clip_threshold: float
    d0: float
    d_coef: float
    dampening: float
    decay_rate: float
    decouple: bool
    differentiable: bool
    eps: float
    eps2: float
    foreach: bool
    fsdp_in_use: bool
    fused: bool
    fused_back_pass: bool
    growth_rate: float
    initial_accumulator_value: int
    initial_accumulator: float
    is_paged: bool
    log_every: int
    lr_decay: float
    max_unorm: float
    maximize: bool
    min_8bit_size: int
    quant_block_size: int
    momentum: float
    nesterov: bool
    no_prox: bool
    optim_bits: int
    percentile_clipping: int
    r: float
    relative_step: bool
    safeguard_warmup: bool
    scale_parameter: bool
    stochastic_rounding: bool
    use_bias_correction: bool
    use_triton: bool
    warmup_init: bool
    weight_decay: float
    gf_consol: float
    noise: bool
    sigmag_peak: float
    ratio_coh: bool
    lazy_gate: bool
    lazy_active_thresh: float
    autotune_table: str
    autotune_reprobe_band: float
    autotune_gamma_snr: float
    dissipation: float
    autotune_beta1_on: float
    autotune_beta1_coh: float
    autotune_servo_per_epoch: int         # noise-seed seeder window divisor: NSR window = steps_per_epoch // this (>= ~32-update floor) Default 3; 1 = legacy once-per-epoch. Concord only.
    concord_conv_full_vhat: bool          # CONV layers only: full per-element Adam v_hat instead of rank-1 v_row*v_col; fixes conv over-cook. Costs one fp32 [out,in*k*k] buffer/conv. Default OFF. Concord only.
    concord_evap_slack: float             # cf-aware evap floor: kill gates on min(coh, coh_raw+slack); cf can spare a coord from evaporation by at most slack above raw coherence (invariant-7 noise floor, deflates boil_cf). Default 0.25; 0 = raw-coh kill. Concord only.
    concord_train_cond_embed: bool        # ADVANCED/RISKY: train SDXL time_embedding + add_embedding MLPs (frozen by default -- training them over-cooks conditioning norms -> free-run collapse). Default OFF. Concord only.
    warmup: int
    lr_min_frac: float
    step_cap: float
    gf_trust_delta_sq: float
    min_leak: float
    evap_build_min: float
    lamb_trust: bool
    lamb_cap: float
    lamb_clip: float
    beta2_epoch_window: bool
    vhat_warmstart: bool
    bias_correct_v: bool
    coh_vhat: bool
    coh_kappa: float
    dissipation_fill_ramp: bool
    telescope_epoch_window: bool
    two_fast: bool
    bracket_d: float
    evict_valve: bool
    evict_gain: float
    evict_cf_gate: bool
    heldout_router: bool
    router_coh_noise: bool
    perfcoh_partition: bool
    perfcoh_tau: float
    noise_seed_servo: bool
    nsr_per_row: bool
    kaiming_init: bool
    kaiming_scale: float
    chase_epoch_window: bool
    chase_alpha: float
    alpha_v_fast: float
    ratio_chase_floor: float
    ratio_chase_floor_min: float
    ratio_leak_floor: float
    ratio_leak_floor_min: float
    weight_lr_power: float
    decoupled_decay: bool
    fixed_decay: bool
    weight_decouple: bool
    rectify: bool
    degenerated_to_sgd: bool
    k: int
    xi: float
    n_sma_threshold: int
    ams_bound: bool
    adanorm: bool
    adam_debias: bool
    slice_p: int
    cautious: bool
    weight_decay_by_lr: True
    prodigy_steps: 0
    use_speed: False
    split_groups: True
    split_groups_mean: True
    factored: True
    factored_fp32: True
    use_stableadamw: True
    use_cautious: False
    use_grams: False
    use_adopt: False
    d_limiter: True
    use_schedulefree: True
    use_orthograd: False
    nnmf_factor: False
    orthogonal_gradient: False
    use_atan2: False
    use_AdEMAMix: False
    beta3_ema: float
    alpha_grad: float
    beta1_warmup: int
    min_beta1: float
    Simplified_AdEMAMix: False
    kourkoutas_beta: False
    schedulefree_c: float
    ns_steps: int
    MuonWithAuxAdam: False
    muon_hidden_layers: str
    muon_adam_regex: False
    muon_adam_lr: float
    muon_te1_adam_lr: float
    muon_te2_adam_lr: float
    muon_adam_config: dict
    rms_rescaling: True
    normuon_variant: False
    beta2_normuon: float
    low_rank_ortho: False
    ortho_rank: int
    accelerated_ns: False
    cautious_wd: False
    approx_mars: False
    auto_kappa_p: False
    compile: False

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("optimizer", Optimizer.ADAMW, Optimizer, False))
        data.append(("adam_w_mode", False, bool, False))
        data.append(("alpha", None, float, True))
        data.append(("amsgrad", False, bool, False))
        data.append(("beta1", None, float, True))
        data.append(("beta2", None, float, True))
        data.append(("beta3", None, float, True))
        data.append(("bias_correction", False, bool, False))
        data.append(("block_wise", False, bool, False))
        data.append(("capturable", False, bool, False))
        data.append(("centered", False, bool, False))
        data.append(("clip_threshold", None, float, True))
        data.append(("d0", None, float, True))
        data.append(("d_coef", None, float, True))
        data.append(("dampening", None, float, True))
        data.append(("decay_rate", None, float, True))
        data.append(("decouple", False, bool, False))
        data.append(("differentiable", False, bool, False))
        data.append(("eps", None, float, True))
        data.append(("eps2", None, float, True))
        data.append(("foreach", False, bool, True))  # Disabled, because it uses too much VRAM
        data.append(("fsdp_in_use", False, bool, False))
        data.append(("fused", False, bool, False))
        data.append(("fused_back_pass", False, bool, False))
        data.append(("growth_rate", None, float, True))
        data.append(("initial_accumulator_value", None, int, True))
        data.append(("initial_accumulator", None, float, True))
        data.append(("is_paged", False, bool, False))
        data.append(("log_every", None, int, True))
        data.append(("lr_decay", None, float, True))
        data.append(("max_unorm", None, float, True))
        data.append(("maximize", False, bool, False))
        data.append(("min_8bit_size", None, int, True))
        data.append(("quant_block_size", None, int, True))
        data.append(("momentum", None, float, True))
        data.append(("nesterov", False, bool, False))
        data.append(("no_prox", False, bool, False))
        data.append(("optim_bits", None, int, True))
        data.append(("percentile_clipping", None, int, True))
        data.append(("r", None, float, True))
        data.append(("relative_step", False, bool, False))
        data.append(("safeguard_warmup", False, bool, False))
        data.append(("scale_parameter", False, bool, False))
        data.append(("stochastic_rounding", True, bool, False))
        data.append(("use_bias_correction", False, bool, False))
        data.append(("use_triton", False, bool, False))
        data.append(("warmup_init", False, bool, False))
        data.append(("weight_decay", None, float, True))
        # --- Concord winner knobs (read by concord_ot.make_concord_config; None -> winner default) ---
        data.append(("gf_consol", None, float, True))
        data.append(("noise", None, bool, True))
        data.append(("sigmag_peak", None, float, True))
        data.append(("ratio_coh", None, bool, True))
        data.append(("lazy_gate", None, bool, True))
        data.append(("lazy_active_thresh", None, float, True))
        data.append(("autotune_table", None, str, True))
        data.append(("autotune_reprobe_band", None, float, True))
        data.append(("autotune_gamma_snr", None, float, True))
        data.append(("autotune_gamma_snr_on", None, bool, True))
        data.append(("dissipation", None, float, True))
        data.append(("autotune_beta1_on", None, float, True))
        data.append(("autotune_beta1_coh", None, float, True))
        data.append(("autotune_servo_per_epoch", None, int, True))
        data.append(("concord_conv_full_vhat", None, bool, True))
        data.append(("concord_evap_slack", None, float, True))
        data.append(("concord_train_cond_embed", None, bool, True))
        data.append(("warmup", None, int, True))
        data.append(("lr_min_frac", None, float, True))
        data.append(("step_cap", None, float, True))
        data.append(("gf_trust_delta_sq", None, float, True))
        data.append(("min_leak", None, float, True))
        data.append(("evap_build_min", None, float, True))
        data.append(("lamb_trust", None, bool, True))
        data.append(("lamb_cap", None, float, True))
        data.append(("lamb_clip", None, float, True))
        data.append(("beta2_epoch_window", None, bool, True))
        data.append(("vhat_warmstart", None, bool, True))
        data.append(("bias_correct_v", None, bool, True))
        data.append(("coh_vhat", None, bool, True))
        data.append(("coh_kappa", None, float, True))
        data.append(("dissipation_fill_ramp", None, bool, True))
        data.append(("telescope_epoch_window", None, bool, True))
        data.append(("two_fast", None, bool, True))
        data.append(("bracket_d", None, float, True))
        data.append(("evict_valve", None, bool, True))
        data.append(("evict_gain", None, float, True))
        data.append(("evict_cf_gate", None, bool, True))
        data.append(("heldout_router", None, bool, True))
        data.append(("router_coh_noise", None, bool, True))
        data.append(("perfcoh_partition", None, bool, True))
        data.append(("perfcoh_tau", None, float, True))
        data.append(("noise_seed_servo", None, bool, True))
        data.append(("nsr_per_row", None, bool, True))
        data.append(("kaiming_init", None, bool, True))
        data.append(("kaiming_scale", None, float, True))
        data.append(("chase_epoch_window", None, bool, True))
        data.append(("chase_alpha", None, float, True))
        data.append(("alpha_v_fast", None, float, True))
        data.append(("ratio_chase_floor", None, float, True))
        data.append(("ratio_chase_floor_min", None, float, True))
        data.append(("ratio_leak_floor", None, float, True))
        data.append(("ratio_leak_floor_min", None, float, True))
        data.append(("weight_lr_power", None, float, True))
        data.append(("decoupled_decay", False, bool, False))
        data.append(("fixed_decay", False, bool, False))
        data.append(("rectify", False, bool, False))
        data.append(("degenerated_to_sgd", False, bool, False))
        data.append(("k", None, int, True))
        data.append(("xi", None, float, True))
        data.append(("n_sma_threshold", None, int, True))
        data.append(("ams_bound", False, bool, False))
        data.append(("adanorm", False, bool, False))
        data.append(("adam_debias", False, bool, False))
        data.append(("slice_p", None, int, True))
        data.append(("cautious", False, bool, False))
        data.append(("weight_decay_by_lr", True, bool, False))
        data.append(("prodigy_steps", None, int, True))
        data.append(("use_speed", False, bool, False))
        data.append(("split_groups", True, bool, False))
        data.append(("split_groups_mean", True, bool, False))
        data.append(("factored", True, bool, False))
        data.append(("factored_fp32", True, bool, False))
        data.append(("use_stableadamw", True, bool, False))
        data.append(("use_cautious", False, bool, False))
        data.append(("use_grams", False, bool, False))
        data.append(("use_adopt", False, bool, False))
        data.append(("d_limiter", True, bool, True))
        data.append(("use_schedulefree", True, bool, True))
        data.append(("use_orthograd", False, bool, False))
        data.append(("nnmf_factor", False, bool, False))
        data.append(("orthogonal_gradient", False, bool, False))
        data.append(("use_atan2", False, bool, False))
        data.append(("use_AdEMAMix", False, bool, False))
        data.append(("beta3_ema", None, float, True))
        data.append(("alpha_grad", None, float, True))
        data.append(("beta1_warmup", None, int, True))
        data.append(("min_beta1", None, float, True))
        data.append(("Simplified_AdEMAMix", False, bool, False))
        data.append(("kourkoutas_beta", False, bool, False))
        data.append(("schedulefree_c", None, float, True))
        data.append(("ns_steps", None, int, True))
        data.append(("MuonWithAuxAdam", False, bool, False))
        data.append(("muon_hidden_layers", None, str, True))
        data.append(("muon_adam_regex", False, bool, False))
        data.append(("muon_adam_lr", None, float, True))
        data.append(("muon_te1_adam_lr", None, float, True))
        data.append(("muon_te2_adam_lr", None, float, True))
        data.append(("muon_adam_config", {}, dict, True))
        data.append(("rms_rescaling", True, bool, True))
        data.append(("normuon_variant", False, bool, False))
        data.append(("beta2_normuon", None, float, True))
        data.append(("low_rank_ortho", False, bool, False))
        data.append(("ortho_rank", None, int, True))
        data.append(("accelerated_ns", False, bool, False))
        data.append(("cautious_wd", False, bool, False))
        data.append(("approx_mars", False, bool, False))
        data.append(("auto_kappa_p", False, bool, False))
        data.append(("compile", False, bool, False))

        return TrainOptimizerConfig(data)


class TrainModelPartConfig(BaseConfig):
    model_name: str
    include: bool
    train: bool
    stop_training_after: int
    stop_training_after_unit: TimeUnit
    learning_rate: float
    weight_dtype: DataType
    dropout_probability: float #this is text encoder caption dropout!
    train_embedding: bool
    attention_mask: bool
    guidance_scale: float

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("model_name", "", str, False))
        data.append(("include", True, bool, False))
        data.append(("train", True, bool, False))
        data.append(("stop_training_after", None, int, True))
        data.append(("stop_training_after_unit", TimeUnit.NEVER, TimeUnit, False))
        data.append(("learning_rate", None, float, True))
        data.append(("weight_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("dropout_probability", 0.0, float, False))
        data.append(("train_embedding", True, bool, False))
        data.append(("attention_mask", False, bool, False))
        data.append(("guidance_scale", 1.0, float, False))

        return TrainModelPartConfig(data)


class TrainEmbeddingConfig(BaseConfig):
    uuid: str
    model_name: str
    placeholder: str
    train: bool
    stop_training_after: int
    stop_training_after_unit: TimeUnit
    token_count: int | None
    initial_embedding_text: str
    is_output_embedding: bool
    group: str
    concord_invert_init: bool

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("uuid", str(uuid.uuid4()), str, False))
        data.append(("model_name", "", str, False))
        data.append(("placeholder", "<embedding>", str, False))
        data.append(("train", True, bool, False))
        data.append(("stop_training_after", None, int, True))
        data.append(("stop_training_after_unit", TimeUnit.NEVER, TimeUnit, False))
        data.append(("token_count", 1, int, True))
        data.append(("initial_embedding_text", "*", str, False))
        data.append(("is_output_embedding", False, bool, False))
        # Concord group-subspace label: a free-text group name (e.g. "character",
        # "style") that ties this embedding's trainable rows into a subspace group.
        # Empty = ungrouped (no separation/flatten). Consumed only when the Concord
        # group-subspace toggles are on; inert otherwise.
        data.append(("group", "", str, False))
        # Concord output-space (CLIP-inversion) init: when the initial embedding text is
        # LONGER than token_count, optimize the token vectors so their contextualized output
        # matches the full phrase's (instead of truncating the tail). Default off = truncate.
        data.append(("concord_invert_init", False, bool, False))

        return TrainEmbeddingConfig(data)

class QuantizationConfig(BaseConfig):
    layer_filter: str
    layer_filter_preset: str
    layer_filter_regex: bool
    svd_dtype: DataType
    svd_rank: int
    cache_dir: str

    @staticmethod
    def default_values():
        data = []

        # name, default value, data type, nullable
        data.append(("layer_filter", "", str, False))
        data.append(("layer_filter_preset", "full", str, False))
        data.append(("layer_filter_regex", False, bool, False))
        data.append(("svd_dtype", DataType.NONE, DataType, False))
        data.append(("svd_rank", 16, int, False))
        data.append(("cache_dir", None, str, True))
        return QuantizationConfig(data)

class TrainConfig(BaseConfig):
    training_method: TrainingMethod
    model_type: ModelType
    debug_mode: bool
    debug_dir: str
    workspace_dir: str
    cache_dir: str
    tensorboard: bool
    tensorboard_expose: bool
    tensorboard_always_on: bool
    tensorboard_port: str
    validation: bool
    validate_after: float
    validate_after_unit: TimeUnit
    continue_last_backup: bool
    prevent_overwrites: bool
    include_train_config: ConfigPart

    # multi-GPU
    multi_gpu: bool
    device_indexes: str
    gradient_reduce_prevision: GradientReducePrecision
    fused_gradient_reduce: bool
    async_gradient_reduce: bool
    async_gradient_reduce_buffer: int

    # model settings
    base_model_name: str
    output_dtype: DataType
    output_model_format: ModelFormat
    output_model_destination: str
    gradient_checkpointing: GradientCheckpointingMethod
    enable_async_offloading: bool
    enable_activation_offloading: bool
    layer_offload_fraction: float
    force_circular_padding: bool
    compile: bool

    # data settings
    concept_file_name: str
    concord_sanitize_tokens: str          # comma-separated single-token words to zero (sanitize)
    concord_cuda_graph: bool              # EXPERIMENTAL opt-in: graph the UNet step (default off)
    concord_graph_te: bool                # default-on: capture the TEXT ENCODERS inside the UNet graph (encode_text->UNet in one capture) so they train in the captured backward. Set False to route the TEs through the EAGER bridge instead -- only the UNet is graphed (the proven pattern), the TEs stay eager (on the fused kernel) so the sampler can offload them. Use False if in-graph TE capture breaks sampling/resume (the captured graph pins the encoders).
    concord_fused_matmul: bool           # default-on: dequant packed_w inside the matmul, drops the bf16 weight cache (~5 GB); works WITH gradient accumulation (accum is driven by the apply kernel's consolidate gate, orthogonal to fused vs cached)
    concord_packed_embeddings: bool       # default-on: train new-token embeddings via the norm-preserving packed self-stepping core (ConcordPackedEmbedding) instead of plain SGD -- pins the deploy norm to the vocab median (anti-overfit). Concord optimizer only.
    concord_bucket_contiguous: bool       # default-on: order aspect-ratio buckets as contiguous blocks (random block order per epoch) instead of globally shuffling batches across shapes -- avoids CUDA-graph recapture churn + allocator fragmentation when bucketing under the graph. latent_caching path only; no-op with a single bucket.
    concord_te_anchor: bool               # MODE selector for Concord CLIP-L training (under Concord + text_encoder.train). False (DEFAULT, since 2026-06-18) = train CLIP-L via the WINNER recipe, exactly like the UNet (even-split load, live coherence gate alpha_v>0, drift_cancel_C>0, gf_consol>0 evaporation, NO wd_anchor, NO pinned v_slow). True (OPT-IN) = frozen-v_slow anchor -- pretrained pinned in v_slow, a 16-bit fast/slow delta self-steps, wd_anchor pulls it back toward pretrained (low-drift, but no live gate/dissipation). Uses the text_encoder LR field. NOTE the flip: configs that OMIT this now get winner (was frozen); set True to keep the old anchor behavior. (Winner-TE is unvalidated vs the frozen anchor -- watch for CLIP drift/collapse.)
    concord_te2_anchor: bool              # SAME mode selector as concord_te_anchor but for CLIP-G (text_encoder_2), under text_encoder_2.train. False (default) = WINNER recipe (train like the UNet); True (opt-in) = frozen-v_slow anchor. Own lr (text_encoder_2 LR field). CLIP-G is 694M, so the packed int-storage saving is large in either mode.
    concord_te_wd_anchor: float           # strength of the elastic pull of the TE delta toward the pretrained anchor (kernel wd_anchor). ~0.5 = gentle (validated); 0 = no anchor (plain packed drift).
    concord_te_chase_alpha: float         # s_fast->s_slow chase (consolidation) rate for the anchored TEs (kernel alpha). 0.1 = the shared UNet/WINNER default. The TE deploys with s_fast KEPT, so this sets only the coarse-int8(s_slow) / fine-int16(s_fast) split, not the deployed weight: lower keeps small anchored excursions in the fine s_fast (more precise); 0 = no chase (all learning stays in s_fast). Raise only if deltas are large enough to need s_slow's extra range.
    concepts: list[ConceptConfig]
    aspect_ratio_bucketing: bool
    latent_caching: bool
    clear_cache_before_training: bool

    # training settings
    learning_rate_scheduler: LearningRateScheduler
    custom_learning_rate_scheduler: str | None
    # Dict keys are literally called "key" and "value"; not a tuple because
    # of restrictions with ConfigList.
    scheduler_params: list[dict[str, str]]
    learning_rate: float
    learning_rate_warmup_steps: float
    learning_rate_cycles: float
    learning_rate_min_factor: float
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    ema: EMAMode
    ema_decay: float
    ema_update_step_interval: int
    dataloader_threads: int
    train_device: str
    temp_device: str
    train_dtype: DataType
    fallback_train_dtype: DataType
    enable_autocast_cache: bool
    only_cache: bool
    resolution: str
    frames: str
    mse_strength: float
    mae_strength: float
    log_cosh_strength: float
    huber_strength: float
    huber_delta: float
    vb_loss_strength: float
    loss_weight_fn: LossWeight
    loss_weight_strength: float
    dropout_probability: float #this is LoRA dropout!
    loss_scaler: LossScaler
    learning_rate_scaler: LearningRateScaler
    clip_grad_norm: float

    #layer filter
    layer_filter: str  # comma-separated
    layer_filter_preset: str
    layer_filter_regex: bool

    # noise
    offset_noise_weight: float
    generalized_offset_noise: bool
    perturbation_noise_weight: float
    rescale_noise_scheduler_to_zero_terminal_snr: bool
    force_v_prediction: bool
    force_epsilon_prediction: bool
    timestep_distribution: TimestepDistribution
    min_noising_strength: float
    max_noising_strength: float
    constant_snr_floor: float

    noising_weight: float
    noising_bias: float

    timestep_shift: float
    dynamic_timestep_shifting: bool
    concord_epoch_cache_release: bool
    concord_sample_deploy: bool
    concord_m6a_meter: bool               # LOG-ONLY dissipation-space DIVERSITY meter (M6a = killed coherent mass in the hypothesis-infancy band [|s_fast|<evap_build_min] / consolidated s_slow energy). Rises when a climbing kappa evaporates coherent cross-example evidence before it consolidates (CPU-MNIST validated; memgap/waste are blind to this). DIAGNOSTIC only -- does NOT gate the servo. Default OFF (logs m6a on the [loss] line + TB when on). Concord optimizer only.
    concord_embedding_anchor: bool
    concord_embedding_preserve_norm: bool  # Default True: pin each packed-embedding token's DEPLOY norm to its OWN seeded base norm (norm PRESERVED). False = legacy pin to the vocab MEDIAN, which homogenizes -- cuts high-norm tokens, boosts low-norm -- a norm-identity rewrite that reads as huge spurious "movement". Concord packed embeddings only.
    concord_train_caption_vocab: bool   # opt-in (Concord only): train the base-vocab tokens that ACTUALLY appear in dataset captions via the packed per-token Concord path + the 'emb' dissipation servo, seeded from base.weight; default off = base vocab frozen. Distinct from added-token training (train_any_embedding()).
    concord_caption_vocab_anchor: bool  # anchor caption-vocab rows in v_slow (deploy=init+gated delta). Default False: anchor=True zeroes alpha_v_fast/drift_cancel_C so the emb servo's coherence climb never engages -- caption tokens seeded from a live base row want the leak ON.
    concord_caption_vocab_content_only: bool  # Default True: separate CONTENT from function-word "glue" -- drop whole-word STOPWORDS (the/of/and/... curated set) + pure punctuation/digits, but KEEP sub-word fragments (theophite->the+oph+ite compose a named concept) and content words. Axis is content-vs-stopword, NOT fragment-vs-word. The norm artifact is a SEPARATE fix (concord_embedding_preserve_norm). Concord caption-vocab only.
    concord_caption_vocab_min_count: int  # (b) drop caption tokens appearing fewer than N times across all training captions (incidental-token floor). Default 1 = keep all. Concord caption-vocab only.
    concord_emb_deflate: bool  # Common-mode CONCENTRATION gate: each epoch the attribution meter SVDs the embedding accumulator and names the top shared components; OWNER rows (affinity >= 0.6) keep their component, non-owner "passenger" rows have gamma of their projection removed -- the component MIGRATES to its owners instead of smearing across co-occurring tokens. Only STABLE (direction-matched vs prev epoch) + OWNED (affinity cliff, not flat) components are gated; state persists across exit-42 segments via the concord_commonmode.json sidecar (owners stored by token NAME -- row order is not segment-stable). Default False = bit-exact legacy. Requires the eager/bridge TE path (concord_graph_te off). Concord packed embeddings only.
    concord_emb_deflate_gamma: float  # Fraction of an owned common-mode component removed from NON-owner rows (owners always keep 100%). 0.5 = passengers lose half their shared projection; 1.0 = hard orthogonalization for passengers.
    concord_emb_deflate_modes: int  # How many owned common-mode components to shrink per epoch: a cap on the armed count. The meter always examines and logs the top 8; only the top-N ELIGIBLE (energetic + stable + owned, in energy order) are actually gated. 1 = shrink only the single strongest owned mode; 8 = shrink all that qualify. Range 1-8 (to disable, turn concord_emb_deflate off).
    concord_emb_group_separate: bool  # Group-subspace BETWEEN-group separation: project each embedding group's update off the span of the OTHER groups' current vectors (an N-group generalization of the style shield), so a character's update stays out of style-space and vice versa. Groups come from each additional embedding's `group` label; needs >=2 groups. Eager/bridge TE path only. Default False = off.
    concord_emb_group_separate_gamma: float  # Fraction of the cross-group projection removed (0.5 = half, 1.0 = hard orthogonalization). Validated safe to 1.0 at production embed dim (the fidelity cost seen at tiny dim is a capacity artifact), but default conservative and raise on the A/B.
    concord_emb_group_flatten: bool  # Group-subspace WITHIN-group flatten: Newton-Schulz across each group's member rows so members become mutually distinct and the group common-mode is demoted (Muon-ish). Groups come from the `group` label; needs a group with >=2 members. Eager/bridge TE path only. Default False = off.
    concord_emb_group_flatten_gamma: float  # Interpolation toward the flattened (orthogonalized) update per step (0.5 = halfway, 1.0 = full spectral reshape at preserved magnitude). Higher = more distinct members; validated as a clean win with no fidelity cost.
    concord_emb_noise_seed: bool  # Per-row NSR seeding for the trainable embedding token rows (exp52): each row's dissipation seeded from its own evidence-clocked noise-to-signal ratio, read from the cores' existing _accum/_power/_seen accumulators at epoch boundaries. Rare-but-consistent tokens get low lam (evidence survives to consolidate); confused tokens get the ceiling. Median row anchors at the configured emb dissipation. Refuses to run with the dissipation servos. Default False.
    concord_hardneg_meter: bool  # Gap-guarded hard-negative mining TELEMETRY (meter ONLY -- no sampler effect). Every Nth update, re-run the current batch's forward under three weight views (deploy / arm-L / arm-H, deterministic noise) and log per-example deploy loss, branch disagreement, and the would-be-mined set (high loss among branch-AGREEING examples). Requires two_fast + the held-out router for the views to mean anything; unsupported under fused matmul (self-disables). CPU receipts: exp48 [epic-williamson] / exps 33-34 [mechanics]. Default False.
    concord_hardneg_every: int  # Audit cadence in UPDATE steps for the hard-negative meter (3 extra no_grad forwards of one batch per audit). Default 128.
    concord_uncond_mean: bool  # Sampling: when the negative prompt is EMPTY, use the DATASET-MEAN conditioning (workspace/concord_uncond_mean.safetensors, from scripts/concord_uncond_mean.py) as the uncond instead of the zero/empty encode. CFG then guides away from the dataset-typical image, so the cond-uncond differential excludes the dataset-mean shift (fixes brown-mud/diversity collapse when the uncond branch is untrained). Explicit negative prompts always win. Default False.
    concord_uncond_pass: bool  # Training (graph path, BOTH modes -- TE bridge and TE-in-graph): partition the uncond objective off the caption-dropout path. On `concord_uncond_pass_rate` of micro-steps (deterministic hash of the resumed global step, so same-seed A/B and restart-resume keep identical fire patterns), run an EXTRA tick-only graph replay with the conditioning gated to ZERO inside the captured region (cond_flag device scalar); the backward scales the TE/embedding gradient by the same zero, so embeddings keep 100% of their gradient events (unlike caption dropout, which zeroes their gradient on dropped samples). NOTE the pass is ADDITIVE: ~(1+rate) total gradient mass vs cond-only; to match a caption-dropout rate r use rate = r/(1-r). Use INSTEAD of caption dropout (set that to 0). ⚠ still not GPU-validated. Default False = bit-identical.
    concord_uncond_pass_rate: float  # Fraction of micro-steps that fire the extra zeroed-conditioning uncond replay (see concord_uncond_pass). ~matches a caption-dropout rate; 0.15 default. Higher = more uncond training (and more compute: each fire is one extra reduced UNet fwd+bwd -- no TE, attn2 collapses under zero conditioning).
    concord_embedding_delay_epochs: float  # "divot": freeze packed embeddings for the first N epochs so the UNet digs its basin against the pristine anchors; tokens then release on a fresh warmup (cosine still ends at the horizon). Counters fast-variable slaving (measured: 1 epoch @ lr 1e-3 slewed tokens ~50 deg off init). Doubles as the auto-drive calibration window.
    concord_embedding_auto_drive: bool  # at divot release, normalize each token's rate by sighting count: drive_i = (median(n)/n_i)^freq_exponent (decade clamp), so per-epoch motion tracks D_i = ||coherent grad sum||/n_i -- the data's own per-appearance evidence. Converged-but-frequent tokens slow down because D is small (not boosted for a small total); rare-but-far tokens get a frequency boost. Drive scaling preserves per-token lambda (evap_frac is a fraction of the buffer). Requires delay_epochs > 0 for a calibration window.
    concord_embedding_freq_exponent: float  # beta in drive = (median(n)/n)^beta. Hierarchical tokens (style tokens containing object tokens) assign shared features BY frequency: the style rightfully wins shared content because it integrates it n_style/n_obj times faster; coherence keeps object content with the object. beta=1 flattens per-epoch rates -- attribution parity, shared features split by noise (style/object clobbering); beta=0 is raw dynamics (correct attribution, but hot tokens fry). beta=0.5 equalizes the NOISE motion across tokens (D_noise ~ 1/sqrt(n), so sqrt(n)*D_noise is constant) while justified motion keeps a sqrt(frequency) advantage -- styles still win shared features, fry suppressed.
    concord_embedding_window_report: bool  # diagnostic (default off): accumulate the incoherent power Sigma||g||^2 over the divot alongside the coherent sum, and print a per-token Wiener posterior around the init at calibration -- rho (signal fraction / how pinned-down the token's true value is) and w (relative window half-width, shrinking as 1/sqrt(n) and as the UNet sharpens). Surfaces which tokens the data has localized vs. which are still wandering. Passive (never touches the update); requires delay_epochs > 0 for a measurement window.
    concord_token_only_dropout: float  # probability (0..1) that an example's caption is replaced with ONLY its trainable tokens (all context words dropped), forcing the token to carry the concept itself rather than leaning on the caption (counters the decorative/address-shared token). Comes into effect only AFTER the embedding divot releases (tokens are training); applies only to examples that contain a trainable token; never applied to validation/sampling. 0 = off.
    concord_words_only_dropout: float  # probability (0..1) that an example's caption has its trainable tokens STRIPPED, context words kept -- the complement of token-only. Both at 1/3 = the thirds scheme (full / tokens-only / words-only): separates the embedding-attributable gradient stream from the caption stream so the optimizer's gate certifies two coherent components instead of taxing a standing mixture forever (2026-07-12 telemetry audit: velocity-telescope anti-alignment at flat loss; exp 68's irreducible-conflict regime). Same gating and eligibility as token-only; one shared per-example draw keeps both TEs on the same composition. 0 = off.
    concord_dropout_injected_only: bool  # scope both caption-dropout modes to the INJECTED (added-placeholder) tokens. With caption-vocab training on, the trainable id set (kind == 2) includes surrounding caption words -- token-only would then keep half the caption and words-only would strip ordinary words. This subtracts the plane's caption_tids so tokens-only keeps just the placeholders and words-only strips just them. No-op when caption vocab is off; OFF = legacy behavior (all trainable ids). Recommended ON whenever concord_train_caption_vocab is on.
    concord_couple_te_dropout: bool  # COUPLE the two text-encoder CFG dropout masks: ONE shared per-example mask drives BOTH encoders at a SINGLE rate (max of the two configured), so a "dropped" example is genuinely unconditional and there are ZERO partial-conditioning states (hard lock). Fixes the independent-draw footgun: with both text_encoder(.2).dropout_probability set (e.g. 0.3/0.3), independent masks make true-unconditional collapse to p1*p2 (0.09) while 2*p*(1-p)=0.42 of examples get ONE encoder zeroed -- inference-nonexistent half-conditioned states that shred CFG training. Locked mode => a clean max(p1,p2) fraction fully-unconditional, the rest fully-conditional. 0/off = legacy independent draws (unchanged).
    concord_contrast_arms: bool  # CONTRASTIVE ARMS (exp 71/72), GRAPH-NATIVE. Pairs the SAME image full-vs-dropped across the two held-out arms as TWO replays of the one captured CUDA graph inside concord_graph._contrast_step: FULL caption -> arm L (tick), token-DROPPED -> arm H (consolidate), same timestep/noise (shared seed). The 2fast gap floor reads e_L-e_H = the token's CONTEXT and evaporates it, banking the token's context-invariant core; the token's gradient is in both arms and consolidates. REQUIRES concord_cuda_graph ON (the eager path is not implemented -> inert without the graph), gradient_accumulation_steps == 1 (the single _contrast_step IS the update; the setup guard permits heldout_router+accum==1 under contrast), heldout_router ON, and the TE bridge (a nonzero caption dropout, which forces should_graph_te False). Cost ~2x TE-encode + 2x UNet-replay per update (NOT epoch-doubled). NOT YET GPU-VALIDATED. TRADEOFF: same-image arms lose the router's cross-split anti-memorization corroboration. 0/off = unchanged. at the SAME timestep/noise -- full caption on the arm-L micro (even), token-dropped on the arm-H micro (odd). The 2fast gap floor (e_L-e_H added to the coherence noise) then sees the token's CONTEXT contribution as cross-arm disagreement and evaporates it, banking the token's context-invariant core; the token's own gradient is in both arms (agree) and consolidates. Requires gradient_accumulation_steps == 2 AND heldout_router on. The odd micro reuses the even micro's batch+seed (its own fetched batch is unused -> ~half the loader data per pair). Overrides the stochastic thirds dropout while on (dropout becomes deterministic per arm). TRADEOFF: same-image arms lose the router's cross-split anti-memorization corroboration (exp 72 caveat). 0/off = unchanged.
    concord_embedding_quality_orthogonal: bool  # master switch for the quality-tag shield. Mark some trainable embeddings as low-quality "tags" (concord_embedding_quality_tags); they train FREELY and absorb the defect across the bad images they caption (becoming a droppable / negative-promptable knob), while every OTHER trainable embedding's gradient is projected off the tags' learned subspace so it learns subject content from a bad image but not its badness. Overlap is fine -- a subject can sit anywhere, it just can't be PUSHED along the tag directions. The tag basis is mean-centered (CLIP embeddings are anisotropic; raw projection would strip the shared component every embedding needs) and rebuilt each step from the tags' current deploy vectors (they're learning). Subjects are also projected off the final basis at save (clean artifact + residual report). Requires concord_embedding_quality_tags to match at least one trainable AND leave at least one non-tag subject.
    concord_embedding_quality_tags: str  # comma/newline-separated PLACEHOLDERS of the trainable embeddings to treat as low-quality tags (the sinks). Everything trainable that is NOT listed is a shielded subject. Empty -> the shield is off regardless of the toggle.
    concord_embedding_quality_mode: str
    concord_embedding_style_tags: str  # comma/newline-separated placeholders or caption words to treat as STYLE tags: they train freely as the style sink, and every other trainable row (quality tags included) is HARD-projected (two-sided) off their learned span each step -- enforcing style/subject factorization symmetrically. Independent of the quality shield; empty = off. Requires concord_embedding_quality_orthogonal master switch ON (shared plumbing).  # "hard" = subject steps fully orthogonal to the tag subspace (no motion along it). "one_sided" = block only the component moving TOWARD a tag direction, leaving a subject free to move away (toward good). Default hard.
    concord_antithetic_timesteps: bool
    concord_antithetic_noise: bool
    concord_antithetic_same_example: bool
    resolution_aware_loss_weight: bool    # opt-in: fold a spectrum-principled per-image cap into the min-SNR loss weight. A natural image has a ~1/f^2 power spectrum, so an image upscaled by linear factor f (= sqrt(crop/orig area)) carries no real signal beyond its native Nyquist and its representable SNR is ~ gamma/f^2; capping min-SNR gamma there (= gamma * original_area/crop_area) stops the model being rewarded for reconstructing fake interpolated high-freq at low-noise timesteps (upscaling artifacts). Requires loss_weight_fn=MIN_SNR_GAMMA; no-op for native/downscaled images. General SDXL feature, not Concord-specific.

    # unet
    unet: TrainModelPartConfig

    # prior
    prior: TrainModelPartConfig

    # transformer
    transformer: TrainModelPartConfig
    quantization: QuantizationConfig

    # text encoder
    text_encoder: TrainModelPartConfig
    text_encoder_layer_skip: int

    # text encoder 2
    text_encoder_2: TrainModelPartConfig
    text_encoder_2_layer_skip: int
    text_encoder_2_sequence_length: int

    # text encoder 3
    text_encoder_3: TrainModelPartConfig
    text_encoder_3_layer_skip: int

    # text encoder 4
    text_encoder_4: TrainModelPartConfig
    text_encoder_4_layer_skip: int

    # vae
    vae: TrainModelPartConfig

    # effnet encoder
    effnet_encoder: TrainModelPartConfig

    # decoder
    decoder: TrainModelPartConfig

    # decoder text encoder
    decoder_text_encoder: TrainModelPartConfig

    # decoder vqgan
    decoder_vqgan: TrainModelPartConfig

    # masked training
    masked_training: bool
    unmasked_probability: float
    unmasked_weight: float
    normalize_masked_area_loss: bool
    masked_prior_preservation_weight: float

    # custom conditioning image
    custom_conditioning_image: bool

    # embedding
    embedding_learning_rate: float
    preserve_embedding_norm: bool
    embedding: TrainEmbeddingConfig
    additional_embeddings: list[TrainEmbeddingConfig]
    embedding_weight_dtype: DataType

    # lora
    peft_type: PeftType
    lora_model_name: str
    lora_rank: int
    lora_alpha: float
    lora_decompose: bool
    lora_decompose_norm_epsilon: bool
    lora_decompose_output_axis: bool
    lora_weight_dtype: DataType
    bundle_additional_embeddings: bool

    # oft
    oft_block_size: int
    oft_block_share: bool
    oft_scaled: bool

    # lokr
    lokr_dim: int
    lokr_decompose_both: bool
    lokr_decompose_factor: int
    lokr_use_tucker: bool
    lokr_weight_decompose: bool
    lokr_dora_on_output: bool
    lokr_full_matrix: bool
    lokr_vec_trick: bool

    # optimizer
    optimizer: TrainOptimizerConfig
    optimizer_defaults: dict[str, TrainOptimizerConfig]

    # sample settings
    sample_definition_file_name: str
    samples: list[SampleConfig]
    sample_after: float
    sample_after_unit: TimeUnit
    sample_skip_first: int
    sample_image_format: ImageFormat
    sample_video_format: VideoFormat
    sample_audio_format: AudioFormat
    samples_to_tensorboard: bool
    non_ema_sampling: bool

    # cloud settings
    cloud: CloudConfig

    # backup settings
    backup_after: float
    backup_after_unit: TimeUnit
    rolling_backup: bool
    rolling_backup_count: int
    backup_before_save: bool
    save_every: int
    save_every_unit: TimeUnit
    save_skip_first: int
    save_filename_prefix: str

    # secrets - not saved into config file
    secrets: SecretsConfig

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(
            data,
            config_version=10,
            config_migrations={
                0: self.__migration_0,
                1: self.__migration_1,
                2: self.__migration_2,
                3: self.__migration_3,
                4: self.__migration_4,
                5: self.__migration_5,
                6: self.__migration_6,
                7: self.__migration_7,
                8: self.__migration_8,
                9: self.__migration_9,
            }
        )

    def __migration_0(self, data: dict) -> dict:
        optimizer_settings = {}
        migrated_data = {}
        for key, value in data.items():
            # move optimizer settings to sub object
            if key == 'optimizer':
                optimizer_settings['optimizer'] = value
            elif key.startswith('optimizer'):
                optimizer_settings[key.removeprefix('optimizer_')] = value
            else:
                migrated_data[key] = value

        if 'optimizer' in optimizer_settings:
            migrated_data['optimizer'] = optimizer_settings
            migrated_data['optimizer_defaults'] = {
                optimizer_settings['optimizer']: deepcopy(optimizer_settings)
            }

        return migrated_data

    def __migration_1(self, data: dict) -> dict:
        migrated_data = {
            "unet": {},
            "prior": {},
            "text_encoder": {},
            "text_encoder_2": {},
            "vae": {},
            "effnet_encoder": {},
            "decoder": {},
            "decoder_text_encoder": {},
            "decoder_vqgan": {},
            "embeddings": [{}],
        }

        for key, value in data.items():
            if key == "train_unet":
                migrated_data["unet"]["train"] = value
            elif key == "train_unet_epochs":
                migrated_data["unet"]["stop_training_after"] = value
                migrated_data["unet"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "unet_learning_rate":
                migrated_data["unet"]["learning_rate"] = value
            elif key == "unet_weight_dtype":
                migrated_data["unet"]["weight_dtype"] = value

            elif key == "train_prior":
                migrated_data["prior"]["train"] = value
            elif key == "prior_model_name":
                migrated_data["prior"]["model_name"] = value
            elif key == "train_prior_epochs":
                migrated_data["prior"]["stop_training_after"] = value
                migrated_data["prior"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "prior_learning_rate":
                migrated_data["prior"]["learning_rate"] = value
            elif key == "prior_weight_dtype":
                migrated_data["prior"]["weight_dtype"] = value

            elif key == "train_text_encoder":
                migrated_data["text_encoder"]["train"] = value
            elif key == "train_text_encoder_epochs":
                migrated_data["text_encoder"]["stop_training_after"] = value
                migrated_data["text_encoder"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "text_encoder_learning_rate":
                migrated_data["text_encoder"]["learning_rate"] = value
            elif key == "text_encoder_weight_dtype":
                migrated_data["text_encoder"]["weight_dtype"] = value

            elif key == "train_text_encoder_2":
                migrated_data["text_encoder_2"]["train"] = value
            elif key == "train_text_encoder_2_epochs":
                migrated_data["text_encoder_2"]["stop_training_after"] = value
                migrated_data["text_encoder_2"]["stop_training_after_unit"] = TimeUnit.EPOCH
            elif key == "text_encoder_2_learning_rate":
                migrated_data["text_encoder_2"]["learning_rate"] = value
            elif key == "text_encoder_2_weight_dtype":
                migrated_data["text_encoder_2"]["weight_dtype"] = value

            elif key == "vae_model_name":
                migrated_data["vae"]["model_name"] = value
            elif key == "vae_weight_dtype":
                migrated_data["vae"]["weight_dtype"] = value

            elif key == "effnet_encoder_model_name":
                migrated_data["effnet_encoder"]["model_name"] = value
            elif key == "effnet_encoder_weight_dtype":
                migrated_data["effnet_encoder"]["weight_dtype"] = value

            elif key == "decoder_model_name":
                migrated_data["decoder"]["model_name"] = value
            elif key == "decoder_weight_dtype":
                migrated_data["decoder"]["weight_dtype"] = value

            elif key == "decoder_text_encoder_weight_dtype":
                migrated_data["decoder_text_encoder"]["weight_dtype"] = value

            elif key == "decoder_vqgan_weight_dtype":
                migrated_data["decoder_vqgan"]["weight_dtype"] = value

            elif key == "embedding_model_names" and len(value) > 0:
                migrated_data["embeddings"][0]["model_name"] = value[0]
            elif key == "token_count":
                migrated_data["embeddings"][0]["token_count"] = value
            elif key == "initial_embedding_text":
                migrated_data["embeddings"][0]["initial_embedding_text"] = value

            else:
                migrated_data[key] = value

        return migrated_data

    def __migration_2(self, data: dict) -> dict:
        migrated_data = data.copy()
        min_snr_gamma = migrated_data.pop("min_snr_gamma", 0.0)
        model_type = ModelType(migrated_data.get("model_type", ModelType.STABLE_DIFFUSION_15))
        if min_snr_gamma:
            migrated_data["loss_weight_fn"] = LossWeight.MIN_SNR_GAMMA
            migrated_data["loss_weight_strength"] = min_snr_gamma
        elif model_type.is_wuerstchen():
            migrated_data["loss_weight_fn"] = LossWeight.P2
            migrated_data["loss_weight_strength"] = 1.0

        return migrated_data

    def __migration_3(self, data: dict) -> dict:
        migrated_data = data.copy()

        noising_weight = migrated_data.pop("noising_weight", 0.0)
        noising_bias = migrated_data.pop("noising_bias", 0.5)

        if noising_weight != 0:
            migrated_data["timestep_distribution"] = TimestepDistribution.SIGMOID
            migrated_data["noising_weight"] = noising_weight
            migrated_data["noising_bias"] = noising_bias - 0.5
        else:
            migrated_data["timestep_distribution"] = TimestepDistribution.UNIFORM
            migrated_data["noising_weight"] = 0.0
            migrated_data["noising_bias"] = 0.0

        return migrated_data

    def __migration_4(self, data: dict) -> dict:
        migrated_data = data.copy()

        gradient_checkpointing = migrated_data.pop("gradient_checkpointing", True)

        if gradient_checkpointing:
            migrated_data["gradient_checkpointing"] = GradientCheckpointingMethod.ON
        else:
            migrated_data["gradient_checkpointing"] = GradientCheckpointingMethod.OFF

        return migrated_data

    def __migration_5(self, data: dict) -> dict:
        migrated_data = data.copy()

        if "save_after" in migrated_data:
            migrated_data["save_every"] = migrated_data.pop("save_after")
        if "save_after_unit" in migrated_data:
            migrated_data["save_every_unit"] = migrated_data.pop("save_after_unit")

        return migrated_data

    def __migration_6(self, data: dict) -> dict:
        migrated_data = data.copy()

        # None is not a valid value, but there was a bug that allowed it, so old config files can have it set to None:
        if (
            "lora_layer_preset" in migrated_data
            and migrated_data["lora_layer_preset"] is None
        ):
            migrated_data["lora_layer_preset"] = "full"

        return migrated_data

    def __migration_7(self, data: dict) -> dict:
        migrated_data = data.copy()

        if "lora_layers" in migrated_data:
            migrated_data["layer_filter"] = migrated_data.pop("lora_layers")
        if "lora_layer_preset" in migrated_data:
            migrated_data["layer_filter_preset"] = migrated_data.pop("lora_layer_preset")
        if "lora_layers_regex" in migrated_data:
            migrated_data["layer_filter_regex"] = migrated_data.pop("lora_layers_regex")

        return migrated_data

    def __migration_8(self, data: dict) -> dict:
        migrated_data = data.copy()

        if migrated_data["model_type"] != "STABLE_CASCADE_1" and migrated_data["model_type"] != "WUERSTCHEN_2":
            migrated_data["transformer"] = migrated_data["prior"]

        return migrated_data

    def __migration_9(self, data: dict) -> dict:
        migrated_data = data.copy()

        def replace_dtype(part: str):
            if part in migrated_data and migrated_data[part]["weight_dtype"] == "NONE":
                migrated_data[part]["weight_dtype"] = migrated_data["weight_dtype"]
        replace_dtype("unet")
        replace_dtype("prior")
        replace_dtype("transformer")
        replace_dtype("text_encoder")
        replace_dtype("text_encoder_2")
        replace_dtype("text_encoder_3")
        replace_dtype("text_encoder_4")
        replace_dtype("vae")
        replace_dtype("effnet_encoder")
        replace_dtype("decoder")
        replace_dtype("decoder_text_encoder")
        replace_dtype("decoder_vqgan")
        migrated_data.pop("weight_dtype")

        return migrated_data

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes(
            self.train_dtype,
            self.fallback_train_dtype,
            self.unet.weight_dtype,
            self.prior.weight_dtype,
            self.transformer.weight_dtype,
            self.text_encoder.weight_dtype,
            self.text_encoder_2.weight_dtype,
            self.text_encoder_3.weight_dtype,
            self.text_encoder_4.weight_dtype,
            self.vae.weight_dtype,
            self.effnet_encoder.weight_dtype,
            self.decoder.weight_dtype,
            self.decoder_text_encoder.weight_dtype,
            self.decoder_vqgan.weight_dtype,
            self.lora_weight_dtype,
            self.embedding_weight_dtype,
        )

    def model_names(self) -> ModelNames:
        return ModelNames(
            base_model=self.base_model_name,
            prior_model=self.prior.model_name,
            transformer_model=self.transformer.model_name,
            effnet_encoder_model=self.effnet_encoder.model_name,
            decoder_model=self.decoder.model_name,
            text_encoder_4=self.text_encoder_4.model_name,
            vae_model=self.vae.model_name,
            lora=self.lora_model_name,
            embedding=EmbeddingName(self.embedding.uuid, self.embedding.model_name) \
                if self.training_method == TrainingMethod.EMBEDDING else None,
            additional_embeddings=[EmbeddingName(embedding.uuid, embedding.model_name) for embedding in
                                   self.additional_embeddings],
            include_text_encoder=self.text_encoder.include,
            include_text_encoder_2=self.text_encoder_2.include,
            include_text_encoder_3=self.text_encoder_3.include,
            include_text_encoder_4=self.text_encoder_4.include,
        )

    def train_any_embedding(self) -> bool:
        return ((self.training_method == TrainingMethod.EMBEDDING) and not self.embedding.is_output_embedding) \
            or any((embedding.train and not embedding.is_output_embedding) for embedding in self.additional_embeddings)

    def train_caption_vocab(self) -> bool:
        # Concord caption-vocab trains base-vocab token EMBEDDINGS only; the encoder WEIGHTS stay
        # frozen (text_encoder.train gates the WINNER swap + requires_grad separately). It must
        # count for the device + caching gates -- the TE forward has to RUN LIVE for the embedding
        # rows to get a gradient -- but train_any_embedding() only covers added-token / EMBEDDING-
        # method training, so caption-vocab needs its own predicate. "Train just the embeddings."
        return (is_concord_family(self.optimizer.optimizer)
                and bool(getattr(self, "concord_train_caption_vocab", False)))

    def train_any_output_embedding(self) -> bool:
        return ((self.training_method == TrainingMethod.EMBEDDING) and self.embedding.is_output_embedding) \
            or any((embedding.train and embedding.is_output_embedding) for embedding in self.additional_embeddings)

    def train_text_encoder_or_embedding(self) -> bool:
        return (self.text_encoder.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding()) \
            or self.train_caption_vocab()

    def train_text_encoder_2_or_embedding(self) -> bool:
        return (self.text_encoder_2.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder_2.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding()) \
            or self.train_caption_vocab()

    def train_text_encoder_3_or_embedding(self) -> bool:
        return (self.text_encoder_3.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder_3.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding())

    def train_text_encoder_4_or_embedding(self) -> bool:
        return (self.text_encoder_4.train and self.training_method != TrainingMethod.EMBEDDING
                and not self.embedding.is_output_embedding) \
            or ((self.text_encoder_4.train_embedding or not self.model_type.has_multiple_text_encoders())
                and self.train_any_embedding())

    def all_embedding_configs(self):
        if self.training_method == TrainingMethod.EMBEDDING:
            return self.additional_embeddings + [self.embedding]
        else:
            return self.additional_embeddings

    def get_last_backup_path(self) -> str | None:
        backups_path = os.path.join(self.workspace_dir, "backup")
        if os.path.exists(backups_path):
            backup_paths = sorted(
                [path for path in os.listdir(backups_path) if
                 os.path.isdir(os.path.join(backups_path, path))],
                reverse=True,
            )

            if backup_paths:
                last_backup_path = backup_paths[0]
                return os.path.join(backups_path, last_backup_path)

        return None

    def to_settings_dict(self, secrets: bool) -> dict:
        config = TrainConfig.default_values().from_dict(self.to_dict())

        config.concepts = None
        config.samples = None

        config_dict = config.to_dict()
        if not secrets:
            config_dict.pop('secrets',None)
        return config_dict

    def to_pack_dict(self, secrets: bool) -> dict:
        config = TrainConfig.default_values().from_dict(self.to_dict())

        if config.concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = json.load(f)
                for i in range(len(concepts)):
                    concepts[i] = ConceptConfig.default_values().from_dict(concepts[i])
                config.concepts = concepts

        if config.samples is None:
            with open(config.sample_definition_file_name, 'r') as f:
                samples = json.load(f)
                for i in range(len(samples)):
                    samples[i] = SampleConfig.default_values(config.model_type).from_dict(samples[i])
                config.samples = samples

        config_dict = config.to_dict()
        if not secrets:
            config_dict.pop('secrets',None)
        return config_dict

    def to_unpacked_config(self) -> 'TrainConfig':
        config = TrainConfig.default_values().from_dict(self.to_dict())
        config.concepts = None
        config.samples = None
        return config

    @staticmethod
    def default_values() -> 'TrainConfig':
        data = []

        # name, default value, data type, nullable

        # general settings
        data.append(("training_method", TrainingMethod.FINE_TUNE, TrainingMethod, False))
        data.append(("model_type", ModelType.STABLE_DIFFUSION_15, ModelType, False))
        data.append(("debug_mode", False, bool, False))
        data.append(("debug_dir", "debug", str, False))
        data.append(("workspace_dir", "workspace/run", str, False))
        data.append(("cache_dir", "workspace-cache/run", str, False))
        data.append(("tensorboard", True, bool, False))
        data.append(("tensorboard_expose", False, bool, False))
        data.append(("tensorboard_always_on", False, bool, False))
        data.append(("tensorboard_port", 6006, int, False))
        data.append(("validation", False, bool, False))
        data.append(("validate_after", 1, int, False))
        data.append(("validate_after_unit", TimeUnit.EPOCH, TimeUnit, False))
        data.append(("continue_last_backup", False, bool, False))
        data.append(("prevent_overwrites", False, bool, False))
        data.append(("include_train_config", ConfigPart.NONE, ConfigPart, False))

        #multi-GPU
        data.append(("multi_gpu", False, bool, False))
        data.append(("device_indexes", "", str, False))
        data.append(("gradient_reduce_precision", GradientReducePrecision.FLOAT_32_STOCHASTIC, GradientReducePrecision, False))
        data.append(("fused_gradient_reduce", True, bool, False))
        data.append(("async_gradient_reduce", True, bool, False))
        data.append(("async_gradient_reduce_buffer", 100, int, False))

        # model settings
        data.append(("base_model_name", "stable-diffusion-v1-5/stable-diffusion-v1-5", str, False))
        data.append(("output_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("output_model_format", ModelFormat.SAFETENSORS, ModelFormat, False))
        data.append(("output_model_destination", "models/model.safetensors", str, False))
        data.append(("gradient_checkpointing", GradientCheckpointingMethod.ON, GradientCheckpointingMethod, False))
        data.append(("enable_async_offloading", True, bool, False))
        data.append(("enable_activation_offloading", True, bool, False))
        data.append(("layer_offload_fraction", 0.0, float, False))
        data.append(("force_circular_padding", False, bool, False))
        data.append(("compile", False, bool, False))

        # data settings
        data.append(("concept_file_name", "training_concepts/concepts.json", str, False))
        data.append(("concord_sanitize_tokens", "", str, False))
        data.append(("concord_cuda_graph", False, bool, False))
        data.append(("concord_graph_te", True, bool, False))
        data.append(("concord_fused_matmul", True, bool, False))
        data.append(("concord_packed_embeddings", True, bool, False))
        data.append(("concord_bucket_contiguous", True, bool, False))
        data.append(("concord_te_anchor", False, bool, False))
        data.append(("concord_te2_anchor", False, bool, False))
        data.append(("concord_te_wd_anchor", 0.5, float, False))
        data.append(("concord_te_chase_alpha", 0.1, float, False))
        data.append(("concepts", None, list[ConceptConfig], True))
        data.append(("aspect_ratio_bucketing", True, bool, False))
        data.append(("latent_caching", True, bool, False))
        data.append(("clear_cache_before_training", True, bool, False))

        # training settings
        data.append(("learning_rate_scheduler", LearningRateScheduler.CONSTANT, LearningRateScheduler, False))
        data.append(("custom_learning_rate_scheduler", None, str, True))
        data.append(("scheduler_params", [], list[dict[str, str]], True))
        data.append(("learning_rate", 3e-6, float, False))
        data.append(("learning_rate_warmup_steps", 200.0, float, False))
        data.append(("learning_rate_cycles", 1.0, float, False))
        data.append(("learning_rate_min_factor", 0.0, float, False))
        data.append(("epochs", 100, int, False))
        data.append(("batch_size", 1, int, False))
        data.append(("gradient_accumulation_steps", 1, int, False))
        data.append(("ema", EMAMode.OFF, EMAMode, False))
        data.append(("ema_decay", 0.999, float, False))
        data.append(("ema_update_step_interval", 5, int, False))
        data.append(("dataloader_threads", 2, int, False))
        data.append(("train_device", default_device.type, str, False))
        data.append(("temp_device", "cpu", str, False))
        data.append(("train_dtype", DataType.FLOAT_16, DataType, False))
        data.append(("fallback_train_dtype", DataType.BFLOAT_16, DataType, False))
        data.append(("enable_autocast_cache", True, bool, False))
        data.append(("only_cache", False, bool, False))
        data.append(("resolution", "512", str, False))
        data.append(("frames", "25", str, False))
        data.append(("mse_strength", 1.0, float, False))
        data.append(("mae_strength", 0.0, float, False))
        data.append(("log_cosh_strength", 0.0, float, False))
        data.append(("huber_strength", 0.0, float, False))
        data.append(("huber_delta", 1.0, float, False))
        data.append(("vb_loss_strength", 1.0, float, False))
        data.append(("loss_weight_fn", LossWeight.CONSTANT, LossWeight, False))
        data.append(("loss_weight_strength", 5.0, float, False))
        data.append(("dropout_probability", 0.0, float, False))
        data.append(("loss_scaler", LossScaler.NONE, LossScaler, False))
        data.append(("learning_rate_scaler", LearningRateScaler.NONE, LearningRateScaler, False))
        data.append(("clip_grad_norm", 1.0, float, True))

        # noise
        data.append(("offset_noise_weight", 0.0, float, False))
        data.append(("generalized_offset_noise", False, bool, False))
        data.append(("perturbation_noise_weight", 0.0, float, False))
        data.append(("rescale_noise_scheduler_to_zero_terminal_snr", False, bool, False))
        data.append(("force_v_prediction", False, bool, False))
        data.append(("force_epsilon_prediction", False, bool, False))
        data.append(("min_noising_strength", 0.0, float, False))
        data.append(("max_noising_strength", 1.0, float, False))
        data.append(("timestep_distribution", TimestepDistribution.UNIFORM, TimestepDistribution, False))
        data.append(("constant_snr_floor", 0.1, float, False))
        data.append(("noising_weight", 0.0, float, False))
        data.append(("noising_bias", 0.0, float, False))
        data.append(("timestep_shift", 1.0, float, False))
        data.append(("dynamic_timestep_shifting", False, bool, False))
        data.append(("resolution_aware_loss_weight", False, bool, False))
        data.append(("concord_epoch_cache_release", True, bool, False))
        data.append(("concord_sample_deploy", True, bool, False))
        data.append(("concord_m6a_meter", False, bool, False))
        data.append(("concord_embedding_anchor", True, bool, False))
        data.append(("concord_embedding_preserve_norm", True, bool, False))
        data.append(("concord_train_caption_vocab", False, bool, False))
        data.append(("concord_caption_vocab_anchor", False, bool, False))
        data.append(("concord_caption_vocab_content_only", True, bool, False))
        data.append(("concord_caption_vocab_min_count", 1, int, False))
        data.append(("concord_emb_deflate", False, bool, False))
        data.append(("concord_emb_deflate_gamma", 0.5, float, False))
        data.append(("concord_emb_deflate_modes", 8, int, False))
        data.append(("concord_emb_group_separate", False, bool, False))
        data.append(("concord_emb_group_separate_gamma", 0.5, float, False))
        data.append(("concord_emb_group_flatten", False, bool, False))
        data.append(("concord_emb_group_flatten_gamma", 0.5, float, False))
        data.append(("concord_emb_noise_seed", False, bool, False))
        data.append(("concord_hardneg_meter", False, bool, False))
        data.append(("concord_hardneg_every", 128, int, False))
        data.append(("concord_uncond_mean", False, bool, False))
        data.append(("concord_uncond_pass", False, bool, False))
        data.append(("concord_uncond_pass_rate", 0.15, float, False))
        data.append(("concord_embedding_delay_epochs", 1.0, float, False))
        data.append(("concord_embedding_auto_drive", True, bool, False))
        data.append(("concord_embedding_freq_exponent", 0.5, float, False))
        data.append(("concord_embedding_window_report", False, bool, False))
        data.append(("concord_token_only_dropout", 0.0, float, False))
        data.append(("concord_words_only_dropout", 0.0, float, False))
        data.append(("concord_dropout_injected_only", False, bool, False))
        data.append(("concord_contrast_arms", False, bool, False))
        data.append(("concord_contrast_fraction", 1.0, float, False))
        data.append(("concord_couple_te_dropout", False, bool, False))
        data.append(("concord_embedding_quality_orthogonal", False, bool, False))
        data.append(("concord_embedding_quality_tags", "", str, False))
        data.append(("concord_embedding_quality_mode", "hard", str, False))
        data.append(("concord_embedding_style_tags", "", str, False))
        data.append(("concord_antithetic_timesteps", False, bool, False))
        data.append(("concord_antithetic_noise", False, bool, False))
        data.append(("concord_antithetic_same_example", False, bool, False))


        # unet
        unet = TrainModelPartConfig.default_values()
        unet.train = True
        unet.stop_training_after = 0
        unet.learning_rate = None
        data.append(("unet", unet, TrainModelPartConfig, False))

        # prior
        prior = TrainModelPartConfig.default_values()
        prior.model_name = ""
        prior.train = True
        prior.stop_training_after = 0
        prior.learning_rate = None
        data.append(("prior", prior, TrainModelPartConfig, False))

        # transformer
        transformer = TrainModelPartConfig.default_values()
        transformer.model_name = ""
        transformer.train = True
        transformer.stop_training_after = 0
        transformer.learning_rate = None
        data.append(("transformer", transformer, TrainModelPartConfig, False))

        #quantization layer filter
        quantization = QuantizationConfig.default_values()
        data.append(("quantization", quantization, QuantizationConfig, False))

        # text encoder
        text_encoder = TrainModelPartConfig.default_values()
        text_encoder.train = True
        text_encoder.stop_training_after = 30
        text_encoder.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder.learning_rate = None
        data.append(("text_encoder", text_encoder, TrainModelPartConfig, False))
        data.append(("text_encoder_layer_skip", 0, int, False))
        data.append(("text_encoder_sequence_length", 512, int, True))

        # text encoder 2
        text_encoder_2 = TrainModelPartConfig.default_values()
        text_encoder_2.train = True
        text_encoder_2.stop_training_after = 30
        text_encoder_2.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder_2.learning_rate = None
        data.append(("text_encoder_2", text_encoder_2, TrainModelPartConfig, False))
        data.append(("text_encoder_2_layer_skip", 0, int, False))
        data.append(("text_encoder_2_sequence_length", 77, int, True))

        # text encoder 3
        text_encoder_3 = TrainModelPartConfig.default_values()
        text_encoder_3.train = True
        text_encoder_3.stop_training_after = 30
        text_encoder_3.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder_3.learning_rate = None
        data.append(("text_encoder_3", text_encoder_3, TrainModelPartConfig, False))
        data.append(("text_encoder_3_layer_skip", 0, int, False))

        # text encoder 4
        text_encoder_4 = TrainModelPartConfig.default_values()
        text_encoder_4.train = True
        text_encoder_4.stop_training_after = 30
        text_encoder_4.stop_training_after_unit = TimeUnit.EPOCH
        text_encoder_4.learning_rate = None
        data.append(("text_encoder_4", text_encoder_4, TrainModelPartConfig, False))
        data.append(("text_encoder_4_layer_skip", 0, int, False))

        # vae
        vae = TrainModelPartConfig.default_values()
        vae.model_name = ""
        data.append(("vae", vae, TrainModelPartConfig, False))

        # effnet encoder
        effnet_encoder = TrainModelPartConfig.default_values()
        effnet_encoder.model_name = ""
        data.append(("effnet_encoder", effnet_encoder, TrainModelPartConfig, False))

        # decoder
        decoder = TrainModelPartConfig.default_values()
        decoder.model_name = ""
        data.append(("decoder", decoder, TrainModelPartConfig, False))

        # decoder text encoder
        decoder_text_encoder = TrainModelPartConfig.default_values()
        data.append(("decoder_text_encoder", decoder_text_encoder, TrainModelPartConfig, False))

        # decoder vqgan
        decoder_vqgan = TrainModelPartConfig.default_values()
        data.append(("decoder_vqgan", decoder_vqgan, TrainModelPartConfig, False))

        # masked training
        data.append(("masked_training", False, bool, False))
        data.append(("unmasked_probability", 0.1, float, False))
        data.append(("unmasked_weight", 0.1, float, False))
        data.append(("normalize_masked_area_loss", False, bool, False))
        data.append(("masked_prior_preservation_weight", 0.0, float, False))
        data.append(("custom_conditioning_image", False, bool, False))

        #layer filter
        data.append(("layer_filter", "", str, False))
        data.append(("layer_filter_preset", "full", str, False))
        data.append(("layer_filter_regex", False, bool, False))

        # embedding
        data.append(("embedding_learning_rate", None, float, True))
        data.append(("preserve_embedding_norm", False, bool, False))
        data.append(("embedding", TrainEmbeddingConfig.default_values(), TrainEmbeddingConfig, False))
        data.append(("additional_embeddings", [], list[TrainEmbeddingConfig], False))
        data.append(("embedding_weight_dtype", DataType.FLOAT_32, DataType, False))

        # cloud
        data.append(("cloud", CloudConfig.default_values(), CloudConfig, False))

        # lora
        data.append(("peft_type", PeftType.LORA, PeftType, False))
        data.append(("lora_model_name", "", str, False))
        data.append(("lora_rank", 16, int, False))
        data.append(("lora_alpha", 1.0, float, False))
        data.append(("lora_decompose", False, bool, False))
        data.append(("lora_decompose_norm_epsilon", True, bool, False))
        data.append(("lora_decompose_output_axis", False, bool, False))
        data.append(("lora_weight_dtype", DataType.FLOAT_32, DataType, False))
        data.append(("bundle_additional_embeddings", True, bool, False))

        # oft
        data.append(("oft_block_size", 32, int, False))
        data.append(("oft_block_share", False, bool, False))
        data.append(("oft_scaled", False, bool, False))

        # lokr
        data.append(("lokr_dim", 16, int, False))
        data.append(("lokr_decompose_both", False, bool, False))
        data.append(("lokr_decompose_factor", -1, int, False))
        data.append(("lokr_use_tucker", False, bool, False))
        data.append(("lokr_weight_decompose", False, bool, False))
        data.append(("lokr_dora_on_output", True, bool, False))
        data.append(("lokr_full_matrix", False, bool, False))
        data.append(("lokr_vec_trick", True, bool, False))

        # optimizer
        data.append(("optimizer", TrainOptimizerConfig.default_values(), TrainOptimizerConfig, False))
        data.append(("optimizer_defaults", {}, dict[str, TrainOptimizerConfig], False))

        # sample settings
        data.append(("sample_definition_file_name", "training_samples/samples.json", str, False))
        data.append(("samples", None, list[SampleConfig], True))
        data.append(("sample_after", 10, int, False))
        data.append(("sample_after_unit", TimeUnit.MINUTE, TimeUnit, False))
        data.append(("sample_skip_first", 0, int, False))
        data.append(("sample_image_format", ImageFormat.JPG, ImageFormat, False))
        data.append(("sample_video_format", VideoFormat.MP4, VideoFormat, False))
        data.append(("sample_audio_format", AudioFormat.MP3, AudioFormat, False))
        data.append(("samples_to_tensorboard", True, bool, False))
        data.append(("non_ema_sampling", True, bool, False))

        # backup settings
        data.append(("backup_after", 30, int, False))
        data.append(("backup_after_unit", TimeUnit.MINUTE, TimeUnit, False))
        data.append(("rolling_backup", False, bool, False))
        data.append(("rolling_backup_count", 3, int, False))
        data.append(("backup_before_save", True, bool, False))
        data.append(("save_every", 0, int, False))
        data.append(("save_every_unit", TimeUnit.NEVER, TimeUnit, False))
        data.append(("save_skip_first", 0, int, False))
        data.append(("save_filename_prefix", "", str, False))

        # secrets
        secrets = SecretsConfig.default_values()
        data.append(("secrets", secrets, SecretsConfig, False))

        return TrainConfig(data)
