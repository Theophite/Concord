import modules.util.multi_gpu_util as multi
from modules.model.BaseModel import BaseModel
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig, TrainOptimizerConfig
from modules.util.enum.Optimizer import Optimizer, is_concord_family
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.muon_util import build_muon_adam_key_fn
from modules.util.torch_util import optimizer_to_device_

import torch


def change_optimizer(train_config: TrainConfig) -> TrainOptimizerConfig:
    optimizer = train_config.optimizer.optimizer

    optimizer_config = TrainOptimizerConfig.default_values()
    optimizer_config.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[optimizer])
    optimizer_config.optimizer = optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        saved_optimizer_config = train_config.optimizer_defaults[str(optimizer)]
        optimizer_config.from_dict(saved_optimizer_config.to_dict())

    return optimizer_config


def load_optimizer_defaults(train_config: TrainConfig) -> TrainOptimizerConfig:
    optimizer = train_config.optimizer.optimizer

    optimizer_config = TrainOptimizerConfig.default_values()
    optimizer_config.from_dict(OPTIMIZER_DEFAULT_PARAMETERS[optimizer])
    optimizer_config.optimizer = optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        train_config.optimizer_defaults.pop(str(optimizer))

    return optimizer_config


def update_optimizer_config(train_config: TrainConfig):
    optimizer = train_config.optimizer.optimizer

    if str(optimizer) in train_config.optimizer_defaults:
        saved_optimizer_config = train_config.optimizer_defaults[str(optimizer)]
        saved_optimizer_config.from_dict(train_config.optimizer.to_dict())
    else:
        optimizer_donfig = TrainOptimizerConfig.default_values()
        optimizer_donfig.from_dict(train_config.optimizer.to_dict())
        train_config.optimizer_defaults[str(optimizer)] = optimizer_donfig


def init_model_parameters(
        model: BaseModel,
        parameters: NamedParameterGroupCollection,
        train_device: torch.device,
):
    model.parameters = parameters

    # Concord can leave the parameter-group collection EMPTY: every trainable weight is
    # managed outside the base optimizer (UNet/TE are packed self-stepping BUFFERS, packed
    # embeddings self-step, aux norms/biases frozen). The base SGD, the LR scheduler, and
    # the tensorboard LR report all assume >=1 group and crash (empty-optimizer / zip
    # mismatch). Add ONE throwaway trainable group so the whole pipeline stays consistent
    # end-to-end (collection == optimizer == scheduler == report); it never receives a
    # gradient (not in the graph), so its SGD step is a no-op. Concord-only: a genuinely
    # empty collection for any other optimizer is a real "nothing to train" error.
    if is_concord_family(model.train_config.optimizer.optimizer) and not parameters.parameters():
        _ph = torch.nn.Parameter(torch.zeros(1, device=train_device))
        model._concord_placeholder_param = _ph      # keep a reference alive
        parameters.add_group(NamedParameterGroup(
            unique_name="concord_placeholder", parameters=[_ph],
            learning_rate=model.train_config.learning_rate))

    #random (LoRA) initialisation can differ, broadcast from GPU #0 to all others
    #to be safe, do that before the optimizer is created because the optimizer could take copies
    multi.broadcast_parameters(parameters.parameters(), train_device)

    layer_key_fn = None
    if model.train_config.optimizer.MuonWithAuxAdam:
        print("INFO: Creating layer keys for MuonWithAuxAdam.")
        layer_key_fn = build_muon_adam_key_fn(model, model.train_config)

    model.optimizer = create.create_optimizer(
        parameters, model.optimizer_state_dict, model.train_config, layer_key_fn
    )

    if model.optimizer is not None:
        optimizer_to_device_(model.optimizer, train_device)
    model.optimizer_state_dict = None

    if multi.is_master():
        model.ema = create.create_ema(parameters.parameters(), model.ema_state_dict, model.train_config)
    else:
        model.ema = None
    model.ema_state_dict = None

    if model.optimizer is not None and any('optim_type' in g for g in model.optimizer.param_groups):
        new_param_group_mapping = []
        for group in model.optimizer.param_groups:
            original_name = group.get('name')

            optim_type = group.get('optim_type', 'unknown')
            unique_name = f"{original_name}_{optim_type}"
            new_param_group_mapping.append(unique_name)
        model.param_group_mapping = new_param_group_mapping
    else:
        model.param_group_mapping = parameters.unique_name_mapping


# Optimizer Key map with defaults
OPTIMIZER_DEFAULT_PARAMETERS = {
    Optimizer.ADAFACTOR: {
        "eps": 1e-30,
        "eps2": 1e-3,
        "clip_threshold": 1.0,
        "decay_rate": -0.8,
        "beta1": None,
        "weight_decay": 0.0,
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
        "stochastic_rounding": True,
        "fused_back_pass": False,
    },
    Optimizer.ADAGRAD: {
        "lr_decay": 0,
        "weight_decay": 0,
        "initial_accumulator_value": 0,
        "eps": 1e-10,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.ADAGRAD_8BIT: {
        "lr_decay": 0,
        "weight_decay": 0,
        "initial_accumulator_value": 0,
        "eps": 1e-10,
        "optim_bits": 8,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "fused_back_pass": False,
    },
    Optimizer.ADAM_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.ADAMW_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.MUON: {
        "momentum": 0.95,
        "weight_decay": 0.0,
        "MuonWithAuxAdam": True,
        "muon_hidden_layers": None,
        "muon_adam_regex": False,
        "muon_adam_lr": 3e-4,
        "muon_te1_adam_lr": None,
        "muon_te2_adam_lr": None,
        "muon_adam_config": {},
    },
    Optimizer.AdEMAMix_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-8,
        "alpha": 5,
        "weight_decay": 1e-2,
        "min_8bit_size": 4096,
        "is_paged": False,
    },
    Optimizer.AdEMAMix: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-8,
        "alpha": 5,
        "weight_decay": 1e-2,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "is_paged": False,
    },
    Optimizer.ADOPT: {
        "beta1": 0.9,
        "beta2": 0.9999,
        "weight_decay": 0.0,
        "decoupled_decay": False,
        "fixed_decay": False,
        "cautious": False,
        "eps": 1e-6,
    },
    Optimizer.LAMB: {
        "bias_correction": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "adam_w_mode": True,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": False,
        "max_unorm": 1.0,
    },
    Optimizer.LAMB_8BIT: {
        "bias_correction": True,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "adam_w_mode": True,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": False,
        "max_unorm": 1.0,
    },
    Optimizer.LARS: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "max_unorm": 0.02,
    },
    Optimizer.LARS_8BIT: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "max_unorm": 0.02,
    },
    Optimizer.LION_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
    },
    Optimizer.RMSPROP: {
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False,
        "optim_bits": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.RMSPROP_8BIT: {
        "alpha": 0.99,
        "eps": 1e-8,
        "weight_decay": 0,
        "momentum": 0,
        "centered": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.SGD_8BIT: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
    },
    Optimizer.SCHEDULE_FREE_ADAMW: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "foreach": False,
    },
    Optimizer.SCHEDULE_FREE_SGD: {
        "momentum": 0,
        "weight_decay": 1e-2,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "foreach": False,
    },
    Optimizer.PRODIGY: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": None,
        "eps": 1e-8,
        "weight_decay": 0,
        "decouple": True,
        "use_bias_correction": False,
        "safeguard_warmup": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
        "slice_p": 11,
    },
    Optimizer.PRODIGY_PLUS_SCHEDULE_FREE: {
        "beta1": 0.9,
        "beta2": 0.99,
        "beta3": None,
        "weight_decay": 0.0,
        "weight_decay_by_lr": True,
        "use_bias_correction": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "prodigy_steps": 0,
        "use_speed": False,
        "eps": 1e-8,
        "split_groups": True,
        "split_groups_mean": False,
        "factored": True,
        "factored_fp32": True,
        "fused_back_pass": False,
        "use_stableadamw": True,
        "use_cautious": False,
        "use_grams": False,
        "use_adopt": False,
        "d_limiter": True,
        "stochastic_rounding": True,
        "use_schedulefree": True,
        "schedulefree_c": 0.0,
        "use_orthograd": False,
    },
    Optimizer.DADAPT_ADA_GRAD: {
        "momentum": 0,
        "log_every": 0,
        "weight_decay": 0.0,
        "eps": 0.0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
    },
    Optimizer.DADAPT_ADAN: {
        "beta1": 0.98,
        "beta2": 0.92,
        "beta3": 0.99,
        "eps": 1e-8,
        "weight_decay": 0.02,
        "no_prox": False,
        "log_every": 0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
    },
    Optimizer.DADAPT_ADAM: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "log_every": 0,
        "decouple": False,
        "use_bias_correction": False,
        "d0": 1e-6,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_SGD: {
        "momentum": 0.0,
        "weight_decay": 0,
        "log_every": 0,
        "d0": 1e-6,
        "growth_rate": float('inf'),
        "fsdp_in_use": False,
    },
    Optimizer.DADAPT_LION: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "log_every": 0,
        "d0": 1e-6,
        "fsdp_in_use": False,
    },
    Optimizer.ADAM: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "stochastic_rounding": False,
        "fused_back_pass": False,
    },
    Optimizer.ADAMW: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": True,
        "stochastic_rounding": False,
        "fused_back_pass": False,
    },
    Optimizer.SGD: {
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "foreach": False,
        "maximize": False,
        "differentiable": False,
    },
    # Concord: optimizer-visible params are the aux SGD's (norms/biases/embeddings); the
    # swapped UNet layers self-step in backward. lr comes from the main learning_rate field.
    # The winner knobs default to the validated sf_060 configuration.
    # The panel shows only the live physical knobs. gf_consol (kappa, subsumed by
    # the dimensionless dissipation) and ratio_coh (the gate IS the mechanism; off
    # is a debug state) remain config-file keys with engine defaults, not panel
    # entries. The probe-gated beta1 pair (autotune_beta1_on / _coh) IS now a panel
    # entry -- experimental, off by default, autotuner-mediated.
    Optimizer.CONCORD: {
        "momentum": 0.9,
        "weight_decay": 0,
        "noise": True,
        "sigmag_peak": 0.6,
        "lazy_gate": False,
        "lazy_active_thresh": 0.0001,
        "warmup": 100,
        "lr_min_frac": 0.2,
        "step_cap": 10.0,
        "gf_trust_delta_sq": 1.0,
        "min_leak": 0.1,
        "evap_build_min": 128.0,
        "lamb_trust": False,
        "lamb_cap": 0.0025,
        "lamb_clip": 4.0,
        "beta2": 0.999,
        "beta2_epoch_window": True,
        "vhat_warmstart": True,
        "bias_correct_v": False,
        "coh_vhat": True,
        "coh_kappa": 1.0,
        "dissipation_fill_ramp": True,
        "telescope_epoch_window": True,
        # Concord 2-fast (split-tick bracket): OFF by default -- the default
        # config IS the validated winner (byte-identical kernel path). When
        # on, the fine field runs as two int8 arms at bracketed friction
        # lam*(1-/+d); arm gap/sum are read-only meters in v1.
        "two_fast": False,
        "bracket_d": 0.25,
        # Eviction valve (both formats): on sign disagreement between the
        # fine mass and the position, demote position mass to the arms at
        # the forward chase rate -- evidence-gated weight decay by
        # detailed balance (positions persist only under >2/3 sign-
        # agreement; converged weights pay nothing). Fixes the monotone
        # norm growth. Delta-gated (exp 26): fires on disagreement with the
        # LEARNED DELTA s_slow-v_slow, so the pretrained PRIOR (common mode)
        # is protected -- raw-gating eroded it (fine-tune sample scramble).
        # Rate = evict_gain (exp-26d knee 0.66). ON by default now that the
        # delta fix is validated (CPU exp 25-26f: prior retention 31->49% vs
        # raw at ~unchanged fine-tune). NOT yet on-GPU-A/B-validated.
        "evict_valve": True,
        "evict_gain": 0.66,
        "evict_cf_gate": False,  # high-gain enabler (exp 26f); ON locks gain to 1.0
        "heldout_router": False,  # 2fast held-out arm router: alternate micros -> alternate arms; needs accum>=2 (guarded); the routed gap is a data meter, not a dissipation derivative
        "router_coh_noise": False,  # under heldout_router: routed gap^2 -> coherence noise floor after the cf discount (exp47b)
        "perfcoh_partition": False,  # soft CF gate: leak commits *= exp(-(1-coh)/tau), evict reverts *= complement (probation for marginal admissions)
        "perfcoh_tau": 0.5,  # nats knee of the perfcoh partition (smaller = stricter commit)
        "noise_seed_servo": False,  # set-don't-hunt: per-layer kappa seeded from measured gradient NSR (exp50); the only dissipation controller
        "nsr_per_row": True,  # per-row whitened lam from the arm meter (R_row); residue-masked rows keep layer lam; exp56 closed-loop verdict pending (fix-forward)
        "kaiming_init": False,
        "kaiming_scale": 0.05,
        "chase_epoch_window": False,
        "chase_alpha": 0.0,
        "alpha_v_fast": 0.001,
        # Ratio-coh gate FLOORS (rate granted at coh=0), cosine-decayed start->min over
        # floor_horizon (~1 epoch) by winner_step. chase = evidence bar for consolidation
        # (exps 31-35: fine-tune optimum ~0.3). The LEAK floors are deliberately absent:
        # DERIVED from the chase in make_concord_config (leak_min = chase_min, the
        # winner's ratio-0.9 pairing at any scale; leak_start = 0.999) -- exp73 showed
        # the leak gate is the chase gate's calibration partner and a mis-RATIOED leak
        # floor deadlocks certification globally. NO override exists (the env hatch was
        # removed 2026-07-18; mis-ratio ablations live on the CPU reference only).
        "ratio_chase_floor": 0.9,
        "ratio_chase_floor_min": 0.1,
        # Dimensionless mode ON by default: lam = lr*kappa = 0.025 (the nanoGPT
        # winner; ~kappa 333 at lr 7.5e-5 -- "kappa at diffusion lr needs to be
        # higher"). Clearing the field falls back to the engine kappa default
        # (gf_consol 50). The autotune table is the CPU-calibrated lam-units
        # curve (exp 5/6/11); its kappa column is divided by lr at build.
        # Caveat carried from the calibration: the COHERENCE column is calibrated
        # on the CPU task -- cross-domain transfer of the lam curve is the
        # hypothesis these defaults exist to test. Re-probe band arms the exp-11d
        # one-sided live watchdog (re-probe only on a coherence DROP).
        "dissipation": 0.025,
        "autotune_table": "[[0.387,0],[0.314,0.1],[0.288,0.2],[0.274,0.4],[0.256,0.4]]",
        "autotune_reprobe_band": 0.02,
        "autotune_gamma_snr_on": True,
        "autotune_gamma_snr": None,
        # Coherence-gated momentum (beta1), probe-selected -- experimental, off by
        # default. Surfaced as panel knobs so the gated-beta1 A/B doesn't need a
        # hand-edited config. Only fires with a non-empty Autotune Table (the tuner
        # must be built) and only on layers whose probed coherence clears the threshold.
        "autotune_beta1_on": 0.0,
        "autotune_beta1_coh": 0.35,
        "autotune_servo_per_epoch": 3,
        "concord_conv_full_vhat": False,
        "concord_evap_slack": 0.25,
        "concord_train_cond_embed": False,
    },
    Optimizer.LION: {
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 0.0,
        "use_triton": False,
    },
    Optimizer.CAME: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-30,
        "eps2": 1e-16,
        "weight_decay": 1e-2,
        "stochastic_rounding": False,
        "use_cautious": False,
        "fused_back_pass": False,
    },
    Optimizer.CAME_8BIT: {
        "beta1": 0.9,
        "beta2": 0.999,
        "beta3": 0.9999,
        "eps": 1e-30,
        "eps2": 1e-16,
        "weight_decay": 1e-2,
        "stochastic_rounding": False,
        "fused_back_pass": False,
        "min_8bit_size": 16384,
        "quant_block_size": 2048
    },
    Optimizer.ADAMW_ADV: {
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": 1e-8,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "use_atan2": False,
        "orthogonal_gradient": False,
        "use_AdEMAMix": False,
        "beta3_ema": 0.9999,
        "alpha": 5,
        "kourkoutas_beta": False,
    },
    Optimizer.ADOPT_ADV: {
        "beta1": 0.9,
        "beta2": 0.9999,
        "eps": 1e-6,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "use_atan2": True,
        "orthogonal_gradient": False,
        "use_AdEMAMix": False,
        "beta3_ema": 0.9999,
        "alpha": 5,
        "Simplified_AdEMAMix": False,
        "alpha_grad": 100.0,
        "kourkoutas_beta": False,
    },
    Optimizer.PRODIGY_ADV: {
        "beta1": 0.9,
        "beta2": 0.99,
        "beta3": None,
        "eps": 1e-8,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "d0": 1e-6,
        "d_coef": 1.0,
        "growth_rate": float('inf'),
        "slice_p": 11,
        "prodigy_steps": 0,
        "d_limiter": False,
        "use_atan2": False,
        "orthogonal_gradient": False,
        "use_AdEMAMix": False,
        "beta3_ema": 0.9999,
        "alpha": 5,
        "Simplified_AdEMAMix": False,
        "alpha_grad": 100.0,
        "kourkoutas_beta": False,
    },
    Optimizer.SIGNSGD_ADV: {
        "momentum": 0.95,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "orthogonal_gradient": False,
        "Simplified_AdEMAMix": False,
        "alpha_grad": 100.0,
    },
    Optimizer.LION_ADV: {
        "beta1": 0.9,
        "beta2": 0.99,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "clip_threshold": None,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "orthogonal_gradient": False,
        "auto_kappa_p": True,
    },
    Optimizer.MUON_ADV: {
        "beta1": 0.9,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "accelerated_ns": False,
        "ns_steps": 5,
        "low_rank_ortho": False,
        "ortho_rank": 128,
        "rms_rescaling": True,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "MuonWithAuxAdam": True,
        "muon_hidden_layers": None,
        "muon_adam_regex": False,
        "muon_adam_lr": 1e-6,
        "muon_te1_adam_lr": None,
        "muon_te2_adam_lr": None,
        "nesterov": True,
        "Simplified_AdEMAMix": False,
        "alpha_grad": 100.0,
        "normuon_variant": True,
        "beta2_normuon": 0.95,
        "orthogonal_gradient": False,
        "approx_mars": False,
        "muon_adam_config": {},
    },
    Optimizer.ADAMUON_ADV: {
        "beta1": 0.95,
        "beta2": 0.95,
        "eps": 1e-8,
        "cautious_wd": False,
        "weight_decay": 0.0,
        "accelerated_ns": False,
        "ns_steps": 5,
        "low_rank_ortho": False,
        "ortho_rank": 128,
        "rms_rescaling": True,
        "nnmf_factor": False,
        "stochastic_rounding": True,
        "compile": False,
        "fused_back_pass": False,
        "MuonWithAuxAdam": True,
        "muon_hidden_layers": None,
        "muon_adam_regex": False,
        "muon_adam_lr": 1e-6,
        "muon_te1_adam_lr": None,
        "muon_te2_adam_lr": None,
        "nesterov": False,
        "use_atan2": False,
        "Simplified_AdEMAMix": False,
        "alpha_grad": 100.0,
        "normuon_variant": True,
        "orthogonal_gradient": False,
        "approx_mars": False,
        "muon_adam_config": {},
    },
    Optimizer.ADABELIEF: {
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-16,
        "weight_decay": 0,
        "amsgrad": False,
        "decoupled_decay": True,
        "fixed_decay": False,
        "rectify": True,
        "degenerated_to_sgd": True,
    },
    Optimizer.TIGER: {
        "beta1": 0.965,
        "weight_decay": 0.01,
        "decoupled_decay": True,
        "fixed_decay": False,
    },
    Optimizer.AIDA: {
        "beta1": 0.9,
        "beta2": 0.999,
        "k": 2,
        "xi": 1e-20,
        "weight_decay": 0.0,
        "decoupled_decay": False,
        "fixed_decay": False,
        "rectify": False,
        "n_sma_threshold": 5,
        "degenerated_to_sgd": True,
        "ams_bound": False,
        "r": 0.95,
        "adanorm": False,
        "adam_debias": False,
        "eps": 1e-8,
    },
    Optimizer.YOGI: {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.0,
        "decoupled_decay": True,
        "fixed_decay": False,
        "r": 0.95,
        "adanorm": False,
        "adam_debias": False,
        "initial_accumulator": 1e-6,
        "eps": 1e-3,
    },
}

# CONCORD_STEPLESS shares CONCORD's whole panel/param surface (same knobs, same
# tooltips via KEY_DETAIL_MAP): the lineages differ only in which kernel module
# kernel_select binds. A live dict copy keeps the two entries in sync by
# construction (and the AST-parsing doc/leak-floor tests see only the literal).
OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.CONCORD_STEPLESS] = dict(
    OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.CONCORD])
