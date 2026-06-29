"""concord_core.state — THE single home for every module-global config scalar,
device-tensor cache, meter buffer, and their setters.

RESPONSIBILITY
    Own EVERY live-mutable global so there is exactly ONE true object per
    (name, device). kernels.py and layers.py reach these ONLY through this module
    (``import state as S``; read ``S._COH_KAPPA`` as a LIVE attribute, never
    ``from state import _COH_KAPPA`` — that snapshots at import time: R1).

    This is the HIGHEST-RISK module (snapshot hazard R1, cache-identity R5,
    SR-determinism R6) and therefore MIGRATES LAST (STEP 6).

MUTATION-CLASS TAXONOMY (INV-A — the protection mechanism differs per class; every
global moved here MUST be classified, not lumped as "launch-baked scalar"):
    CLASS 1 — setter-driven scalars (set_X(v) rebinds a module global):
        _USE_FIXED_COH, _USE_COH_VHAT, _COH_KAPPA, _EVAP_SLACK, _MIN_LEAK,
        _EVAP_BUILD_MIN, _RATIO_COH (+floors), _LAZY_GATE, _LAZY_THRESH,
        _LAMB_TRUST, _GAP_FEEDBACK, _SIGMAG_NOISE, _SIGMAG_ISO, _COH_WEIGHTED_V,
        _BIAS_CORRECT_V, _V_BC, _GATE_GAIN.
        Protected by: setter -> shim PEP-562 __getattr__ -> S.-live-read + the L0
        set_X(v)->assert gate. COVERED by the plan's R1 mitigation.
    CLASS 2 — DIRECT-ATTR-ASSIGNED (NO setter; written as ppb._X = v):
        _FUSED_MATMUL  (also the env mirror CONCORD_FUSED_MATMUL).
        *** STRESS-FIX SF1 (MANDATORY) ***  __getattr__ CANNOT protect this: an
        external write ``ppb._FUSED_MATMUL = True`` creates a REAL shim attribute
        that SHADOWS __getattr__, and the production write loop never touches
        `state`, so layers reading S._FUSED_MATMUL stay at the import-time env
        default -> fused/cached SILENTLY INVERTS. FIX: add a real
        ``set_fused_matmul(bool)`` setter here (single source of truth), route the
        two writers through it (StableDiffusionXLFineTuneSetup.py:149 and
        test_autotuner_cpu.py:450/459), and L0-gate it (attr-assign on shim ->
        assert BOTH layers AND state observe it). See REFACTOR_PLAN.md SF1.
    CLASS 3 — REBOUND non-scalar dicts (global X; X = dict(...)):
        _REB_STATS  (reset_reb_stats does ``global _REB_STATS; _REB_STATS = dict(...)``
        at PB:2139-2142 -> a `from state import _REB_STATS` in kernels.py would
        snapshot the pre-reset object). rebalance_packed in kernels.py must read it
        as S._REB_STATS. No external consumer (bounded impact) but ENUMERATE it.
        _GRADW_DIAG is rebind-SAFE (in-place dict mutation) but is read in the
        apply launcher (kernels.py) and has a live consumer (concord_ot.py:1067 via
        read_gradw_diag) -> also read as S._GRADW_DIAG. (Holes 3 & 4.)

CACHE-IDENTITY INVARIANT (R5 — define each EXACTLY ONCE; kernels/layers reach via
    state functions only): _LR_SCALAR_CACHE, _EPS_SCALAR_CACHE, _NAMED_SCALAR_CACHE,
    _CONSOLIDATE_FLAGS, _STEP_COUNTERS, _REB_SEED_CACHE, the LAMB caches/bufs,
    _RATIO_*_FLOOR_T, _SIGMAG_SIGMA_T, _V_BC_BUFS, _VHAT_MEAN_BUFS, _MEMGAP_BUFS,
    _BOIL_BUFS, _PERLAYER_METERS.

KEYING ASYMMETRY (R5/R6 — KEEP, do NOT unify):
    _STEP_COUNTERS keyed by raw str(device)   (PB:472)
    _CONSOLIDATE_FLAGS keyed by _dev_key (normalizes cuda<->cuda:0)  (PB:486-499)
    One shared _STEP_COUNTERS feeds BOTH apply_packed_sgd and apply_packed_adamw so
    the SR salt stays in lock-step (R6).

MIGRATES FROM prototype_packed_b.py (PB) — DO NOT MOVE CODE YET; line-range map only:
    PB:173-197    _FUSED_MATMUL env flag, _FUSED_SCRATCH, _get_fused_scratch
                  (+ STRESS-FIX SF1: NEW set_fused_matmul setter goes here)
    PB:469-476    _STEP_COUNTERS, _get_step_counter   (raw str(device) key — KEEP)
    PB:479-507    _CONSOLIDATE_FLAGS, _dev_key, _get_consolidate_flag,
                  set_consolidate   (_dev_key key — KEEP asymmetry)
    PB:510-526    _LR_SCALAR_CACHE, _ensure_lr_tensor
    PB:529-544    _EPS_SCALAR_CACHE, _ensure_eps_tensor
    PB:547-563    _NAMED_SCALAR_CACHE, _ensure_named_scalar
    PB:1277-1341  _USE_FIXED_COH/set_fixed_coh, _USE_COH_VHAT/set_coh_vhat,
                  _COH_KAPPA/set_coh_kappa, _EVAP_SLACK/set_evap_slack,
                  _MIN_LEAK/set_min_leak, _EVAP_BUILD_MIN/set_evap_build_min
    PB:1344-1419  LAMB: _LAMB_TRUST/_LAMB_CAP/_LAMB_CLIP,
                  _LAMB_SCALE_CACHE/_WNORM_SQ/_STEPNORM_SQ, set_lamb_trust,
                  _lamb_scale_buf/_lamb_wnorm_sq_buf/_lamb_stepnorm_sq_buf
    PB:1422-1548  METERS: _MEMGAP_BUFS/_BOIL_BUFS, _boil_buf/read_boil/
                  _memgap_buf/read_memgap; _PERLAYER_METERS,
                  register_layer_meters/_lookup_layer_meters/clear_layer_meters/
                  read_layer_boil/read_layer_memgap
    PB:1551-1600  _GATE_GAIN/set_gate_gain, _SIGMAG_*/set_sigmag_noise/
                  set_sigmag_sigma/_get_sigmag_sigma
    PB:1603-1705  _GAP_FEEDBACK/_GAP_SCALE/set_gap_feedback,
                  _COH_WEIGHTED_V/set_coh_weighted_v,
                  _RATIO_COH+floors/set_ratio_coh/set_ratio_coh_floors/
                  _ensure_floor_tensors
    PB:1663-1678  _LAZY_GATE/_LAZY_THRESH/set_lazy_gate/set_lazy_thresh
                  (NOTE: this sub-range is interleaved inside 1603-1705 above;
                   move the whole 1603-1705 block, this line is a pointer.)
    PB:1708-1782  _BIAS_CORRECT_V/_V_BC_BUFS/_VHAT_MEAN_BUFS, _v_bc_buf/
                  _vhat_mean_buf/set_bias_correct_v/set_v_bias_correction/
                  bias_correction_factor; _GRADW_DIAG/_GRADW_KEYS/read_gradw_diag
    PB:2134-2162  _REB_STATS/reset_reb_stats/get_reb_stats,
                  _REB_SEED_CACHE/_ensure_reb_seed_tensor

RE-EXPORT (shim must expose ALL of these; concord_winner re-exports the setter
    family; concord_ot reads the launch-mutable scalars for health/active_config):
    set_consolidate, register_layer_meters, read_boil, read_memgap,
    read_gradw_diag, _get_step_counter,
    set_bias_correct_v / _v_bc_buf / bias_correction_factor / set_v_bias_correction,
    _lamb_scale_buf / _lamb_wnorm_sq_buf / _lamb_stepnorm_sq_buf /
        _LAMB_TRUST / _LAMB_CAP / _LAMB_CLIP,
    set_ratio_coh / set_ratio_coh_floors / set_fixed_coh / set_lazy_gate /
        set_lazy_thresh / set_min_leak / set_evap_build_min / set_lamb_trust /
        set_coh_vhat / set_coh_kappa / set_evap_slack / set_sigmag_noise /
        set_sigmag_sigma,
    set_fused_matmul (NEW, SF1),
    and the launch-mutable scalars read live by concord_ot:
        _FUSED_MATMUL, _EVAP_BUILD_MIN, _USE_COH_VHAT, _COH_KAPPA.

MIGRATION: STEP 6 (LAST move before kernels). Gate: L0 (re-export chain + the
    set_X(v)->assert ppb._X==v snapshot-catcher, the single most important
    automated check) + L0' (LIVE-MODULE variant SF/Hole-5: assert the module that
    HOLDS the read observes the change, not just the shim) + L1 (full CPU suite)
    + targeted L3 micro-gate (flip evap_slack/coh_vhat/coh_kappa, one apply step,
    hash packed_w vs baseline).
"""
