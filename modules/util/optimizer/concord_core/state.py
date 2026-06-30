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

MIGRATES FROM prototype_packed_b.py (PB) — see **REFACTOR_PLAN.md §3** for the
    AUTHORITATIVE, reconciled PB line-range map (single source of truth). The per-line
    ranges that used to be duplicated here were the original setup-task numbers against a
    4150-line PB and are SUPERSEDED — the source is now 4176 lines after the 2026-06-29
    M6a / 6-wide-boil / servo-ceiling drift. NOTE the meters are now 6-WIDE (_BOIL_BUFS =
    zeros(6): [0..3] boil/waste incl. the coh_evap-weighted [3], [4]/[5] = the M6a
    diversity meter num/denom — read by concord_ot, not by read_boil). Any ``PB:NNN``
    still cited elsewhere in this docstring is ILLUSTRATIVE only; re-verify against §3
    before moving. DO NOT MOVE CODE YET.

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
    compare the unpacked packed_w fields s_fast/s_slow/v_slow vs baseline within the
    near-bit-exact per-field envelope — NOT a raw bit-hash; the kernel is not
    bit-reproducible, see REFACTOR_PLAN.md §7).
"""
