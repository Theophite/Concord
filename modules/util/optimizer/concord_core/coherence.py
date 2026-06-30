"""concord_core.coherence — host-side coherence reconstruction helpers.

RESPONSIBILITY
    The two pure-host (float-tensor) coherence functions used by servo and by
    callers that need the kernel's Wiener gain computed outside a launch:
        gate_coherence_from_fields  (the kernel's USE_FIXED_COH Wiener gain,
                                     computed host-side from unpacked int fields)
        measure_coherence
    A LEAF module: depends only on torch + constants (no kernels, no state).
    CPU-import-safe enough for the CPU coherence golden (test_coherence_cpu).

CAUTION (INV-B / D1 — multiple coherence definitions must stay numerically locked;
    NOT unified in this behavior-preserving pass):
        in-kernel coh_raw (un-cf-discounted Wiener) vs cf-discounted coh vs
        _COH_WEIGHTED_V vs the USE_RATIO_COH non-FIXED branch vs THESE host
        helpers. They are intentionally distinct today. A FIFTH definition lives
        OUTSIDE this split and outside every golden: concord_ot.py:_metrics
        (~1104-1136) re-derives the kernel coherence in pure host torch, hard-coding
        ``- 15.0`` (MANTISSA_BIAS), ``* 128.0`` (S_SLOW_FACTOR), and the kernel's
        3% v_hat floor as ``0.03 * vh.mean()``, reading _ppb._COH_KAPPA/
        _USE_COH_VHAT/_EVAP_BUILD_MIN through the shim. The refactor must keep the
        shim resolving those live; do NOT touch the constants without auditing that
        host re-derivation (it has no golden -> it drifts silently). See D1 (D2nd-pass).

MIGRATES FROM prototype_packed_b.py (PB) — see **REFACTOR_PLAN.md §3** for the
    AUTHORITATIVE, reconciled PB line-range map (single source of truth). The per-line
    ranges that used to be duplicated here were the original setup-task numbers against a
    4150-line PB and are SUPERSEDED — the source is now 4176 lines after the 2026-06-29
    M6a / 6-wide-boil / servo-ceiling drift. Any ``PB:NNN`` still cited elsewhere in this
    docstring is ILLUSTRATIVE only; re-verify against §3 before moving. DO NOT MOVE CODE YET.

RE-EXPORT (shim must expose): gate_coherence_from_fields, measure_coherence
    (servo imports measure_coherence).

MIGRATION: STEP 4 (leaf; only servo + measure_coherence consumers).
    Gate: L0 + L1 (test_coherence_cpu).
"""
