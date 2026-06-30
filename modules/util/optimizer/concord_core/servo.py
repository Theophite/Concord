"""concord_core.servo — the dissipation autotuners.

RESPONSIBILITY
    DissipationAutoTuner (table-driven; the SHIPPED-default path — defaults wire a
    populated autotune_table AND autotune_servo=False, so the TABLE is active) and
    EpochDissipationServo (the per-layer table-free dissipation autotuner; the doc
    says it supersedes the table). Pure host control logic over the boil/memgap
    meters; imports torch + ``import state`` + ``from coherence import measure_coherence``.
    No kernels, no triton -> CPU-golden-friendly (test_servo_cpu, test_autotuner_cpu).

CAUTION (D5 — table-vs-servo redundancy is currently AMBIGUOUS; do NOT resolve in
    this pass): doc says servo replaces the table; shipped defaults activate the
    table. NOT a clean removal — second pass must decide which supersedes which.

MIGRATES FROM prototype_packed_b.py (PB) — see **REFACTOR_PLAN.md §3** for the
    AUTHORITATIVE, reconciled PB line-range map (single source of truth). The per-line
    ranges that used to be duplicated here were the original setup-task numbers against a
    4150-line PB and are SUPERSEDED — the source is now 4176 lines after the 2026-06-29
    M6a / 6-wide-boil / servo-ceiling drift (EpochDissipationServo.step changed: the
    cf-ceiling was removed for protected_boil — see §3). Any ``PB:NNN`` still cited
    elsewhere in this docstring is ILLUSTRATIVE only; re-verify against §3 before moving.
    DO NOT MOVE CODE YET.

RE-EXPORT (shim must expose): DissipationAutoTuner, EpochDissipationServo
    (both have external consumers).

MIGRATION: STEP 5 (after coherence.py). ``import state``,
    ``from coherence import measure_coherence``.
    Gate: L0 + L1 (test_servo_cpu, test_autotuner_cpu — run the trajectory-snapshot
    diff at atol=0; G3a servo._kappa/_step/_last_dir/_agg_boil trajectory +
    G3b autotuner commit/re-probe events).
"""
