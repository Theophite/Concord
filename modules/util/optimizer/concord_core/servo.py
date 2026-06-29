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

MIGRATES FROM prototype_packed_b.py (PB) — DO NOT MOVE CODE YET; line-range map only:
    PB:3493-3649  DissipationAutoTuner
    PB:3652-3945  EpochDissipationServo

RE-EXPORT (shim must expose): DissipationAutoTuner, EpochDissipationServo
    (both have external consumers).

MIGRATION: STEP 5 (after coherence.py). ``import state``,
    ``from coherence import measure_coherence``.
    Gate: L0 + L1 (test_servo_cpu, test_autotuner_cpu — run the trajectory-snapshot
    diff at atol=0; G3a servo._kappa/_step/_last_dir/_agg_boil trajectory +
    G3b autotuner commit/re-probe events).
"""
