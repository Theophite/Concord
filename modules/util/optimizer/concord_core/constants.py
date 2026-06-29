"""concord_core.constants — pure numeric constants and the host-side drift-cancel helper.

RESPONSIBILITY
    The leaf module of the tree. Holds the int8/int16 quantization bounds, the
    mantissa bias, the slow-field scale factors, and ``compute_drift_cancel_C``.
    ZERO dependencies (no torch / no triton import at module load) — this file
    must stay importable on a CPU-only / no-CUDA host so the CPU golden gate
    (L0/L1) and the doc-tests can read it.

DOC-CONTRACT (do not silently change — gated by L2 / G7):
    MANTISSA_BIAS == 15, S_SLOW_FACTOR == 128, V_SLOW_FACTOR == 128 are asserted
    by VALUE in the doc tests and CONCORD.md. compute_drift_cancel_C is doc-tested
    by value. A variable rename or value change here is a CONTRACT BREAK.

MIGRATES FROM prototype_packed_b.py (PB) — DO NOT MOVE CODE YET; line-range map only:
    PB:45-49    MANTISSA_BIAS, INT8_MIN/INT8_MAX, INT16_MIN/INT16_MAX,
                S_SLOW_FACTOR, V_SLOW_FACTOR
    PB:52-102   compute_drift_cancel_C

RE-EXPORT (the shim + concord_winner + concord_embedding_packed must see these):
    MANTISSA_BIAS, INT16_MIN, INT16_MAX, S_SLOW_FACTOR, V_SLOW_FACTOR,
    compute_drift_cancel_C
    (concord_embedding_packed.py:18-19 reads INT16_MIN/MAX/S_SLOW_FACTOR/V_SLOW_FACTOR;
     it also reads ConcordLinearPackedB's class-level MANTISSA_BIAS — see layers.py.)

MIGRATION: STEP 2 (first move; lowest coupling). After moving, the monolith does
    ``from constants import *`` (bare; the concord dir is already on sys.path).
    Gate: L0 + L1 + L2 (values are doc-tested -> must stay green).
"""
