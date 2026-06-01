"""Concord -- packed-int storage optimizer (public API).

A bare ``ConcordLinearPackedB`` / ``ConcordConv2dPackedB`` is the validated
optimizer (rank-1 v-hat AdamW + fixed coherence gate). The weight update is
fused into the backward pass. See ``packed_b.py`` and ``README.md``.
"""
from .packed_b import (
    ConcordLinearPackedB,
    ConcordConv2dPackedB,
    compute_drift_cancel_C,
    set_fixed_coh,
    set_gate_gain,
    reset_reb_stats,
    get_reb_stats,
    S_SLOW_FACTOR,
    V_SLOW_FACTOR,
    MANTISSA_BIAS,
)

__all__ = [
    "ConcordLinearPackedB",
    "ConcordConv2dPackedB",
    "compute_drift_cancel_C",
    "set_fixed_coh",
    "set_gate_gain",
    "reset_reb_stats",
    "get_reb_stats",
    "S_SLOW_FACTOR",
    "V_SLOW_FACTOR",
    "MANTISSA_BIAS",
]
