"""concord_core.kernels — every @triton.jit kernel + its thin launch wrapper +
the in-Python pack/unpack helpers + the (gated, default-off) denom diagnostic.

RESPONSIBILITY
    All GPU compute. This file imports triton/torch at load (so it is NOT
    CPU-import-safe by itself — only reached on the GPU path / GPU goldens).

    LIVE-BINDING DISCIPLINE (R1 — the central invariant of this refactor):
        ``import state as S`` and read launch-baked globals as ``S._COH_KAPPA``
        (live attribute access) at LAUNCH time. NEVER ``from state import _COH_KAPPA``
        (that snapshots the value at import and silently desyncs after a setter).
        ``from constants import *`` is fine (constants never mutate).

    Cross-module LIVE reads in this file (S.-prefixed; verified):
        apply_packed_adamw (PB:1785-1987) reads
            S._COH_KAPPA, S._EVAP_SLACK, S._MIN_LEAK, S._EVAP_BUILD_MIN,
            S._GATE_GAIN, S._LAZY_THRESH, S._GAP_FEEDBACK, S._GAP_SCALE,
            S._USE_FIXED_COH, S._USE_COH_VHAT, S._RATIO_COH, S._LAZY_GATE,
            S._LAMB_TRUST   (live reads confirmed PB:1947-1986).
        apply launcher also touches S._GRADW_DIAG (PB:1832,1860-1864 — Hole 4,
            rebind-safe but ENUMERATED) and CALLS _denom_diagnostic at PB:1913.
        rebalance_packed reads S._REB_STATS (PB:2197,2210-2215 — Hole 3, REBOUND
            class; must be S.-live, never `from state import _REB_STATS`).

DOC-CONTRACT (R3 / STEP 9 / L2 — the kernel BODY text is the contract):
    test_doc_kernel.py greps these byte-identical strings AND their order from the
    file that holds the kernel (currently concord/prototype_packed_b.py:116-117 via
    PPB_SRC; repointed in STEP 9 to wherever the kernel text physically lives):
        "noise = d_fs - drift_cancel_C * d_sv"
        "noise_in_w = noise * scale_fwd"
        "v_proxy = noise_in_w * noise_in_w"            (i_noise)
        "v_proxy = v_proxy + gf_trust_delta_sq * v_hat" (i_vhat; i_noise < i_vhat)
        "denom_p = tl.exp2(precond_p * tl.log2(v_proxy + eps))"
    NO variable renames in this pass — the strings ARE the contract.

MIGRATES FROM prototype_packed_b.py (PB) — see **REFACTOR_PLAN.md §3** for the
    AUTHORITATIVE, reconciled PB line-range map (single source of truth). The per-line
    ranges that used to be duplicated here were the original setup-task numbers against a
    4150-line PB and are SUPERSEDED — the source is now 4176 lines after the 2026-06-29
    M6a / 6-wide-boil / servo-ceiling drift (the adamw kernel now writes the coh_evap-
    weighted boil [3] + the M6a [4]/[5] atomics; doc-grepped strings are matched by
    CONTENT, not line). Any ``PB:NNN`` still cited elsewhere in this docstring is
    ILLUSTRATIVE only; re-verify against §3 before moving. DO NOT MOVE CODE YET.

RE-EXPORT (shim must expose): materialize_packed_bf16, apply_packed_sgd,
    apply_packed_adamw, rebalance_packed, _lamb_scale_kernel (consumed externally).

MIGRATION: STEP 7 (after state.py). ``import state as S; from constants import *``.
    Gate: L0 + L1 (CPU import/grep, doc-string presence) + the FULL L3 GPU gate
    (G1 config matrix + G2 trajectory). NOTE: the apply kernel is NOT bit-reproducible —
    a single step diverges ~70/2048 ints from identical inputs, EVEN under CUDA-graph
    replay (HW float-reduction order, 2026-06-29). So the GPU gate is NEAR-bit-exact:
    refactored stays within the baseline's per-field self-jitter envelope × margin (deploy
    fields s_slow/v_slow/weight_buf tight, s_fast loose); only the HOST gates are atol=0.
    See REFACTOR_PLAN.md §7. THIS IS THE EQUIVALENCE-VERDICT STEP.
"""
