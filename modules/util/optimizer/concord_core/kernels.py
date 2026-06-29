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

MIGRATES FROM prototype_packed_b.py (PB) — DO NOT MOVE CODE YET; line-range map only:
    PB:105-117    _hash_uniform
    PB:120-170    _materialize_packed_bf16_kernel + materialize_packed_bf16
    PB:202-323    _FUSED_AUTOTUNE_CONFIGS, _fused_packed_linear_kernel /
                  _fused_packed_gradx_kernel + fused_packed_linear / fused_packed_gradx
    PB:326-466    _apply_packed_sgd_kernel
    PB:566-598    apply_packed_sgd
    PB:626-642    _lamb_scale_kernel
    PB:645-1180   _apply_packed_adamw_kernel  (THE monster; doc-grepped strings live
                  at PB:792-794, 805-823, 876-898 — KEEP byte-identical)
    PB:1183-1266  _DENOM_DIAG, _denom_diagnostic  (gated enabled=False; D6: move
                  with gate INTACT, do not delete)
    PB:1785-1987  apply_packed_adamw  (the launcher; reads S.* live — see above)
    PB:1995-2226  _rebalance_packed_decide_kernel + rebalance_packed
                  (reads S._REB_STATS — Hole 3)

RE-EXPORT (shim must expose): materialize_packed_bf16, apply_packed_sgd,
    apply_packed_adamw, rebalance_packed, _lamb_scale_kernel (consumed externally).

MIGRATION: STEP 7 (after state.py). ``import state as S; from constants import *``.
    Gate: L0 + L1 (CPU import/grep, doc-string presence) + the FULL L3 GPU gate
    (G1 config matrix at atol=0 + G2 trajectory at atol=0). THIS IS THE
    BIT-EXACTNESS VERDICT STEP.
"""
