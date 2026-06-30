"""concord_core.layers — the four self-stepping packed nn.Module layers.

RESPONSIBILITY
    The swap targets: FusedConcordLinearPackedB, ConcordLinearPackedB,
    FusedConcordConv2dPackedB, ConcordConv2dPackedB. Each owns the int32-packed
    weight storage, the bf16 cache, the slow/fast fields, and drives the kernels
    in its forward/backward. Imports torch/triton transitively via kernels.

    LIVE-BINDING DISCIPLINE (R1 + STRESS-FIX SF2 / Hole 2 — the plan scoped the
    S.-rule to kernels.py ONLY; it MUST extend to layers.py too). ``import state as S``
    and read these as LIVE S.* attributes at the verified sites — a
    ``from state import _X`` here would pass the shim L0 re-export check yet freeze
    the value at its import-time default (the L0 gate reads the SHIM namespace,
    not layers.py's internal binding):
        S._SIGMAG_NOISE   (PB:2338)
        S._SIGMAG_ISO     (PB:2341)
        S._LAZY_GATE      (PB:2356)
        S._LAZY_THRESH    (PB:2363)
        S._COH_WEIGHTED_V (PB:2382, 3069, 3311)        — also the D2 dead-path read
        S._FUSED_MATMUL   (PB:2256, 2257, 2962, 2969, 3223, 3224 — 6 sites; CLASS-2
                           direct-attr-assigned, see SF1: read via S.set_fused_matmul
                           single source, NOT a snapshot)
    NOISE PATH (Hole 5 / golden gap #1): the shipped config runs noise ON
    (swap_unet_to_winner -> set_sigmag_noise(True, isotropic=True)), so the layer
    forward hits torch.randn_like(gwf) at PB:2344 with the draw at PB:2348 EVERY
    step. This is in the LAYER FORWARD, not the kernel -> G1's kernel-branch matrix
    cannot reach it and G2 must capture torch's global RNG state (or run noise-off
    + a separate draw-order check) to cover it. See REFACTOR_PLAN.md SF2.

    CLASS-CONST RE-DECLARATIONS (KEEP — external consumer): ConcordLinearPackedB
    re-declares MANTISSA_BIAS/EXP_MIN/EXP_MAX/MAX_M on the class at PB:2479-2482;
    concord_embedding_packed.py:21-22 reads them off the class. Do NOT collapse to
    module constants.

MIGRATES FROM prototype_packed_b.py (PB) — see **REFACTOR_PLAN.md §3** for the
    AUTHORITATIVE, reconciled PB line-range map (single source of truth). The per-line
    ranges that used to be duplicated here were the original setup-task numbers against a
    4150-line PB and are SUPERSEDED — the source is now 4176 lines after the 2026-06-29
    M6a / 6-wide-boil / servo-ceiling drift. Any ``PB:NNN`` still cited elsewhere in this
    docstring (the SF2 live-read sites, the class-const re-declaration) is ILLUSTRATIVE
    only; re-verify against §3 before moving. DO NOT MOVE CODE YET.

RE-EXPORT (shim must expose): ConcordLinearPackedB, ConcordConv2dPackedB
    (the two non-fused classes are read by concord_embedding_packed + concord_ot
    swap; the Fused* pair are reached via the swap too).

MIGRATION: STEP 8 (after kernels.py — layers imports kernels).
    ``import state as S; from kernels import *; from constants import *``.
    Gate: L0 + L1 (G4-CPU load/deploy math, test_doc_deploy/format) + L3
    (G4-construct + G2 re-run, NEAR-bit-exact within the per-field envelope — the kernel
    is not bit-reproducible; see REFACTOR_PLAN.md §7).
"""
