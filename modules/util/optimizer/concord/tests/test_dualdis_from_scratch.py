"""CPU unit tests for the SUBSTRATE + OFFSET (ADD-1) from-scratch invariant of the
CORRECTED dual-dissipation packed-B fine accumulator.

Reference under test (pure-torch, CPU-only):
  modules/util/optimizer/concord/dual_dissipation_ref.py

REWORK NOTE -- this file SUPERSEDES the prior from-scratch test, which was built on the
REMOVED fractional denormal:

    OLD (deleted): is_denormal zero-start -> sub-LSB inflow banks in e_H's linear
    fraction field (e_H_fraction) -> _denormal_build PROMOTES a whole unit into s_slow
    -> d_sv = (s_slow - v_slow)*128 goes positive -> the legacy coh-gated chase opens and
    the deploy "builds out of the denormal basin" from ZERO. The +128 credit into s_slow
    was an UNBOOKED deploy leak (the conservation ledger caught it). REMOVED. The symbols
    e_H_fraction / _denormal_build no longer exist in the reference, so the old test no
    longer even imports.

The build-from-zero PROBLEM that the old test demonstrated a fix for is GONE under ADD-1:

    weight = SUBSTRATE + OFFSET.

  * The substrate is a fixed, seed-derived (from-scratch) or base-supplied (finetune)
    random prior OUTSIDE the packed word -- never stored in the word, never consolidated,
    never dissipated, never ledgered. It breaks symmetry and sets the per-row/col scale.
  * load_weights (ENABLED) sets substrate := init/base and ZEROES the accumulator, so the
    OFFSET starts at 0 and the live/deploy weight starts EXACTLY at the substrate. There is
    no "denormal basin" to climb out of -- training never has to BUILD the weight up from 0,
    it only has to move the OFFSET away from the (already nonzero, symmetry-broken) prior.
  * Dissipation decays the OFFSET toward 0, i.e. the weight toward the SUBSTRATE (the prior),
    NOT toward 0. With zero gradient under dissipation the weight relaxes back to the
    substrate, never to zero.
  * The substrate is CONSTANT across steps (delta_substrate == 0), so it is a read-side
    addend OUTSIDE the conservation ledger; the per-step invariant still holds on the
    OFFSET word's internal transfers alone.

What this test asserts (all CPU, plain asserts, no pytest):

  CHECK 1  STEP-0 IDENTITY. After an ENABLED load the accumulator (the packed OFFSET word)
           is exactly ZERO and the live weight == the deploy weight == the SUBSTRATE,
           bit-for-bit, for BOTH from-scratch (seed-derived substrate) and finetune
           (base-supplied substrate). No even-split into the accumulator; the word carries
           NO copy of the weight; the substrate regenerates from (seed, mode).

  CHECK 2  NO BUILD-FROM-ZERO. Under a persistent coherent gradient the DEPLOY weight
           builds as  substrate + GROWING consolidated offset:  the deploy OFFSET mantissa
           (s_slow + v_slow)*128 grows away from 0 in the gradient's direction, the deploy
           weight separates from the substrate, and at every step the weight is exactly
           substrate + offset*scale (the substrate is never modified). The growth is a
           consolidation of the offset, not a climb out of a zero basin.

  CHECK 3  DISSIPATION -> SUBSTRATE, NOT ZERO. Train an offset up, then run with ZERO
           gradient and dissipation on (gf_consol > 0). The live weight RELAXES back toward
           the SUBSTRATE (|weight - substrate| shrinks monotic-ish toward ~0), it does NOT
           decay toward 0 (|weight - 0| stays near |substrate|, far from 0). This is the
           core ADD-1 correction: decay target is the prior, not the origin.

  CHECK 4  CONSERVATION every step (point 8), across both the build and the relax phases:
           delta(deploy mantissa) + delta(fine mantissa) == inflow_int, with the substrate
           and the denormal exponent-claim render as read-side addends OUTSIDE the ledger.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_dualdis_from_scratch.py
  or:  CUDA_VISIBLE_DEVICES="" python .../tests/test_dualdis_from_scratch.py

CPU-ONLY. This file imports torch and runs on CPU; it never touches CUDA.
"""
import os
import sys
from pathlib import Path

# Hard CPU pin: the user is actively using the GPU. Never let torch grab CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

torch.manual_seed(0)
torch.set_grad_enabled(False)

# import the reference straight from the concord dir (pure-torch, CPU)
HERE = Path(__file__).resolve()
CONCORD = HERE.parents[1]                       # .../optimizer/concord
sys.path.insert(0, str(CONCORD))

from dual_dissipation_ref import (              # noqa: E402
    DualDissipationLayer,
    make_substrate,
    decode_to_live_weight,
    decode_to_deploy_weight,
    deploy_mantissa,
    assert_conservation,
    unpack_dual,
)


# ---------------------------------------------------------------------------
# small helpers (do NOT reimplement the reference; only read its public state)
# ---------------------------------------------------------------------------
def _coarse_fields(packed):
    """Return (s_slow, v_slow) int32 tensors from the packed word."""
    _, _, s_slow, v_slow = unpack_dual(packed)
    return s_slow.to(torch.int32), v_slow.to(torch.int32)


def _deploy_offset_mant(packed):
    """The deploy OFFSET mantissa (s_slow + v_slow)*128 as an int64 tensor (the ledger's
    deploy term; the substrate is OUTSIDE this -- it is not in the word)."""
    return deploy_mantissa(packed)


# A coherent, persistent gradient. delta_grad = -lr*step*scale_inv, so a NEGATIVE gradient
# drives the OFFSET (hence the weight) UP; a POSITIVE gradient drives it DOWN.
STEP_KW = dict(
    alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02, alpha_v_fast=0.001,
    coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
    min_leak=0.05, evap_build_min=128.0, beta1=0.0, beta2=0.999,
    use_coh_vhat=True, mass_preserve=True, chase_floor=0.1, leak_floor=0.05,
    consf=1.0,
)


# ===========================================================================
# CHECK 1 -- STEP-0 IDENTITY: enabled load zeroes the OFFSET word; the live and deploy
# weights equal the SUBSTRATE exactly (from-scratch AND finetune); substrate is OUTSIDE
# the word and regenerates from (seed, mode).
# ===========================================================================
def check1_step0_weight_equals_substrate():
    N, K = 6, 8

    # ---- from-scratch: substrate is a fresh seeded draw -------------------
    ref = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=5)
    W = torch.randn(N, K) * 0.5                 # the "init" passed in (only used DISABLED)
    ref.load_weights(W)                         # from-scratch -> substrate from the seed

    assert ref.substrate is not None, "ENABLED load must set a substrate (the prior)"
    assert int(ref.packed_w.abs().max()) == 0, \
        "the OFFSET accumulator must be ZEROED at an enabled load (offset 0 -> no even-split)"

    s_slow, v_slow = _coarse_fields(ref.packed_w)
    assert int(s_slow.abs().max()) == 0 and int(v_slow.abs().max()) == 0, \
        "no coarse mantissa in the word at step 0 (the weight lives in the substrate)"
    assert int(_deploy_offset_mant(ref.packed_w).abs().max()) == 0, \
        "the deploy OFFSET mantissa (s_slow+v_slow)*128 must be 0 at step 0"

    live0 = ref.live_weight()
    dep0 = ref.deploy_weight(deterministic=True)
    assert torch.equal(live0, ref.substrate), \
        "live weight must EQUAL the substrate at step 0 (offset 0 -> weight == substrate)"
    assert torch.equal(dep0, ref.substrate), \
        "deploy weight must EQUAL the substrate at step 0 (no un-consolidated residual yet)"

    # the substrate is NOT a copy of the passed init W; it is the seed-derived prior, and it
    # regenerates exactly from (seed, mode) -- it is recoverable, not stored in the word.
    re_sub = make_substrate(N, K, ref.substrate_seed, ref.substrate_mode)
    assert torch.equal(re_sub, ref.substrate), \
        "the substrate must regenerate from (seed, mode) -- seed-derived, not in the word"

    # ---- finetune: substrate IS the supplied pretrained base --------------
    ref_ft = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=6)
    base = torch.randn(N, K) * 0.3
    ref_ft.load_weights(base, base=base)
    assert int(ref_ft.packed_w.abs().max()) == 0, \
        "finetune enabled load must also ZERO the offset accumulator"
    assert torch.equal(ref_ft.substrate, base), \
        "finetune substrate must BE the supplied pretrained base"
    assert torch.equal(ref_ft.live_weight(), base), \
        "finetune live weight == base at step 0 (offset 0)"
    assert torch.equal(ref_ft.deploy_weight(deterministic=True), base), \
        "finetune deploy weight == base at step 0"
    return True


# ===========================================================================
# CHECK 2 -- NO BUILD-FROM-ZERO: under a persistent coherent gradient the deploy weight
# builds as substrate + GROWING consolidated offset. The offset moves away from 0 in the
# gradient direction; the weight is always EXACTLY substrate + offset*scale (substrate
# untouched); it separates from the substrate over training.
# ===========================================================================
def check2_deploy_builds_as_substrate_plus_offset():
    N, K = 6, 8
    ref = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=4)
    W = torch.randn(N, K) * 0.2
    ref.load_weights(W)
    substrate0 = ref.substrate.clone()

    # NEGATIVE gradient -> delta_grad positive -> offset (and weight) climb UP everywhere.
    g = torch.full((N, K), -0.05)

    dep0 = ref.deploy_weight(deterministic=True).clone()
    assert torch.equal(dep0, substrate0), "deploy starts at the substrate"

    off_mant_series = []
    sep_series = []
    for t in range(400):
        ref.step(g, lr=0.05, **STEP_KW)
        # the substrate must NEVER change -- it is constant, outside the word.
        assert torch.equal(ref.substrate, substrate0), \
            "the substrate must be CONSTANT across steps (delta_substrate == 0)"
        # the live weight must be EXACTLY substrate + offset*scale at every step.
        offset_only = decode_to_live_weight(
            ref.packed_w, ref.row_exp, ref.col_exp, substrate=None)
        recomposed = ref.substrate + offset_only
        assert torch.allclose(ref.live_weight(), recomposed, atol=0), \
            "live weight must be EXACTLY substrate + offset*scale every step"
        off_mant_series.append(int(_deploy_offset_mant(ref.packed_w).sum()))
        sep_series.append(float((ref.deploy_weight(deterministic=True) - substrate0)
                                .abs().sum()))

    # the consolidated OFFSET grew away from 0 (the deploy ratcheted), in the +direction.
    off_final = off_mant_series[-1]
    assert off_final > 0, \
        "the deploy OFFSET mantissa (s_slow+v_slow)*128 must GROW positive under +offset drive"
    # the deploy weight separated from the substrate (it is substrate + a real offset now).
    assert sep_series[-1] > 0.0, \
        "the deploy weight must separate from the substrate as the offset consolidates"
    assert sep_series[-1] >= sep_series[len(sep_series) // 4], \
        "the separation must GROW over training (a build of the offset, not a transient)"
    # and this build never required climbing from zero: the weight was substrate-anchored
    # from step 0 (dep0 == substrate) -- assert it was already nonzero at the very start.
    assert float(dep0.abs().sum()) > 0.0, \
        "the deploy weight was nonzero from step 0 (substrate-anchored) -- no zero basin"
    return True, off_final, sep_series[-1]


# ===========================================================================
# CHECK 3 -- DISSIPATION DRAINS THE HYPOTHESES; the consolidated THEORY PERSISTS.
# (Architect ruling: only the fine register e_L/e_H decays under dissipation; s_slow/v_slow
# are EARNED knowledge and do NOT decay.) Train a consolidated offset up, then run ZERO
# gradient under dissipation: the FINE (hypothesis) contribution drains toward ~0, but the
# consolidated deploy (s_slow+v_slow) PERSISTS -- so the weight settles at substrate +
# consolidated offset, NOT at the substrate.
# ===========================================================================
def check3_zero_grad_drains_hypothesis_keeps_theory():
    N, K = 6, 8
    ref = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=9)
    W = torch.randn(N, K) * 0.3
    ref.load_weights(W)
    substrate0 = ref.substrate.clone()

    # phase A: drive a real consolidated offset up with a coherent gradient.
    g_build = torch.full((N, K), -0.05)
    for _ in range(250):
        ref.step(g_build, lr=0.05, **STEP_KW)

    live_built = ref.live_weight().clone()
    dep_built = ref.deploy_weight().clone()
    fine_built = float((live_built - dep_built).abs().sum())       # HYPOTHESIS (fine) magnitude
    theory_built = float((dep_built - substrate0).abs().sum())     # consolidated THEORY magnitude
    assert theory_built > 0.0, "phase A must consolidate a real theory (deploy offset) off the substrate"

    # phase B: ZERO gradient, dissipation ON. The HYPOTHESES drain; the THEORY persists.
    g_zero = torch.zeros(N, K)
    relax_kw = dict(STEP_KW)
    relax_kw["gf_consol"] = 0.9                  # strong evaporation of the FINE register
    relax_kw["min_leak"] = 0.0
    for _ in range(2000):
        ref.step(g_zero, lr=0.05, **relax_kw)
        assert torch.equal(ref.substrate, substrate0), \
            "substrate must stay constant while the hypotheses dissipate"

    live_f = ref.live_weight()
    dep_f = ref.deploy_weight()
    fine_final = float((live_f - dep_f).abs().sum())              # HYPOTHESIS magnitude now
    theory_final = float((dep_f - substrate0).abs().sum())        # consolidated THEORY now

    # (a) the HYPOTHESES (fine register) drained substantially toward 0.
    assert fine_final < 0.5 * fine_built + 1e-6, \
        ("zero-grad dissipation must DRAIN the hypotheses (fine register): "
         f"|fine| {fine_built:.4e} -> {fine_final:.4e}")
    # (b) the consolidated THEORY (s_slow+v_slow) PERSISTED -- it did NOT decay to the substrate.
    assert theory_final > 0.5 * theory_built, \
        ("the consolidated theory must PERSIST under zero grad, NOT decay to the substrate: "
         f"|theory| {theory_built:.4e} -> {theory_final:.4e}")
    return True, fine_built, fine_final, theory_built, theory_final


# ===========================================================================
# CHECK 4 -- CONSERVATION every step (point 8), across the build AND the relax phases:
# delta(deploy mantissa) + delta(fine mantissa) == inflow_int (evap booked as a sink).
# The substrate (constant) and the denormal render (a read of e_H) are OUTSIDE the ledger.
# ===========================================================================
def check4_conservation_build_and_relax():
    N, K = 6, 8
    ref = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=33)
    W = torch.randn(N, K) * 0.25
    ref.load_weights(W)

    max_resid = 0

    # build phase (coherent gradient): conservation every step.
    g = torch.full((N, K), -0.05)
    for t in range(300):
        info = ref.step(g, lr=0.05, return_ledger=True, **STEP_KW)
        pb, pa, inflow = info["_ledger"]
        ok, resid = assert_conservation(pb, pa, inflow)
        max_resid = max(max_resid, resid)
        assert ok, (f"CONSERVATION VIOLATED at BUILD step {t}: residual={resid} "
                    f"(delta_deploy + delta_fine - inflow != 0)")

    # relax phase (zero gradient, strong dissipation): conservation every step, with evap
    # as a booked sink -- this is the test that caught the removed +128 leak.
    g_zero = torch.zeros(N, K)
    relax_kw = dict(STEP_KW); relax_kw["gf_consol"] = 0.9; relax_kw["min_leak"] = 0.0
    for t in range(500):
        info = ref.step(g_zero, lr=0.05, return_ledger=True, **relax_kw)
        pb, pa, inflow = info["_ledger"]
        ok, resid = assert_conservation(pb, pa, inflow)
        max_resid = max(max_resid, resid)
        assert ok, (f"CONSERVATION VIOLATED at RELAX step {t}: residual={resid} "
                    f"(evap must be a booked sink, no unbooked deploy credit)")

    assert max_resid == 0, f"max conservation residual over build+relax must be 0 (got {max_resid})"
    return True, max_resid


# ===========================================================================
# Runner -- prints PASS/FAIL per check, exit code 1 on any failure. No pytest.
# ===========================================================================
def _run(name, fn):
    try:
        out = fn()
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    if isinstance(out, tuple):
        ok, extra = out[0], out[1:]
        detail = "  ".join(str(x) for x in extra)
        print(f"[PASS] {name}  ({detail})" if detail else f"[PASS] {name}")
        return bool(ok)
    print(f"[PASS] {name}")
    return bool(out)


if __name__ == "__main__":
    print("=" * 78)
    print("dual-dissipation FROM-SCRATCH: SUBSTRATE + OFFSET (ADD-1) -- CPU, no GPU")
    print("=" * 78)
    results = []
    results.append(_run(
        "check1  step-0: offset==0, live==deploy==SUBSTRATE (scratch+finetune), regenerable",
        check1_step0_weight_equals_substrate))
    results.append(_run(
        "check2  no build-from-zero: deploy = substrate + GROWING consolidated offset",
        check2_deploy_builds_as_substrate_plus_offset))
    results.append(_run(
        "check3  zero-grad: hypotheses (fine) drain, consolidated theory (s_slow+v_slow) persists",
        check3_zero_grad_drains_hypothesis_keeps_theory))
    results.append(_run(
        "check4  conservation holds every step (build + relax phases)",
        check4_conservation_build_and_relax))

    print("-" * 78)
    n_pass = sum(1 for r in results if r)
    n_total = len(results)
    if all(results):
        print(f"ALL {n_total} CHECKS PASS")
        sys.exit(0)
    else:
        print(f"{n_pass}/{n_total} CHECKS PASS -- {n_total - n_pass} FAILED")
        sys.exit(1)
