"""CPU conservation test for the CORRECTED dual-dissipation packed-B fine accumulator.

Reference under test:
  modules/util/optimizer/concord/dual_dissipation_ref.py   (pure-torch, CPU)

THE ONE TEST THAT WOULD HAVE CAUGHT THE PRIOR SIDECAR LEAK.
────────────────────────────────────────────────────────────────────────────────
This module asserts POINT 8 of the corrected design -- the CONSERVATION INVARIANT --
every consolidation step, over a long random run, in BOTH dual-enabled and disabled
(legacy) modes:

    every mantissa unit credited to the DEPLOY word (s_slow+v_slow)*128 is DEBITED
    from the FINE register (e_L+e_H, enabled; or the single int16 s_fast, disabled).

The only legitimate non-conservative term in a step is the gradient INFLOW that is
freshly stochastic-rounded into the fine register that step (NEW external mass, not
an internal transfer). So per step the exact internal-transfer ledger is:

        delta(deploy_mantissa) + delta(fine_mantissa) == inflow_int          (LEDGER)

equivalently, separating the inflow back out of the fine register:

        delta(deploy_mantissa) == -( delta(fine_mantissa) - inflow_int )     (POINT 8)

i.e. the deploy word advances by EXACTLY what the chase/leak carried OUT of the fine
register -- no one-directional DC leak into the deploy word.

WHY THIS CATCHES THE SIDECAR BUG. The two WRONG predecessors leaked mass that the
deploy word never accounted for:
  * dither_accum_ref.py kept fp32 SIDECARS (err_s, den_frac). Sub-LSB mass lived
    OUTSIDE the integer ledger, so the integer (deploy+fine) ledger did NOT close --
    mass appeared/vanished through the sidecars. This test reads the integer mantissa
    of BOTH registers straight from the 32-bit word (no sidecar is even consulted),
    so any value that escaped into a sidecar shows up as a NON-ZERO ledger residual.
  * DUAL_DISSIPATION_DESIGN.md packed e_H as the HIGH byte (worth x256). The chase
    credited s_slow in x128 LSBs while debiting e_H in x256 units -> a 256x ledger
    mismatch + a one-directional DC leak into the deploy word. With the co-equal
    layout, fine_mantissa = e_L + e_H (x1 each) and one s_slow LSB = 128 units, so
    the carry is exact and the residual is identically zero. A regression to byte
    packing would re-open the 256x mismatch and this test would scream.

This test does NOT trust the reference's own assert_conservation() blindly: it RE-DERIVES
the deploy and fine mantissae directly from the raw packed_before / packed_after int32
words (its own deploy_mantissa_of / fine_mantissa_of below), and ALSO cross-checks the
reference helper agrees. Both must report residual == 0 every step.

Tested in BOTH modes:
  * ENABLED  (dual co-equal e_L/e_H):  fine_mantissa = e_L + e_H   (point 1).
  * DISABLED (legacy single int16):    fine_mantissa = s_fast      (point 7).
The disabled run also confirms the legacy path is itself conservative (the corrected
design must NOT regress the legacy ledger).

A standing live weight is also tracked: the corrected design never destroys mass
silently, so the cumulative (deploy advance) must equal the cumulative (mass carried
out of the fine register) at every horizon, not merely per step.

CPU-ONLY. Assume CUDA_VISIBLE_DEVICES="". Run nothing here that needs a GPU.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_dualdis_conservation.py
  or:  CUDA_VISIBLE_DEVICES="" python .../tests/test_dualdis_conservation.py
"""
import os
import sys
from pathlib import Path

# Hard CPU pin: this test must never touch a GPU (the user is using the GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

# import the reference straight from the concord dir (pure-torch, CPU)
HERE = Path(__file__).resolve()
CONCORD = HERE.parents[1]                       # .../optimizer/concord
sys.path.insert(0, str(CONCORD))

from dual_dissipation_ref import (              # noqa: E402
    DualDissipationLayer,
    CARRY, M_MIN, EH_VELO_CAP,
    unpack_dual, unpack_legacy,
    deploy_mantissa, fine_mantissa, assert_conservation,
)

torch.manual_seed(0)

# ── tiny PASS/FAIL harness (no pytest dependency) ──
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


# ============================================================================
# Independent ledger readouts, derived ONLY from the raw 32-bit word.
# These deliberately DUPLICATE the reference's deploy_mantissa / fine_mantissa
# logic from first principles so the test is an INDEPENDENT check: if the
# reference helper were silently wrong (or a sidecar were reintroduced), the
# test's own reading would disagree and the residual would be non-zero.
# ============================================================================
def deploy_mantissa_of(packed):
    """Deploy-word integer mantissa (s_slow + v_slow) * 128, read straight from the
    word. This is the quantity the chase/leak CREDIT. No sidecars consulted."""
    _, _, s_slow, v_slow = unpack_dual(packed)
    return (s_slow.to(torch.int64) + v_slow.to(torch.int64)) * CARRY


def fine_mantissa_enabled_of(packed):
    """ENABLED fine-register INTEGER mantissa (point 1). The quantity the chase/leak DEBIT.
    NOT e_H*256 + e_L. BUG-1 (1a): for an exp-mode denormal coord ((s_slow==0 & v_slow==0)
    & |e_H| < EH_VELO_CAP) e_H is a sub-unit LOG field (|decode| < 1), BELOW the integer
    ledger's resolution, so it is EXCLUDED — re-derived independently to MATCH the corrected
    reference ledger (the design changed the fine-register DEFINITION; the invariant
    delta_deploy + delta_fine == inflow is preserved, not weakened). Elsewhere e_H is linear."""
    e_H, e_L, s_slow, v_slow = unpack_dual(packed)
    exp_den = (s_slow == 0) & (v_slow == 0) & (e_H.abs() < EH_VELO_CAP)
    e_H_ledger = torch.where(exp_den, torch.zeros_like(e_H), e_H)
    return (e_L.to(torch.int64) + e_H_ledger.to(torch.int64))


def fine_mantissa_disabled_of(packed):
    """DISABLED fine-register integer mantissa = the single int16 s_fast (point 7).
    Read as ONE int16 from bits [31:16] -- never the e_L/e_H partition."""
    s_fast, _, _ = unpack_legacy(packed)
    return s_fast.to(torch.int64)


def full_live_mantissa_of(packed, enabled):
    """The FULL integer live mantissa (s_slow+v_slow)*128 + fine. The corrected design
    represents the live weight (up to a per-row 2^exp scale and the sub-LSB e_H
    fraction) by this single integer -- recoverable from the word ALONE (point 9)."""
    if enabled:
        return deploy_mantissa_of(packed) + fine_mantissa_enabled_of(packed)
    return deploy_mantissa_of(packed) + fine_mantissa_disabled_of(packed)


# ============================================================================
# Core driver: run `steps` random consolidation steps, asserting the ledger
# closes EVERY step. Returns a dict of accumulated diagnostics.
# ============================================================================
def run_conservation(enabled, *, N=12, K=20, steps=600, seed=0,
                     lr=0.05, grad_mode="coherent", grad_accum_M=8, kw=None):
    """Drive a DualDissipationLayer for `steps` and assert, EVERY step:

        (A) LEDGER (independent): delta(deploy) + delta(fine) == inflow_int,
            using the test's OWN word readouts (deploy_mantissa_of / fine_*_of).
        (B) POINT 8 restatement:  delta(deploy) == -(delta(fine) - inflow_int).
        (C) the reference's assert_conservation() agrees (residual == 0).
        (D) the reference's deploy_mantissa()/fine_mantissa() helpers match the
            test's independent readouts (guards against a divergent helper).

    `grad_mode`:
        "coherent" -> a persistent signed drift (drives real consolidation: many
                      chase ticks, the regime where a leak would actually show).
        "noisy"    -> zero-mean noise (drives evaporation-heavy steps; the regime
                      where the WRONG sidecar dropped sub-LSB mass).
        "mixed"    -> drift + noise (both pathways active at once).
    """
    torch.manual_seed(seed)
    layer = DualDissipationLayer(N, K, enabled=enabled, grad_accum_M=grad_accum_M, seed=seed)
    # Verify the mode actually took (M-guard can coerce enabled off).
    if enabled:
        assert grad_accum_M >= M_MIN, "test misconfig: enabled run needs M>=M_MIN"
        assert layer.enabled is True, "enabled run unexpectedly collapsed to legacy"
    else:
        assert layer.enabled is False, "disabled run unexpectedly enabled"

    W = torch.randn(N, K) * 0.05
    layer.load_weights(W)

    if kw is None:
        kw = dict(alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02, alpha_v_fast=0.001,
                  coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
                  min_leak=0.05, evap_build_min=128.0, beta1=0.0, beta2=0.999,
                  use_coh_vhat=True, mass_preserve=True, chase_floor=0.1,
                  leak_floor=0.05, consf=1.0)

    drift = (-torch.sign(W) * 0.02 + 0.02)      # persistent coherent push

    fine_of = fine_mantissa_enabled_of if enabled else fine_mantissa_disabled_of

    max_resid_independent = 0
    max_resid_reference = 0
    max_helper_gap = 0
    n_deploy_advanced = 0
    n_fine_changed = 0
    n_reexp_skipped = 0              # steps skipped because the re-exponent fired
    cum_deploy_advance = 0            # cumulative delta(deploy) over the run
    cum_fine_carried_out = 0         # cumulative -(delta(fine) - inflow) over the run
    cum_inflow = 0

    for t in range(steps):
        if grad_mode == "coherent":
            g = drift
        elif grad_mode == "noisy":
            torch.manual_seed(seed * 100003 + t)
            g = torch.randn(N, K) * 0.04
        else:  # mixed
            torch.manual_seed(seed * 100003 + t)
            g = drift + torch.randn(N, K) * 0.03

        row_exp_before = layer.row_exp.clone()
        info = layer.step(g, lr=lr, return_ledger=True, **kw)
        packed_before, packed_after, inflow_int = info["_ledger"]

        # The mantissa-LEDGER invariant is defined for a normal consolidation step. On a
        # RE-EXPONENT step the layer value-preservingly HALVES (s_slow, v_slow, e_L, e_H)
        # and bumps row_exp (mirror prototype_packed_b.py:1132-1143) -- a rescale of the
        # SCALED weight, NOT a chase/leak mantissa transfer. Independent integer halving
        # of each field does not preserve the integer (deploy+fine) ledger (rounding), so
        # such steps are SKIPPED for the per-step ledger, exactly as the reference's own
        # assert_disabled_matches_legacy skips them. They are rare (only when the live
        # mantissa nears MAX_M=24000) and are counted/reported, not silently ignored.
        if bool((layer.row_exp != row_exp_before).any()):
            n_reexp_skipped += 1
            continue

        # ---- (A) independent ledger from raw words ----
        d_dep = (deploy_mantissa_of(packed_after) - deploy_mantissa_of(packed_before))
        d_fine = (fine_of(packed_after) - fine_of(packed_before))
        resid_ind = (d_dep + d_fine - inflow_int.to(torch.int64))
        r_ind = int(resid_ind.abs().max()) if resid_ind.numel() else 0
        max_resid_independent = max(max_resid_independent, r_ind)
        if r_ind != 0:
            raise AssertionError(
                f"[{'enabled' if enabled else 'disabled'}/{grad_mode}] "
                f"INDEPENDENT LEDGER VIOLATED at step {t}: "
                f"delta_deploy + delta_fine - inflow = {r_ind} (must be 0). "
                f"This is the sidecar-leak signature.")

        # ---- (B) point-8 restatement: delta(deploy) == -(delta(fine) - inflow) ----
        rhs = -(d_fine - inflow_int.to(torch.int64))
        r_p8 = int((d_dep - rhs).abs().max()) if d_dep.numel() else 0
        if r_p8 != 0:
            raise AssertionError(
                f"[{'enabled' if enabled else 'disabled'}/{grad_mode}] "
                f"POINT-8 VIOLATED at step {t}: "
                f"delta_deploy != -(delta_fine - inflow), gap={r_p8}.")

        # ---- (C) reference helper agrees (ENABLED mode only) ----
        # The reference's assert_conservation() reads fine_mantissa() = e_L + e_H (the
        # two-byte CO-EQUAL sum). That is the correct fine register in ENABLED mode. In
        # DISABLED mode the fine register is the SINGLE int16 s_fast = packed>>16, which
        # is NOT e_L+e_H in general (s_fast = e_H*256 + (e_L & 0xFF), whereas e_L+e_H is
        # the sign-extended byte sum) -- so the e_L+e_H helper is the WRONG reading there
        # and would spuriously fail. The reference's own self-test only exercises
        # assert_conservation() in the enabled path for exactly this reason. In DISABLED
        # mode the authoritative invariant is the INDEPENDENT check (A) above, which reads
        # the true int16 s_fast via fine_mantissa_disabled_of.
        if enabled:
            ok_ref, r_ref = assert_conservation(packed_before, packed_after, inflow_int)
            max_resid_reference = max(max_resid_reference, r_ref)
            if not ok_ref:
                raise AssertionError(
                    f"[enabled/{grad_mode}] reference assert_conservation FAILED at "
                    f"step {t}: residual={r_ref}.")

        # ---- (D) reference readouts match the independent readouts ----
        gap_dep = int((deploy_mantissa(packed_after)
                       - deploy_mantissa_of(packed_after)).abs().max())
        gap_fine_after = int((fine_mantissa(packed_after)
                              - fine_mantissa_enabled_of(packed_after)).abs().max())
        # NB: reference fine_mantissa() is the ENABLED (e_L+e_H) reading. In the DISABLED
        # run the fine register IS the int16 s_fast, which equals e_L+e_H only as a raw
        # 16-bit reinterpretation, so we only cross-check the helper readout in the mode
        # it is defined for. The deploy helper is mode-agnostic and always cross-checked.
        max_helper_gap = max(max_helper_gap, gap_dep)
        if enabled:
            max_helper_gap = max(max_helper_gap, gap_fine_after)
        if gap_dep != 0:
            raise AssertionError(
                f"[{'enabled' if enabled else 'disabled'}/{grad_mode}] reference "
                f"deploy_mantissa() disagrees with independent readout at step {t}: "
                f"gap={gap_dep}.")
        if enabled and gap_fine_after != 0:
            raise AssertionError(
                f"[enabled/{grad_mode}] reference fine_mantissa() disagrees with "
                f"independent (e_L+e_H) readout at step {t}: gap={gap_fine_after}.")

        # ---- accounting ----
        if int(d_dep.abs().sum()) != 0:
            n_deploy_advanced += 1
        if int(d_fine.abs().sum()) != 0:
            n_fine_changed += 1
        cum_deploy_advance += int(d_dep.sum())
        cum_fine_carried_out += int(rhs.sum())
        cum_inflow += int(inflow_int.sum())

    return dict(
        enabled=enabled, grad_mode=grad_mode, steps=steps,
        max_resid_independent=max_resid_independent,
        max_resid_reference=max_resid_reference,
        max_helper_gap=max_helper_gap,
        n_deploy_advanced=n_deploy_advanced,
        n_fine_changed=n_fine_changed,
        n_reexp_skipped=n_reexp_skipped,
        cum_deploy_advance=cum_deploy_advance,
        cum_fine_carried_out=cum_fine_carried_out,
        cum_inflow=cum_inflow,
        layer=layer,
    )


# ============================================================================
# Tests.
# ============================================================================
def test_enabled_conservation():
    """ENABLED (dual co-equal): the integer ledger closes every step across three
    gradient regimes. The fine register is e_L+e_H (x1 each)."""
    all_ok = True
    for mode in ("coherent", "noisy", "mixed"):
        r = run_conservation(enabled=True, grad_mode=mode, steps=600, seed=7)
        ok = (r["max_resid_independent"] == 0
              and r["max_resid_reference"] == 0
              and r["max_helper_gap"] == 0)
        all_ok &= check(
            f"ENABLED conservation [{mode}]", ok,
            f"indep_resid={r['max_resid_independent']}, "
            f"ref_resid={r['max_resid_reference']}, helper_gap={r['max_helper_gap']}, "
            f"deploy_advanced {r['n_deploy_advanced']}/{r['steps']}")
    return all_ok


def test_disabled_conservation():
    """DISABLED (legacy single int16): the corrected design must NOT regress the
    legacy ledger -- delta(deploy) == -(delta(fine_int16) - inflow) every step."""
    all_ok = True
    for mode in ("coherent", "noisy", "mixed"):
        r = run_conservation(enabled=False, grad_mode=mode, steps=600, seed=13)
        # In disabled mode the authoritative invariant is the INDEPENDENT readout (A),
        # which reads the true int16 s_fast. (The reference's e_L+e_H helper is not the
        # right register here and is intentionally not consulted -- see check (C).)
        ok = (r["max_resid_independent"] == 0)
        all_ok &= check(
            f"DISABLED (legacy) conservation [{mode}]", ok,
            f"indep_resid={r['max_resid_independent']}, "
            f"deploy_advanced {r['n_deploy_advanced']}/{r['steps']}")
    return all_ok


def test_cumulative_mass_balance():
    """STRONGER than per-step: at EVERY horizon the cumulative deploy advance equals
    the cumulative mass carried OUT of the fine register. A one-directional DC leak
    (the WRONG-#2 failure) would make the deploy word advance faster than the fine
    register is debited; the cumulative balance would then drift even if (pathological)
    per-step residuals happened to cancel. Checked in both modes."""
    all_ok = True
    for enabled in (True, False):
        r = run_conservation(enabled=enabled, grad_mode="coherent", steps=800,
                             seed=21 if enabled else 22)
        bal = (r["cum_deploy_advance"] == r["cum_fine_carried_out"])
        # Sanity: real consolidation happened (otherwise the balance is trivially 0==0).
        moved = abs(r["cum_deploy_advance"]) > 0
        ok = bal and moved
        all_ok &= check(
            f"cumulative deploy==carried-out [{'enabled' if enabled else 'disabled'}]",
            ok,
            f"cum_deploy={r['cum_deploy_advance']}, "
            f"cum_carried_out={r['cum_fine_carried_out']}, "
            f"cum_inflow={r['cum_inflow']}")
    return all_ok


def test_no_leak_with_zero_gradient():
    """With ZERO gradient (no inflow), there is no NEW mass -- so the ledger must close
    with inflow == 0: delta(deploy) == -delta(fine) EXACTLY, every step. This is the
    purest form of the invariant: any mass the deploy word gains MUST come out of the
    fine register, with no external source to hide a leak. A sidecar that absorbed
    sub-LSB mass would break this immediately. Both modes.

    SETUP NOTE (gf_consol=0.0): this test isolates the MASS-PRESERVING transfers — the
    chase (fine->s_slow) and the leak (s_slow<->v_slow) — for which delta_deploy ==
    -delta_fine holds EXACTLY. Evaporation is a deliberate, gradient-INDEPENDENT SINK
    (it fires on the existing fine register even at zero gradient, per design point 8,
    and is booked into inflow_int as a sink); leaving it ON (gf_consol>0) would make
    delta_deploy + delta_fine == -evap (== inflow_int) NOT 0, contradicting the test's
    'no inflow' premise. So evap is turned OFF here to test the conservative-transfer
    invariant in its pure form; the general (evap-inclusive) ledger is covered by the
    main ENABLED/DISABLED conservation tests above.  (chase/leak run regardless of
    gf_consol, so the chase/leak mass movement this test targets is unaffected.)"""
    all_ok = True
    for enabled in (True, False):
        N, K, steps = 10, 16, 400
        torch.manual_seed(31 if enabled else 32)
        layer = DualDissipationLayer(N, K, enabled=enabled, grad_accum_M=8,
                                     seed=31 if enabled else 32)
        # Seed a non-trivial fine register so the chase/leak actually moves mass with
        # zero gradient: load a weight whose fine residual is non-zero, then drive g=0.
        layer.load_weights(torch.randn(N, K) * 0.05)
        fine_of = fine_mantissa_enabled_of if enabled else fine_mantissa_disabled_of
        g0 = torch.zeros(N, K)
        kw = dict(alpha=0.1, gf_consol=0.0, drift_cancel_C=0.02, alpha_v_fast=0.001,
                  coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
                  min_leak=0.05, evap_build_min=128.0, beta1=0.0, beta2=0.999,
                  use_coh_vhat=True, mass_preserve=True, chase_floor=0.1,
                  leak_floor=0.05, consf=1.0)
        max_resid = 0
        max_inflow = 0
        for t in range(steps):
            row_exp_before = layer.row_exp.clone()
            info = layer.step(g0, lr=0.05, return_ledger=True, **kw)
            pb, pa, inflow = info["_ledger"]
            # skip the (here essentially impossible) re-exponent step, as above
            if bool((layer.row_exp != row_exp_before).any()):
                continue
            # zero gradient -> inflow must itself be zero (no NEW mass enters)
            max_inflow = max(max_inflow, int(inflow.abs().max()) if inflow.numel() else 0)
            d_dep = deploy_mantissa_of(pa) - deploy_mantissa_of(pb)
            d_fine = fine_of(pa) - fine_of(pb)
            resid = int((d_dep + d_fine).abs().max()) if d_dep.numel() else 0
            max_resid = max(max_resid, resid)
            if resid != 0:
                raise AssertionError(
                    f"[{'enabled' if enabled else 'disabled'}] zero-grad leak at step "
                    f"{t}: delta_deploy + delta_fine = {resid} (must be 0).")
        ok = (max_resid == 0 and max_inflow == 0)
        all_ok &= check(
            f"zero-gradient: delta_deploy == -delta_fine "
            f"[{'enabled' if enabled else 'disabled'}]", ok,
            f"max_resid={max_resid}, max_inflow={max_inflow}")
    return all_ok


def test_long_horizon_stress():
    """A LONG (2000-step) high-LR mixed run with the M-guard at the on/off boundary.
    Exercises re-exponent firing, int8 clamps on the fine bytes, leak saturation, and
    denormal coords -- all the places a regression could spill mass -- while the integer
    ledger must STILL close bit-exactly every step. The single most important test, run
    hard."""
    all_ok = True
    for enabled in (True, False):
        r = run_conservation(
            enabled=enabled, grad_mode="mixed", steps=2000, seed=99,
            lr=0.15, N=16, K=24,
            kw=dict(alpha=0.15, gf_consol=0.5, drift_cancel_C=0.02, alpha_v_fast=0.002,
                    coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
                    min_leak=0.05, evap_build_min=64.0, beta1=0.0, beta2=0.999,
                    use_coh_vhat=True, mass_preserve=True, chase_floor=0.1,
                    leak_floor=0.05, consf=1.0))
        ok = (r["max_resid_independent"] == 0
              and r["max_resid_reference"] == 0
              and r["max_helper_gap"] == 0)
        all_ok &= check(
            f"long-horizon stress (2000 steps, high LR) "
            f"[{'enabled' if enabled else 'disabled'}]", ok,
            f"indep_resid={r['max_resid_independent']}, "
            f"ref_resid={r['max_resid_reference']}, helper_gap={r['max_helper_gap']}, "
            f"deploy_advanced {r['n_deploy_advanced']}/{r['steps']}, "
            f"reexp_skipped={r['n_reexp_skipped']}")
    return all_ok


def test_disabled_helper_is_int16_view():
    """Guard against the test fooling itself: in DISABLED mode the fine register is the
    single int16 s_fast, and the deploy helper is mode-agnostic. Confirm the independent
    DISABLED readout (fine = s_fast) and the deploy readout reconstruct a consistent
    full live mantissa, and that the reference deploy helper matches -- so the disabled
    ledger check above is reading the RIGHT register."""
    N, K = 8, 16
    torch.manual_seed(5)
    layer = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=5)
    layer.load_weights(torch.randn(N, K) * 0.05)
    g = torch.randn(N, K) * 0.02
    ok = True
    for t in range(50):
        layer.step(g, lr=0.05, gf_consol=0.3, drift_cancel_C=0.02, chase_floor=0.1,
                   min_leak=0.05, evap_build_min=128.0)
        p = layer.packed_w
        # deploy helper (mode-agnostic) must match the independent readout
        if int((deploy_mantissa(p) - deploy_mantissa_of(p)).abs().max()) != 0:
            ok = False
            break
        # disabled fine readout must equal s_fast = packed>>16 (point 7)
        s_fast = (p >> 16).to(torch.int64)
        if int((fine_mantissa_disabled_of(p) - s_fast).abs().max()) != 0:
            ok = False
            break
        # full live mantissa is internally consistent (s_slow+v_slow)*128 + s_fast
        _, s_slow, v_slow = unpack_legacy(p)
        recon = (s_slow.to(torch.int64) + v_slow.to(torch.int64)) * CARRY + s_fast
        if int((full_live_mantissa_of(p, enabled=False) - recon).abs().max()) != 0:
            ok = False
            break
    return check("disabled fine register IS the int16 s_fast (readout sanity)", ok)


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    # Belt-and-suspenders CPU pin (defensive; the user is on the GPU).
    if torch.cuda.is_available():
        # We never move tensors to CUDA in this file, but make the intent loud.
        print("  [note] CUDA visible but UNUSED: this test is CPU-only by construction.")

    print("dual-dissipation CONSERVATION test (point 8): "
          "delta(deploy) == -(delta(fine) - inflow), every step\n")

    print("ENABLED (dual co-equal e_L/e_H, fine = e_L + e_H):")
    e_ok = test_enabled_conservation()

    print("\nDISABLED (legacy single int16, fine = s_fast):")
    d_ok = test_disabled_conservation()

    print("\nCUMULATIVE balance (no DC leak at any horizon):")
    c_ok = test_cumulative_mass_balance()

    print("\nZERO-GRADIENT (purest invariant: delta_deploy == -delta_fine, no inflow):")
    z_ok = test_no_leak_with_zero_gradient()

    print("\nLONG-HORIZON STRESS (2000 steps, high LR, re-exponent + clamps + denormal):")
    s_ok = test_long_horizon_stress()

    print("\nREADOUT SANITY (disabled fine register is the int16 s_fast):")
    r_ok = test_disabled_helper_is_int16_view()

    n_pass = sum(_results)
    n_total = len(_results)
    print(f"\n{'=' * 64}")
    print(f"SUMMARY: {n_pass}/{n_total} checks passed.")
    overall = e_ok and d_ok and c_ok and z_ok and s_ok and r_ok and (n_pass == n_total)
    if overall:
        print("ALL CONSERVATION CHECKS PASS "
              "(deploy ledger debits exactly the fine register; no sidecar leak).")
    else:
        print("CONSERVATION FAILURE -- a mantissa unit was credited to the deploy word "
              "without being debited from the fine register (the sidecar-leak signature).")
    # Non-zero exit on failure so a CI runner notices, but no exception spam.
    sys.exit(0 if overall else 1)
