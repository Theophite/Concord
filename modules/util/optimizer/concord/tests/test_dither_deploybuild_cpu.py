"""CPU unit test for the dither-accum redesign HEADLINE (no GPU, no Triton).

THE HEADLINE (the whole reason the redesign exists, design doc DITHER_ACCUM_DESIGN.md
§1 + §3.5 + self-test 3): under a SUSTAINED COHERENT update with an EVAPORATION DRAIN
acting on the fine residual, the NEW sigma-delta deploy carry ADVANCES the deploy
weight ~every step (it TRACKS the inflow), whereas the LEGACY "climb-to-128" chase
STALLS under the SAME drain -- the residual `s_fast` never crosses the 128 carry
threshold (and with a near-zero coherence gate the chase never bootstraps at all), so
the coarse/deploy word never moves.

We model BOTH paths from the SAME init, SAME static coherent gradient, SAME evap drain
(`gf_consol`), differing ONLY in the carry mechanism:

  * NEW    = DitherAccumRef(dither_enabled=True).step(..., chase_floor>0)
             -- the reference under test; sigma-delta carry with a chase-gate FLOOR.
  * LEGACY = dither_accum_ref._legacy_step(..., chase_floor=0)
             -- the reference's OWN literal single-int16-s_fast legacy stepper, pure
                coherence chase gate (no floor), exactly the kernel path the redesign
                replaces.

deploy-lag := steps / (number of steps on which the DEPLOY weight changed). lag==1
means it ratchets every step (perfect tracking); lag==steps means a full stall.
We quantify deploy-lag(NEW) << deploy-lag(LEGACY).

Everything runs on the reference's PUBLIC surface (DitherAccumRef.step / .deploy_weight
/ .load_weights) plus its own `_legacy_step` helper -- no re-derivation of the kernel.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_dither_deploybuild_cpu.py
  or: CUDA_VISIBLE_DEVICES="" python modules/util/optimizer/concord/tests/test_dither_deploybuild_cpu.py
"""
import sys
from pathlib import Path

import torch

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

from dither_accum_ref import (
    DitherAccumRef,
    _legacy_step,
    unpack_word,
    _scale_fwd,
    CARRY,
)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Shared regime: SUSTAINED COHERENT update + EVAP DRAIN, identical for both paths.
#   * g is a constant positive drift -> every coord sees a consistent-sign inflow
#     EVERY step (a sustained coherent update); it is never exactly 0 for any coord.
#   * gf_consol>0 turns on the legacy evaporation that drains the fine residual
#     (evap_frac = lr*gf_consol*(1-coh_raw)), the exact mechanism the design doc
#     §1 blames for the legacy deploy stall.
# ─────────────────────────────────────────────────────────────────────────────
SEED = 7
N, K = 8, 16
STEPS = 300
GF_CONSOL = 1.0           # the SAME evaporation drain for BOTH paths
DRIFT_C = 0.02            # the SAME drift_cancel_C for BOTH paths

torch.manual_seed(42)
W0 = torch.randn(N, K) * 0.05
G = torch.ones(N, K) * 0.012     # sustained coherent drift, consistent sign, never 0

# kwargs shared by both steppers (only the carry mechanism / chase_floor differs)
COMMON = dict(
    alpha=0.1, gf_consol=GF_CONSOL, drift_cancel_C=DRIFT_C, coh_kappa=1.0,
    min_leak=0.0, evap_build_min=128.0,
)
# extra kwargs the bare _legacy_step needs spelled out (DitherAccumRef.step defaults them)
LEGACY_EXTRA = dict(
    alpha_v_fast=0.001, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
    beta2=0.999, use_coh_vhat=True, mass_preserve=True, leak_floor=0.0,
)


def _deploy_from_packed(packed, row_exp, col_exp):
    """Pure-coarse legacy deploy weight (s_slow+v_slow)*128*scale, for the legacy
    path which has no DitherAccumRef wrapper. Matches deploy_weight() with
    den_frac_bits==0 (the disabled/legacy deploy)."""
    _, s_slow, v_slow = unpack_word(packed)
    return (s_slow + v_slow).to(torch.float32) * CARRY * _scale_fwd(row_exp, col_exp)


# ── confirm the regime really is a sustained coherent inflow (precondition) ──
check("regime: sustained coherent inflow -- every coord has a nonzero, fixed-sign drift",
      bool((G != 0).all()) and bool((G.sign() == G.sign()[0, 0]).all()),
      f"g_const={float(G[0,0]):.3f}, all coords same sign, none zero")


# ─────────────────────────────────────────────────────────────────────────────
# NEW path: DitherAccumRef.step with a chase-gate FLOOR (the sigma-delta carry).
# ─────────────────────────────────────────────────────────────────────────────
new = DitherAccumRef(N, K, dither_enabled=True, seed=SEED)
new.load_weights(W0)
dep0_new = new.deploy_weight().clone()
prev_new = dep0_new.clone()
adv_new = 0
sfast_max_new = 0
for t in range(STEPS):
    info = new.step(G, lr=0.05, chase_floor=0.1, **COMMON)
    cur = new.deploy_weight()
    if float((cur - prev_new).abs().sum()) > 0.0:
        adv_new += 1
    prev_new = cur.clone()
    sfast_max_new = max(sfast_max_new, info["s_fast_abs_max"])
moved_new = float((new.deploy_weight() - dep0_new).abs().sum())
lag_new = STEPS / max(adv_new, 1)


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY path: the reference's OWN _legacy_step (climb-to-128 chase, NO floor),
# driven from the SAME init / SAME g / SAME evap drain.
# ─────────────────────────────────────────────────────────────────────────────
leg = DitherAccumRef(N, K, dither_enabled=False, seed=SEED)
leg.load_weights(W0)
packed = leg.packed_w.clone()
row_exp = leg.row_exp.clone()
col_exp = leg.col_exp.clone()
v_row = leg.v_row.clone()
v_col = leg.v_col.clone()
dep0_leg = _deploy_from_packed(packed, row_exp, col_exp).clone()
prev_leg = dep0_leg.clone()
adv_leg = 0
sfast_mean_noevap_end = None      # for the residual-drain reality check below
for t in range(1, STEPS + 1):
    packed, v_row, v_col = _legacy_step(
        packed, row_exp, col_exp, G, 0.05,
        v_row=v_row, v_col=v_col, step=t, seed=SEED,
        chase_floor=0.0, **COMMON, **LEGACY_EXTRA,
    )
    cur = _deploy_from_packed(packed, row_exp, col_exp)
    if float((cur - prev_leg).abs().sum()) > 0.0:
        adv_leg += 1
    prev_leg = cur.clone()
moved_leg = float((_deploy_from_packed(packed, row_exp, col_exp) - dep0_leg).abs().sum())
sfast_mean_leg_end = float(unpack_word(packed)[0].abs().to(torch.float32).mean())
lag_leg = STEPS / max(adv_leg, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Residual-drain reality check: re-run the LEGACY path with NO evap (gf_consol=0)
# and confirm the evap genuinely SHRINKS the fine residual s_fast (so "evap drain
# on the residual" is a real, measured effect, not a degenerate no-inflow case).
# ─────────────────────────────────────────────────────────────────────────────
leg2 = DitherAccumRef(N, K, dither_enabled=False, seed=SEED)
leg2.load_weights(W0)
p2 = leg2.packed_w.clone()
re2 = leg2.row_exp.clone(); ce2 = leg2.col_exp.clone()
vr2 = leg2.v_row.clone(); vc2 = leg2.v_col.clone()
NOEVAP = dict(alpha=0.1, gf_consol=0.0, drift_cancel_C=DRIFT_C, coh_kappa=1.0,
              min_leak=0.0, evap_build_min=128.0)
for t in range(1, STEPS + 1):
    p2, vr2, vc2 = _legacy_step(p2, re2, ce2, G, 0.05, v_row=vr2, v_col=vc2,
                                step=t, seed=SEED, chase_floor=0.0,
                                **NOEVAP, **LEGACY_EXTRA)
sfast_mean_noevap_end = float(unpack_word(p2)[0].abs().to(torch.float32).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Report + asserts
# ─────────────────────────────────────────────────────────────────────────────
print("== HEADLINE: sustained coherent update + evap drain ==")
print(f"   NEW   : advanced {adv_new}/{STEPS}  deploy_moved={moved_new:.4e}  "
      f"deploy-lag={lag_new:.3f}  max|s_fast|={sfast_max_new}")
print(f"   LEGACY: advanced {adv_leg}/{STEPS}  deploy_moved={moved_leg:.4e}  "
      f"deploy-lag={lag_leg:.1f}  end mean|s_fast|={sfast_mean_leg_end:.1f}")
print(f"   evap-drain on residual (LEGACY end mean|s_fast|): "
      f"no-evap={sfast_mean_noevap_end:.1f} -> evap={sfast_mean_leg_end:.1f}")

# 1. The evap genuinely drains the fine residual (the design doc's stall mechanism
#    is operative, not vacuous): with the drain on, the legacy residual is much
#    smaller than with it off.
check("evap drain is real: legacy end mean|s_fast| is much smaller WITH the drain than without",
      sfast_mean_noevap_end > 3.0 * sfast_mean_leg_end and sfast_mean_leg_end < sfast_mean_noevap_end,
      f"no-evap={sfast_mean_noevap_end:.1f} > 3x evap={sfast_mean_leg_end:.1f}")

# 2. NEW TRACKS: the deploy advances on (nearly) every step under the active drain.
check("NEW deploy TRACKS the coherent inflow: advances on >250/300 steps",
      adv_new > 250, f"advanced {adv_new}/{STEPS}, lag={lag_new:.3f}")
check("NEW deploy actually moved a nonzero amount under the drain",
      moved_new > 0.0, f"deploy_moved={moved_new:.4e}")
check("NEW deploy-lag ~= 1 (ratchets ~every step)",
      lag_new < 1.2, f"lag_new={lag_new:.3f}")

# 3. LEGACY STALLS under the SAME drain: the climb-to-128 chase never carries, so the
#    deploy word never net-advances (the residual is held below the 128 threshold /
#    the coherence gate never bootstraps). We allow a sub-LSB-scale SR flicker (a lone
#    boundary carry on ~1 step across seeds) -- the claim is a STALL, not perfect
#    immobility -- but the legacy motion is >=1000x smaller than the NEW tracking motion.
check("LEGACY deploy STALLS under the same drain: net deploy motion is >=1000x smaller than NEW",
      moved_leg <= moved_new / 1000.0, f"deploy_moved_legacy={moved_leg:.4e} vs new={moved_new:.4e}")
check("LEGACY deploy advanced on FEW steps (stall)",
      adv_leg < 30, f"advanced {adv_leg}/{STEPS}, lag={lag_leg:.1f}")

# 4. THE QUANTIFIED CONTRAST: deploy-lag(NEW) << deploy-lag(LEGACY).
check("deploy-lag(NEW) << deploy-lag(LEGACY): legacy lag is >= 30x the new lag",
      lag_leg >= 30.0 * lag_new,
      f"lag_legacy={lag_leg:.1f} >= 30x lag_new={lag_new:.3f}  (ratio={lag_leg/lag_new:.1f}x)")


n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} dither-deploybuild CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
