"""CPU unit tests for the dissipation autotuner install (no GPU, no Triton launch).

Covers the two pieces added for the dimensionless-dissipation install:

  1-3  the exp-11d one-sided re-probe watchdog in DissipationAutoTuner
       (commit -> hold -> re-probe ONLY on a coherence DROP; rises are the
       friction working and must NOT trigger -- the exp-11b/c failure mode)
  4    make_concord_config passes dissipation / autotune_reprobe_band through
  5    the dimensionless arithmetic: gf_consol = lam/lr and the table kappa
       column converted by 1/lr (mirrors ConcordController / _build_autotuner)

Run with the OneTrainer venv python:
  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_autotuner_cpu.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import prototype_packed_b as ppb

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

TABLE = [(0.387, 0.0), (0.314, 100.0), (0.288, 200.0), (0.274, 400.0), (0.256, 400.0)]


def drive(tuner, schedule):
    """Run tuner.step(t) over a synthetic coherence schedule (a function t -> coh),
    monkeypatching the module-level meter. Returns the list of commit events."""
    commits = []
    orig = ppb.measure_coherence
    t_now = {"t": 0}
    ppb.measure_coherence = lambda m: schedule(t_now["t"])
    try:
        for t in range(schedule.total):
            t_now["t"] = t
            k = tuner.step(t)
            if k is not None:
                commits.append((t, k))
    finally:
        ppb.measure_coherence = orig
    return commits


def mklayers(n=3):
    return [SimpleNamespace(gf_consol=None, beta1=None) for _ in range(n)]


# ---- 1. watchdog: clean probe commits low kappa; later DROP re-probes and
# ----    recommits high; layers carry the new value -------------------------
def sched1(t):
    if t < 200:
        return 0.40           # clean stream -> commit kappa = 0
    return 0.262              # noise arrives -> deep drop -> re-probe -> ~400
sched1.total = 400

layers = mklayers()
tuner = ppb.DissipationAutoTuner(layers, probe_start=0, probe_end=30, table=TABLE,
                                 probe_kappa=50.0, measure_every=10, verbose=False,
                                 beta1_on=0.0, reprobe_band=0.02)
commits = drive(tuner, sched1)
check("1a clean probe commits kappa=0", len(commits) >= 1 and commits[0][1] == 0.0,
      f"commits={commits}")
check("1b drop triggers exactly one re-probe", tuner.reprobes == 1,
      f"reprobes={tuner.reprobes}")
check("1c re-commit lands high (>=300)", len(commits) == 2 and commits[1][1] >= 300.0,
      f"commits={commits}")
check("1d layers carry the re-committed kappa",
      all(m.gf_consol == commits[-1][1] for m in layers))

# ---- 2. one-sided: a coherence RISE must NOT re-probe ----------------------
def sched2(t):
    return 0.40 if t < 200 else 0.55      # friction working / cleaner stream
sched2.total = 400

tuner2 = ppb.DissipationAutoTuner(mklayers(), probe_start=0, probe_end=30, table=TABLE,
                                  probe_kappa=50.0, measure_every=10, verbose=False,
                                  beta1_on=0.0, reprobe_band=0.02)
commits2 = drive(tuner2, sched2)
check("2  rise does not re-probe (one-sided)",
      tuner2.reprobes == 0 and len(commits2) == 1, f"reprobes={tuner2.reprobes}")

# ---- 3. reprobe_band=None: legacy one-commit, watchdog fully inert ---------
tuner3 = ppb.DissipationAutoTuner(mklayers(), probe_start=0, probe_end=30, table=TABLE,
                                  probe_kappa=50.0, measure_every=10, verbose=False,
                                  beta1_on=0.0)
commits3 = drive(tuner3, sched1)          # same drop schedule
check("3  band=None stays one-commit through a drop",
      tuner3.reprobes == 0 and len(commits3) == 1 and tuner3._baseline is None)

# ---- 3b. watchdog arm delay: secular-relaxation drops inside the blackout are
# ----     absorbed into the baseline; real drops after it fire normally ------
def sched3b(t):
    if t < 100:
        return 0.40            # clean probe -> commit
    if t < 500:
        return 0.30            # secular fall INSIDE the blackout (telescope)
    return 0.20                # real drop after the blackout
sched3b.total = 700

tuner3b = ppb.DissipationAutoTuner(mklayers(), probe_start=0, probe_end=30, table=TABLE,
                                   probe_kappa=50.0, measure_every=10, verbose=False,
                                   beta1_on=0.0, reprobe_band=0.02, watchdog_min_t=300)
commits3b = drive(tuner3b, sched3b)
check("3b blackout drop absorbed; post-blackout drop re-probes once",
      tuner3b.reprobes == 1 and len(commits3b) == 2 and commits3b[1][0] > 500,
      f"reprobes={tuner3b.reprobes} commits={commits3b}")

# ---- 4. config plumb-through ------------------------------------------------
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer"))
from concord_ot import make_concord_config

oc = SimpleNamespace(dissipation=0.025, autotune_reprobe_band=0.02,
                     step_cap=5.0, gf_trust_delta_sq=0.5)
cfg = make_concord_config(7.5e-5, oc)
check("4a dissipation picked", cfg.dissipation == 0.025)
check("4b reprobe band picked", cfg.autotune_reprobe_band == 0.02)
check("4c step cap / trust region picked",
      cfg.step_cap == 5.0 and cfg.gf_trust_delta_sq == 0.5)
cfg_default = make_concord_config(7.5e-5, SimpleNamespace())
check("4d defaults: dissipation/band None (off), cap/trust at winner",
      cfg_default.dissipation is None and cfg_default.autotune_reprobe_band is None
      and cfg_default.step_cap == 10.0 and cfg_default.gf_trust_delta_sq == 1.0)

# ---- 5. dimensionless arithmetic (mirror of controller init / table build) --
lr = 7.5e-5
lam = 0.025
gf = lam / max(lr, 1e-12)
check("5a lam/lr: lam=0.025 @ lr=7.5e-5 -> kappa=333.3", abs(gf - 1000.0 / 3.0) < 1e-6,
      f"kappa={gf:.2f}")
conv = [(c, k / max(lr, 1e-12)) for c, k in TABLE]
check("5b table column scales by 1/lr",
      conv[1] == (0.314, 100.0 / lr) and conv[0][1] == 0.0)
check("5c stability ceiling reads in lam: lr*kappa = lam < 2",
      lr * gf == lam and lam < 2.0)

# ---- 6. guard ordering: in dissipation mode the lr*kappa<2 guard must see the
# ----    CONVERTED table — a lam >= 2 entry has lr*lam_raw ~ 1e-4 and would
# ----    slip an unstable ceiling through if the guard ran pre-conversion -----
import json
unstable = json.dumps([[0.387, 0.0], [0.256, 2.5]])      # lam ceiling 2.5 >= 2
bad = SimpleNamespace(config=SimpleNamespace(
    autotune_table=unstable, dissipation=0.025, lr=lr, gf_consol=333.0,
    alpha=0.01, warmup=0, autotune_beta1_on=0.0, autotune_beta1_coh=0.35,
    autotune_reprobe_band=None), total_steps=1000, layers=[],
    _autotune_pending=True)
from concord_ot import ConcordController
try:
    ConcordController._build_autotuner(bad)
    check("6  lam>=2 table entry raises in dissipation mode", False, "no raise")
except ValueError as e:
    check("6  lam>=2 table entry raises in dissipation mode",
          "unstable" in str(e), str(e).split(":")[0])

# ---- 6b/6c. probe placement: auto-defer past warmup/transient; disable when
# ----        a clean window cannot fit in the first half of the run ----------
def probe_rig(total_steps, warmup=100, alpha=0.1, alpha_v_fast=0.001):
    return SimpleNamespace(config=SimpleNamespace(
        autotune_table=json.dumps([[0.387, 0.0], [0.256, 0.3]]),
        dissipation=0.1, lr=lr, gf_consol=1333.0, alpha=alpha, warmup=warmup,
        alpha_v_fast=alpha_v_fast,
        autotune_beta1_on=0.0, autotune_beta1_coh=0.35,
        autotune_reprobe_band=None), total_steps=total_steps,
        layers=[SimpleNamespace(gf_consol=None, beta1=None)],
        _autotune_pending=True, autotuner=None)

rig = probe_rig(1178)          # the reported run: 4% = 47; telescope floor 500
ConcordController._build_autotuner(rig)
check("6b probe auto-defers past the telescope relaxation (47 -> 500)",
      rig.autotuner is not None and rig.autotuner.probe_start == 500
      and rig.autotuner.probe_end == 570,
      f"window=[{getattr(rig.autotuner, 'probe_start', None)},"
      f"{getattr(rig.autotuner, 'probe_end', None)})")
rig = probe_rig(220)           # deferred window cannot fit in the first half
ConcordController._build_autotuner(rig)
check("6c too-short run disables the tuner instead of mis-committing",
      rig.autotuner is None)
rig = probe_rig(20000)         # 4% = 800 >= all floors: untouched window
ConcordController._build_autotuner(rig)
check("6d long-run window untouched",
      rig.autotuner.probe_start == 800 and rig.autotuner.probe_end == 2000)

# ---- 7. shipped GUI defaults are sane: dimensionless mode on, table in lam
# ----    units (descending coh, ceiling < 2), watchdog armed ----------------
# (optimizer_util can't be imported standalone -- importing it first trips the
# create.py <-> modelSetup circular import the app avoids by import order --
# so read the literal CONCORD defaults dict out of the source via ast.)
import ast

def gui_concord_defaults():
    tree = ast.parse((OT / "modules" / "util" / "optimizer_util.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) \
                and any(getattr(t, "id", "") == "OPTIMIZER_DEFAULT_PARAMETERS" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Attribute) and k.attr == "CONCORD":
                    return {kk.value: ast.literal_eval(vv)
                            for kk, vv in zip(v.keys, v.values)}
    raise AssertionError("CONCORD defaults dict not found")

gd = gui_concord_defaults()
check("7a default dissipation is the nanoGPT winner lam",
      gd["dissipation"] == 0.025)
tab = [(float(c), float(k)) for c, k in json.loads(gd["autotune_table"])]
check("7b default table coh strictly descending",
      all(c1 > c2 for (c1, _), (c2, _) in zip(tab, tab[1:])))
check("7c default table lam ceiling stable (< 2)",
      max(k for _, k in tab) < 2.0, f"ceiling={max(k for _, k in tab)}")
check("7d default table converts to finite kappa at SDXL lr",
      max(k / 7.5e-5 for _, k in tab) == 0.4 / 7.5e-5)
check("7e watchdog armed by default", gd["autotune_reprobe_band"] == 0.02)
check("7f subsumed knobs pruned from the panel (config-file only)",
      not any(k in gd for k in
              ("gf_consol", "ratio_coh", "autotune_beta1_on", "autotune_beta1_coh")))
check("7f2 lazy pair restored to the panel (orthogonal to build threshold)",
      gd["lazy_gate"] is False and gd["lazy_active_thresh"] == 0.0001)
check("7g step cap / trust region exposed at winner values",
      gd["step_cap"] == 10.0 and gd["gf_trust_delta_sq"] == 1.0)

# ---- 8. gamma-SNR dissipation modulation (controller hook, CPU tensors) -----
import torch
from concord_ot import ConcordController

def snr_hook_rig(knee, base_committed, lr=7.5e-5, tuner=True, gf_consol=333.0):
    rig = SimpleNamespace(
        config=SimpleNamespace(autotune_gamma_snr=knee, gf_consol=gf_consol, lr=lr),
        autotuner=SimpleNamespace(committed=base_committed) if tuner else None,
        layers=[SimpleNamespace(_gf_consol_buf=torch.full((1,), -1.0)) for _ in range(3)],
        _snr_mod_announced=True, _LAM_MOD_CAP=ConcordController._LAM_MOD_CAP,
        _current_fill_ramp=1.0)
    return rig

def snr_to_alphas(snrs):
    """alphas_cumprod table such that index i has exactly snr[i] = ac/(1-ac)."""
    return torch.tensor([s / (1.0 + s) for s in snrs], dtype=torch.float64)

def run_hook(rig, snrs):
    ac = snr_to_alphas(snrs)
    ts = torch.arange(len(snrs))
    ConcordController.on_timesteps(rig, ts, ac)
    return [float(m._gf_consol_buf[0]) for m in rig.layers]

# 8a below the knee (snr <= knee) -> m = 1 -> buffers = committed base
rig = snr_hook_rig(5.0, base_committed=200.0)
bufs = run_hook(rig, [1.0, 3.0, 5.0])
check("8a below-knee batch leaves kappa at base",
      all(abs(b - 200.0) < 1e-3 for b in bufs), f"bufs={bufs}")

# 8b above the knee -> m = mean(max(1, snr/knee)); snr {5,10,15} -> (1+2+3)/3 = 2
rig = snr_hook_rig(5.0, base_committed=200.0)
bufs = run_hook(rig, [5.0, 10.0, 15.0])
check("8b above-knee batch scales kappa by mean(max(1, snr/knee))",
      all(abs(b - 400.0) < 1e-2 for b in bufs), f"bufs={bufs}")

# 8c cap: huge snr -> kappa_t clamps at LAM_MOD_CAP/lr (lam_t = 1)
rig = snr_hook_rig(5.0, base_committed=5000.0)
bufs = run_hook(rig, [500.0])
cap = 1.0 / 7.5e-5
check("8c modulated kappa caps at lam=1 (kappa = 1/lr)",
      all(abs(b - cap) < 1.0 for b in bufs), f"bufs={bufs} cap={cap:.0f}")

# 8d probe gating: tuner present but not committed -> hook is silent
rig = snr_hook_rig(5.0, base_committed=None)
bufs = run_hook(rig, [50.0])
check("8d probe window unmodulated (buffers untouched)",
      all(b == -1.0 for b in bufs))

# 8e knee None -> off entirely
rig = snr_hook_rig(None, base_committed=200.0)
bufs = run_hook(rig, [50.0])
check("8e knee=None disables the hook", all(b == -1.0 for b in bufs))

# 8f no tuner -> modulates the config base (fixed-friction run)
rig = snr_hook_rig(5.0, base_committed=None, tuner=False, gf_consol=300.0)
bufs = run_hook(rig, [10.0])      # m = 2
check("8f fixed-friction run modulates gf_consol base",
      all(abs(b - 600.0) < 1e-2 for b in bufs), f"bufs={bufs}")

# ---- 9. min-leak servo floor ------------------------------------------------
def evap_frac(lam, coh, min_leak):
    """Mirror of the kernel clamp: min(lam*(1-coh), 1 - min_leak)."""
    return min(lam * (1.0 - coh), 1.0 - min_leak)

check("9a winner regimes unaffected (clamp never binds at lam<=0.9)",
      evap_frac(0.4, 0.0, 0.1) == 0.4 and evap_frac(0.025, 0.0, 0.1) == 0.025)
check("9b lam=1 at coh=0: survival floored at min_leak",
      abs((1.0 - evap_frac(1.0, 0.0, 0.1)) - 0.1) < 1e-12)
check("9c lam=1.5 at coh=0: no negative factor (ringing removed)",
      abs((1.0 - evap_frac(1.5, 0.0, 0.1)) - 0.1) < 1e-12)
check("9d Wiener filtering intact where coh speaks (clamp inactive)",
      evap_frac(1.0, 0.5, 0.1) == 0.5)
check("9e module default and setter",
      hasattr(ppb, "set_min_leak") and ppb._MIN_LEAK == 0.1)
cfg9 = make_concord_config(7.5e-5, SimpleNamespace(min_leak=0.25))
check("9f pick-through", cfg9.min_leak == 0.25
      and make_concord_config(7.5e-5, SimpleNamespace()).min_leak == 0.1)
check("9g GUI default exposed", gui_concord_defaults()["min_leak"] == 0.1)

# ---- 10. memorization-gap meter (buffer machinery + controller sign) --------
buf = ppb._memgap_buf("cpu")
check("10a buffer zero-init and idempotent accessor",
      float(buf[0]) == 0.0 and ppb._memgap_buf("cpu") is buf)
buf += 0.25                          # stand in for the kernel's atomic adds
check("10b read returns accumulated value and resets",
      ppb.read_memgap("cpu") == 0.25 and float(buf[0]) == 0.0
      and ppb.read_memgap("cpu") == 0.0)
buf += -0.5                          # s_fast anti-aligned with grad (typical)
rig10 = SimpleNamespace(layers=[SimpleNamespace(packed_w=torch.zeros(1))])
gap = ConcordController.read_memorization_gap(rig10)
check("10c controller flips sign: deploy reads HIGHER than live",
      gap == 0.5 and float(buf[0]) == 0.0)
check("10d empty-layers path returns 0.0",
      ConcordController.read_memorization_gap(SimpleNamespace(layers=[])) == 0.0)

# ---- 11. evap build threshold (hypothesis-infancy guard) --------------------
def evap_with_build(lam, coh, d_fs, min_leak=0.1, build_min=128.0):
    """Mirror of the kernel: evaporated amount on one element."""
    frac = min(lam * (1.0 - coh), 1.0 - min_leak)
    build_ok = 1.0 if abs(d_fs) >= build_min else 0.0
    return frac * d_fs * build_ok

check("11a sub-tick velocity is never dissipated (|u| < 128)",
      evap_with_build(1.0, 0.0, 100.0) == 0.0
      and evap_with_build(1.0, 0.0, -127.0) == 0.0)
check("11b committable velocity dissipates normally (|u| >= 128)",
      abs(evap_with_build(0.4, 0.0, 200.0) - 80.0) < 1e-9
      and evap_with_build(1.0, 0.0, -130.0) == -130.0 * 0.9)
check("11c build_min=0 is bit-exact legacy",
      evap_with_build(0.4, 0.0, 1.0, build_min=0.0) == 0.4)
check("11d module default and setter",
      hasattr(ppb, "set_evap_build_min") and ppb._EVAP_BUILD_MIN == 128.0)
cfg11 = make_concord_config(7.5e-5, SimpleNamespace(evap_build_min=64.0))
check("11e pick-through", cfg11.evap_build_min == 64.0
      and make_concord_config(7.5e-5, SimpleNamespace()).evap_build_min == 128.0)
check("11f GUI default exposed", gui_concord_defaults()["evap_build_min"] == 128.0)

# ---- 12. telescope-fill dissipation ramp ------------------------------------
import math as _math
fr = ConcordController._fill_ramp
check("12a ramp 0 at t=0; 63% at tau; ~1 late",
      fr(0, 0.001) == 0.0
      and abs(fr(500, 0.001) - (1 - _math.exp(-1))) < 1e-12
      and fr(5000, 0.001) > 0.9999)
check("12b pinned anchor (alpha_v=0) -> no ramp", fr(1000, 0.0) == 1.0)
rig12 = snr_hook_rig(5.0, base_committed=200.0)
rig12._current_fill_ramp = 0.5
bufs = run_hook(rig12, [10.0])     # m = 2; base 200 * 2 * ramp 0.5 = 200
check("12c gamma-SNR composes with the fill ramp",
      all(abs(b - 200.0) < 1e-2 for b in bufs), f"bufs={bufs}")
cfg12 = make_concord_config(7.5e-5, SimpleNamespace(dissipation_fill_ramp=False))
check("12d pick-through and default ON",
      cfg12.dissipation_fill_ramp is False
      and make_concord_config(7.5e-5, SimpleNamespace()).dissipation_fill_ramp is True)
check("12e GUI default exposed", gui_concord_defaults()["dissipation_fill_ramp"] is True)

# ---- 13. boil meter (false-kill audit) ---------------------------------------
bbuf = ppb._boil_buf("cpu")
check("13a buffer zero-init and idempotent accessor",
      float(bbuf.sum()) == 0.0 and ppb._boil_buf("cpu") is bbuf)
# formula mirror: killed=(2,1) in W units, coh=(0,1); chase flow energy 15 ->
#   aligned = 1 ; total kill = 5 ; boil = 0.2 ; waste = 5/(5+15) = 0.25
for killed, coh_v in ((2.0, 0.0), (1.0, 1.0)):
    bbuf[0] += killed * killed * coh_v
    bbuf[1] += killed * killed
bbuf[2] += 15.0
a, b, c = ppb.read_boil("cpu")
check("13b energy decomposition and reset",
      a == 1.0 and b == 5.0 and c == 15.0 and float(bbuf.sum()) == 0.0)
bbuf[0] += 1.0; bbuf[1] += 5.0; bbuf[2] += 15.0
rig13 = SimpleNamespace(layers=[SimpleNamespace(packed_w=torch.zeros(1))])
boil13, waste13 = ConcordController.read_flow_audit(rig13)
check("13c controller boil and waste fractions",
      abs(boil13 - 0.2) < 1e-12 and abs(waste13 - 0.25) < 1e-12)
check("13d empty window -> (None, None); no layers -> (None, None)",
      ConcordController.read_flow_audit(rig13) == (None, None)
      and ConcordController.read_flow_audit(SimpleNamespace(layers=[])) == (None, None))
# lag-tax signature is representable: kills with NO drift recognition + small
# chase flow -> waste high, boil ~ 0
bbuf[1] += 8.0; bbuf[2] += 2.0
boil13b, waste13b = ConcordController.read_flow_audit(rig13)
check("13e lag-tax signature: high waste, boil ~ 0",
      boil13b == 0.0 and waste13b == 0.8)

# ---- 14. telescope epoch window ----------------------------------------------
def ew_rig(flag=True, av=0.001):
    return SimpleNamespace(
        config=SimpleNamespace(telescope_epoch_window=flag, alpha_v_fast=av),
        emb_cores=[], emb_delay_epochs=0.0, emb_delay_steps=0, steps_per_epoch=0.0,
        layers=[SimpleNamespace(alpha=0.1, alpha_v_fast=av, drift_cancel_C=0.0,
                                mass_preserve_v=True) for _ in range(2)])

rig14 = ew_rig()
ConcordController.apply_epoch_window(rig14, 471.0)
av_new = 1.0 / (2.0 * 471.0)
check("14a alpha_v pinned to the revisit period (1/(2*SPE))",
      abs(rig14.config.alpha_v_fast - av_new) < 1e-15
      and all(abs(m.alpha_v_fast - av_new) < 1e-15 for m in rig14.layers))
check("14b C* re-derived per layer (mass-preserve form)",
      all(abs(m.drift_cancel_C
              - ppb.compute_drift_cancel_C(0.1, av_new, mass_preserve=True)) < 1e-15
          for m in rig14.layers))
check("14c at the current SDXL size the change is ~6% (near no-op)",
      abs(av_new / 0.001 - 1.0) < 0.07)
rig14b = ew_rig(flag=False)
ConcordController.apply_epoch_window(rig14b, 471.0)
check("14d flag off -> untouched",
      rig14b.config.alpha_v_fast == 0.001
      and all(m.alpha_v_fast == 0.001 for m in rig14b.layers))
rig14c = ew_rig()
ConcordController.apply_epoch_window(rig14c, 0)
check("14e degenerate SPE -> untouched", rig14c.config.alpha_v_fast == 0.001)
cfg14 = make_concord_config(7.5e-5, SimpleNamespace(telescope_epoch_window=False))
check("14f pick-through and default ON",
      cfg14.telescope_epoch_window is False
      and make_concord_config(7.5e-5, SimpleNamespace()).telescope_epoch_window is True)
check("14g GUI default exposed", gui_concord_defaults()["telescope_epoch_window"] is True)

# ---- 15. embedding-core registration ------------------------------------------
def fake_core():
    return SimpleNamespace(gf_consol=0.0,
                           packed_w=torch.tensor([[0x7FFF1234, 0x00015678]],
                                                 dtype=torch.int32))

core15 = fake_core()
rig15 = SimpleNamespace(config=SimpleNamespace(dissipation=0.1, gf_consol=1333.0,
                                               lr=7.5e-5),
                        emb_cores=[], emb_lr=0.0, layers=[])
planes15 = [{"cp": SimpleNamespace(trainable=SimpleNamespace(core=core15))},
            {"cp": SimpleNamespace(trainable=None)}]
ConcordController.register_embedding_cores(rig15, planes15, 1e-3)
check("15a friction at the EMBEDDING lr: kappa_emb = lam/lr_emb",
      len(rig15.emb_cores) == 1 and abs(core15.gf_consol - 100.0) < 1e-9,
      f"kappa_emb={core15.gf_consol}")
# 15b: deploy bridge masks + restores embedding cores (CPU tensors; fused mode
# so the materialize branch is skipped)
_fused = ppb._FUSED_MATMUL
ppb._FUSED_MATMUL = True
try:
    before = core15.packed_w.clone()
    stash = ConcordController.materialize_unet_deploy(rig15)
    masked = core15.packed_w.clone()
    rig15._deploy_scratch = None
    ConcordController.restore_unet_deploy(rig15, stash)
    after = core15.packed_w.clone()
finally:
    ppb._FUSED_MATMUL = _fused
check("15b bridge masks embedding s_fast and restores bit-exactly",
      bool((masked == (before & 0xFFFF)).all()) and bool((after == before).all()))

# ---- 16. embedding anchor mode -----------------------------------------------
from concord_embedding_packed import ConcordPackedEmbedding

# CPU rig: the constructor/_resync paths launch the Triton materialize kernel
# (GPU-only); it only refreshes the bf16 scratch and is irrelevant to the
# bit-level assertions below -- no-op it for the test.
_orig_resync = ppb.ConcordLinearPackedB._resync_weight_buf
_orig_mat = ppb.materialize_packed_bf16
ppb.ConcordLinearPackedB._resync_weight_buf = lambda self: None
ppb.materialize_packed_bf16 = lambda *a, **k: k.get("out")

def unpack(pw):
    return (pw >> 16), ((pw << 16) >> 24), ((pw << 24) >> 24)   # s_fast, s_slow, v_slow

torch.manual_seed(0)
init_vec = torch.randn(2, 16) * 0.05
emb_a = ConcordPackedEmbedding(2, 16, device="cpu", lr=1e-3, target_norm=0.5)
emb_a.init_tokens(init=init_vec.clone(), anchor=True)
sf, ss, vs = unpack(emb_a.core.packed_w)
check("16a anchor mode: position lives in v_slow, s_slow empty",
      int(ss.abs().sum()) == 0 and int(vs.abs().sum()) > 0)
check("16b leak frozen and C* zeroed",
      emb_a.core.alpha_v_fast == 0.0 and emb_a.core.drift_cancel_C == 0.0)
dep = emb_a.deploy_weight().float()
cos = torch.nn.functional.cosine_similarity(dep, init_vec, dim=1)
check("16c deploy ~ init direction at the pinned norm",
      bool((cos > 0.95).all()) and bool(((dep.norm(dim=1) - 0.5).abs() < 0.05).all()),
      f"cos={cos.min():.3f} norms={dep.norm(dim=1).tolist()}")
before = emb_a.core.packed_w.clone()
emb_a._pin_norm(torch.arange(2))
check("16d per-step pin is a no-op under anchor (no requant churn)",
      bool((emb_a.core.packed_w == before).all()))
emb_l = ConcordPackedEmbedding(2, 16, device="cpu", lr=1e-3, target_norm=0.5)
emb_l.init_tokens(init=init_vec.clone(), anchor=False)
sf2, ss2, vs2 = unpack(emb_l.core.packed_w)
check("16e legacy mode unchanged: position in s_slow",
      int(ss2.abs().sum()) > 0 and int(vs2.abs().sum()) == 0)
ppb.ConcordLinearPackedB._resync_weight_buf = _orig_resync
ppb.materialize_packed_bf16 = _orig_mat

# ---- 17. embedding release delay (divot), auto-drive calibration, sigma globals
from concord_embedding_packed import _PackedEmbStep
from concord_winner import winner_step

# 17a-c: the shifted schedule clock (pure function -> resume-safe)
check("17a delay 0 is the identity clock",
      ConcordController._emb_clock(123, 1000, 0) == (123, 1000))
check("17b frozen during the divot (step < delay)",
      ConcordController._emb_clock(942, 1000, 943) is None
      and ConcordController._emb_clock(0, 1000, 943) is None)
check("17c released on a shifted clock: fresh warmup, cosine ends at horizon",
      ConcordController._emb_clock(943, 9432, 943) == (0, 8489)
      and ConcordController._emb_clock(9431, 9432, 943) == (8488, 8489))

# 17d: delay epochs -> steps resolution at horizon finalize (store happens
# BEFORE the telescope gate, so it works with the epoch window off)
rig17 = SimpleNamespace(config=SimpleNamespace(telescope_epoch_window=False),
                        emb_cores=[object()], emb_delay_epochs=1.0,
                        emb_delay_steps=0, steps_per_epoch=0.0, layers=[])
ConcordController.apply_epoch_window(rig17, 943)
check("17d delay resolved at finalize even with epoch window off",
      rig17.emb_delay_steps == 943 and rig17.steps_per_epoch == 943.0)

# 17e: auto-drive calibration math -- drive corrects FREQUENCY ONLY
# (median(n)/n, decade clamp); per-sighting distance D stays the data's signal.
# Regression: a CONVERGED-but-frequent token (small total A, huge n) must be
# DAMPED, not boosted -- normalizing to total change got this backwards.
class _FakeTr(SimpleNamespace):
    def set_drive(self, d):
        self.drive = d.clone()
acc = torch.zeros(4, 8)
seen = torch.zeros(4)
acc[0, 0], seen[0] = 1.0, 100.0   # converged+hot: D=0.01, n=100 -> 10/100 -> clamp 0.2
acc[1, 0], seen[1] = 1.0, 10.0    # median:        D=0.1,  n=10  -> 1.0
acc[2, 0], seen[2] = 2.0, 2.0     # rare+far:      D=1.0,  n=2   -> 10/2 = 5.0
                                  # row 3 unseen: n=0 -> stays 1, excluded from median
rigc = SimpleNamespace(emb_trainables=[_FakeTr(_accum=acc, _seen=seen, drive=None)],
                       emb_row_names=["hotconv", "mid", "rarefar", "unseen"],
                       _emb_drive_applied=False)
ConcordController._finalize_embedding_calibration(rigc)
got = rigc.emb_trainables[0].drive
check("17e calibration: frequency-normalized (converged-hot damped, rare-far "
      "boosted), unseen stays 1",
      rigc._emb_drive_applied and got is not None
      and torch.allclose(got, torch.tensor([0.2, 1.0, 5.0, 1.0])),
      f"drive={None if got is None else got.tolist()}")
rige = SimpleNamespace(emb_trainables=[_FakeTr(_accum=torch.zeros(2, 8),
                                               _seen=torch.zeros(2), drive=None)],
                       emb_row_names=["a", "b"], _emb_drive_applied=False)
ConcordController._finalize_embedding_calibration(rige)
check("17e2 empty accumulator (resumed past divot) -> drive untouched",
      rige._emb_drive_applied and rige.emb_trainables[0].drive is None)

# 17f: per-token drive buffer + backward ordering (accumulate RAW, apply SCALED)
ppb.ConcordLinearPackedB._resync_weight_buf = lambda self: None
ppb.materialize_packed_bf16 = lambda *a, **k: k.get("out")
try:
    emb17 = ConcordPackedEmbedding(2, 16, device="cpu", lr=1e-3, target_norm=0.5)
    ones_ok = bool((emb17._drive == 1.0).all())
    emb17.set_drive([1.0, 2.0])
    set_ok = emb17._drive.shape == (2, 1) and float(emb17._drive[1]) == 2.0
    try:
        emb17.set_drive([1.0])
        len_ok = False
    except ValueError:
        len_ok = True
    applied = []
    emb17.core.apply_grad_step = lambda g: applied.append(g.clone())
    emb17._anchored = True                      # one-shot pin -> no-op
    grad = torch.randn(3, 16)                   # token 1 appears TWICE this batch
    ctx = SimpleNamespace(saved_tensors=(torch.tensor([0, 1, 1]),), mod=emb17)
    _PackedEmbStep.backward(ctx, grad.clone())
    G_raw = torch.stack([grad[0], grad[1] + grad[2]])
    raw_ok = (torch.allclose(emb17._accum, G_raw)               # raw, pre-drive
              and torch.allclose(emb17._seen, torch.tensor([1.0, 2.0])))
    scaled_ok = (torch.allclose(applied[0][0], G_raw[0])
                 and torch.allclose(applied[0][1], 2.0 * G_raw[1]))
finally:
    ppb.ConcordLinearPackedB._resync_weight_buf = _orig_resync
    ppb.materialize_packed_bf16 = _orig_mat
check("17f set_drive: defaults to ones, sets [K,1], rejects wrong length",
      ones_ok and set_ok and len_ok)
check("17g backward: accumulator sees RAW grad, kernel sees drive-scaled grad",
      raw_ok and scaled_ok)

# 17h: schedule-only winner_step leaves the module globals alone (the sigma
# clobber: emb group's noise=False used to zero sigma for the whole model)
cfg17 = SimpleNamespace(lr=1e-3, warmup=5, sigmag_peak=0.6, lr_min_frac=0.05,
                        noise=True, ratio_chase_floor=0.9, ratio_chase_floor_min=0.9,
                        ratio_leak_floor=0.98, ratio_leak_floor_min=0.98)
_sig_before = ppb._SIGMAG_SIGMA
try:
    ppb.set_sigmag_sigma(0.42)
    lr_only = winner_step(10, 100, [], peak_lr=0.0, noise=False, config=cfg17,
                          update_globals=False)
    untouched = abs(ppb._SIGMAG_SIGMA - 0.42) < 1e-12
    winner_step(10, 100, [], peak_lr=0.0, noise=False, config=cfg17)
    clobbered = ppb._SIGMAG_SIGMA == 0.0
finally:
    ppb.set_sigmag_sigma(_sig_before)
check("17h update_globals=False is schedule-only (sigma survives a noise=False "
      "secondary group); default still writes", untouched and clobbered
      and lr_only == 0.0)

print()
ok = all(results)
print(f"{'ALL PASS' if ok else 'FAILURES'} ({sum(results)}/{len(results)})")
sys.exit(0 if ok else 1)
