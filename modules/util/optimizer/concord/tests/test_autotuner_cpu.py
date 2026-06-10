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
def probe_rig(total_steps, warmup=100, alpha=0.1):
    return SimpleNamespace(config=SimpleNamespace(
        autotune_table=json.dumps([[0.387, 0.0], [0.256, 0.3]]),
        dissipation=0.1, lr=lr, gf_consol=1333.0, alpha=alpha, warmup=warmup,
        autotune_beta1_on=0.0, autotune_beta1_coh=0.35,
        autotune_reprobe_band=None), total_steps=total_steps,
        layers=[SimpleNamespace(gf_consol=None, beta1=None)],
        _autotune_pending=True, autotuner=None)

rig = probe_rig(1178)          # the reported run: 4% = 47 < warmup 100
ConcordController._build_autotuner(rig)
check("6b short-run probe auto-defers past warmup (47 -> 100)",
      rig.autotuner is not None and rig.autotuner.probe_start == 100
      and rig.autotuner.probe_end == 170,
      f"window=[{getattr(rig.autotuner, 'probe_start', None)},"
      f"{getattr(rig.autotuner, 'probe_end', None)})")
rig = probe_rig(220)           # deferred window [100,113) > 110 = half the run
ConcordController._build_autotuner(rig)
check("6c too-short run disables the tuner instead of mis-committing",
      rig.autotuner is None)
rig = probe_rig(5000)          # 4% = 200 >= warmup: untouched window
ConcordController._build_autotuner(rig)
check("6d long-run window untouched",
      rig.autotuner.probe_start == 200 and rig.autotuner.probe_end == 500)

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
        _snr_mod_announced=True, _LAM_MOD_CAP=ConcordController._LAM_MOD_CAP)
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

print()
ok = all(results)
print(f"{'ALL PASS' if ok else 'FAILURES'} ({sum(results)}/{len(results)})")
sys.exit(0 if ok else 1)
