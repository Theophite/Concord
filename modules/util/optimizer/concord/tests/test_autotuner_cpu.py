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

oc = SimpleNamespace(dissipation=0.025, autotune_reprobe_band=0.02)
cfg = make_concord_config(7.5e-5, oc)
check("4a dissipation picked", cfg.dissipation == 0.025)
check("4b reprobe band picked", cfg.autotune_reprobe_band == 0.02)
cfg_default = make_concord_config(7.5e-5, SimpleNamespace())
check("4c defaults are None (off)",
      cfg_default.dissipation is None and cfg_default.autotune_reprobe_band is None)

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

print()
ok = all(results)
print(f"{'ALL PASS' if ok else 'FAILURES'} ({sum(results)}/{len(results)})")
sys.exit(0 if ok else 1)
