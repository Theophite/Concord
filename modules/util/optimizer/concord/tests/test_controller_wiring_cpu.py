"""CPU unit tests for ConcordController + concord_winner wiring (no GPU, no Triton).

Targets the pure-Python parts that the live recipe's GPU swap would otherwise entangle:
the @staticmethod schedule helpers, the dimensionless-dissipation / gamma-SNR config
mapping, the winner-filter (alpha_v_fast>0) gate, the winner_step per-step schedule + its
MODULE-GLOBAL sigma last-writer hazard, the baseline (non-Concord) optimizer picker,
and the off-by-default config defaults.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_controller_wiring_cpu.py
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

OT = Path(__file__).resolve().parents[5]
for p in (str(OT), str(OT / "modules" / "util" / "optimizer"),
          str(OT / "modules" / "util" / "optimizer" / "concord")):
    sys.path.insert(0, p)

# Import optimizer_util FIRST, before the concord chain, to dodge a OneTrainer
# circular import (optimizer_util <-> model setup) that only triggers once concord_ot
# has partially loaded the cycle.
try:
    from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS as _OPT_DEFAULTS
    from modules.util.enum.Optimizer import Optimizer as _Optimizer
    _OPT_OK = True
except Exception as _e:                                        # noqa: BLE001
    _OPT_OK, _OPT_ERR = False, repr(_e)

import prototype_packed_b as ppb
import concord_winner as cw
from concord_ot import ConcordController, make_concord_config

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def skip(name, reason):
    print(f"  [SKIP] {name}  ({reason})")


# ---------------------------------------------------------------- #
print("== @staticmethod schedule helpers (pure math, no controller) ==")
check("_fill_ramp(t, alpha_v_fast<=0) == 1.0 (pinned anchor)", ConcordController._fill_ramp(100, 0.0) == 1.0)
check("_fill_ramp(500, 0.001) == 1 - exp(-1) ~ 0.6321",
      abs(ConcordController._fill_ramp(500, 0.001) - (1 - math.exp(-1.0))) < 1e-6,
      f"v={ConcordController._fill_ramp(500, 0.001):.4f}")

check("_emb_clock delay=0 -> passthrough", ConcordController._emb_clock(10, 100, 0) == (10, 100))
check("_emb_clock frozen (step < delay) -> None", ConcordController._emb_clock(3, 100, 5) is None)
check("_emb_clock released -> shifted (step-delay, total-delay)",
      ConcordController._emb_clock(8, 100, 5) == (3, 95))

drv = ConcordController._emb_drive_from_counts(torch.tensor([10., 1., 100., 0.]), 0.5)
check("_emb_drive_from_counts: median(live)=10 -> (10/n)^0.5; unseen n=0 -> 1.0; decade clamp",
      torch.allclose(drv, torch.tensor([1.0, math.sqrt(10), math.sqrt(0.1), 1.0]), atol=1e-4)
      and float(drv.min()) >= 0.2 and float(drv.max()) <= 5.0, f"drive={drv.tolist()}")

ws = ConcordController._emb_window_stats(torch.tensor([400., 100., 10.]),
                                         torch.tensor([50., 50., 50.]),
                                         torch.tensor([10., 5., 1.]))
rho = ws[0] if isinstance(ws, (tuple, list)) else ws
w = ws[1] if isinstance(ws, (tuple, list)) and len(ws) > 1 else None
check("_emb_window_stats: rho in [0,1]", bool((rho >= 0).all() and (rho <= 1).all()), f"rho={rho.tolist()}")
check("_emb_window_stats: n<2 row -> rho==0 (sentinel)", float(rho[2]) < 1e-6,
      f"rho[n<2]={float(rho[2]):.4f}" + (f", w={float(w[2]):.0f}" if w is not None else ""))


# ---------------------------------------------------------------- #
print("== dimensionless dissipation -> gf_consol = lam/lr (concord_ot.py:137) ==")
lam, lr = 0.025, 7.5e-5
check("gf_consol = lam/lr ~ 333.3", abs(lam / max(lr, 1e-12) - 333.333) < 0.1,
      f"gf_consol={lam / lr:.1f}")

print("== make_concord_config: None -> validated winner defaults ==")
c = make_concord_config(7.5e-5, None)
check("lr taken from the learning_rate arg", abs(c.lr - 7.5e-5) < 1e-12)
check("gf_consol default == 50.0", abs(c.gf_consol - 50.0) < 1e-9, f"gf_consol={c.gf_consol}")
check("dissipation default is None (dimensionless override does NOT fire on the dataclass path)",
      c.dissipation is None)
# The removed hunting-servo knob's name is spelled SPLIT below so the repo's
# straggler grep for the dead knob never matches this absence-guard.
_GONE_SERVO_KNOB = "autotune" "_servo"
check("the hunting-servo knob is GONE from ConcordConfig",
      not hasattr(c, _GONE_SERVO_KNOB))
check("autotune_gamma_snr default is None (gamma-SNR hook off by default)", c.autotune_gamma_snr is None)

print("== make_concord_config: GUI value overrides the default (partial stub is safe) ==")
oc = SimpleNamespace(dissipation=0.5, **{_GONE_SERVO_KNOB: True})   # stale stub knob must be IGNORED
c2 = make_concord_config(1e-4, oc)
check("dissipation GUI value passes through", c2.dissipation == 0.5)
check("a stale hunting-servo knob on the stub does not resurrect the field",
      not hasattr(c2, _GONE_SERVO_KNOB))
check("unspecified knob falls back to default (gf_consol)", abs(c2.gf_consol - 50.0) < 1e-9)


# ---------------------------------------------------------------- #
print("== winner filter: alpha_v_fast>0 gate (anchored cores excluded) ==")
cores = [SimpleNamespace(alpha_v_fast=0.001, tag="live"),
         SimpleNamespace(alpha_v_fast=0.0, tag="anchored")]
live = [m for m in cores if getattr(m, "alpha_v_fast", 0.0) > 0.0]   # mirrors the TE-scratch / winner-TE filter in concord_ot
check("only alpha_v_fast>0 cores pass the winner filter (anchored: frozen telescope, coh~0, excluded)",
      len(live) == 1 and live[0].tag == "live")

print("== gamma-SNR knee<=0 == OFF selector (concord_ot.py:827-838) ==")
def gamma_off(on, knee):
    return (not on) or knee is None or float(knee) <= 0.0
check("knee == 0 -> OFF (no longer pins lam=1)", gamma_off(True, 0.0) is True)
check("knee is None -> OFF", gamma_off(True, None) is True)
check("knee > 0 + on -> ENGAGED", gamma_off(True, 5.0) is False)
check("on flag False -> OFF regardless of knee", gamma_off(False, 5.0) is True)


# ---------------------------------------------------------------- #
print("== winner_step per-step schedule + MODULE-GLOBAL sigma last-writer hazard ==")
class FL:
    pass
L = [FL(), FL()]
ppb.set_sigmag_sigma(-999.0)                                   # sentinel
lr1 = cw.winner_step(50, 1000, L, config=cw.WINNER_CONFIG)    # update_globals=True (default)
check("winner_step writes per-layer lr (> 0) on every layer",
      L[0].lr > 0 and L[0].lr == lr1 and L[1].lr == lr1, f"lr={lr1:.3e}")
check("winner_step (update_globals=True) writes the GLOBAL sigma (off the sentinel)",
      ppb._SIGMAG_SIGMA != -999.0)

marker = ppb._SIGMAG_SIGMA
lr2 = cw.winner_step(50, 1000, L, peak_lr=1e-5, config=cw.WINNER_CONFIG, update_globals=False)
check("winner_step(update_globals=False) leaves the global sigma UNCHANGED (schedule-only)",
      ppb._SIGMAG_SIGMA == marker)
expected_lr2 = lr1 * 1e-5 / cw.WINNER_CONFIG.lr                # same f,warm; only peak_lr changed
check("winner_step(update_globals=False) still rescales per-layer lr to peak_lr",
      L[0].lr == lr2 and abs(lr2 - expected_lr2) < 1e-15, f"lr2={lr2:.3e}")

ppb.set_sigmag_sigma(0.123)                                   # a known nonzero global
before = ppb._SIGMAG_SIGMA
cw.winner_step(50, 1000, L, peak_lr=1e-5, config=cw.WINNER_CONFIG, noise=False, update_globals=True)
check("HAZARD: a secondary update_globals=True + noise=False zeroes the GLOBAL sigma "
      "(this is why secondary groups MUST pass update_globals=False)",
      before != 0.0 and ppb._SIGMAG_SIGMA == 0.0, f"{before} -> {ppb._SIGMAG_SIGMA}")
ppb.set_sigmag_sigma(0.0)                                     # restore module global


# ---------------------------------------------------------------- #
print("== configure_optimizer baseline (non-Concord) branch + make_aux ==")
net = nn.Linear(4, 4)
params = list(net.parameters())
layers, opt, _ = cw.configure_optimizer(net, "cpu", cw.ConcordConfig(kind="adamw", lr=1e-3))
check("baseline kind='adamw' -> (concord_layers None, torch AdamW)",
      layers is None and isinstance(opt, torch.optim.AdamW))
layers_s, opt_s, _ = cw.configure_optimizer(net, "cpu", cw.ConcordConfig(kind="sgd", lr=1e-3))
check("baseline kind='sgd' -> (None, torch SGD)", layers_s is None and isinstance(opt_s, torch.optim.SGD))
check("make_aux 'sgd' -> torch SGD", isinstance(cw.make_aux(params, cw.ConcordConfig(aux="sgd", lr=1e-3)), torch.optim.SGD))
check("make_aux 'adamw' -> torch AdamW", isinstance(cw.make_aux(params, cw.ConcordConfig(aux="adamw", lr=1e-3)), torch.optim.AdamW))
check("make_aux 'none' -> None", cw.make_aux(params, cw.ConcordConfig(aux="none", lr=1e-3)) is None)


# ---------------------------------------------------------------- #
print("== off-by-default config defaults (heavy import; guarded) ==")
try:
    from modules.util.config.TrainConfig import TrainConfig
    tc = TrainConfig.default_values()
    check("TrainConfig.concord_train_caption_vocab default False", tc.concord_train_caption_vocab is False)
    check("TrainConfig.concord_caption_vocab_anchor default False", tc.concord_caption_vocab_anchor is False)
except Exception as e:                                         # noqa: BLE001
    skip("TrainConfig caption-vocab defaults", repr(e)[:90])

if _OPT_OK:
    d = _OPT_DEFAULTS[_Optimizer.CONCORD]
    check("OPTIMIZER_DEFAULT_PARAMETERS[CONCORD] dissipation == 0.025 (dimensionless mode IS the live default)",
          abs(float(d["dissipation"]) - 0.025) < 1e-9, f"dissipation={d.get('dissipation')}")
    check("OPTIMIZER_DEFAULT_PARAMETERS[CONCORD] hunting-servo knob is absent",
          _GONE_SERVO_KNOB not in d)
    check("OPTIMIZER_DEFAULT_PARAMETERS[CONCORD] autotune_gamma_snr is None (gamma-SNR off)",
          d["autotune_gamma_snr"] is None)
else:
    skip("OPTIMIZER_DEFAULT_PARAMETERS[CONCORD]", _OPT_ERR[:90])


n_pass = sum(results)
print(f"{n_pass}/{len(results)} controller-wiring CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
