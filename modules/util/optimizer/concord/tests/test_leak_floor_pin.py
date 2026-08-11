"""CPU unit test: the DERIVED leak floor (no GPU, no Triton).

2026-07-18, exp73 (research repo, experiments/cpu_dynamics): the leak gate is the
chase gate's CALIBRATION PARTNER (under drift mu/u = L*alpha*gc/gl; C* encodes the
chase:leak rate RATIO -- a 10x mis-ratio deadlocks certification at every drift
rate). History: first the leak floor was PINNED to the winner constants (0.999, 0.1);
then the user asked for "a way to set the chase and automatically get a good leak",
so the pin became a DERIVATION in make_concord_config:
    leak_min   = chase_min   (the winner's own ratio-0.9 pairing, at any scale)
    leak_start = 0.999       (withhold-judgment-early ignition, chase-independent)
Winner-default chase (0.9, 0.1) reproduces the validated winner leak (0.999, 0.1)
BYTE-IDENTICALLY. Chase floors are clamped to [0,1] (>1 inverts the affine gate).
GUI leak fields remain deleted. There is NO override: the CONCORD_LEAK_FLOOR env
hatch was removed at the user's request ("remove the option for it to be any
different") -- a set-but-ignored env var warns loudly; mis-ratio ablations remain
reproducible on the CPU reference (exp73) only.

This test locks all of that in:
  1  no optimizer_config          -> winner (0.999, 0.1), SILENT (byte-identical default)
  2  user chase (0.03, 0.01)      -> leak FOLLOWS: (0.999, 0.01)
  3  user chase + stale winner-leak carried in config -> derived + loud IGNORED note
     (the production run's exact relaunch case)
  4  carried leak matching the derived values -> SILENT (no log spam)
  5  CONCORD_LEAK_FLOOR set -> IGNORED (still derived) + loud no-longer-supported warning
  6  chase clamp: (1.5, -0.2) -> (1.0, 0.0) + loud note; leak follows the CLAMPED min
  7  UI surface: leak keys gone from KEY_DETAIL_MAP + OPTIMIZER_DEFAULT_PARAMETERS
     (AST-parsed); chase keys REMAIN (the one dial)
  8  ConcordConfig dataclass still carries winner (0.999, 0.1) (leak_start source)

Run with the OneTrainer venv python:
  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_leak_floor_pin.py
"""
import ast
import contextlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer"))

from concord_ot import make_concord_config
from concord_winner import ConcordConfig

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _make(oc=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = make_concord_config(1e-4, oc)
    return cfg, buf.getvalue()


def _lf(cfg):
    return (cfg.ratio_leak_floor, cfg.ratio_leak_floor_min)


WINNER = (0.999, 0.1)
_saved_env = os.environ.pop("CONCORD_LEAK_FLOOR", None)
try:
    # 1: no optimizer_config -> derivation reproduces the winner, silent
    cfg, out = _make(None)
    check("1 default -> winner leak (byte-identical)", _lf(cfg) == WINNER, f"{_lf(cfg)}")
    check("1 default -> silent", "leak floor" not in out and "ratio_leak_floor" not in out)

    # 2: user chase -> leak follows (min = chase_min, start stays 0.999)
    oc = SimpleNamespace(ratio_chase_floor=0.03, ratio_chase_floor_min=0.01)
    cfg, out = _make(oc)
    check("2 chase (0.03, 0.01) -> leak (0.999, 0.01)", _lf(cfg) == (0.999, 0.01), f"{_lf(cfg)}")
    check("2 chase passthrough intact",
          (cfg.ratio_chase_floor, cfg.ratio_chase_floor_min) == (0.03, 0.01))

    # 3: THE PRODUCTION RELAUNCH CASE -- user chase + stale winner-leak in the config
    oc = SimpleNamespace(ratio_chase_floor=0.03, ratio_chase_floor_min=0.01,
                         ratio_leak_floor=0.999, ratio_leak_floor_min=0.1)
    cfg, out = _make(oc)
    check("3 stale leak carried -> derived wins", _lf(cfg) == (0.999, 0.01), f"{_lf(cfg)}")
    check("3 stale leak carried -> loud IGNORED note", "IGNORED" in out and "DERIVED" in out)

    # 4: carried leak matching the derived values -> silent
    oc = SimpleNamespace(ratio_chase_floor=0.03, ratio_chase_floor_min=0.01,
                         ratio_leak_floor=0.999, ratio_leak_floor_min=0.01)
    cfg, out = _make(oc)
    check("4 matching carried leak -> silent", "IGNORED" not in out and _lf(cfg) == (0.999, 0.01))

    # 5: the env hatch is GONE -- setting it changes nothing and warns loudly
    os.environ["CONCORD_LEAK_FLOOR"] = "1.0,1.0"
    cfg, out = _make(None)
    check("5 env set -> STILL derived (winner default)", _lf(cfg) == WINNER, f"{_lf(cfg)}")
    check("5 env set -> loud no-longer-supported warning", "NO LONGER SUPPORTED" in out)
    oc = SimpleNamespace(ratio_chase_floor=0.03, ratio_chase_floor_min=0.01)
    cfg, out = _make(oc)
    check("5 env set + user chase -> STILL derived", _lf(cfg) == (0.999, 0.01), f"{_lf(cfg)}")
    del os.environ["CONCORD_LEAK_FLOOR"]

    # 6: chase clamp to [0,1]; the derived leak follows the CLAMPED value
    oc = SimpleNamespace(ratio_chase_floor=1.5, ratio_chase_floor_min=-0.2)
    cfg, out = _make(oc)
    check("6 chase clamped to [0,1]",
          (cfg.ratio_chase_floor, cfg.ratio_chase_floor_min) == (1.0, 0.0),
          f"{(cfg.ratio_chase_floor, cfg.ratio_chase_floor_min)}")
    check("6 clamp is loud", "CLAMPED" in out)
    check("6 leak follows the clamped min", _lf(cfg) == (0.999, 0.0), f"{_lf(cfg)}")

    # 7: the UI surfaces no longer expose the leak option
    opw = (OT / "modules" / "ui" / "OptimizerParamsWindow.py").read_text(encoding="utf-8")
    check("7 KEY_DETAIL_MAP leak entries gone",
          "'ratio_leak_floor'" not in opw and '"ratio_leak_floor"' not in opw)
    check("7 chase-floor entries REMAIN (the one dial)", "'ratio_chase_floor'" in opw)
    outil = (OT / "modules" / "util" / "optimizer_util.py").read_text(encoding="utf-8")
    node = None
    for n in ast.walk(ast.parse(outil)):
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "OPTIMIZER_DEFAULT_PARAMETERS" for t in n.targets
        ):
            node = n.value
            break
    check("7 defaults table found", node is not None)
    keys = set()
    if node is not None:
        # the CONCORD entry's dict literal: the one whose keys include the chase floor
        for v in ast.walk(node):
            if isinstance(v, ast.Dict):
                ks = {k.value for k in v.keys if isinstance(k, ast.Constant)}
                if "ratio_chase_floor" in ks:
                    keys = ks
                    break
    check("7 CONCORD defaults: leak floors REMOVED",
          bool(keys) and "ratio_leak_floor" not in keys and "ratio_leak_floor_min" not in keys,
          f"{len(keys)} keys")
    check("7 CONCORD defaults: chase floors kept",
          "ratio_chase_floor" in keys and "ratio_chase_floor_min" in keys)

    # 8: the derivation's anchors (dataclass) still carry the winner schedule
    d = ConcordConfig()
    check("8 ConcordConfig dataclass = winner (0.999, 0.1)",
          (d.ratio_leak_floor, d.ratio_leak_floor_min) == WINNER)
finally:
    if _saved_env is not None:
        os.environ["CONCORD_LEAK_FLOOR"] = _saved_env
    else:
        os.environ.pop("CONCORD_LEAK_FLOOR", None)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
