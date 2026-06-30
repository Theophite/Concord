"""CONSOLIDATION GATE (CPU, atol=0): the WINNER dict and the CONCORD_DEFAULTS panel
literal must EQUAL the projection of a single ConcordConfig() (config_defaults.py),
modulo the documented intentional per-surface overrides. This is the backstop that
lets the four config layers be collapsed to one source without changing any shipped
default -- if a literal and the dataclass drift apart, this fails.

Run (CPU; safe while live with CUDA_VISIBLE_DEVICES=""):
  CUDA_VISIBLE_DEVICES="" venv/Scripts/python.exe \
      modules/util/optimizer/concord_core/tests/test_config_projection.py
"""
import sys
from pathlib import Path

OT = Path(__file__).resolve().parents[5]
_OPT = OT / "modules" / "util" / "optimizer"
_CONCORD = _OPT / "concord"
_CORE = _OPT / "concord_core"
for _p in (str(OT), str(_OPT), str(_CONCORD), str(_CORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ast  # noqa: E402
import config_defaults as cd          # noqa: E402  (concord_core, bare import)
import concord_winner as cw           # noqa: E402  (the canonical ConcordConfig, for now)


def _gui_concord_defaults():
    """Read OPTIMIZER_DEFAULT_PARAMETERS[CONCORD] out of optimizer_util.py SOURCE via
    ast -- importing optimizer_util standalone trips the create.py<->modelSetup circular
    import (same reason + technique as test_autotuner_cpu.py:196). No framework import,
    only {Optimizer} implicitly (we match the `.CONCORD` attribute name)."""
    tree = ast.parse((OT / "modules" / "util" / "optimizer_util.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) \
                and any(getattr(t, "id", "") == "OPTIMIZER_DEFAULT_PARAMETERS" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Attribute) and k.attr == "CONCORD":
                    return {kk.value: ast.literal_eval(vv) for kk, vv in zip(v.keys, v.values)}
    raise AssertionError("CONCORD defaults dict not found in optimizer_util.py")


def test_winner_is_projection():
    got = cd.winner_from(cw.ConcordConfig())
    assert cw.WINNER == got, _diff("WINNER", cw.WINNER, got)


def test_concord_defaults_is_projection():
    panel = _gui_concord_defaults()
    got = cd.concord_defaults_from(cw.ConcordConfig())
    assert panel == got, _diff("CONCORD_DEFAULTS", panel, got)


def _diff(name, a, b):
    keys = set(a) | set(b)
    lines = [f"{name} literal != projection:"]
    for k in sorted(keys):
        av, bv = a.get(k, "<MISSING>"), b.get(k, "<MISSING>")
        if av != bv:
            lines.append(f"  {k}: literal={av!r}  projection={bv!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    test_winner_is_projection()
    test_concord_defaults_is_projection()
    print("[test_config_projection] PASS: WINNER and CONCORD_DEFAULTS are projections of ConcordConfig()")
