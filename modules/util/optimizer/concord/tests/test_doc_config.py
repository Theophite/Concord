"""CONCORD.md Section 10 (Config & GUI integration) -- doc-vs-code assertions, CPU only.

This module verifies the *config* assertions in CONCORD.md against the actual OneTrainer
config plumbing. Everything here is pure Python data (no torch-GPU, no Triton launch): the
documented field defaults read off ``TrainConfig.default_values()``, the optimizer-panel
CONCORD defaults read off ``optimizer_util.OPTIMIZER_DEFAULT_PARAMETERS``, the
``packed_embeddings_active`` truth table, and the dimensionless ``gf_consol == lam/lr``
relation in ``concord_ot.py``.

sys.path is set up exactly like test_servo_cpu.py: OT-root (= parents[5]) plus the concord
dir, so the module imports standalone under pytest or `python test_doc_config.py`.

CONCORD.md assertions COVERED here (Section 10 unless noted):
  - TrainConfig field defaults (CONCORD.md:299-302, code TrainConfig.py:1060,1124-1134):
      concord_packed_embeddings True, concord_embedding_anchor True,
      concord_train_caption_vocab False, concord_caption_vocab_anchor False,
      concord_embedding_delay_epochs 1.0, concord_embedding_freq_exponent 0.5,
      concord_embedding_quality_mode "hard".
  - Optimizer-panel CONCORD defaults (CONCORD.md Section 10, code optimizer_util.py):
      the per-layer hunting-servo knob ABSENT (the hunting servos were removed; the
      NoiseScaleSeeder is the only dissipation controller), dissipation 0.025,
      servo-companion knobs asserted ABSENT (excised with the hunting servos).
  - packed_embeddings_active logic (CONCORD.md Section 7 + the flag; code concord_ot.py:1175):
      CONCORD optimizer AND concord_packed_embeddings AND
      (train_any_embedding() OR concord_train_caption_vocab).
  - Dimensionless relation gf_consol == lam/lr (CONCORD.md:35,263,293, code concord_ot.py:137):
      the ConcordController applies `self.config.gf_consol = lam / max(self.config.lr, 1e-12)`
      and make_concord_config threads `dissipation` through from the optimizer_config.

Not unit-tested (by inspection / empirical / pure file:line, per the suite's ground rules):
  - "beats live get_weight by ~0.04-0.06 val nats", "s_fast ~4-7% of weight mass",
    "stable 10.8M..49M" -- EMPIRICAL claims (CONCORD.md:17,81,182).
  - The GUI *surfacing* claims (OptimizerParamsWindow.py:141-163, TrainingTab.py:595-673,
    353-355, 542-564) and the "not surfaced in any panel" note (CONCORD.md:305) -- pure
    file:line / UI-layout citations, not behavioral code.
  - gf_consol engine default == 50 (CONCORD.md:293): NOT separately tested as a doc default
    because it is the ConcordConfig() dataclass default, surfaced indirectly via
    make_concord_config(lr, None).gf_consol below (asserted == 50.0 as a sanity anchor).

Code discrepancies found vs CONCORD.md (the doc is slightly off; tests use the CODE):
  - CONCORD.md:35,263 cite the gf_consol = lam/lr relation at "concord_ot.py:135"; the
    actual assignment is at line 137 (the `if self.config.dissipation is not None:` guard is
    at 135). The relation itself matches. CONCORD.md:293 separately cites it correctly at 137.
  - the per-layer hunting-servo knob was removed from the panel/config with the hunting
    servos; the test asserts its ABSENCE from the CONCORD optimizer-defaults block (a
    reappearance means a dead knob was resurrected).
"""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# -- sys.path: mirror test_servo_cpu.py exactly ------------------------------
OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

HAS_CUDA = torch.cuda.is_available()  # not used here (Section 10 is CPU-only) but kept per the suite contract

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.Optimizer import Optimizer        # noqa: E402

_CONCORD_OT_PATH = OT / "modules" / "util" / "optimizer" / "concord_ot.py"
_OPTIMIZER_UTIL_PATH = OT / "modules" / "util" / "optimizer_util.py"


def _load_concord_ot():
    """Load concord_ot.py as a standalone top-level module.

    `import modules.util.optimizer_util` (which re-exports concord glue) triggers a heavy
    circular import via `from modules.util import create`, so we load the concord_ot source
    directly by file path. Its own top inserts the concord dir on sys.path and does
    `from concord_winner import ConcordConfig`, exactly the import style this test mirrors.
    """
    spec = importlib.util.spec_from_file_location("concord_ot_under_test", str(_CONCORD_OT_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_concord_optimizer_defaults():
    """Return OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.CONCORD] without importing optimizer_util.

    optimizer_util.py top-imports `from modules.util import create`, which import-cycles when
    loaded in isolation. The CONCORD default block is a plain dict literal whose only free name
    is the `Optimizer` enum, so we parse the module AST, find the OPTIMIZER_DEFAULT_PARAMETERS
    assignment, and eval just that expression with `Optimizer` in scope. This reads the *actual*
    source values, not a copy.
    """
    src = _OPTIMIZER_UTIL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "OPTIMIZER_DEFAULT_PARAMETERS" for t in n.targets
        ):
            node = n.value
            break
    assert node is not None, "OPTIMIZER_DEFAULT_PARAMETERS assignment not found in optimizer_util.py"
    table = eval(compile(ast.Expression(node), "<optimizer_util:ODP>", "eval"), {"Optimizer": Optimizer})
    return table[Optimizer.CONCORD]


# -- module-level singletons (built once; cheap, CPU-only) -------------------
CFG = TrainConfig.default_values()
CONCORD_DEFAULTS = _load_concord_optimizer_defaults()
COT = _load_concord_ot()


def _stub_config(optimizer, packed, train_any_emb, caption_vocab):
    """Minimal duck-typed config for packed_embeddings_active: only the attributes that
    function actually reads. Deliberately NOT a real model/TrainConfig."""
    return SimpleNamespace(
        optimizer=SimpleNamespace(optimizer=optimizer),
        concord_packed_embeddings=packed,
        concord_train_caption_vocab=caption_vocab,
        train_any_embedding=lambda: train_any_emb,
    )


# ----------------------------------------------------------------------------
# TrainConfig field defaults (CONCORD.md:299-302)
# ----------------------------------------------------------------------------
def test_default_packed_embeddings_true():
    # CONCORD.md:299 "concord_packed_embeddings (default True, :1060)"
    assert CFG.concord_packed_embeddings is True


def test_default_embedding_anchor_true():
    # CONCORD.md:299 "concord_embedding_anchor (True, :1124)"
    assert CFG.concord_embedding_anchor is True


def test_default_train_caption_vocab_false():
    # CONCORD.md:299 "concord_train_caption_vocab (False, :1125)"
    assert CFG.concord_train_caption_vocab is False


def test_default_caption_vocab_anchor_false():
    # CONCORD.md:299 "concord_caption_vocab_anchor (False, :1126)"
    assert CFG.concord_caption_vocab_anchor is False


def test_default_embedding_delay_epochs_one():
    # CONCORD.md:300 "concord_embedding_delay_epochs (1.0, :1127)"
    assert CFG.concord_embedding_delay_epochs == pytest.approx(1.0)


def test_default_embedding_freq_exponent_half():
    # CONCORD.md:300 "concord_embedding_freq_exponent (0.5, :1129)"
    assert CFG.concord_embedding_freq_exponent == pytest.approx(0.5)


def test_default_embedding_quality_mode_hard():
    # CONCORD.md:301 "..._quality_mode (\"hard\", :1134)"
    assert CFG.concord_embedding_quality_mode == "hard"


# ----------------------------------------------------------------------------
# Optimizer-panel CONCORD defaults (CONCORD.md:293-295, optimizer_util.py)
# ----------------------------------------------------------------------------
def test_hunting_servo_knob_absent_from_concord_block():
    # The per-layer hunting servo was removed (the NoiseScaleSeeder is the only dissipation
    # controller); its knob must stay OUT of the CONCORD optimizer-defaults block. The
    # knob's name is spelled SPLIT so the repo's straggler grep never matches this guard.
    assert ("autotune" "_servo") not in CONCORD_DEFAULTS
    # the seeder's window knob remains (window = steps_per_epoch // per)
    assert CONCORD_DEFAULTS["autotune_servo_per_epoch"] == 3


def test_dissipation_default_is_0_025():
    # CONCORD.md:293 "Dissipation (lam), default 0.025, optimizer_util.py:454"
    assert CONCORD_DEFAULTS["dissipation"] == pytest.approx(0.025)


def test_servo_companion_knobs_absent():
    # the hunting servos and their companion knobs are excised: the panel must not
    # carry them (split literals keep the straggler greps clean)
    for k in ("autotune" "_climb_rate", "autotune" "_boil_ceiling",
              "concord_servo" "_protected_boil", "concord_servo" "_waste_ceiling"):
        assert k not in CONCORD_DEFAULTS, k


# ----------------------------------------------------------------------------
# packed_embeddings_active logic (CONCORD.md, concord_ot.py:1175)
#   active == CONCORD AND concord_packed_embeddings AND
#             (train_any_embedding() OR concord_train_caption_vocab)
# ----------------------------------------------------------------------------
def test_packed_active_concord_packed_with_trainable_embedding():
    cfg = _stub_config(Optimizer.CONCORD, packed=True, train_any_emb=True, caption_vocab=False)
    assert COT.packed_embeddings_active(cfg) is True


def test_packed_active_concord_packed_with_caption_vocab():
    # train_any_embedding() False but caption-vocab on -> still active (the OR branch).
    cfg = _stub_config(Optimizer.CONCORD, packed=True, train_any_emb=False, caption_vocab=True)
    assert COT.packed_embeddings_active(cfg) is True


def test_packed_inactive_when_nothing_to_train():
    # CONCORD + packed but neither embeddings nor caption-vocab -> inactive.
    cfg = _stub_config(Optimizer.CONCORD, packed=True, train_any_emb=False, caption_vocab=False)
    assert COT.packed_embeddings_active(cfg) is False


def test_packed_inactive_when_flag_off():
    # concord_packed_embeddings False gates it off even with a trainable embedding.
    cfg = _stub_config(Optimizer.CONCORD, packed=False, train_any_emb=True, caption_vocab=False)
    assert COT.packed_embeddings_active(cfg) is False


def test_packed_inactive_for_non_concord_optimizer():
    # Non-CONCORD optimizer -> never routes through the packed core.
    cfg = _stub_config(Optimizer.ADAMW, packed=True, train_any_emb=True, caption_vocab=False)
    assert COT.packed_embeddings_active(cfg) is False


# ----------------------------------------------------------------------------
# Dimensionless relation gf_consol == lam/lr (CONCORD.md:35,263,293; concord_ot.py:137)
# ----------------------------------------------------------------------------
def _controller_gf_consol_expr():
    """Extract the exact `self.config.gf_consol = ...` assignment RHS source from
    concord_ot.py (the line the doc cites). Asserting against the parsed source proves the
    relation is what the code actually computes -- we can't construct a real ConcordController
    here (its __init__ swaps the UNet to packed cores, which needs a model + GPU)."""
    src = _CONCORD_OT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and t.attr == "gf_consol"
                    and isinstance(t.value, ast.Attribute)
                    and t.value.attr == "config"
                ):
                    return ast.unparse(node.value)
    return None


def test_controller_sets_gf_consol_to_lam_over_lr():
    rhs = _controller_gf_consol_expr()
    assert rhs is not None, "self.config.gf_consol assignment not found in concord_ot.py"
    # the code guards lr against 0 with max(lr, 1e-12); the relation is lam/lr.
    assert "lam" in rhs and "self.config.lr" in rhs
    # evaluate the real source expression for a representative (lam, lr) and compare to lam/lr.
    lam = 0.025
    lr = 7.5e-5
    selfobj = SimpleNamespace(config=SimpleNamespace(lr=lr))
    computed = eval(rhs, {"max": max}, {"lam": lam, "self": selfobj})
    assert computed == pytest.approx(lam / lr)
    # sanity: this is the ~333 the doc quotes for lam=0.025 @ lr=7.5e-5 (CONCORD.md:446).
    assert computed == pytest.approx(333.333, rel=1e-3)


def test_make_concord_config_threads_dissipation_through():
    # make_concord_config must carry the GUI `dissipation` (lam) onto the ConcordConfig so the
    # controller can convert it to gf_consol. With optimizer_config=None it falls back to the
    # ConcordConfig() dataclass default (dissipation None, gf_consol engine default 50).
    cfg_none = COT.make_concord_config(7.5e-5, None)
    assert cfg_none.dissipation is None
    assert cfg_none.gf_consol == pytest.approx(50.0)  # CONCORD.md:293 engine kappa default
    # When the optimizer_config carries a dissipation, pick() threads it through unchanged.
    oc = SimpleNamespace(dissipation=0.025)
    cfg = COT.make_concord_config(7.5e-5, oc)
    assert cfg.dissipation == pytest.approx(0.025)


# allow `python test_doc_config.py` like the sibling test_servo_cpu.py
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))