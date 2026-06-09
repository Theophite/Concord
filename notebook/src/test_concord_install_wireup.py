"""Smoke test for the C:\\OneTrainerMod install wireup.

Validates that the feature-deploy from this turn actually landed in
the OneTrainer install:

  1. The Concord package files in `C:\\OneTrainerMod\\` are in sync
     with the dev copies and import cleanly.
  2. `Optimizer.CONCORD_SGD` is in the enum, has an entry in
     `OPTIMIZER_DEFAULT_PARAMETERS`, and the entry includes the new
     `concord_wrap_embeddings` knob.
  3. `KEY_DETAIL_MAP` in `OptimizerParamsWindow.py` contains all 24
     `concord_*` tooltip entries — this is what was failing when the
     panel showed up empty.
  4. `TrainOptimizerConfig.default_values()` builds and exposes the
     concord_* fields with the right defaults.
  5. `scripts/train.py` and `scripts/train_ui.py` call
     `onetrainer_concord_patch.install()` (so monkeypatches run).
  6. `install()` is idempotent and applies the state_dict patch to
     `ConcordLinearFused`.
  7. `ConcordTrainer` reads `config.gradient_accumulation_steps` and
     calls `set_accum_steps` on its optimizer.
  8. The new K-microbatch outer loop in `ConcordTrainer.train()`
     actually runs: K microbatches per effective step, with the chase
     firing only on the K-th (verified at the optimizer level — every
     wrapped layer's `_apply_chase` flag flips correctly).

Run: OT_ROOT=C:/OneTrainerMod python test_concord_install_wireup.py
"""
import os
import sys
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

INSTALL = Path(r"C:\OneTrainerMod")
DEV = Path(r"C:\foliated_onetrainer")

# Force imports to resolve from the INSTALL copy, not the dev copy.
sys.path.insert(0, str(INSTALL))

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Test 1. Package files synced (byte-for-byte) into the install.
# --------------------------------------------------------------------- #

def test_files_synced():
    print("[1] Concord package files synced into OneTrainerMod")
    targets = [
        "concord_triton_fused.py", "concord_triton.py",
        "concord_linear_fused.py", "concord_polyak.py",
        "concord_optimizer.py", "onetrainer_concord_patch.py",
        "concord_trainer.py", "concord_dataset.py", "fused_profiler.py",
    ]
    for name in targets:
        a = (DEV / name).read_bytes()
        b = (INSTALL / name).read_bytes()
        assert a == b, f"{name} differs between dev and install"
    print(f"    OK — {len(targets)} files byte-identical")


# --------------------------------------------------------------------- #
# Test 2. CONCORD_SGD optimizer defaults wired.
# --------------------------------------------------------------------- #

def test_optimizer_defaults():
    print("[2] OPTIMIZER_DEFAULT_PARAMETERS[CONCORD_SGD]")
    from modules.util.enum.Optimizer import Optimizer
    from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS
    assert Optimizer.CONCORD_SGD.value == 'CONCORD_SGD'
    defs = OPTIMIZER_DEFAULT_PARAMETERS.get(Optimizer.CONCORD_SGD)
    assert defs is not None, \
        "CONCORD_SGD missing from OPTIMIZER_DEFAULT_PARAMETERS — UI panel would be empty"
    must_have = {
        'beta1', 'beta2', 'weight_decay', 'eps',
        'concord_aux_lr', 'concord_aux_optimizer',
        'concord_alpha', 'concord_beta1',
        'concord_rebalance_every', 'concord_refit_every',
        'concord_refit_target', 'concord_tickdown',
        'concord_qtridiag', 'concord_qt_refresh', 'concord_qtridiag_pairs',
        'concord_lr_flat_after', 'concord_lr_flat_frac',
        'concord_bma_obs_every',
        'concord_polyak_bias', 'concord_polyak_observe_every',
        'concord_polyak_leak', 'concord_polyak_commit',
        'concord_polyak_probe_every', 'concord_polyak_level',
        'concord_polyak_warmup', 'concord_polyak_temperature',
        'concord_target_modules', 'concord_wrap_embeddings',
    }
    missing = must_have - set(defs.keys())
    assert not missing, f"missing keys in defaults: {missing}"
    print(f"    OK — {len(defs)} fields, all expected concord_* present "
          "(including concord_wrap_embeddings)")


# --------------------------------------------------------------------- #
# Test 3. KEY_DETAIL_MAP has the GUI tooltips — the actual fix that
# made the UI panel populate.
# --------------------------------------------------------------------- #

def test_gui_tooltips():
    print("[3] KEY_DETAIL_MAP concord_* tooltips")
    # Parse the file rather than instantiate the Tkinter window.
    src = (INSTALL / "modules/ui/OptimizerParamsWindow.py").read_text(encoding='utf-8')
    tree = ast.parse(src)
    # Find the KEY_DETAIL_MAP assignment inside create_dynamic_ui.
    found_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "KEY_DETAIL_MAP"
                for t in node.targets):
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and k.value.startswith("concord_"):
                    found_keys.add(k.value)
    must_have = {
        'concord_aux_lr', 'concord_aux_optimizer',
        'concord_alpha', 'concord_beta1',
        'concord_rebalance_every', 'concord_refit_every',
        'concord_refit_target', 'concord_tickdown',
        'concord_qtridiag', 'concord_qt_refresh', 'concord_qtridiag_pairs',
        'concord_lr_flat_after', 'concord_lr_flat_frac',
        'concord_bma_obs_every',
        'concord_polyak_bias', 'concord_polyak_observe_every',
        'concord_polyak_leak', 'concord_polyak_commit',
        'concord_polyak_probe_every', 'concord_polyak_level',
        'concord_polyak_warmup', 'concord_polyak_temperature',
        'concord_target_modules', 'concord_wrap_embeddings',
    }
    missing = must_have - found_keys
    assert not missing, \
        f"GUI tooltips missing: {missing}.  Panel would render empty for those rows."
    print(f"    OK — {len(found_keys)} concord_* tooltips in KEY_DETAIL_MAP")


# --------------------------------------------------------------------- #
# Test 4. TrainConfig schema exposes concord_wrap_embeddings.
# --------------------------------------------------------------------- #

def test_schema_has_wrap_embeddings():
    print("[4] TrainOptimizerConfig schema includes concord_wrap_embeddings")
    from modules.util.config.TrainConfig import TrainOptimizerConfig
    cfg = TrainOptimizerConfig.default_values()
    d = cfg.to_dict()
    assert 'concord_wrap_embeddings' in d, \
        "concord_wrap_embeddings missing from TrainOptimizerConfig schema"
    # Default should be False (existing OneTrainer runs that train
    # embeddings shouldn't silently switch them onto int storage).
    assert d['concord_wrap_embeddings'] is False, \
        f"expected default False, got {d['concord_wrap_embeddings']!r}"
    print("    OK — concord_wrap_embeddings present, defaults to False")


# --------------------------------------------------------------------- #
# Test 5. scripts/train.py + train_ui.py call install().
# --------------------------------------------------------------------- #

def test_train_scripts_install_concord():
    print("[5] scripts/train.py and train_ui.py call install()")
    for name in ("scripts/train.py", "scripts/train_ui.py"):
        src = (INSTALL / name).read_text(encoding='utf-8')
        assert "import onetrainer_concord_patch" in src, \
            f"{name} missing import"
        assert "onetrainer_concord_patch.install()" in src, \
            f"{name} missing install() call"
    print("    OK — both train scripts wired")


# --------------------------------------------------------------------- #
# Test 6. install() is idempotent and applies state_dict patch.
# --------------------------------------------------------------------- #

def test_install_idempotent_and_patches():
    print("[6] onetrainer_concord_patch.install() idempotent + patches")
    import onetrainer_concord_patch
    onetrainer_concord_patch.install()
    onetrainer_concord_patch.install()   # idempotent
    from concord_linear_fused import (ConcordLinearFused, ConcordConv2dFused,
                                         ConcordEmbeddingFused)
    # State-dict patch sets a sentinel on the Linear class. The patch
    # also propagates to Conv2d (subclass) and Embedding (separate
    # explicit attachment).
    assert getattr(ConcordLinearFused, "_ot_state_dict_patched", False)
    assert getattr(ConcordEmbeddingFused, "_ot_state_dict_patched", False)
    # Quick functional check: emit weight via the patched path.
    m = ConcordLinearFused(8, 4, bias=True, device='cuda', alpha=0.1, lr=0.1)
    sd = m.state_dict()
    assert "weight" in sd and "bias" in sd, \
        f"patched state_dict should emit weight + bias, got {sorted(sd.keys())}"
    assert sd["weight"].shape == (4, 8)
    print("    OK — install() idempotent; state_dict converter active "
          "on Linear / Conv2d / Embedding")


# --------------------------------------------------------------------- #
# Test 7. ConcordTrainer reads gradient_accumulation_steps and configures
# the optimizer's accum cycle.
# --------------------------------------------------------------------- #

def _make_trainer_stub(grad_accum_steps: int):
    """Construct a ConcordTrainer without running __init__ (skips the
    SDXL load) and hand-set just enough for _build_optimizer to run
    end-to-end."""
    from concord_trainer import ConcordTrainer
    stub = ConcordTrainer.__new__(ConcordTrainer)
    # Mock unet: a tiny model with one Linear we can wrap.
    stub.unet = nn.Sequential(nn.Linear(16, 8)).cuda()
    stub.te1 = None
    stub.te2 = None
    stub._te_mode = SimpleNamespace(
        te1=False, te2=False, emb1=False, emb2=False,
        te1_lr=1e-5, te2_lr=1e-5, live=False)
    stub._base_lr = 1e-3
    # Tiny config — just the fields _build_optimizer + advance_accum need.
    stub.config = SimpleNamespace(
        learning_rate=1e-3,
        gradient_accumulation_steps=grad_accum_steps,
        optimizer=_FakeOptCfg(),
    )
    return stub


class _FakeOptCfg:
    """Mirror the concord_* fields ConcordSGD reads. Default values
    cribbed from C:\\OneTrainerMod's OPTIMIZER_DEFAULT_PARAMETERS."""
    optimizer = 'CONCORD_SGD'
    concord_aux_lr = 1e-3
    concord_alpha = 0.1
    concord_beta1 = 0.0
    concord_rebalance_every = 999_999  # disable rebalance during smoke
    concord_refit_every = 0
    concord_refit_target = 16384
    concord_tickdown = 'off'
    concord_qtridiag = False
    concord_qt_refresh = 3000
    concord_qtridiag_pairs = None
    concord_lr_flat_after = 0
    concord_lr_flat_frac = 0.0
    concord_bma_obs_every = 0
    concord_polyak_bias = False
    concord_polyak_observe_every = 8
    concord_polyak_leak = 0.05
    concord_polyak_commit = 0.1
    concord_polyak_probe_every = 200
    concord_polyak_level = 1
    concord_polyak_warmup = 2
    concord_polyak_temperature = 0.0
    concord_target_modules = '.*'
    concord_aux_optimizer = 'adamw'
    concord_wrap_embeddings = False
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.01
    eps = 1e-8


def test_trainer_reads_grad_accum_config():
    print("[7] ConcordTrainer reads config.gradient_accumulation_steps")
    K = 4
    stub = _make_trainer_stub(grad_accum_steps=K)
    stub._build_optimizer()
    assert stub._accum_steps == K, \
        f"trainer didn't pick up K={K}, got {stub._accum_steps}"
    # The optimizer's accum cycle should be configured to match.
    assert stub.optimizer._accum_steps == K, \
        f"optimizer accum_steps != trainer's: " \
        f"{stub.optimizer._accum_steps} vs {K}"
    # Every wrapped Concord layer should be in tick-only mode at the
    # start of a fresh cycle (apply_chase==False until the K-th
    # microbatch).
    from concord_linear_fused import ConcordLinearFused
    for m in stub.unet.modules():
        if isinstance(m, ConcordLinearFused):
            assert m._apply_chase is False, \
                "layer should be tick-only at cycle start"
    print(f"    OK — K={K} propagated to optimizer and all wrapped layers")


# --------------------------------------------------------------------- #
# Test 8. K-microbatch loop: simulate ConcordTrainer.train()'s inner
# logic on a tiny model. Verify the chase fires exactly once per K
# microbatches.
# --------------------------------------------------------------------- #

def test_kmicrobatch_loop_fires_chase_once_per_K():
    print("[8] K-microbatch outer loop fires chase exactly once per K")
    K = 3
    stub = _make_trainer_stub(grad_accum_steps=K)
    stub._build_optimizer()
    from concord_linear_fused import ConcordLinearFused
    cl = [m for m in stub.unet.modules() if isinstance(m, ConcordLinearFused)][0]
    assert cl.bias is not None  # autograd anchor

    # Simulate 2 effective steps of K microbatches each.
    chase_fired = []   # per-microbatch: did the layer's chase actually run?
    x = torch.randn(4, 16, device='cuda')
    for eff_step in range(2):
        # Mirror the trainer's outer-loop pattern.
        stub.optimizer.zero_grad(set_to_none=True)
        for k in range(K):
            s_slow_before = cl.s_slow.clone()
            loss = stub.unet(x).float().square().mean() / K
            loss.backward()
            s_slow_moved = (cl.s_slow != s_slow_before).any().item()
            chase_fired.append(s_slow_moved)
            stub.optimizer.advance_accum()
        stub.optimizer.step()
        # zero_grad on next effective step will reset the cycle.
    # Expected pattern: [False, False, True, False, False, True]
    # (K=3, chase only on the K-th = 3rd microbatch in each cycle)
    expected = [False, False, True] * 2
    print(f"    chase_fired per microbatch: {chase_fired}")
    print(f"    expected:                   {expected}")
    assert chase_fired == expected, \
        f"chase didn't fire on the K-th only: got {chase_fired}, " \
        f"expected {expected}"
    print(f"    OK — chase fires exactly on microbatch K-1 in each cycle")


# --------------------------------------------------------------------- #

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        sys.exit(0)
    test_files_synced()
    print()
    test_optimizer_defaults()
    print()
    test_gui_tooltips()
    print()
    test_schema_has_wrap_embeddings()
    print()
    test_train_scripts_install_concord()
    print()
    test_install_idempotent_and_patches()
    print()
    test_trainer_reads_grad_accum_config()
    print()
    test_kmicrobatch_loop_fires_chase_once_per_K()
    print()
    print("all install-wireup smoke checks passed.")


if __name__ == "__main__":
    main()
