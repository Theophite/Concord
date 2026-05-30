"""Smoke test for the Concord text-encoder + embedding training paths.

Verifies the ConcordTrainer-side wiring (mode detection, freeze policy,
parameter-group construction, multi-subtree wrapping, gradient flow)
without spinning up a real SDXL load. Uses tiny stub modules that
mimic CLIP's structure (text_model.embeddings.token_embedding + a
couple of Linears).

Covers:
  1. `_compute_te_mode` returns the right flags for each combination
     of `text_encoder{,_2}.train` / `train_embedding` / `learning_rate`.
  2. `_apply_te_freeze_policy` sets requires_grad correctly:
       - train=True  → entire TE requires_grad.
       - train_embedding=True (and train=False) → only the
         token_embedding's weight is trainable; Linears stay frozen.
       - both False → fully frozen + eval.
  3. Multi-subtree wrapping via `concord_optimizer.create_concord_optimizer`:
       - UNet-only: only unet Linears wrapped.
       - UNet+TE1+TE2: every named subtree's Linears wrapped.
       - Embedding-only: TE NOT wrapped (would mutate frozen Linears
         via concord's int update); embedding.weight ends up in aux.
  4. End-to-end gradient flow when TE1 and TE1's embedding both train:
     the Concord TE Linear's int state drifts AND the token_embedding's
     fp32 Parameter drifts after a few backward+step iterations.

Run: OT_ROOT=/path/to/OneTrainer python test_concord_te_training.py
     (CUDA required — Concord's kernels are CUDA-only.
      OT_ROOT must point at an OneTrainer checkout so we can import
      modules.trainer.BaseTrainer; ConcordTrainer's class definition
      depends on it even though this test only constructs a stub via
      __new__ + manual attribute set.)
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
_OT_ROOT = os.environ.get("OT_ROOT")
if _OT_ROOT:
    sys.path.insert(0, _OT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

import onetrainer_concord_patch
onetrainer_concord_patch.install()

from concord_linear_fused import ConcordLinearFused, ConcordConv2dFused
from concord_optimizer import create_concord_optimizer
from concord_trainer import ConcordTrainer


# --------------------------------------------------------------------- #
# Stubs: tiny modules shaped like CLIP-L / OpenCLIP-G
# --------------------------------------------------------------------- #

class _StubTE(nn.Module):
    """CLIP-shaped tiny text encoder. The attribute path
    text_model.embeddings.token_embedding matches what
    ConcordTrainer._te_token_embedding() looks for. The Linears here
    play the role of attention q/k/v/out_proj + MLP fc1/fc2."""
    def __init__(self, vocab=128, dim=32):
        super().__init__()
        self.text_model = nn.Module()
        self.text_model.embeddings = nn.Module()
        self.text_model.embeddings.token_embedding = nn.Embedding(vocab, dim)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        h = self.text_model.embeddings.token_embedding(input_ids)
        h = F.relu(self.proj1(h))
        h = self.proj2(h)
        h = self.norm(h)
        return h.mean(dim=1)   # pool over seq


class _StubUNet(nn.Module):
    def __init__(self, dim=32, n_class=10):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, n_class)
    def forward(self, h):
        return self.l2(F.relu(self.l1(h)))


# --------------------------------------------------------------------- #
# Configs shaped like OneTrainer's TrainConfig
# --------------------------------------------------------------------- #

class _TECfg:
    def __init__(self, train=False, train_embedding=False,
                 learning_rate=None):
        self.train = train
        self.train_embedding = train_embedding
        self.learning_rate = learning_rate


class _TrainCfg:
    def __init__(self, te1_cfg=None, te2_cfg=None, lr=1e-3):
        self.learning_rate = lr
        self.text_encoder = te1_cfg or _TECfg()
        self.text_encoder_2 = te2_cfg or _TECfg()


class _OptCfg:
    """Mirror the concord_* fields create_concord_optimizer reads."""
    optimizer = 'CONCORD_SGD'
    concord_aux_lr = 1e-3
    concord_alpha = 0.1
    concord_beta1 = 0.0
    concord_rebalance_every = 8
    concord_refit_every = 0
    concord_refit_target = 16384
    concord_tickdown = 'off'
    concord_qtridiag = False
    concord_qt_refresh = 3000
    concord_qtridiag_pairs = None
    concord_lr_flat_after = 0
    concord_lr_flat_frac = 1.0
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
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.01
    eps = 1e-8


def _make_trainer_stub(train_cfg):
    """Minimal ConcordTrainer instance — just enough to call
    _compute_te_mode / _apply_te_freeze_policy. Bypasses
    BaseTrainer.__init__ (which would want callbacks + commands)."""
    stub = ConcordTrainer.__new__(ConcordTrainer)
    stub.config = train_cfg
    stub._base_lr = float(train_cfg.learning_rate)
    return stub


# --------------------------------------------------------------------- #
# Test 1. _compute_te_mode covers every combination of the four flags
# --------------------------------------------------------------------- #

def test_compute_te_mode():
    print("[1] _compute_te_mode flag combinations")
    cases = [
        # (te1.train, te1.emb, te2.train, te2.emb, expected.live)
        (False, False, False, False, False),
        (True,  False, False, False, True),
        (False, True,  False, False, True),
        (False, False, True,  False, True),
        (False, False, False, True,  True),
        (True,  False, True,  False, True),
        (True,  True,  False, False, True),    # redundant but accepted
        (False, True,  False, True,  True),
    ]
    for t1, e1, t2, e2, live in cases:
        cfg = _TrainCfg(
            te1_cfg=_TECfg(train=t1, train_embedding=e1, learning_rate=2e-5),
            te2_cfg=_TECfg(train=t2, train_embedding=e2, learning_rate=3e-5),
            lr=1e-3)
        stub = _make_trainer_stub(cfg)
        m = stub._compute_te_mode()
        assert m.te1 == t1 and m.emb1 == e1, \
            f"flag mismatch for te1: got {m}"
        assert m.te2 == t2 and m.emb2 == e2, \
            f"flag mismatch for te2: got {m}"
        assert m.live == live, f"live={m.live} expected {live} for {(t1,e1,t2,e2)}"
        # Per-component LR plumbing
        assert m.te1_lr == 2e-5, f"te1_lr={m.te1_lr}"
        assert m.te2_lr == 3e-5, f"te2_lr={m.te2_lr}"
    # LR fallback to base_lr when text_encoder.learning_rate is None
    cfg_no_te_lr = _TrainCfg(
        te1_cfg=_TECfg(train=True, learning_rate=None), lr=4e-4)
    m = _make_trainer_stub(cfg_no_te_lr)._compute_te_mode()
    assert m.te1_lr == 4e-4, \
        f"missing TE lr should fall back to base_lr; got {m.te1_lr}"
    print("    OK — 8 combinations + LR fallback")


# --------------------------------------------------------------------- #
# Test 2. _apply_te_freeze_policy sets requires_grad correctly
# --------------------------------------------------------------------- #

def test_apply_te_freeze_policy():
    print("[2] _apply_te_freeze_policy: requires_grad per mode")
    stub = _make_trainer_stub(_TrainCfg())

    # Mode A: train full
    te = _StubTE().cuda()
    stub._apply_te_freeze_policy(te, train_full=True, train_embedding=False,
                                   tag="TE-A")
    n_train = sum(1 for p in te.parameters() if p.requires_grad)
    n_total = sum(1 for _ in te.parameters())
    assert n_train == n_total, \
        f"train_full: expected all {n_total} params trainable, got {n_train}"
    assert te.training, "train_full should put TE in train() mode"

    # Mode B: embedding-only
    te = _StubTE().cuda()
    stub._apply_te_freeze_policy(te, train_full=False, train_embedding=True,
                                   tag="TE-B")
    tok = te.text_model.embeddings.token_embedding
    assert tok.weight.requires_grad, \
        "embedding-only should leave token_embedding trainable"
    assert not te.proj1.weight.requires_grad, \
        "embedding-only: Linears must stay frozen"
    assert not te.training, \
        "embedding-only should keep TE in eval mode"

    # Mode C: fully frozen
    te = _StubTE().cuda()
    stub._apply_te_freeze_policy(te, train_full=False, train_embedding=False,
                                   tag="TE-C")
    n_train = sum(1 for p in te.parameters() if p.requires_grad)
    assert n_train == 0, f"all-frozen: 0 expected, got {n_train}"
    assert not te.training, "fully frozen should be eval()"

    print("    OK — train_full / embedding-only / frozen all set "
          "requires_grad correctly")


# --------------------------------------------------------------------- #
# Test 3. Multi-subtree wrapping via create_concord_optimizer
# --------------------------------------------------------------------- #

def _count_concord(root):
    return sum(1 for m in root.modules()
               if isinstance(m, (ConcordLinearFused, ConcordConv2dFused)))


def _emb_weight_in_aux(opt, emb_weight):
    """True iff the given Parameter is in any aux-optimizer param group."""
    for g in opt._aux.param_groups:
        for p in g['params']:
            if p is emb_weight:
                return True
    return False


def test_unet_only_wrap():
    print("[3a] UNet-only wrap: TE Linears untouched")
    unet, te1, te2 = _StubUNet().cuda(), _StubTE().cuda(), _StubTE().cuda()
    container = SimpleNamespace(unet=unet)
    onetrainer_concord_patch.cache_model(container)
    pd = [{'name': 'unet', 'params': list(unet.parameters()),
           'lr': 1e-4, 'initial_lr': 1e-4}]
    create_concord_optimizer(pd, _TrainCfg(), _OptCfg())
    assert _count_concord(unet) >= 2, "unet should be wrapped"
    assert _count_concord(te1) == 0, "te1 should NOT be wrapped"
    assert _count_concord(te2) == 0, "te2 should NOT be wrapped"
    print(f"    OK — unet wrapped ({_count_concord(unet)} layers), TEs untouched")


def test_unet_plus_tes_wrap():
    print("[3b] UNet + TE1 + TE2 wrap")
    unet, te1, te2 = _StubUNet().cuda(), _StubTE().cuda(), _StubTE().cuda()
    container = SimpleNamespace(unet=unet, text_encoder_1=te1,
                                  text_encoder_2=te2)
    onetrainer_concord_patch.cache_model(container)
    pd = [
        {'name': 'unet', 'params': list(unet.parameters()),
         'lr': 1e-4, 'initial_lr': 1e-4},
        {'name': 'text_encoder_1', 'params': list(te1.parameters()),
         'lr': 1e-5, 'initial_lr': 1e-5},
        {'name': 'text_encoder_2', 'params': list(te2.parameters()),
         'lr': 1e-5, 'initial_lr': 1e-5},
    ]
    create_concord_optimizer(pd, _TrainCfg(), _OptCfg())
    assert _count_concord(unet) >= 2 and _count_concord(te1) >= 1 and \
            _count_concord(te2) >= 1, \
            "all three subtrees should be wrapped"
    print(f"    OK — unet={_count_concord(unet)} te1={_count_concord(te1)} "
          f"te2={_count_concord(te2)} concord layers")


def test_embedding_only_wrap():
    print("[3c] embedding-only: TE NOT wrapped, embedding.weight in aux")
    unet, te1 = _StubUNet().cuda(), _StubTE().cuda()
    # Freeze TE Linears (mirroring what _apply_te_freeze_policy would do
    # for embedding-only mode).
    for p in te1.parameters():
        p.requires_grad_(False)
    tok = te1.text_model.embeddings.token_embedding
    tok.weight.requires_grad_(True)
    container = SimpleNamespace(unet=unet, text_encoder_1=te1)
    onetrainer_concord_patch.cache_model(container)
    pd = [
        {'name': 'unet', 'params': list(unet.parameters()),
         'lr': 1e-4, 'initial_lr': 1e-4},
        # embedding-only group uses a name that does NOT match any
        # nn.Module attribute on the container, so the wrap-root filter
        # ignores it. The embedding.weight Parameter still ends up in
        # the aux optimizer via the param_dicts flow.
        {'name': 'text_encoder_1_embedding', 'params': [tok.weight],
         'lr': 1e-3, 'initial_lr': 1e-3},
    ]
    opt = create_concord_optimizer(pd, _TrainCfg(), _OptCfg())
    assert _count_concord(te1) == 0, \
        "TE must NOT be wrapped in embedding-only mode (concord's int " \
        "update would mutate frozen Linears each backward)"
    assert _emb_weight_in_aux(opt, tok.weight), \
        "token_embedding.weight should be in the aux optimizer"
    print(f"    OK — te1 has {_count_concord(te1)} concord layers (expected 0); "
          f"embedding.weight in aux")


# --------------------------------------------------------------------- #
# Test 4. End-to-end gradient flow with TE + embedding training
# --------------------------------------------------------------------- #

def test_te_e2e_training():
    print("[4] E2E: TE Linears + TE embedding both train via the optimizer")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    unet, te1 = _StubUNet().cuda(), _StubTE().cuda()
    container = SimpleNamespace(unet=unet, text_encoder_1=te1)
    onetrainer_concord_patch.cache_model(container)
    pd = [
        {'name': 'unet', 'params': list(unet.parameters()),
         'lr': 1e-3, 'initial_lr': 1e-3},
        {'name': 'text_encoder_1', 'params': list(te1.parameters()),
         'lr': 1e-3, 'initial_lr': 1e-3},
    ]
    opt = create_concord_optimizer(pd, _TrainCfg(), _OptCfg())

    # Snapshot a TE concord Linear's effective weight + the TE
    # embedding's fp32 Parameter so we can verify both moved.
    te_concord = [m for m in te1.modules()
                  if isinstance(m, ConcordLinearFused)][0]
    w_before = te_concord.weight.detach().clone()
    emb_before = te1.text_model.embeddings.token_embedding.weight.detach().clone()

    tokens = torch.randint(0, 128, (8, 5), device='cuda')
    y = torch.randint(0, 10, (8,), device='cuda')
    losses = []
    for _ in range(10):
        h = te1(tokens)
        logits = unet(h)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0], \
        f"loss should decrease: {losses[0]} -> {losses[-1]}"
    w_after = te_concord.weight.detach()
    emb_after = te1.text_model.embeddings.token_embedding.weight.detach()
    te_lin_drift = (w_after - w_before).abs().max().item()
    emb_drift = (emb_after - emb_before).abs().max().item()
    assert te_lin_drift > 0, \
        "TE concord Linear weight should have changed (concord int update)"
    assert emb_drift > 0, \
        "TE token_embedding should have changed (aux AdamW update)"
    print(f"    OK — loss {losses[0]:.3f} -> {losses[-1]:.3f}; "
          f"TE Linear concord drift = {te_lin_drift:.2e}, "
          f"TE embedding drift = {emb_drift:.2e}")


# --------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------- #

def main():
    if not torch.cuda.is_available():
        print("CUDA not available; concord requires CUDA. Skipping.")
        sys.exit(0)
    test_compute_te_mode()
    print()
    test_apply_te_freeze_policy()
    print()
    test_unet_only_wrap()
    print()
    test_unet_plus_tes_wrap()
    print()
    test_embedding_only_wrap()
    print()
    test_te_e2e_training()
    print()
    print("all TE / embedding training smoke checks passed.")


if __name__ == "__main__":
    main()
