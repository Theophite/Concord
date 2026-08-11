"""CPU unit tests for the token control-plane + caption-vocab path (no GPU, no Triton).

This whole subsystem is OFF in a pure SDXL fine-tune: the base vocab is frozen, no
trainable embeddings exist, and caption-vocab is default-off. These tests exercise the
pure-torch / pure-Python routing + scanning logic WITHOUT building a packed core (which
would launch Triton in __init__) -- i.e. ControlPlaneEmbedding with trainable=None, the
init resolvers, the caption-id scanner, and the master `packed_embeddings_active` gate.

Run:  venv/Scripts/python.exe modules/util/optimizer/concord/tests/test_controlplane_caption_cpu.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

OT = Path(__file__).resolve().parents[5]
for p in (str(OT), str(OT / "modules" / "util" / "optimizer"),
          str(OT / "modules" / "util" / "optimizer" / "concord")):
    sys.path.insert(0, p)

import control_plane as cpmod
from control_plane import ControlPlaneEmbedding, resolve_init
from concord_embedding_packed import resolve_token_init

# concord_ot pulls the full controller graph; guard it so a heavy-import failure
# only skips the caption-vocab + gate checks.
try:
    import concord_ot
    from modules.util.enum.Optimizer import Optimizer
    CONCORD_OT_OK = True
except Exception as _e:                                    # noqa: BLE001
    CONCORD_OT_OK = False
    _CONCORD_OT_ERR = repr(_e)

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def skip(name, reason):
    print(f"  [SKIP] {name}  ({reason})")


class _Toks:
    def __init__(self, ids): self.input_ids = ids

class FakeTok:
    """Minimal CLIP-like tokenizer: whitespace split -> ids via a small vocab."""
    VOCAB = {"cat": 10, "red": 12, "dog": 11, "the": 5,
             "special": 49407, "oov": 1500, "penis": [7, 8]}   # multi-token, ids fit the fixture
    all_special_ids = [49407]
    vocab_size = 1000
    def __call__(self, text, add_special_tokens=False, truncation=False):
        ids = []
        for w in str(text).split():
            v = self.VOCAB.get(w)
            if v is None:
                continue
            ids += v if isinstance(v, list) else [v]
        return _Toks(ids)


# ---------------------------------------------------------------- #
print("== ControlPlaneEmbedding.forward routing (trainable=None: kinds 0/1 only, no kernel) ==")
torch.manual_seed(0)
base = nn.Embedding(8, 4)
cp = ControlPlaneEmbedding(base)
cp.set_zero(2)                                   # id 2 -> static zero
fixed_vec = torch.arange(4.0) + 100.0
cp.set_fixed(3, fixed_vec)                        # id 3 -> fixed vector
out = cp.forward(torch.tensor([[0, 2, 3, 5]])).detach()
check("routing shape is (*input, dim)", tuple(out.shape) == (1, 4, 4), f"shape={tuple(out.shape)}")
check("kind 0 (id 0) -> base.weight[0]", torch.equal(out[0, 0], base.weight[0].detach()))
check("kind 1 zero (id 2) -> zeros", torch.equal(out[0, 1], torch.zeros(4)))
check("kind 1 fixed (id 3) -> fixed vector", torch.equal(out[0, 2], fixed_vec))
check("kind 0 (id 5) -> base.weight[5]", torch.equal(out[0, 3], base.weight[5].detach()))

print("== set_zero / set_fixed / _grow buffer mutation ==")
check("set_zero -> kind==1, idx==0", int(cp.kind[2]) == 1 and int(cp.idx[2]) == 0)
check("set_fixed -> kind==1, idx points at the appended static row",
      int(cp.kind[3]) == 1 and torch.equal(cp.static_vals[cp.idx[3]], fixed_vec))
cp._grow(100)
check("_grow extends kind/idx, new ids default to kind 0 (base)",
      cp.kind.shape[0] >= 101 and int(cp.kind[100]) == 0, f"len={cp.kind.shape[0]}")


# ---------------------------------------------------------------- #
print("== resolve_init (control_plane: explicit init, NO subword-mean) ==")
big = nn.Embedding(50, 4)
bw = big.weight
check("tensor init -> reshaped [dim]", torch.equal(resolve_init(torch.ones(4), None, bw), torch.ones(4)))
check("'zero' init -> zeros", bool((resolve_init("zero", None, bw) == 0).all()))
check("None init -> zeros", bool((resolve_init(None, None, bw) == 0).all()))
r = resolve_init("random", None, bw)
check("'random' init -> [dim] nonzero", tuple(r.shape) == (4,) and bool((r != 0).any()))
check("single-token word -> base_weight[id]",
      torch.equal(resolve_init("cat", FakeTok(), bw), bw[10].float()))
try:
    resolve_init("penis", FakeTok(), bw)            # 2 CLIP tokens
    check("multi-token init word -> ValueError", False, "no error raised")
except ValueError:
    check("multi-token init word -> ValueError", True)

print("== resolve_token_init (embedding: str -> SUBWORD-MEAN, differs from resolve_init) ==")
emb = nn.Embedding(50, 4)
out_ri = resolve_token_init(["cat", torch.ones(4), None], FakeTok(), emb, device="cpu")
check("resolve_token_init shape [K, dim]", tuple(out_ri.shape) == (3, 4), f"shape={tuple(out_ri.shape)}")
check("str spec -> mean of subword-id rows", torch.allclose(out_ri[0], emb.weight[[10]].float().mean(0)))
check("tensor spec -> explicit vector", torch.equal(out_ri[1], torch.ones(4)))
# multi-token word does NOT error here (subword-mean), unlike resolve_init
out_multi = resolve_token_init(["penis"], FakeTok(), emb, device="cpu")
check("multi-token word subword-means (no error, unlike resolve_init)",
      torch.allclose(out_multi[0], emb.weight[[7, 8]].float().mean(0)))


# ---------------------------------------------------------------- #
print("== collect_caption_token_ids (caption scan, special/OOV stripped) ==")
if not CONCORD_OT_OK:
    skip("collect_caption_token_ids", "concord_ot import failed: " + _CONCORD_OT_ERR)
    skip("collect_caption_token_ids fallback", "concord_ot import failed")
    skip("packed_embeddings_active gate", "concord_ot import failed")
else:
    tmp = tempfile.mkdtemp(prefix="concord_capvocab_")
    open(os.path.join(tmp, "a.png"), "wb").close()
    with open(os.path.join(tmp, "a.txt"), "w", encoding="utf-8") as f:
        f.write("cat red special oov\n")            # special(49407)+oov(1500) must be stripped
    open(os.path.join(tmp, "dog.png"), "wb").close()  # no sidecar -> contributes nothing in 'sample'

    cfg_sample = SimpleNamespace(
        concepts=[{"enabled": True, "path": tmp, "include_subdirectories": False,
                   "text": {"prompt_source": "sample"}}],
        concept_file_name=None)
    ids = concord_ot.collect_caption_token_ids(cfg_sample, FakeTok())
    check("'sample' sidecar scan -> {cat, red}; special+OOV stripped", ids == {10, 12}, f"ids={sorted(ids)}")

    cfg_fname = SimpleNamespace(
        concepts=[{"enabled": True, "path": tmp, "include_subdirectories": False,
                   "text": {"prompt_source": "filename"}}],
        concept_file_name=None)
    ids_f = concord_ot.collect_caption_token_ids(cfg_fname, FakeTok())
    check("'filename' source -> token from the stem (dog -> 11)", 11 in ids_f, f"ids={sorted(ids_f)}")

    disabled = SimpleNamespace(
        concepts=[{"enabled": False, "path": tmp, "text": {"prompt_source": "sample"}}],
        concept_file_name=None)
    check("disabled concept contributes nothing",
          concord_ot.collect_caption_token_ids(disabled, FakeTok()) == set())

    print("== collect_caption_token_ids fallback (concepts None) ==")
    none_cfg = SimpleNamespace(concepts=None, concept_file_name=None)
    check("concepts=None, no file -> empty set (warns)",
          concord_ot.collect_caption_token_ids(none_cfg, FakeTok()) == set())

    bad = os.path.join(tmp, "not_json.txt")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("this is not json {{{")
    bad_cfg = SimpleNamespace(concepts=None, concept_file_name=bad)
    check("concepts=None, malformed concept file -> degrades to empty set",
          concord_ot.collect_caption_token_ids(bad_cfg, FakeTok()) == set())

    # ------------------------------------------------------------ #
    print("== packed_embeddings_active master gate (OFF in a pure fine-tune) ==")
    def mk_cfg(packed, train_any, capvocab):
        return SimpleNamespace(
            optimizer=SimpleNamespace(optimizer=Optimizer.CONCORD),
            concord_packed_embeddings=packed,
            train_any_embedding=lambda: train_any,
            concord_train_caption_vocab=capvocab)
    check("OFF: flag off -> inactive", concord_ot.packed_embeddings_active(mk_cfg(False, False, False)) is False)
    check("OFF: flag on but nothing to train -> inactive (pure fine-tune)",
          concord_ot.packed_embeddings_active(mk_cfg(True, False, False)) is False)
    check("ON: flag on + train_any_embedding -> active",
          concord_ot.packed_embeddings_active(mk_cfg(True, True, False)) is True)
    check("ON: flag on + caption-vocab -> active",
          concord_ot.packed_embeddings_active(mk_cfg(True, False, True)) is True)
    non_concord = SimpleNamespace(optimizer=SimpleNamespace(optimizer=None),
                                  concord_packed_embeddings=True,
                                  train_any_embedding=lambda: True,
                                  concord_train_caption_vocab=True)
    check("OFF: non-Concord optimizer -> inactive",
          concord_ot.packed_embeddings_active(non_concord) is False)


n_pass = sum(results)
print(f"{n_pass}/{len(results)} control-plane/caption-vocab CPU checks passed")
sys.exit(0 if n_pass == len(results) else 1)
