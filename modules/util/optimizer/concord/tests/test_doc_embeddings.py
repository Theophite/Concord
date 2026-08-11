"""CONCORD.md Section 7 (token embeddings & control plane) -- assertions checked
against the ACTUAL code in concord_embedding_packed.py + control_plane.py.

Standalone (mirrors test_servo_cpu.py's sys.path setup): OT root = parents[5] and
the concord dir are inserted so `import prototype_packed_b` / `control_plane` work.

WHAT THIS MODULE COVERS (CONCORD.md Section 7 assertions, grounded in code):

  (a) ControlPlaneEmbedding routes each id by `kind` (control_plane.py:107-123):
        - kind 0 (base/frozen) passes through to base.weight  -> test_routing_base_passthrough
        - kind 1 (static zero / fixed) reads static_vals       -> test_routing_static_zero,
                                                                   test_routing_static_fixed
        - kind 2 (trainable) routes to cp.trainable            -> test_attach_trainable_kind_routing [CUDA]
      The `.weight` shim returns the unchanged base vocab (control_plane.py:70-75)
                                                            -> test_weight_shim_is_base
  (b) THE FIX (init_tokens non-anchor, concord_embedding_packed.py:218-225, fixed
      2026-06-20): load_weights packs the mantissa into the SLOW path so
      deploy = consolidated_weight() ~= the seed (NOT ~0, NOT a 2^40 blowup), norm
      pinned to the vocab-median target.                    -> test_nonanchor_init_deploys_seed [CUDA]
      The doc's "slow-path load -> s_slow == v_slow (gap-zero), |s_fast|<=64" claim
      (:241) is verified on load_weights directly, and the gap-stays-small claim
      post-init.            -> test_loadweights_slow_path_gap_zero_and_residual [CUDA],
                               test_nonanchor_post_init_gap_small_relative [CUDA]
  (c) _pin_norm on the load_weights state yields a well-defined norm and does NOT
      saturate row_exp (concord_embedding_packed.py:240-268). -> test_pin_norm_well_defined [CUDA]
  (d) XFAIL -- the ANCHOR init path (init_tokens anchor=True, :199-217) currently
      deploys ~0: it re-derives v_slow from the post-load s_fast (which is only the
      <=64 fine RESIDUAL, NOT the mantissa), so v_slow rounds to 0, s_slow stays 0,
      and consolidated_weight() = (s_slow+v_slow)*128 == 0. The doc itself documents
      the non-anchor variant of exactly this re-split bug as fixed; the anchor path
      still has it. Written as the test that SHOULD pass.   -> test_anchor_init_deploys_seed [CUDA, xfail]

NOT UNIT-TESTED (by inspection / empirical -- per task constraints):
  - "beats the live get_weight() by ~0.04-0.06 val nats", "s_fast settles to ~4-7%
    of weight mass" (CONCORD.md:81,182,194,247): EMPIRICAL training-outcome claims.
  - "_seen counts only gradient-bearing occurrences", "inflated row 0's count ~140x"
    (:247): backward runs through _PackedEmbStep, which drives the GPU kernel via
    core.apply_grad_step -- a Triton launch + a full backward; not a coarse, robust
    CPU/GPU unit assertion (verified by inspection at concord_embedding_packed.py:61-63).
  - "direct kernel launch, never a nested torch.autograd.backward()" (:239): a
    capture-safety property of _PackedEmbStep.backward (:84-89), verified by code
    inspection (the call is core.apply_grad_step(...), not core(x).backward(...)).
  - Caption-vocab / row_map sentinel / materialize_packed_embeddings_to_vectors
    (:249): those symbols live in concord_ot.py setup, not in this section's two
    modules; out of scope for an embedding/control-plane unit test.

DOC-vs-CODE DISCREPANCIES NOTED (the doc is imprecise, not the code):
  - CONCORD.md:241 cites the slow-path/|s_fast|<=64 behavior as "init_tokens, :218-225";
    in the actual file that is the NON-anchor branch (lines 218-225 indeed), correct.
  - CONCORD.md:99-101 (control_plane) describes forward routing "(forward, :107-123)";
    the line range matches the actual ControlPlaneEmbedding.forward. Correct.
  - The doc's `consolidated_weight()` "(s_slow+v_slow)*128*2^exp" (:175-180) matches
    prototype_packed_b.py:2582-2589 exactly (m_slow = s_slow*128 + v_slow*128).
"""
import sys
from pathlib import Path

import pytest
import torch

OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import torch.nn as nn  # noqa: E402

import prototype_packed_b as ppb  # noqa: E402,F401  (parity with test_servo_cpu.py)
from concord_embedding_packed import ConcordPackedEmbedding  # noqa: E402
from control_plane import ControlPlaneEmbedding  # noqa: E402
from prototype_packed_b import ConcordLinearPackedB  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
_CUDA = pytest.mark.skipif(not HAS_CUDA,
                           reason="ConcordLinearPackedB.__init__ launches the Triton "
                                  "materialize kernel (GPU-only); embedding ctor needs CUDA")


# -- (a) ControlPlaneEmbedding routing by kind -- CPU (no trainable needed) --------

def _cpu_base(dim=8, vocab=20):
    base = nn.Embedding(vocab, dim)
    with torch.no_grad():
        base.weight.copy_(torch.arange(vocab * dim, dtype=torch.float32).reshape(vocab, dim) * 0.01)
    return base


def test_routing_initial_all_base():
    # Fresh control plane: every id is kind 0 (base/frozen), idx 0.
    cp = ControlPlaneEmbedding(_cpu_base())
    assert int(cp.kind.sum().item()) == 0
    assert int(cp.idx.sum().item()) == 0
    assert cp.trainable is None


def test_routing_base_passthrough():
    # kind 0 ids forward exactly to base.weight.
    base = _cpu_base()
    cp = ControlPlaneEmbedding(base)
    ids = torch.tensor([[3, 1, 9]])
    out = cp.forward(ids)
    assert out.shape == (1, 3, base.weight.shape[1])
    assert torch.allclose(out[0, 0], base.weight[3])
    assert torch.allclose(out[0, 2], base.weight[9])


def test_routing_static_zero():
    # set_zero -> kind 1, idx 0 (static row 0 == zero vector); forward emits 0.
    base = _cpu_base()
    cp = ControlPlaneEmbedding(base)
    cp.set_zero(5)
    assert int(cp.kind[5]) == 1
    assert int(cp.idx[5]) == 0
    out = cp.forward(torch.tensor([[5]]))
    assert float(out.abs().sum()) == 0.0


def test_routing_static_fixed():
    # set_fixed -> kind 1, a NEW static_vals row holding the fixed vector.
    base = _cpu_base()
    dim = base.weight.shape[1]
    cp = ControlPlaneEmbedding(base)
    vec = torch.full((dim,), 3.0)
    cp.set_fixed(7, vec)
    assert int(cp.kind[7]) == 1
    assert int(cp.idx[7]) == 1                       # row 0 is the zero row; fixed lands at row 1
    out = cp.forward(torch.tensor([[7]]))
    assert torch.allclose(out[0, 0], vec)


def test_routing_mixed_batch_independent():
    # base + zero + fixed in one batch route independently.
    base = _cpu_base()
    dim = base.weight.shape[1]
    cp = ControlPlaneEmbedding(base)
    cp.set_zero(5)
    cp.set_fixed(7, torch.full((dim,), 2.0))
    out = cp.forward(torch.tensor([[3, 5, 7]]))
    assert torch.allclose(out[0, 0], base.weight[3])    # base passthrough
    assert float(out[0, 1].abs().sum()) == 0.0          # zero
    assert torch.allclose(out[0, 2], torch.full((dim,), 2.0))  # fixed


def test_weight_shim_is_base():
    # The .weight shim returns the UNCHANGED base vocab (control_plane.py:70-75).
    base = _cpu_base()
    cp = ControlPlaneEmbedding(base)
    assert cp.weight is base.weight
    cp.set_zero(5)                                       # routing change must not touch the vocab
    assert cp.weight is base.weight
    assert torch.equal(cp.weight, base.weight)


# -- (a, cont) trainable routing (kind 2) -- needs the packed core -> CUDA ----------

@_CUDA
def test_attach_trainable_kind_routing():
    dim, vocab = 16, 40
    base = nn.Embedding(vocab, dim).to("cuda")
    with torch.no_grad():
        base.weight.copy_(torch.randn(vocab, dim, device="cuda"))
    median = base.weight.float().norm(dim=1).median().item()
    cp = ControlPlaneEmbedding(base)
    tids = [10, 12]
    inits = base.weight[tids].detach().clone()
    cp.attach_trainable(tids, inits, lr=5e-3, target_norm=median, anchor=False)
    # ids routed to kind 2, idx == attach order.
    assert int(cp.kind[10]) == 2 and int(cp.idx[10]) == 0
    assert int(cp.kind[12]) == 2 and int(cp.idx[12]) == 1
    assert isinstance(cp.trainable, ConcordPackedEmbedding)
    assert cp.trainable.K == len(tids)
    # forward: base id still passes through; a trainable id is served by cp.trainable
    # (deploy path), so it differs from the raw base row.
    out = cp.forward(torch.tensor([[3, 10]], device="cuda"))
    assert torch.allclose(out[0, 0].float(), base.weight[3].float())
    assert not torch.allclose(out[0, 1].float(), base.weight[10].float(), atol=1e-3)


# -- (b) THE FIX: non-anchor init_tokens deploys ~= the seed (not 0, not a blowup) --

def _seeded_embedding(K=4, dim=16, anchor=False, seed=2):
    torch.manual_seed(seed)
    base = torch.randn(40, dim, device="cuda")
    median = base.norm(dim=1).median().item()
    s = base[:K].clone()
    emb = ConcordPackedEmbedding(K, dim, device="cuda", lr=5e-3, target_norm=median)
    emb.init_tokens(init=s.clone(), anchor=anchor)
    return emb, s, median


@_CUDA
def test_nonanchor_init_deploys_seed():
    # The 2026-06-20 fix: deploy is the seed's direction at the median norm --
    # NOT ~0 (the old re-split bug collapsed it) and NOT a 2^40 blowup.
    emb, seed, median = _seeded_embedding(anchor=False)
    dep = emb.deploy_weight().float()
    assert torch.isfinite(dep).all()
    norms = dep.norm(dim=1)
    # every row well clear of zero, and pinned to ~the vocab median (not exploded).
    assert (norms > 0.25 * median).all(), norms.tolist()
    assert (norms < 4.0 * median).all(), norms.tolist()
    assert torch.allclose(norms, torch.full_like(norms, median), rtol=0.2), norms.tolist()
    # direction preserved: deploy points along the seed.
    cos = torch.cosine_similarity(dep, seed.float(), dim=1)
    assert (cos > 0.9).all(), cos.tolist()


@_CUDA
def test_loadweights_slow_path_gap_zero_and_residual():
    # CONCORD.md:241 / load_weights docstring (prototype_packed_b.py:2467-2475): the
    # mantissa is packed into the SLOW path with an EVEN coarse split, so the telescope
    # gap d_sv = s_slow - v_slow is in {-1,0,1} (gap-zero) and only the fine residual
    # (|s_fast| <= 64) stays in s_fast. Tested on load_weights DIRECTLY -- the op the
    # doc cites -- before init_tokens' _pin_norm re-rounds the fields.
    torch.manual_seed(3)
    dim, K = 16, 4
    core = ConcordLinearPackedB(dim, K, bias=False, device="cuda")
    W = torch.randn(K, dim, device="cuda")
    core.load_weights(W)
    s_fast, s_slow, v_slow = core.get_state()
    assert (s_slow - v_slow).abs().max().item() <= 1, \
        "even coarse split -> |s_slow - v_slow| in {-1,0,1}"
    assert int(s_fast.abs().max().item()) <= 64, \
        "only the fine residual (|s_fast| <= 64) stays in s_fast"
    # and the dropped-s_fast deploy still reproduces W (slow path carries the mantissa).
    dep = core.consolidated_weight().float()
    assert (dep - W).norm() / W.norm().clamp_min(1e-12) < 0.05


@_CUDA
def test_nonanchor_post_init_gap_small_relative():
    # After the full non-anchor init_tokens (load_weights + _pin_norm), the telescope
    # gap is still ~0 RELATIVE to the consolidated mass: ||d_sv|| << ||deploy||, the
    # block-float meaning of the doc's "s_slow == v_slow, gap-zero". (_pin_norm re-rounds
    # all three fields independently, so the integer gap can reach 2, but it stays tiny
    # against the slow magnitude.)
    emb, _seed, _median = _seeded_embedding(anchor=False)
    _s_fast, s_slow, v_slow = emb.core.get_state()
    d_sv = (s_slow - v_slow).float() * ppb.S_SLOW_FACTOR        # gap in mantissa units
    m_slow = (s_slow + v_slow).float() * ppb.S_SLOW_FACTOR      # consolidated mantissa
    ratio = d_sv.norm() / m_slow.norm().clamp_min(1e-12)
    assert ratio.item() < 0.1, f"||d_sv|| / ||deploy|| = {ratio.item():.4f} not ~0"


# -- (c) _pin_norm: well-defined norm, no row_exp saturation -----------------------

@_CUDA
def test_pin_norm_well_defined():
    # _pin_norm (run inside non-anchor init_tokens) pins the deploy norm to target
    # without saturating row_exp against EXP_MIN/EXP_MAX (the symptom of the div-by-~0
    # garbage block-float that the old re-split bug produced).
    emb, _seed, median = _seeded_embedding(anchor=False)
    re = emb.core.row_exp
    assert (re > ConcordLinearPackedB.EXP_MIN).all(), re.tolist()
    assert (re < ConcordLinearPackedB.EXP_MAX).all(), re.tolist()
    dep = emb.deploy_weight().float()
    assert torch.isfinite(dep).all()
    # norm well-defined and at the target.
    assert torch.allclose(dep.norm(dim=1), torch.full((emb.K,), median, device="cuda"),
                          rtol=0.2)


# -- (d) XFAIL -- the ANCHOR init path currently deploys ~0 (doc-flagged latent bug) -

@_CUDA
@pytest.mark.xfail(reason="known, flagged in CONCORD.md: the ANCHOR init path "
                          "(init_tokens anchor=True) re-derives v_slow from the post-load "
                          "s_fast (only the <=64 fine residual, not the mantissa), so v_slow "
                          "rounds to 0 and consolidated_weight()=(s_slow+v_slow)*128==0 -- "
                          "deploy collapses to ~0, the same re-split class of bug the "
                          "non-anchor path was fixed for on 2026-06-20.",
                   strict=True)
def test_anchor_init_deploys_seed():
    # What SHOULD happen: anchor mode freezes the init in v_slow so
    # deploy = init (+ gated delta, zero at step 0) ~= the seed direction, non-zero.
    emb, seed, median = _seeded_embedding(anchor=True)
    # founding-semantics flags ARE set correctly (this part works):
    assert emb.core.alpha_v_fast == 0.0
    assert emb.core.drift_cancel_C == 0.0
    dep = emb.deploy_weight().float()
    # ... but the deployed vector should carry the init -- it currently does NOT (==0).
    assert (dep.norm(dim=1) > 0.25 * median).all(), \
        f"anchor deploy collapsed to ~0: norms={dep.norm(dim=1).tolist()}"
    cos = torch.cosine_similarity(dep, seed.float(), dim=1)
    assert (cos > 0.9).all(), cos.tolist()
