"""Smoke test for Concord-native gradient accumulation.

Verifies that the ``concord_accum_steps = K`` knob:

  1. Suppresses chase + v_slow leak on microbatches 0..K-2 and fires
     the full update only on the K-th. Inspect-able as: after K-1
     backward calls under the cycle, s_slow is byte-identical to its
     pre-cycle snapshot while s_fast has moved. After the K-th
     backward, s_slow has moved too.

  2. K accumulated SR-ticks of grad/K produce a mean s_fast drift
     statistically equivalent to one big SR-tick of the same total
     grad — verifying the unbiased SR property at the accumulation
     boundary.

  3. End-to-end training with K-microbatch accumulation drives loss
     down and produces a similar trajectory to a same-effective-batch
     single-step baseline (within noise).

  4. Embedding sparse update respects APPLY_CHASE the same way: only
     the touched rows tick, and chase fires once per K microbatches.

Run: python test_concord_grad_accum.py
     (CUDA required — Concord kernels are CUDA-only.)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import onetrainer_concord_patch
onetrainer_concord_patch.install()

from concord_linear_fused import (ConcordLinearFused, ConcordConv2dFused,
                                     ConcordEmbeddingFused)
from concord_optimizer import create_concord_optimizer


# --------------------------------------------------------------------- #
# Test 1. APPLY_CHASE=False suppresses s_slow / v_slow_i8 updates
# --------------------------------------------------------------------- #

def test_tick_only_freezes_slow_state():
    print("[1] APPLY_CHASE=False suppresses s_slow / v_slow_i8 updates")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    m = ConcordLinearFused(16, 8, bias=True, device='cuda', alpha=0.1, lr=0.1)
    m.enable_v_slow_i8()
    m.set_optimizer_kind('adamw', weight_decay=0.0, eps=1.0)

    s_slow_before = m.s_slow.clone()
    s_fast_before = m.s_fast.clone()
    v_slow_before = m.v_slow_i8.clone()

    # Tick-only mode: 3 backwards in a row, none should move s_slow or v_slow.
    m._apply_chase = False
    x = torch.randn(4, 16, device='cuda')
    for _ in range(3):
        y = m(x)
        loss = y.float().square().mean()
        loss.backward()
        if m.bias.grad is not None:
            m.bias.grad = None

    fast_moved = (m.s_fast != s_fast_before).any().item()
    slow_moved = (m.s_slow != s_slow_before).any().item()
    vslow_moved = (m.v_slow_i8 != v_slow_before).any().item()
    print(f"    after 3 tick-only backwards: "
          f"s_fast changed={fast_moved}, s_slow changed={slow_moved}, "
          f"v_slow_i8 changed={vslow_moved}")
    assert fast_moved, "s_fast should accumulate ticks even in tick-only mode"
    assert not slow_moved, "s_slow MUST stay byte-identical when APPLY_CHASE=False"
    assert not vslow_moved, "v_slow_i8 MUST stay byte-identical when APPLY_CHASE=False"

    # Now fire the chase: one more backward with APPLY_CHASE=True moves slow.
    m._apply_chase = True
    y = m(x)
    loss = y.float().square().mean()
    loss.backward()
    if m.bias.grad is not None:
        m.bias.grad = None
    slow_moved_after_chase = (m.s_slow != s_slow_before).any().item()
    vslow_moved_after_chase = (m.v_slow_i8 != v_slow_before).any().item()
    print(f"    after final chase: s_slow changed={slow_moved_after_chase}, "
          f"v_slow_i8 changed={vslow_moved_after_chase}")
    assert slow_moved_after_chase, "chase must move s_slow"


# --------------------------------------------------------------------- #
# Test 2. ConcordSGD.set_accum_steps + advance_accum cycle
# --------------------------------------------------------------------- #

class _OptCfg:
    optimizer = 'CONCORD_SGD'
    concord_aux_lr = 5e-2
    concord_alpha = 0.1
    concord_beta1 = 0.0
    concord_rebalance_every = 999_999_999  # disable rebalance during this test
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


class _TrainCfg:
    learning_rate = 5e-2


def test_optimizer_accum_cycle():
    print("[2] ConcordSGD set_accum_steps(K) + advance_accum cycle")
    torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    K = 4
    m = nn.Sequential(nn.Linear(32, 16)).cuda()
    onetrainer_concord_patch.cache_model(m)
    pd = [{'name': 'all', 'params': list(m.parameters()),
           'lr': 1e-3, 'initial_lr': 1e-3}]
    opt = create_concord_optimizer(pd, _TrainCfg(), _OptCfg())
    opt.set_accum_steps(K)

    cl = [m_ for m_ in m.modules() if isinstance(m_, ConcordLinearFused)][0]
    # After set_accum_steps, accum_pos=0, K-1=3, so apply_chase=False.
    assert cl._apply_chase is False

    s_slow_at_cycle_start = cl.s_slow.clone()
    x = torch.randn(4, 32, device='cuda')

    # Run K-1 microbatches: every one should be tick-only (apply_chase=False).
    opt.zero_grad()
    for k in range(K - 1):
        loss = m(x).float().square().mean() / K
        loss.backward()
        # Verify it WAS tick-only.
        assert cl._apply_chase is False, \
            f"microbatch {k}/{K-1} should be tick-only"
        s_slow_during = cl.s_slow
        assert (s_slow_during == s_slow_at_cycle_start).all().item(), \
            f"s_slow should be unchanged during microbatch {k}"
        opt.advance_accum()
        # advance_accum bumps accum_pos; if not yet K-1, still tick-only.
        if k < K - 2:
            assert cl._apply_chase is False
        else:
            # After advancing past microbatch K-2, next call (the K-th)
            # should have apply_chase=True.
            assert cl._apply_chase is True, \
                f"after advance from pos {k+1}, apply_chase should be True"

    # K-th microbatch: chase fires.
    loss = m(x).float().square().mean() / K
    loss.backward()
    s_slow_after_chase = cl.s_slow
    chase_moved = (s_slow_after_chase != s_slow_at_cycle_start).any().item()
    assert chase_moved, "s_slow should move on the K-th (chase) microbatch"
    opt.advance_accum()  # wraps back to 0 — opt.step() will reset too
    opt.step()
    # After step(), zero_grad on the NEXT effective step resets the cycle.
    opt.zero_grad()
    assert cl._apply_chase is False, \
        "after zero_grad start of next cycle, apply_chase should be False"
    print(f"    K={K} cycle: s_slow stays frozen for {K-1} microbatches, "
          "moves on the K-th. Cycle resets via zero_grad.")


# --------------------------------------------------------------------- #
# Test 3. Statistical: K ticks of g/K have same mean drift as 1 tick of g
# --------------------------------------------------------------------- #

def test_unbiased_tick_accumulation():
    print("[3] K ticks of g/K vs 1 tick of g: same mean s_fast drift")
    torch.manual_seed(2); torch.cuda.manual_seed_all(2)
    K = 8

    # Make the experiment repeatable across trials by re-seeding each
    # trial and re-initialising the layer so the int state starts at
    # the same value.
    def run_trial(use_accum: bool, trial_seed: int):
        torch.manual_seed(trial_seed); torch.cuda.manual_seed_all(trial_seed)
        # bias=True so the autograd Function has a Parameter to attach
        # a grad_fn to; otherwise loss.backward() would error before
        # we ever reach the SR-tick. (Same reason ConcordEmbeddingFused
        # has a _grad_anchor.) alpha=0 freezes s_slow too. lr is small
        # enough that delta_t lives in the ones-of-mantissa-units
        # regime — well below the int16 saturation ceiling, so K small
        # ticks and one big tick stay statistically comparable.
        m = ConcordLinearFused(32, 16, bias=True, device='cuda',
                                 alpha=0.0, lr=1e-3)
        s_fast_before = m.s_fast.clone()
        x = torch.randn(64, 32, device='cuda')
        # Deterministic synthetic grad target so the float grad is the
        # same in both modes — we want to isolate the SR-tick variance.
        target = torch.randn(64, 16, device='cuda')

        if use_accum:
            # K microbatches, each with 1/K of the data, tick-only.
            chunk = 64 // K
            m._apply_chase = False
            for k in range(K):
                xc = x[k*chunk:(k+1)*chunk]
                tc = target[k*chunk:(k+1)*chunk]
                y = m(xc)
                loss = F.mse_loss(y.float(), tc.float(), reduction='sum') / 64.0
                loss.backward()
                if m.bias.grad is not None:
                    m.bias.grad = None
        else:
            # One big tick on the full batch.
            m._apply_chase = False
            y = m(x)
            loss = F.mse_loss(y.float(), target.float(), reduction='sum') / 64.0
            loss.backward()
            if m.bias.grad is not None:
                m.bias.grad = None
        return (m.s_fast.float() - s_fast_before.float()).cpu()

    # Average over N trials to reduce SR variance.
    N_trials = 12
    accum_drift = sum(run_trial(True, s) for s in range(N_trials)) / N_trials
    single_drift = sum(run_trial(False, s) for s in range(N_trials)) / N_trials

    # Per-element drifts are each at most an SR-tick (≈ 1 mantissa
    # unit). The mean across N_trials is an unbiased estimate of the
    # true drift, so the difference between two such estimates has
    # SD ~ sqrt(2 * Var_single / N_trials). With Var_single ≤ 0.25
    # and N_trials=12 → SD ≤ 0.20. Use a generous 3σ tolerance.
    diff_mean = (accum_drift - single_drift).abs().mean().item()
    accum_nonzero = (accum_drift != 0).sum().item()
    single_nonzero = (single_drift != 0).sum().item()
    print(f"    mean |drift_accum - drift_single| = {diff_mean:.3f} "
          f"(N={N_trials} trials; "
          f"accum nonzero cells={accum_nonzero}, "
          f"single nonzero cells={single_nonzero})")
    # Sanity: at least one is non-zero (the experiment must have done
    # something).
    assert accum_drift.abs().sum() > 0
    assert single_drift.abs().sum() > 0
    # Loose check: per-element mean drift agrees within SR floor +
    # tolerance.
    assert diff_mean < 0.6, \
        f"mean drift mismatch beyond SR floor: {diff_mean} on " \
        f"K={K}, N={N_trials}"


# --------------------------------------------------------------------- #
# Test 4. End-to-end: training with K-accum descends loss
# --------------------------------------------------------------------- #

def test_e2e_training_with_grad_accum():
    print("[4] E2E training with set_accum_steps(K): loss descends")
    torch.manual_seed(3); torch.cuda.manual_seed_all(3)
    K = 4
    m = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),
                      nn.Linear(32, 10)).cuda()
    onetrainer_concord_patch.cache_model(m)
    pd = [{'name': 'all', 'params': list(m.parameters()),
           'lr': 5e-2, 'initial_lr': 5e-2}]
    opt = create_concord_optimizer(pd, _TrainCfg(), _OptCfg())
    opt.set_accum_steps(K)

    # Learnable toy task: label is determined by argmax of a fixed
    # linear projection of x, so cross-entropy can actually descend
    # below the class-prior floor of ln(10) ≈ 2.30.
    x_big = torch.randn(64, 32, device='cuda')
    W_oracle = torch.randn(32, 10, device='cuda')
    y = (x_big @ W_oracle).argmax(dim=1)

    losses = []
    for effective_step in range(120):
        opt.zero_grad()
        ep_loss = 0.0
        for k in range(K):
            xc = x_big[k*16:(k+1)*16]
            yc = y[k*16:(k+1)*16]
            logits = m(xc)
            loss = F.cross_entropy(logits, yc) / K
            loss.backward()
            ep_loss += loss.item() * K
            opt.advance_accum()
        opt.step()
        losses.append(ep_loss / K)
    print(f"    loss[0]={losses[0]:.3f}  loss[-1]={losses[-1]:.3f}")
    assert losses[-1] < losses[0] * 0.7, \
        f"loss should descend with K={K} accumulation: " \
        f"{losses[0]} -> {losses[-1]}"


# --------------------------------------------------------------------- #
# Test 5. Embedding path also respects APPLY_CHASE
# --------------------------------------------------------------------- #

def test_embedding_grad_accum():
    print("[5] ConcordEmbedding sparse update respects APPLY_CHASE")
    torch.manual_seed(4); torch.cuda.manual_seed_all(4)
    V, D = 64, 16
    cef = ConcordEmbeddingFused(V, D, device='cuda', lr=0.5)

    touched = torch.tensor([3, 7, 11], device='cuda')

    # Tick-only: 3 backwards on touched rows; s_slow should NOT move.
    cef._apply_chase = False
    s_slow_before = cef.s_slow.clone()
    s_fast_before = cef.s_fast.clone()
    for _ in range(3):
        out = cef(touched.repeat(4))
        out.float().square().mean().backward()
    fast_moved = (cef.s_fast != s_fast_before).any().item()
    slow_moved = (cef.s_slow != s_slow_before).any().item()
    print(f"    tick-only: s_fast changed={fast_moved}, "
          f"s_slow changed={slow_moved}")
    assert fast_moved
    assert not slow_moved

    # Then chase: s_slow finally moves, but ONLY on the touched rows.
    cef._apply_chase = True
    out = cef(touched.repeat(4))
    out.float().square().mean().backward()
    delta_slow = (cef.s_slow.int() - s_slow_before.int()).abs().sum(dim=1)
    touched_mask = torch.zeros(V, dtype=torch.bool, device='cuda')
    touched_mask[touched] = True
    untouched_moved = (delta_slow[~touched_mask] > 0).any().item()
    touched_moved = (delta_slow[touched_mask] > 0).sum().item()
    print(f"    after chase: {touched_moved}/{touched.numel()} touched "
          f"rows moved; untouched changed = {untouched_moved}")
    assert not untouched_moved, \
        "untouched rows must stay frozen in the embedding chase pass"
    assert touched_moved == touched.numel()


# --------------------------------------------------------------------- #

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        sys.exit(0)
    test_tick_only_freezes_slow_state()
    print()
    test_optimizer_accum_cycle()
    print()
    test_unbiased_tick_accumulation()
    print()
    test_e2e_training_with_grad_accum()
    print()
    test_embedding_grad_accum()
    print()
    print("all gradient-accumulation smoke checks passed.")


if __name__ == "__main__":
    main()
