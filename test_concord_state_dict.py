"""Smoke test for _patch_concord_state_dict in onetrainer_concord_patch.py.

Verifies:
  1. state_dict() on a concord Linear / Conv2d returns ONLY {weight, bias},
     not the internal int16 buffers.
  2. The emitted weight tensor matches the layer's effective fp32 weight
     (the same reconstruction the BMA observer uses).
  3. load_state_dict() with a standard weight tensor round-trips: the
     re-loaded layer produces the same forward output as the original,
     up to one LSB of SR rounding (the load goes through .load_weights()
     which re-splits the fp32 W into s_slow/s_fast halves).
  4. The legacy buffer keys (s_slow etc.) DO NOT appear in the saved
     state_dict -- the converter would not match them.

Run:  python test_concord_state_dict.py
      (assumes the concord_* modules are in the same dir, which is
      already added to sys.path at the top of this file)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import onetrainer_concord_patch
onetrainer_concord_patch.install()

from concord_linear_fused import ConcordLinearFused, ConcordConv2dFused


def _reconstruct_W(m):
    """Live weight from a concord module's internal state. Must match
    materialize_bf16_weight / the forward kernel: that means INCLUDING
    v_slow_i8 * V_SLOW_FACTOR when the three-accumulator buffer is
    allocated. (Earlier versions of this helper omitted the v_slow
    term, which made the state_dict round-trip test self-blind to a
    real bug where _save_to_state_dict also omitted it — see
    test_three_accum_round_trip below for the regression coverage.)
    """
    exp = (m.row_exp[:, None] + m.col_exp[None, :]
           - m.MANTISSA_BIAS).float()
    mantissa = m.s_slow.to(torch.int32) + m.s_fast.to(torch.int32)
    v_slow_i8 = getattr(m, 'v_slow_i8', None)
    if v_slow_i8 is not None:
        factor = int(getattr(m, 'v_slow_factor', 128))
        mantissa = mantissa + v_slow_i8.to(torch.int32) * factor
    return mantissa.float() * torch.exp2(exp)


def test_linear():
    print("[test_linear] building 32x16 concord Linear...")
    m = ConcordLinearFused(32, 16, bias=True, device='cuda',
                             alpha=0.1, beta1=0.0, lr=0.0)
    W = torch.randn(16, 32, device='cuda') * 0.1
    m.load_weights(W)
    m.bias.data.copy_(torch.randn(16, device='cuda') * 0.01)

    sd = m.state_dict()
    print(f"[test_linear] state_dict keys: {sorted(sd.keys())}")
    expected = {"weight", "bias"}
    forbidden = {"s_slow", "s_fast", "row_exp", "col_exp", "vsign"}
    if set(sd.keys()) != expected:
        # OK to have *some* extras (e.g. nn.Module bookkeeping), but the
        # concord internals must NOT be there.
        leaked = set(sd.keys()) & forbidden
        if leaked:
            raise AssertionError(
                f"Concord internal buffers leaked into state_dict: {leaked}")
        print(f"  (note: extra keys present: "
              f"{set(sd.keys()) - expected})")
    print(f"[test_linear] weight shape: {sd['weight'].shape}  "
          f"dtype: {sd['weight'].dtype}")

    # Spot-check: emitted weight matches internal reconstruction.
    W_internal = _reconstruct_W(m)
    diff = (sd['weight'] - W_internal).abs().max().item()
    assert diff < 1e-6, f"weight mismatch: {diff}"
    print(f"[test_linear] weight matches internal reconstruction: "
          f"max_abs_diff={diff:.2e}")

    # Round-trip: build a fresh concord module and load the saved state.
    m2 = ConcordLinearFused(32, 16, bias=True, device='cuda',
                              alpha=0.1, beta1=0.0, lr=0.0)
    # Initialize m2 to garbage so we know the load actually overwrote.
    m2.load_weights(torch.zeros(16, 32, device='cuda'))
    m2.load_state_dict(sd)

    W2 = _reconstruct_W(m2)
    rt_diff = (W - W2).abs().max().item()
    print(f"[test_linear] round-trip weight max_abs_diff: {rt_diff:.2e} "
          f"(should be < 2 LSB ~= 2/16384 = 1.2e-4)")
    assert rt_diff < 2e-3, f"round-trip drift too large: {rt_diff}"

    b_diff = (m.bias - m2.bias).abs().max().item()
    print(f"[test_linear] round-trip bias max_abs_diff: {b_diff:.2e}")
    assert b_diff < 1e-6, f"bias mismatch: {b_diff}"
    print("[test_linear] OK")


def test_conv():
    print("[test_conv] building 4->8x3x3 concord Conv2d...")
    m = ConcordConv2dFused(4, 8, 3, stride=1, padding=1, bias=True,
                              device='cuda', alpha=0.1, beta1=0.0, lr=0.0)
    W4d = torch.randn(8, 4, 3, 3, device='cuda') * 0.1
    m.load_weights(W4d.reshape(8, 4 * 3 * 3))
    m.bias.data.copy_(torch.randn(8, device='cuda') * 0.01)

    sd = m.state_dict()
    print(f"[test_conv] state_dict keys: {sorted(sd.keys())}")
    forbidden = {"s_slow", "s_fast", "row_exp", "col_exp", "vsign"}
    leaked = set(sd.keys()) & forbidden
    if leaked:
        raise AssertionError(
            f"Concord internal buffers leaked into state_dict: {leaked}")

    print(f"[test_conv] weight shape: {sd['weight'].shape}  "
          f"dtype: {sd['weight'].dtype}")
    assert sd['weight'].shape == (8, 4, 3, 3), \
        f"conv weight should be 4D nn.Conv2d shape, got {sd['weight'].shape}"

    # Internal layer is 2D; saved weight should reshape correctly.
    W_internal_2d = _reconstruct_W(m)
    W_internal_4d = W_internal_2d.reshape(8, 4, 3, 3)
    diff = (sd['weight'] - W_internal_4d).abs().max().item()
    assert diff < 1e-6, f"conv weight mismatch: {diff}"
    print(f"[test_conv] weight matches internal reconstruction: "
          f"max_abs_diff={diff:.2e}")

    # Round-trip.
    m2 = ConcordConv2dFused(4, 8, 3, stride=1, padding=1, bias=True,
                               device='cuda', alpha=0.1, beta1=0.0, lr=0.0)
    m2.load_weights(torch.zeros(8, 4 * 3 * 3, device='cuda'))
    m2.load_state_dict(sd)

    W2 = _reconstruct_W(m2).reshape(8, 4, 3, 3)
    rt_diff = (W4d - W2).abs().max().item()
    print(f"[test_conv] round-trip weight max_abs_diff: {rt_diff:.2e}")
    assert rt_diff < 2e-3, f"round-trip drift too large: {rt_diff}"
    print("[test_conv] OK")


def test_module_tree():
    """The realistic case: an nn.Module that wraps concord layers, and
    we call .state_dict() on the whole tree. The concord children
    should emit weight/bias with the correct prefix."""
    print("[test_module_tree] building tree with conv_in.concord child...")

    class Tree(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = ConcordConv2dFused(4, 8, 3, padding=1,
                                                bias=True, device='cuda',
                                                alpha=0.1, beta1=0.0,
                                                lr=0.0)
            self.fc_out = ConcordLinearFused(8, 10, bias=True,
                                               device='cuda', alpha=0.1,
                                               beta1=0.0, lr=0.0)

    t = Tree()
    t.conv_in.load_weights(
        torch.randn(8, 4 * 3 * 3, device='cuda') * 0.1)
    t.fc_out.load_weights(torch.randn(10, 8, device='cuda') * 0.1)

    sd = t.state_dict()
    keys = sorted(sd.keys())
    print(f"[test_module_tree] state_dict keys: {keys}")
    must_have = {"conv_in.weight", "conv_in.bias",
                 "fc_out.weight", "fc_out.bias"}
    assert must_have.issubset(set(keys)), \
        f"missing: {must_have - set(keys)}"
    must_not_have = {"conv_in.s_slow", "conv_in.s_fast",
                     "fc_out.s_slow", "fc_out.s_fast"}
    leaked = must_not_have & set(keys)
    assert not leaked, f"leaked concord internals: {leaked}"
    print("[test_module_tree] conv_in.weight / fc_out.weight present, "
          "no concord internals leaked. OK")
    print(f"[test_module_tree] conv_in.weight shape: "
          f"{sd['conv_in.weight'].shape}")


def _materialize_live_weight(m):
    """Live weight via the SAME path forward() / .weight uses (the
    Triton recon kernel). Independent of `_reconstruct_W` so we can
    cross-check the two. Returns fp32 for stable comparison."""
    from concord_triton_fused import materialize_bf16_weight
    w_bf16 = materialize_bf16_weight(
        m.s_slow, m.s_fast, m.row_exp, m.col_exp,
        mantissa_bias=m.MANTISSA_BIAS,
        v_slow=getattr(m, 'v_slow_i8', None),
        v_slow_factor=int(getattr(m, 'v_slow_factor', 128)))
    return w_bf16.float()


def test_three_accum_round_trip():
    """Regression for the bug where _save_to_state_dict reconstructed
    W = (s_slow + s_fast) * 2^exp and silently dropped the v_slow term.
    Before the fix this test (which actually loads non-zero values
    into v_slow_i8) caught the divergence between the saved weight
    and the live weight that materialize_bf16_weight emits.

    Construction: 1/3-1/3-1/3 split across s_slow / s_fast / v_slow.
    This is the steady-state at zero gradient AND the Bayesian-prior
    fine-tune init we're proposing, so it's the realistic case to
    cover. With v_slow_i8 carrying a full third of the weight,
    pre-fix saves dropped ~33% of each weight value — diff ≫ bf16
    floor, easy to catch.
    """
    print("[test_three_accum_round_trip] enable_v_slow_i8 + 1/3 split")
    torch.manual_seed(0)
    m = ConcordLinearFused(32, 16, bias=True, device='cuda',
                             alpha=0.1, beta1=0.0, lr=0.0)
    W_target = (torch.randn(16, 32, device='cuda') * 0.05).float()
    m.enable_v_slow_i8()
    with torch.no_grad():
        max_abs_row = W_target.abs().max(dim=1).values.clamp(min=1e-30)
        m.row_exp.copy_(
            torch.ceil(torch.log2(max_abs_row) + 1.0)
            .clamp(m.EXP_MIN, m.EXP_MAX).to(torch.int8))
        m.col_exp.zero_()
        exp = (m.row_exp[:, None] + m.col_exp[None, :]
               - m.MANTISSA_BIAS).float()
        scale = torch.pow(2.0, exp)
        m_total = (W_target / scale).round().to(torch.int32)
        # 1/3 each.
        target_slow = (m_total.float() / 3).round().to(torch.int32)
        target_fast = (m_total.float() / 3).round().to(torch.int32)
        target_v_full = m_total - target_slow - target_fast
        v_i8 = (target_v_full.float() / m.v_slow_factor).round() \
                .to(torch.int32).clamp(-128, 127)
        # Absorb the int8-quantisation residual back into s_slow + s_fast
        # so the live weight matches W_target exactly (within bf16).
        actual_v_full = v_i8 * m.v_slow_factor
        spill = target_v_full - actual_v_full
        spill_half = (spill / 2).round().to(torch.int32)
        target_slow = target_slow + spill_half
        target_fast = target_fast + (spill - spill_half)
        m.s_slow.copy_(target_slow.clamp(m.INT16_MIN, m.INT16_MAX)
                        .to(torch.int16))
        m.s_fast.copy_(target_fast.clamp(m.INT16_MIN, m.INT16_MAX)
                        .to(torch.int16))
        m.v_slow_i8.copy_(v_i8.to(torch.int8))
    v_slow_max = m.v_slow_i8.abs().max().item()
    assert v_slow_max >= 20, \
        f"v_slow_i8 should carry meaningful signal in this test " \
        f"(max should be ~|m_total|/3/factor ≈ 40); got {v_slow_max}"
    print(f"    v_slow_i8 carries 1/3 of weight mass: max |v_slow_i8| "
          f"= {v_slow_max} (range ±127)")

    # The live weight from materialize_bf16_weight (the forward path).
    W_live = _materialize_live_weight(m)
    # The state_dict-emitted weight.
    sd = m.state_dict()
    W_saved = sd['weight'].float()
    # Both helpers should match: _reconstruct_W (with v_slow), the
    # forward kernel (with v_slow), and the state_dict save (with
    # v_slow). The bug was that the state_dict save omitted v_slow,
    # so W_saved would differ from W_live by v_slow_full * 2^exp.
    diff_live_vs_saved = (W_live - W_saved).abs().max().item()
    print(f"    max |W_live - W_saved| = {diff_live_vs_saved:.3e}  "
          f"(bf16 quantisation floor at this magnitude)")
    # Tolerance: bf16's 7-bit mantissa = ~8e-3 worst case at max |W|.
    # The state_dict emits fp32; the forward emits bf16. Difference
    # is purely the bf16 rounding of the forward path.
    assert diff_live_vs_saved < 2e-2, \
        f"state_dict W disagrees with materialize_bf16_weight beyond " \
        f"bf16 rounding: {diff_live_vs_saved}"

    # Round-trip: load into a fresh module that ALSO has v_slow_i8
    # allocated (so the load knows to zero v_slow on the load path).
    m2 = ConcordLinearFused(32, 16, bias=True, device='cuda',
                              alpha=0.1, beta1=0.0, lr=0.0)
    m2.enable_v_slow_i8()
    # Pre-populate m2's v_slow_i8 with garbage so we can verify the
    # load path actually clears it (otherwise the test wouldn't
    # notice if load left stale v_slow content lying around).
    with torch.no_grad():
        m2.v_slow_i8.copy_(
            torch.randint(-50, 50, m2.v_slow_i8.shape,
                          dtype=torch.int8, device='cuda'))
    m2.load_state_dict(sd)
    # After load: m2's live weight should equal the saved W (within
    # SR rounding) and m2.v_slow_i8 should be all zeros (the long-
    # time signal was lost on save — see _load_from_state_dict docs).
    assert (m2.v_slow_i8 == 0).all().item(), \
        "v_slow_i8 should be zeroed on load to keep live weight = " \
        "saved W (the saved W already includes the original v_slow " \
        "contribution; if v_slow_i8 stayed non-zero, it would be " \
        "added on top and the live weight would diverge)."
    W2_live = _materialize_live_weight(m2)
    rt_diff = (W_live - W2_live).abs().max().item()
    print(f"    round-trip live-weight max_abs_diff: {rt_diff:.3e}  "
          f"(SR + bf16 quantisation floor)")
    assert rt_diff < 2e-2, \
        f"round-trip live-weight diff exceeds bf16 floor: {rt_diff}"
    print("[test_three_accum_round_trip] OK — three-accumulator save/"
          "load preserves the live weight and zeroes v_slow on load")


def test_slow_state_distributions():
    """Sweep across plausible (s_slow, s_fast, v_slow) initial
    distributions and verify each round-trips through state_dict.
    Coverage motivated by the fine-tuning prior discussion: at init
    we may want very different mass splits between accumulators
    (e.g. 1/3-1/3-1/3 for the Bayesian-prior fine-tune init vs
    1/2-1/2-0 for the from-scratch init). All should save+load
    cleanly without drift between the live and saved weights.
    """
    print("[test_slow_state_distributions] sweep over initial mass splits")
    torch.manual_seed(1)
    in_f, out_f = 32, 16

    # Build a canonical pretrained-like weight at moderate magnitude,
    # then for each split spec construct a module with that
    # accumulator distribution.
    W_target_fp = (torch.randn(out_f, in_f, device='cuda') * 0.05).float()

    # Each spec is (label, fraction_in_s_slow, fraction_in_s_fast,
    # fraction_in_v_slow). They must sum to 1.0; the test verifies the
    # composite live weight matches W_target and saves/loads cleanly.
    specs = [
        ("from-scratch (1/2, 1/2, 0)",       0.5,    0.5,    0.0),
        ("Bayesian fine-tune (1/3, 1/3, 1/3)", 1/3,  1/3,    1/3),
        ("very-slow-heavy (0, 0, 1)",        0.0,    0.0,    1.0),
        ("slow-heavy (1, 0, 0)",             1.0,    0.0,    0.0),
        ("fast-heavy (0, 1, 0)",             0.0,    1.0,    0.0),
        ("post-warmup (1/4, 1/4, 1/2)",      0.25,   0.25,   0.5),
        ("mid-drift (0.4, 0.5, 0.1)",        0.4,    0.5,    0.1),
    ]

    for label, f_slow, f_fast, f_v in specs:
        assert abs(f_slow + f_fast + f_v - 1.0) < 1e-6, \
            f"spec '{label}' fractions don't sum to 1: {f_slow+f_fast+f_v}"
        m = ConcordLinearFused(in_f, out_f, bias=True, device='cuda',
                                 alpha=0.1, beta1=0.0, lr=0.0)
        m.enable_v_slow_i8()
        # Pick the envelope from the full target weight (so the per-
        # row exponent fits the eventual sum, not just one component).
        with torch.no_grad():
            max_abs_row = W_target_fp.abs().max(dim=1).values.clamp(min=1e-30)
            row_exp = (torch.ceil(torch.log2(max_abs_row) + 1.0)
                       .clamp(m.EXP_MIN, m.EXP_MAX).to(torch.int8))
            m.row_exp.copy_(row_exp)
            m.col_exp.zero_()
            exp = (m.row_exp[:, None] + m.col_exp[None, :]
                   - m.MANTISSA_BIAS).float()
            scale = torch.pow(2.0, exp)
            # Total mantissa we want to represent.
            m_total = (W_target_fp / scale).round().to(torch.int32)
            # Split it according to the spec.
            target_v_full = (m_total.float() * f_v).round().to(torch.int32)
            # v_slow_i8 = target_v_full / V_SLOW_FACTOR, clamped to int8.
            target_v_i8 = (target_v_full // m.v_slow_factor).to(torch.int32).clamp(-128, 127)
            actual_v_full = target_v_i8 * m.v_slow_factor
            # Residual from int8 quantisation goes into s_slow + s_fast
            # at the requested split ratio.
            remainder = m_total - actual_v_full
            target_slow_extra = (remainder.float() * (f_slow / max(f_slow + f_fast, 1e-30))
                                  ).round().to(torch.int32)
            target_slow = (m_total.float() * f_slow).round().to(torch.int32) - actual_v_full + target_slow_extra - target_slow_extra
            # Simpler: just give s_slow its fraction of the original
            # mantissa, s_fast its fraction, and v_slow what's left.
            target_slow_int = (m_total.float() * f_slow).round().to(torch.int32)
            target_fast_int = (m_total.float() * f_fast).round().to(torch.int32)
            target_v_full_int = m_total - target_slow_int - target_fast_int
            target_v_i8 = (target_v_full_int.float() / m.v_slow_factor
                            ).round().to(torch.int32).clamp(-128, 127)
            # Absorb the int8 rounding remainder back into s_slow + s_fast
            # to keep the live weight ≈ W_target exact.
            actual_v_full = target_v_i8 * m.v_slow_factor
            spilled = target_v_full_int - actual_v_full
            spilled_half = (spilled / 2).round().to(torch.int32)
            target_slow_int = target_slow_int + spilled_half
            target_fast_int = target_fast_int + (spilled - spilled_half)
            # Clamp to int16 range.
            m.s_slow.copy_(target_slow_int.clamp(m.INT16_MIN, m.INT16_MAX)
                            .to(torch.int16))
            m.s_fast.copy_(target_fast_int.clamp(m.INT16_MIN, m.INT16_MAX)
                            .to(torch.int16))
            m.v_slow_i8.copy_(target_v_i8.to(torch.int8))

        # Sanity: live weight matches W_target within bf16 / SR rounding.
        W_live = _materialize_live_weight(m)
        recon_diff = (W_live - W_target_fp).abs().max().item()
        # Save / load round-trip.
        sd = m.state_dict()
        m2 = ConcordLinearFused(in_f, out_f, bias=True, device='cuda',
                                  alpha=0.1, beta1=0.0, lr=0.0)
        m2.enable_v_slow_i8()
        m2.load_state_dict(sd)
        W2_live = _materialize_live_weight(m2)
        rt_diff = (W_live - W2_live).abs().max().item()
        v_slow_max = m.v_slow_i8.abs().max().item()
        print(f"    {label:35s}  recon={recon_diff:.2e}  "
              f"rt={rt_diff:.2e}  max|v_slow_i8|={v_slow_max}")
        # bf16 floor at magnitude ~max(|W_target|): W_target * 1e-2 worst case.
        bf16_floor = max(W_target_fp.abs().max().item() * 1e-2, 1e-3)
        assert recon_diff < bf16_floor, \
            f"[{label}] live-weight reconstruction beyond bf16 floor: " \
            f"recon={recon_diff:.3e}  bf16_floor={bf16_floor:.3e}"
        assert rt_diff < bf16_floor, \
            f"[{label}] save/load round-trip beyond bf16 floor: " \
            f"rt={rt_diff:.3e}  bf16_floor={bf16_floor:.3e}"
    print("[test_slow_state_distributions] OK — all 7 distributions "
          "round-trip cleanly")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        sys.exit(0)
    test_linear()
    print()
    test_conv()
    print()
    test_module_tree()
    print()
    test_three_accum_round_trip()
    print()
    test_slow_state_distributions()
    print()
    print("All state_dict patch tests passed.")
