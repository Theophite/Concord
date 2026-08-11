"""Doc-vs-code tests for CONCORD.md sections 1 ("The packed format & accumulator
semantics") and 4 ("Chase, leak, consolidation & the deploy weight") -- the DEPLOY
WEIGHT and the slow-path load.

Every test here is grounded in the live code in ../prototype_packed_b.py: each
asserts what the cited method ACTUALLY computes, so the suite PASSES against the
current source. Where the doc's stated number disagrees with the code, the test
asserts the CODE's value and the discrepancy is recorded below.

These tests call the REAL ConcordLinearPackedB methods (consolidated_weight,
get_weight, load_weights, load_weights_anchor) and the REAL compute_drift_cancel_C.
The kernel-launching parts of __init__ (_ensure_buffers -> materialize_packed_bf16,
a Triton kernel) are GPU-only, so to stay CPU-runnable we drive the pure-tensor
methods on a minimal stand-in object built with object.__new__ that carries only the
buffers those methods touch (packed_w, row_exp, col_exp, MANTISSA_BIAS, EXP_MIN/MAX).
This is faithful: load_weights / consolidated_weight / get_weight reference nothing
else on self, and load_weights' trailing self._resync_weight_buf() is a no-op when
_bf16_weight_buf is absent (it does getattr(..., None) and returns). A parallel
HAS_CUDA path constructs a real ConcordLinearPackedB and re-runs the same checks
through the fully-built object.

================================================================================
CONCORD.md assertions covered by THIS module
================================================================================
Section 1 (packed format & accumulator semantics):
  * Unpacking is sign-extending bit-shift: s_fast = packed >> 16;
    s_slow_i8 = (packed << 16) >> 24; v_slow_i8 = (packed << 24) >> 24
    (get_state / get_weight, prototype_packed_b.py:2561-2596).  [test_unpack_*]
  * S_SLOW_FACTOR == V_SLOW_FACTOR == 128, MANTISSA_BIAS == 15
    (:45-49).  [test_module_constants]
  * Block-float recon: m_eff = s_slow*128 + s_fast + v_slow*128;
    weight = m_eff * 2^(row_exp+col_exp-15) (get_weight, :2561-2569).
    [test_get_weight_blockfloat_formula]
  * "The deploy drops s_fast": consolidated_weight() materializes only
    (s_slow+v_slow)*128*2^exp and is INVARIANT to s_fast (:2573-2589).
    [test_consolidated_*]
Section 4 (chase, leak, consolidation & the deploy weight):
  * consolidated_weight() == (s_slow*128 + v_slow*128) * 2^exp, slow path only
    (:2573-2589).  [test_consolidated_equals_slow_path_formula]
  * load_weights packs the mantissa into the SLOW path: |s_fast| <= 64,
    s_slow == v_slow within 1 (gap-zero, d_sv ~ 0), deploy ~= W from step 0
    (:2437-2482, :2466-2475).  [test_load_weights_*]
  * compute_drift_cancel_C returns the analytic C* at packed-B rates
    (:52-102; constructor default alpha=0.1, alpha_v_fast=0.001, mass_preserve=True).
    [test_drift_cancel_*]

================================================================================
Not unit-tested here (by inspection / empirical / cross-module)
================================================================================
  * EMPIRICAL: consolidated_weight beats live get_weight by ~0.04-0.06 val nats;
    s_fast settles to ~4-7% of weight mass; stable 10.8M..49M params (:17, :81,
    :182). Not falsifiable in a unit test -- inspected only.
  * EMPIRICAL: "doubling the anchor (s_slow + 2*v_slow) overshoots, is worse"
    (:182) -- a training-loss claim, not a structural one.
  * ">bf16 effective precision" (:77) -- an information-theoretic argument about
    the shared per-row/col exponent, not a single computed quantity.
  * The kernel-side decomposition d_fs/d_sv, the Wiener gate, chase/leak ticks,
    and the mass-preserving relaxation at 2*alpha_v_fast (:744-986) launch the
    GPU-only Triton apply kernel; covered (coarsely) by the kernel modules, not
    here. By inspection: d_sv = (s_slow_full - v_slow_full), "momentum is free"
    because it is just the int8-channel difference already in the word (:79).
  * Section 7 embedding ANCHOR init "deploys ~0" latent bug (init_tokens,
    concord_embedding_packed.py) is OUT OF SCOPE for sections 1&4 -- it belongs to
    the embedding module. (For a Linear anchor layer, load_weights_anchor puts the
    COARSE part in v_slow, so consolidated_weight ~= W to 8-bit, NOT ~0; the doc's
    own note at :2489-2490 says to deploy anchor layers via get_weight, which keeps
    the fine residual. test_anchor_consolidated_drops_fine_residual verifies that.)

================================================================================
DISCREPANCIES found between CONCORD.md prose and the code
================================================================================
  * drift_cancel_C default value. CONCORD.md section 2 (:97) and the
    compute_drift_cancel_C docstring HEADER (prototype_packed_b.py:69-70) both
    quote C* ~= 0.0091 "at packed-B rates" -- but that is the NON-mass-preserve
    formula L*rho/(1-L*alpha_vf). The CONSTRUCTOR default is mass_preserve=True
    (prototype_packed_b.py:2251-2253), whose formula L*2rho/(1-2rho) gives
    C* ~= 0.01804 -- which is what every real layer actually carries. Section 4
    (:173) and the docstring's own mass-preserve note (:101) correctly say
    "~0.018 at defaults". So 0.0091 is the LEGACY-branch value; the live default
    is ~0.018. Both are asserted below (mass_preserve True vs False), and the
    constructor's self.drift_cancel_C is checked to be the ~0.018 (True) value.

Run standalone (CPU, no GPU needed for the stand-in path):
  venv/Scripts/python.exe -m pytest \
    modules/util/optimizer/concord/tests/test_doc_deploy.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

# Mirror test_servo_cpu.py's standalone sys.path setup: OT-root = parents[5],
# plus the concord dir, so `import prototype_packed_b` works without installing.
OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import prototype_packed_b as ppb  # noqa: E402
from prototype_packed_b import ConcordLinearPackedB, compute_drift_cancel_C  # noqa: E402

HAS_CUDA = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# CPU stand-in: a minimal object that carries ONLY the buffers the pure-tensor
# methods touch, so we can call the REAL unbound methods without launching the
# Triton kernel that __init__/_ensure_buffers would.
# --------------------------------------------------------------------------- #
class _CpuCore:
    """Carrier for ConcordLinearPackedB's pure-tensor deploy/load methods on CPU."""

    MANTISSA_BIAS = ConcordLinearPackedB.MANTISSA_BIAS  # 15
    EXP_MIN = ConcordLinearPackedB.EXP_MIN
    EXP_MAX = ConcordLinearPackedB.EXP_MAX

    # bind the real, unmodified methods off the class
    consolidated_weight = ConcordLinearPackedB.consolidated_weight
    get_weight = ConcordLinearPackedB.get_weight
    get_state = ConcordLinearPackedB.get_state
    load_weights = ConcordLinearPackedB.load_weights
    load_weights_anchor = ConcordLinearPackedB.load_weights_anchor

    def __init__(self, out_features, in_features):
        # _bf16_weight_buf intentionally ABSENT -> _resync_weight_buf() is a no-op.
        self.packed_w = torch.zeros(out_features, in_features, dtype=torch.int32)
        self.row_exp = torch.zeros(out_features, dtype=torch.int8)
        self.col_exp = torch.zeros(in_features, dtype=torch.int8)

    def _resync_weight_buf(self):
        # No _bf16_weight_buf attribute -> mirrors the real method's None-guard
        # early return; no kernel launch on CPU.
        return None


def _pack(s_fast, s_slow_i8, v_slow_i8):
    """Pack three int fields exactly as load_weights does (:2476-2480)."""
    return (((s_fast & 0xFFFF) << 16)
            | ((s_slow_i8 & 0xFF) << 8)
            | (v_slow_i8 & 0xFF)).to(torch.int32)


def _scale_per_elem(core):
    """2^(row_exp + col_exp - MANTISSA_BIAS), the per-element block-float scale."""
    exp = (core.row_exp[:, None].to(torch.float32)
           + core.col_exp[None, :].to(torch.float32)
           - core.MANTISSA_BIAS)
    return torch.pow(2.0, exp)


# =========================================================================== #
# Section 1: module constants & the unpack / block-float reconstruction
# =========================================================================== #
def test_module_constants():
    # CONCORD.md :45-49 / :72 (MANTISSA_BIAS = 15)
    assert ppb.S_SLOW_FACTOR == 128
    assert ppb.V_SLOW_FACTOR == 128
    assert ppb.MANTISSA_BIAS == 15
    assert ConcordLinearPackedB.MANTISSA_BIAS == 15
    assert (ppb.INT8_MIN, ppb.INT8_MAX) == (-128, 127)
    assert (ppb.INT16_MIN, ppb.INT16_MAX) == (-32768, 32767)


def test_unpack_is_sign_extending_bitshift():
    # CONCORD.md :66 -- s_fast = packed>>16; s_slow_i8=(packed<<16)>>24;
    # v_slow_i8=(packed<<24)>>24, sign-extending. Use negative fields to prove
    # the sign extension (arithmetic shift), not just the masking.
    core = _CpuCore(2, 3)
    s_fast = torch.tensor([[-5000, 12345, -1], [32767, -32768, 7]], dtype=torch.int32)
    s_slow = torch.tensor([[-3, 100, -128], [127, -1, 0]], dtype=torch.int32)
    v_slow = torch.tensor([[-128, 5, 127], [-7, 1, -64]], dtype=torch.int32)
    core.packed_w = _pack(s_fast, s_slow, v_slow)

    gs_fast, gs_slow, gv_slow = core.get_state()
    assert torch.equal(gs_fast, s_fast)
    assert torch.equal(gs_slow, s_slow)
    assert torch.equal(gv_slow, v_slow)


def test_get_weight_blockfloat_formula():
    # CONCORD.md :71-72 -- m_eff = s_slow*128 + s_fast + v_slow*128;
    # weight = m_eff * 2^(row_exp + col_exp - 15). get_weight (:2561-2569).
    core = _CpuCore(2, 2)
    s_fast = torch.tensor([[10, -10], [3, 4]], dtype=torch.int32)
    s_slow = torch.tensor([[1, 2], [-1, 0]], dtype=torch.int32)
    v_slow = torch.tensor([[1, 2], [-1, 0]], dtype=torch.int32)
    core.packed_w = _pack(s_fast, s_slow, v_slow)
    core.row_exp = torch.tensor([3, 5], dtype=torch.int8)
    core.col_exp = torch.tensor([0, 2], dtype=torch.int8)

    m_eff = (s_slow * 128 + s_fast + v_slow * 128).to(torch.float32)
    expected = (m_eff * _scale_per_elem(core)).to(torch.bfloat16)
    assert torch.equal(core.get_weight(), expected)


# =========================================================================== #
# Section 1 & 4: the deploy weight = consolidated_weight() drops s_fast
# =========================================================================== #
def test_consolidated_equals_slow_path_formula():
    # CONCORD.md :175-180 -- consolidated_weight() == (s_slow*128 + v_slow*128)*2^exp.
    core = _CpuCore(2, 2)
    s_fast = torch.tensor([[20, -7], [5, 9]], dtype=torch.int32)
    s_slow = torch.tensor([[3, -2], [1, 4]], dtype=torch.int32)
    v_slow = torch.tensor([[3, -2], [1, 4]], dtype=torch.int32)
    core.packed_w = _pack(s_fast, s_slow, v_slow)
    core.row_exp = torch.tensor([4, 2], dtype=torch.int8)
    core.col_exp = torch.tensor([1, 0], dtype=torch.int8)

    m_slow = (s_slow * 128 + v_slow * 128).to(torch.float32)
    expected = (m_slow * _scale_per_elem(core)).to(torch.bfloat16)
    assert torch.equal(core.consolidated_weight(), expected)


def test_consolidated_drops_s_fast_invariant():
    # CONCORD.md :81, :175 -- "The deploy drops s_fast." consolidated_weight must
    # be INVARIANT to s_fast: a huge s_fast leaves it unchanged vs the slow-only value.
    core = _CpuCore(2, 2)
    s_slow = torch.tensor([[3, -2], [1, 4]], dtype=torch.int32)
    v_slow = torch.tensor([[3, -2], [1, 4]], dtype=torch.int32)
    core.row_exp = torch.tensor([4, 2], dtype=torch.int8)
    core.col_exp = torch.tensor([1, 0], dtype=torch.int8)

    # s_fast = 0 baseline
    core.packed_w = _pack(torch.zeros_like(s_slow), s_slow, v_slow)
    deploy_zero = core.consolidated_weight().clone()

    # s_fast = a big int16 velocity (non-saturating), same slow path
    big = torch.tensor([[30000, -30000], [12345, -9999]], dtype=torch.int32)
    core.packed_w = _pack(big, s_slow, v_slow)
    deploy_big = core.consolidated_weight()

    assert torch.equal(deploy_big, deploy_zero)
    # ... and get_weight (which KEEPS s_fast) DOES change -> proves the difference
    # is genuinely the s_fast drop, not a no-op tensor.
    assert not torch.equal(core.get_weight(), deploy_big)


# =========================================================================== #
# Section 4: load_weights packs the mantissa into the SLOW path
# =========================================================================== #
def _loaded_core(W):
    core = _CpuCore(W.shape[0], W.shape[1])
    core.load_weights(W)
    return core


def _sample_W(seed=0):
    g = torch.Generator().manual_seed(seed)
    # moderate magnitudes (~unit scale) so exponents are well-behaved
    return (torch.randn(8, 16, generator=g) * 0.3)


def test_load_weights_sfast_bounded_by_64():
    # CONCORD.md :75, :241, :447 -- only the sub-128 FINE residual lands in s_fast,
    # so |s_fast| <= 64 for every element (round-to-nearest of m/128 leaves <=64).
    core = _loaded_core(_sample_W(1))
    s_fast, _, _ = core.get_state()
    assert int(s_fast.abs().max()) <= 64


def test_load_weights_gap_zero_d_sv():
    # CONCORD.md :75, :241, :2439, :2468 -- even split s_slow == v_slow within one
    # unit (the initial gap d_sv ~= 0).
    core = _loaded_core(_sample_W(2))
    _, s_slow, v_slow = core.get_state()
    gap = (s_slow - v_slow).abs()
    assert int(gap.max()) <= 1


def test_load_weights_deploy_approx_W():
    # CONCORD.md :17, :75, :182, :2440 -- deploy = consolidated_weight() ~= W from
    # step 0 (slow path carries the coarse mantissa). The only loss is the dropped
    # fine residual |s_fast| <= 64 mantissa units => <= 64 * per-row scale, plus
    # bf16 rounding. Assert that exact, code-grounded bound elementwise.
    W = _sample_W(3)
    core = _loaded_core(W)
    deploy = core.consolidated_weight().to(torch.float32)
    scale = _scale_per_elem(core)
    # dropped fine residual is at most 64 mantissa units; allow +1 for the
    # coarse/even-split integer rounding, and a small bf16 relative cushion.
    tol = 65.0 * scale + 8e-3 * W.abs()
    assert torch.all((deploy - W).abs() <= tol)


def test_load_weights_get_weight_more_exact_than_deploy():
    # CONCORD.md :441 -- "the live weight is exact to 16 bits": get_weight() keeps
    # s_fast and so reconstructs W more tightly than the s_fast-dropping deploy.
    W = _sample_W(4)
    core = _loaded_core(W)
    live = core.get_weight().to(torch.float32)
    deploy = core.consolidated_weight().to(torch.float32)
    err_live = (live - W).abs().sum()
    err_deploy = (deploy - W).abs().sum()
    assert err_live <= err_deploy + 1e-6


def test_anchor_consolidated_drops_fine_residual():
    # CONCORD.md / load_weights_anchor docstring (:2489-2490): for an ANCHOR layer
    # the coarse part lives in v_slow and the fine residual in s_fast, so deploy via
    # get_weight (keeps s_fast) ~= W, while consolidated_weight() (drops s_fast)
    # yields ONLY the coarse 8-bit anchor -> looser. This is the documented "deploy
    # anchor layers via get_weight, not consolidated_weight" note for Linear layers.
    W = _sample_W(5)
    core = _CpuCore(W.shape[0], W.shape[1])
    core.load_weights_anchor(W)
    _, s_slow, v_slow = core.get_state()
    # anchor init: s_slow == 0 (whole coarse mantissa is the frozen v_slow anchor)
    assert int(s_slow.abs().max()) == 0
    live = core.get_weight().to(torch.float32)
    deploy = core.consolidated_weight().to(torch.float32)
    assert (live - W).abs().sum() <= (deploy - W).abs().sum() + 1e-6


# =========================================================================== #
# Section 2 & 4: compute_drift_cancel_C at the packed-B rates
# =========================================================================== #
def test_drift_cancel_mass_preserve_default_is_0p018():
    # CONCORD.md :173 & docstring :101 -- mass_preserve=True (the CONSTRUCTOR
    # default) gives C* = L*2rho/(1-2rho) ~= 0.018 at alpha=0.1, alpha_v_fast=0.001.
    C = compute_drift_cancel_C(0.1, 0.001, mass_preserve=True)
    assert abs(C - 0.018036) < 1e-4


def test_drift_cancel_legacy_branch_is_0p0091():
    # CONCORD.md :97 / docstring header :69-70 quote ~0.0091 -- that is the
    # NON-mass-preserve formula L*rho/(1-L*alpha_vf). Asserting it documents that
    # the 0.0091 in the prose is the LEGACY branch, not the live default.
    C = compute_drift_cancel_C(0.1, 0.001, mass_preserve=False)
    assert abs(C - 0.009082) < 1e-4


def test_drift_cancel_constructor_default_matches_mass_preserve():
    # The live layer carries the mass_preserve=True value (~0.018), NOT the
    # doc-prose 0.0091. ConcordLinearPackedB.__init__ sets
    # self.drift_cancel_C = compute_drift_cancel_C(alpha, alpha_v_fast,
    #                                              mass_preserve=True)  (:2251-2253).
    # Verify without launching the kernel by recomputing from the documented
    # constructor defaults (alpha=0.1, alpha_v_fast=0.001).
    expected = compute_drift_cancel_C(0.1, 0.001, mass_preserve=True)
    assert abs(expected - 0.018036) < 1e-4
    # the legacy/prose value is ~2x smaller -> they are NOT interchangeable
    legacy = compute_drift_cancel_C(0.1, 0.001, mass_preserve=False)
    assert expected > 1.9 * legacy


def test_drift_cancel_zero_alpha_v_fast_kills_C():
    # CONCORD.md :259, :2450-2451 -- freezing the anchor (alpha_v_fast=0) zeroes C*
    # (=> coherence gate collapses to coh==0). The universal "anchor vs winner"
    # selector hinges on this.
    assert compute_drift_cancel_C(0.1, 0.0, mass_preserve=True) == 0.0
    assert compute_drift_cancel_C(0.1, 0.0, mass_preserve=False) == 0.0


# =========================================================================== #
# HAS_CUDA: re-run the structural checks through a FULLY-CONSTRUCTED layer
# (its __init__ launches the Triton materialize kernel, so it is GPU-only).
# Coarse, robust assertions only -- no exact int-quantized numerics.
# =========================================================================== #
@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB.__init__ launches "
                                         "the Triton materialize kernel (GPU-only)")
def test_real_layer_consolidated_drops_s_fast():
    layer = ConcordLinearPackedB(16, 8, bias=False, device="cuda")
    W = (torch.randn(8, 16, device="cuda") * 0.3)
    layer.load_weights(W)

    deploy0 = layer.consolidated_weight().clone()
    # inject a big s_fast into every element, keep the slow channels intact
    sf, ss, vs = layer.get_state()
    big = torch.full_like(sf, 30000)
    layer.packed_w.copy_(_pack(big, ss, vs).to(layer.packed_w.device))
    deploy1 = layer.consolidated_weight()

    assert torch.equal(deploy1, deploy0)          # deploy is invariant to s_fast
    assert not torch.equal(layer.get_weight(), deploy1)  # live weight moved


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB.__init__ launches "
                                         "the Triton materialize kernel (GPU-only)")
def test_real_layer_load_weights_slow_path():
    layer = ConcordLinearPackedB(16, 8, bias=False, device="cuda")
    W = (torch.randn(8, 16, device="cuda") * 0.3)
    layer.load_weights(W)
    sf, ss, vs = layer.get_state()
    assert int(sf.abs().max()) <= 64                 # |s_fast| <= 64
    assert int((ss - vs).abs().max()) <= 1           # gap-zero d_sv ~ 0
    deploy = layer.consolidated_weight().to(torch.float32)
    assert torch.allclose(deploy, W.to(torch.float32), atol=0.05, rtol=0.05)


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB.__init__ launches "
                                         "the Triton materialize kernel (GPU-only)")
def test_real_layer_drift_cancel_C_default():
    # The live default carries the mass_preserve=True value (~0.018), confirming the
    # prose 0.0091 is the legacy branch, not what a constructed layer holds.
    layer = ConcordLinearPackedB(16, 8, bias=False, device="cuda")
    assert abs(float(layer.drift_cancel_C) - 0.018036) < 1e-3
