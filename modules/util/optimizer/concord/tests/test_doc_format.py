"""Verify CONCORD.md Section 1 ("The packed format & accumulator semantics")
against the ACTUAL code in prototype_packed_b.py.

Each test reads the real constants / unpack arithmetic and asserts what the code
ACTUALLY does, so the suite PASSES against the live source.

CONCORD.md assertions COVERED by this module (all CPU, int32 torch ops):
  - S_SLOW_FACTOR == V_SLOW_FACTOR == 128            (doc ":48-49")
  - MANTISSA_BIAS == 15                              (doc "weight = m_eff*2^(...-15)")
  - INT16 range == [-32768, 32767], INT8 range == [-128, 127]
  - the int8 slow channels are ±128-quantized coarse bits (factor 128).
  - Unpacking is sign-extending bit-shift (get_state, ":2594-2596"):
        s_fast    = packed_w >> 16
        s_slow_i8 = (packed_w << 16) >> 24
        v_slow_i8 = (packed_w << 24) >> 24
  - Packing is the inverse OR-of-masked-fields (load_weights, ":2466-2480"):
        ((s_fast & 0xFFFF) << 16) | ((s_slow_i8 & 0xFF) << 8) | (v_slow_i8 & 0xFF)
    and a hand-built pack->unpack round-trip recovers s_fast/s_slow/v_slow.
  - Live mantissa  m_eff = s_slow_i8*128 + s_fast + v_slow_i8*128   (get_weight ":2564-2565")
  - Telescope gap  d_sv  = (s_slow_full - v_slow_full) = (s_slow - v_slow)*128
    ("the two telescopes", ":744-745 / :2531").
  - The deploy weight drops s_fast:
        consolidated_weight() materializes (s_slow + v_slow)*128*2^exp, NO s_fast
        (":2573-2588"); get_weight() KEEPS s_fast (m_eff). The two differ exactly
        by s_fast*2^exp.
  - get_weight() recovers a hand-packed weight to block-float precision
    (m_eff * 2^(row_exp + col_exp - MANTISSA_BIAS), ":2561-2569").
  - load_weights packs the coarse part EVENLY across the two int8 slow channels so
    s_slow + v_slow == coarse and the initial gap d_sv ~= 0 (|s_slow - v_slow| <= 1),
    leaving the fine residual (|s_fast| <= 64) in s_fast (":2466-2480").

Not unit-tested (by inspection / out of scope for this module):
  - EMPIRICAL: "s_fast settles to ~4-7% of weight mass"; consolidated_weight
    "beats the live get_weight by ~0.04-0.06 val nats, stable 10.8M..49M"
    (":2577-2580", doc Section-1 prose) -- empirical claims, not unit-testable.
  - "doubling the anchor (s_slow + 2*v_slow) overshoots, is worse" -- empirical.
  - ">bf16 effective precision" qualitative claim about shared-exponent block-float.
  - The kernel's per-tile scale_fwd/scale_inv = exp2(+/-total_exp) (":735-737") and
    the Wiener gate's use of d_sv as signal (":744-745 sig = drift_cancel_C*d_sv)
    are GPU/kernel paths -- covered (if at all) by the kernel test module, not here.
  - Pure file:line citations (e.g. docstring at ":2181-2182", layout banner ":3-10").

DISCREPANCIES / notes found while grounding these tests:
  - The doc's symbol "s_slow_full" is not a stored field; it is s_slow_i8*128
    (and v_slow_full = v_slow_i8*128). The code computes these inline; there is no
    attribute by that name. Asserted via the arithmetic identity instead.
  - get_weight()/consolidated_weight() return bfloat16, so the s_fast contribution
    is only recoverable to bf16 precision; tests that compare the two weights use a
    bf16-aware tolerance rather than exact equality.
  - ConcordLinearPackedB.__init__ has NO `optimizer_kind` argument (signature is
    (in_features, out_features, bias=True, device='cuda', alpha, beta1, lr)); the
    doc's "optimizer_kind=='adamw' branch" lives in the kernel/recipe path, not the
    constructor. More importantly, __init__ -> _ensure_buffers() launches the
    materialize Triton kernel, so the class CANNOT be instantiated on CPU at all.
    The core Section-1 assertions (pack/unpack arithmetic, the m_eff/d_sv identities,
    the constants) are therefore tested as PURE CPU int32 arithmetic mirroring the
    exact source lines; the tests that exercise the live ConcordLinearPackedB methods
    (get_state/get_weight/consolidated_weight/load_weights) are GPU-gated with
    skipif(not HAS_CUDA) because constructing the layer requires the kernel.

Run standalone (mirrors test_servo_cpu.py sys.path setup):
  venv/Scripts/python.exe -m pytest \
      modules/util/optimizer/concord/tests/test_doc_format.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

# Mirror test_servo_cpu.py: OT-root = parents[5], plus the concord dir, so
# `import prototype_packed_b` resolves standalone.
OT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(OT))
sys.path.insert(0, str(OT / "modules" / "util" / "optimizer" / "concord"))

import prototype_packed_b as ppb  # noqa: E402

HAS_CUDA = torch.cuda.is_available()


# ── helpers ──────────────────────────────────────────────────────────────────
def pack(s_fast, s_slow_i8, v_slow_i8):
    """Pack three int32 tensors exactly as load_weights does (":2476-2480")."""
    return (
        ((s_fast & 0xFFFF) << 16)
        | ((s_slow_i8 & 0xFF) << 8)
        | (v_slow_i8 & 0xFF)
    )


def unpack(packed_w):
    """Unpack via the sign-extending shifts get_state uses (":2594-2596")."""
    s_fast = (packed_w >> 16)
    s_slow_i8 = (packed_w << 16) >> 24
    v_slow_i8 = (packed_w << 24) >> 24
    return s_fast, s_slow_i8, v_slow_i8


# A spread of edge values: signed extremes for each field plus interior values.
_SF = torch.tensor([[12345, -12345], [ppb.INT16_MAX, ppb.INT16_MIN], [0, 64]],
                   dtype=torch.int32)
_SS = torch.tensor([[100, -100], [ppb.INT8_MAX, ppb.INT8_MIN], [0, -1]],
                   dtype=torch.int32)
_VS = torch.tensor([[-50, 50], [ppb.INT8_MIN, ppb.INT8_MAX], [0, 1]],
                   dtype=torch.int32)


# ── 1: scale-factor constants ────────────────────────────────────────────────
def test_slow_factors_are_128():
    # CONCORD.md: "S_SLOW_FACTOR = V_SLOW_FACTOR = 128 (:48-49)".
    assert ppb.S_SLOW_FACTOR == 128
    assert ppb.V_SLOW_FACTOR == 128
    assert ppb.S_SLOW_FACTOR == ppb.V_SLOW_FACTOR


def test_mantissa_bias_is_15():
    # CONCORD.md: "weight = m_eff * 2^(row_exp + col_exp - MANTISSA_BIAS) # MANTISSA_BIAS = 15".
    assert ppb.MANTISSA_BIAS == 15


# ── 2: integer field ranges ──────────────────────────────────────────────────
def test_int16_range():
    # s_fast is the int16 field, "virtually non-saturating".
    assert ppb.INT16_MIN == -32768
    assert ppb.INT16_MAX == 32767


def test_int8_range():
    # both slow channels are int8.
    assert ppb.INT8_MIN == -128
    assert ppb.INT8_MAX == 127


def test_slow_channels_are_128_quantized_int8():
    # The doc: int8 slow channels add +/-128-quantized coarse bits, i.e. each int8
    # LSB carries exactly S_SLOW_FACTOR (=128) mantissa units.
    assert ppb.S_SLOW_FACTOR == 128 and ppb.V_SLOW_FACTOR == 128
    assert (ppb.INT8_MAX - ppb.INT8_MIN) == 255  # full signed-8-bit span


# ── 3: pack -> unpack round-trip recovers each field ─────────────────────────
def test_roundtrip_recovers_all_fields():
    packed = pack(_SF, _SS, _VS)
    assert packed.dtype == torch.int32
    sf, ss, vs = unpack(packed)
    assert torch.equal(sf, _SF), "s_fast not recovered"
    assert torch.equal(ss, _SS), "s_slow_i8 not recovered"
    assert torch.equal(vs, _VS), "v_slow_i8 not recovered"


def test_unpack_is_sign_extending():
    # Negative values in each field must come back negative (the (<<k)>>24 shifts
    # sign-extend an int32 tensor). Use the most-negative representable values.
    sf = torch.tensor([[ppb.INT16_MIN]], dtype=torch.int32)
    ss = torch.tensor([[ppb.INT8_MIN]], dtype=torch.int32)
    vs = torch.tensor([[ppb.INT8_MIN]], dtype=torch.int32)
    rsf, rss, rvs = unpack(pack(sf, ss, vs))
    assert int(rsf) == ppb.INT16_MIN < 0
    assert int(rss) == ppb.INT8_MIN < 0
    assert int(rvs) == ppb.INT8_MIN < 0


def test_fields_are_independent():
    # Mutating one field's bits must not bleed into the others.
    sf = torch.tensor([[0]], dtype=torch.int32)
    ss = torch.tensor([[0]], dtype=torch.int32)
    vs = torch.tensor([[0]], dtype=torch.int32)
    base = pack(sf, ss, vs)
    assert int(base) == 0
    # set s_fast = -1 only
    p = pack(torch.tensor([[-1]], dtype=torch.int32), ss, vs)
    rsf, rss, rvs = unpack(p)
    assert int(rsf) == -1 and int(rss) == 0 and int(rvs) == 0


# ── 4: m_eff and d_sv identities (the doc's accumulator algebra) ─────────────
def test_m_eff_identity():
    # CONCORD.md: m_eff = s_slow_i8*128 + s_fast + v_slow_i8*128  (get_weight :2564-2565).
    sf, ss, vs = unpack(pack(_SF, _SS, _VS))
    m_eff = ss * ppb.S_SLOW_FACTOR + sf + vs * ppb.V_SLOW_FACTOR
    expect = _SS * 128 + _SF + _VS * 128
    assert torch.equal(m_eff, expect)


def test_d_sv_identity():
    # CONCORD.md: d_sv = (s_slow_full - v_slow_full) = (s_slow - v_slow)*128.
    sf, ss, vs = unpack(pack(_SF, _SS, _VS))
    s_slow_full = ss * ppb.S_SLOW_FACTOR
    v_slow_full = vs * ppb.V_SLOW_FACTOR
    d_sv = s_slow_full - v_slow_full
    assert torch.equal(d_sv, (_SS - _VS) * 128)


# ── 5: end-to-end via the real ConcordLinearPackedB unpack methods (GPU) ─────
# NOTE: ConcordLinearPackedB.__init__ -> _ensure_buffers() launches the
# materialize Triton kernel, so the layer can only be built on CUDA. These tests
# are skipped on a CPU-only box; the pure-arithmetic tests above already pin the
# Section-1 algebra without the kernel.
def _make_layer(N=2, K=2):
    """A real ConcordLinearPackedB with a hand-written packed_w (CUDA)."""
    layer = ppb.ConcordLinearPackedB(K, N, bias=False, device='cuda')
    # ConcordLinearPackedB stores packed_w shaped [out, in] = [N, K].
    assert tuple(layer.packed_w.shape) == (N, K), layer.packed_w.shape
    return layer


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_get_state_matches_manual_unpack():
    layer = _make_layer()
    dev = layer.packed_w.device
    sf = torch.tensor([[1234, -5678], [ppb.INT16_MAX, ppb.INT16_MIN]], dtype=torch.int32, device=dev)
    ss = torch.tensor([[10, -20], [ppb.INT8_MAX, ppb.INT8_MIN]], dtype=torch.int32, device=dev)
    vs = torch.tensor([[-5, 7], [ppb.INT8_MIN, ppb.INT8_MAX]], dtype=torch.int32, device=dev)
    layer.packed_w.copy_(pack(sf, ss, vs))
    g_sf, g_ss, g_vs = layer.get_state()
    assert torch.equal(g_sf, sf)
    assert torch.equal(g_ss, ss)
    assert torch.equal(g_vs, vs)


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_get_weight_is_m_eff_blockfloat():
    # get_weight (:2561-2569) = m_eff * 2^(row_exp + col_exp - MANTISSA_BIAS).
    layer = _make_layer()
    dev = layer.packed_w.device
    sf = torch.tensor([[1000, -2000], [300, -400]], dtype=torch.int32, device=dev)
    ss = torch.tensor([[5, -6], [7, -8]], dtype=torch.int32, device=dev)
    vs = torch.tensor([[1, -2], [3, -4]], dtype=torch.int32, device=dev)
    layer.packed_w.copy_(pack(sf, ss, vs))
    layer.row_exp.zero_()
    layer.col_exp.zero_()
    w = layer.get_weight().to(torch.float32)
    m_eff = (ss * 128 + sf + vs * 128).to(torch.float32)
    exp = float(0 + 0 - ppb.MANTISSA_BIAS)
    expect = (m_eff * (2.0 ** exp)).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(w, expect)


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_consolidated_weight_drops_s_fast():
    # consolidated_weight (:2573-2588) = (s_slow + v_slow)*128 * 2^exp -- NO s_fast.
    layer = _make_layer()
    dev = layer.packed_w.device
    sf = torch.tensor([[8000, -8000], [4000, -4000]], dtype=torch.int32, device=dev)
    ss = torch.tensor([[10, -10], [20, -20]], dtype=torch.int32, device=dev)
    vs = torch.tensor([[5, -5], [6, -6]], dtype=torch.int32, device=dev)
    layer.packed_w.copy_(pack(sf, ss, vs))
    layer.row_exp.zero_()
    layer.col_exp.zero_()
    dep = layer.consolidated_weight().to(torch.float32)
    m_slow = (ss * 128 + vs * 128).to(torch.float32)
    exp = float(0 + 0 - ppb.MANTISSA_BIAS)
    expect = (m_slow * (2.0 ** exp)).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(dep, expect), "deploy weight is not the slow-only sum"


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_deploy_differs_from_live_by_s_fast():
    # get_weight keeps s_fast; consolidated_weight drops it. With nonzero s_fast the
    # two weights differ, and the difference is exactly s_fast * 2^exp (to bf16).
    layer = _make_layer()
    dev = layer.packed_w.device
    sf = torch.tensor([[6000, -6000], [3000, -3000]], dtype=torch.int32, device=dev)
    ss = torch.tensor([[2, -2], [4, -4]], dtype=torch.int32, device=dev)
    vs = torch.tensor([[1, -1], [2, -2]], dtype=torch.int32, device=dev)
    layer.packed_w.copy_(pack(sf, ss, vs))
    layer.row_exp.zero_()
    layer.col_exp.zero_()
    live = layer.get_weight().to(torch.float32)
    dep = layer.consolidated_weight().to(torch.float32)
    exp = 2.0 ** float(-ppb.MANTISSA_BIAS)
    s_fast_in_w = sf.to(torch.float32) * exp
    # live and dep are EACH separately rounded to bf16 (8-bit mantissa), so the
    # residual (live - dep) - s_fast carries up to a couple of bf16 ULPs of the
    # *live* magnitude: tol ~ 4 * |live| * 2^-8.
    diff = live - dep
    tol = 4.0 * live.abs() * (2.0 ** -8) + exp
    assert torch.all((diff - s_fast_in_w).abs() <= tol), \
        f"live-deploy should equal s_fast in W units; max err {(diff - s_fast_in_w).abs().max()}"
    # And dropping a nonzero s_fast genuinely changes the weight.
    assert not torch.equal(live, dep)


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_zero_s_fast_makes_live_equal_deploy():
    # With s_fast == 0 there is nothing to drop, so live == deploy exactly.
    layer = _make_layer()
    dev = layer.packed_w.device
    ss = torch.tensor([[3, -3], [5, -5]], dtype=torch.int32, device=dev)
    vs = torch.tensor([[1, -1], [2, -2]], dtype=torch.int32, device=dev)
    layer.packed_w.copy_(pack(torch.zeros_like(ss), ss, vs))
    layer.row_exp.zero_()
    layer.col_exp.zero_()
    assert torch.equal(layer.get_weight(), layer.consolidated_weight())


# ── 6: load_weights even-split semantics (s_slow + v_slow == coarse, d_sv ~= 0) ─
@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_load_weights_even_split_gap_zero():
    # CONCORD.md (:2466-2480): the coarse part is split EVENLY across the two int8
    # slow channels so s_slow + v_slow == coarse and the initial gap d_sv ~= 0
    # (|s_slow - v_slow| <= 1); the fine residual (|s_fast| <= 64) stays in s_fast.
    torch.manual_seed(0)
    N, K = 4, 8
    layer = ppb.ConcordLinearPackedB(K, N, bias=False, device='cuda')
    W = (torch.randn(N, K, device='cuda') * 0.05).to(torch.float32)
    layer.load_weights(W)
    sf, ss, vs = layer.get_state()
    # gap d_sv = (s_slow - v_slow) is in {-1,0,1} per the even split.
    assert int((ss - vs).abs().max()) <= 1, "load_weights did not split the coarse part evenly"
    # fine residual lives in s_fast and is bounded (|.|<=64 per the comment).
    assert int(sf.abs().max()) <= 64, "fine residual leaked outside |s_fast|<=64"
    # round-trips to W within block-float tolerance via get_weight.
    rec = layer.get_weight().to(torch.float32)
    assert torch.allclose(rec, W, atol=5e-3, rtol=0.0), \
        f"load_weights -> get_weight did not recover W (max err {(rec - W).abs().max()})"


@pytest.mark.skipif(not HAS_CUDA, reason="ConcordLinearPackedB ctor launches the materialize Triton kernel (GPU-only)")
def test_load_weights_deploy_matches_live_at_init():
    # Because the mantissa is packed into the SLOW path with only a tiny residual in
    # s_fast, the deploy weight (drops s_fast) ~= the live weight at init.
    torch.manual_seed(1)
    N, K = 4, 8
    layer = ppb.ConcordLinearPackedB(K, N, bias=False, device='cuda')
    W = (torch.randn(N, K, device='cuda') * 0.05).to(torch.float32)
    layer.load_weights(W)
    live = layer.get_weight().to(torch.float32)
    dep = layer.consolidated_weight().to(torch.float32)
    # they differ only by the small fine residual s_fast, so they are close to W.
    assert torch.allclose(dep, live, atol=6e-3, rtol=0.0)
    assert torch.allclose(dep, W, atol=8e-3, rtol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
