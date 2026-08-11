"""concord.substrate_gauss_triton - Triton DEVICE transliteration of
substrate_gauss.gauss_unit, to be INLINED into the dual-dissipation read paths
(fused fwd / gradx bwd / materialize / get_weight) by the SDXL port.

Line-for-line port of the CPU reference (substrate_gauss.py), which is the validated
truth (tests/test_substrate_gauss.py). The hash is a strong Murmur3-fmix32 avalanche
hash in uint32 (the SR xorshift is NOT reused: it has weak cross-input avalanche,
fatal for independent draws - see substrate_gauss.py). Box-Muller's sqrt/log/cos differ
at fp32-transcendental ULP between torch (CPU) and libdevice (GPU), immaterial for an
init addend written into bf16 weights; bit-exact CPU<->GPU is not required (within a run
all read paths use this one hash and agree).

NOT imported by the CPU test suite (keeps any triton import off the live-run box).
GPU-compile in the run-DOWN window confirms uint32 support + tl.log / tl.cos for the
pinned triton version, then a CPU-vs-GPU equivalence check closes the loop.

pos = row*K + col (int32). std = per-layer float32 (xavier/kaiming, host-computed).
"""
import triton
import triton.language as tl

# Must equal substrate_gauss.SUBSTRATE_SALT_{A,B} and the combine/finalizer constants.
SUBSTRATE_SALT_A: tl.constexpr = 0x53554241
SUBSTRATE_SALT_B: tl.constexpr = 0x9E3779B9
_C_SEED: tl.constexpr = 0x9E3779B1
_C_POS: tl.constexpr = 0x85EBCA77
_F1: tl.constexpr = 0x85EBCA6B
_F2: tl.constexpr = 0xC2B2AE35
_U_MIN: tl.constexpr = 1.0 / 16777216.0          # 2**-24, log(0) guard
_TWO_PI: tl.constexpr = 6.283185307179586


@triton.jit
def _hash_uniform(seed, pos, salt):
    """(seed, pos, salt) -> U[0,1), strong avalanche. uint32 => logical >> + mod-2**32
    wrap, matching the numpy-uint32-checked CPU mirror (substrate_gauss._mix32)."""
    s = seed.to(tl.uint32)
    p = pos.to(tl.uint32)
    h = (s * _C_SEED) ^ (p * _C_POS) ^ salt.to(tl.uint32)
    h = h ^ (h >> 16)
    h = h * _F1
    h = h ^ (h >> 13)
    h = h * _F2
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def substrate_gauss(seed, pos, std):
    """Per-element Gaussian substrate (weight units). seed, pos: int32; std: float32.

    Box-Muller (cos branch): z = sqrt(-2 ln u1) * cos(2*pi * u2); substrate = z * std.
    Computed on ALL lanes; the caller masks to from-scratch coords (substrate enabled).
    Inline this expression directly into each read path's inner loop and ADD the result
    to the decoded weight: w = substrate_gauss(seed, pos, std) + m_eff * scale.
    """
    u1 = _hash_uniform(seed, pos, SUBSTRATE_SALT_A)
    u1 = tl.maximum(u1, _U_MIN)                       # log(0) guard
    u2 = _hash_uniform(seed, pos, SUBSTRATE_SALT_B)
    r = tl.sqrt(-2.0 * tl.log(u1))
    return r * tl.cos(_TWO_PI * u2) * std
