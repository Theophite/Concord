"""concord.substrate_gauss — seed-derived GAUSSIAN substrate, generated PER-ELEMENT
from a position hash (NO resident tensor).

The from-scratch substrate is a STATIC ADDITIVE portion of every weight, computed
inside the fused matmul (architect ruling 2026-06-28):

    weight[i,j] = substrate(seed, i, j) + offset(packed word)[i,j] * scale[i,j]

Only the SEED (one int per layer) is stored. The substrate VALUES never exist as a
tensor during training — each read path regenerates them from (seed, pos). This file
is the CPU REFERENCE (pure torch, the validated truth) + the std helpers + the export
materializer. The Triton device transliteration is in substrate_gauss_triton.py and is
inlined into the dual-dissipation read paths (fused fwd / gradx bwd / materialize /
get_weight) so they all agree on the weight.

METHOD — Box-Muller (cos branch) from two SALTED draws of a STRONG avalanche hash:
    u1, u2 = hash(seed, pos, {SALT_A, SALT_B})
    z      = sqrt(-2 * ln u1) * cos(2*pi * u2)           # ~ N(0,1), exact
    substrate = z * std(fan_in, fan_out)                 # weight units
u1 is floored at 2**-24 as the log(0) guard, bounding |z| < ~5.8.

HASH — we do NOT reuse the SR dither hash (prototype_packed_b.py:109-117). That xorshift
has weak cross-input avalanche: a 1-bit input change leaves the HIGH output bits intact,
so adjacent positions / 1-bit-apart salts produce ~0.99-correlated uniforms (verified —
it fails normality + independence). Fine for one dither draw per element, fatal for
generating independent draws here. Instead: a combine (big odd multipliers) + the
Murmur3 fmix32 finalizer. All arithmetic is mod 2**32 (Triton int32/uint32 wrap; the CPU
mirror masks int64 and is cross-checked against numpy uint32 in the test). Bit-exact
CPU<->GPU is NOT required (substrate is delta=0, outside the ledger, an init addend into
bf16 weights; within a run all four read paths use the one Triton hash and agree).

CONSERVATION: the substrate is a CONSTANT additive (delta_substrate == 0 every step),
OUTSIDE both dual-dissipation ledger terms — it never touches the accumulator accounting.

pos CONVENTION: row-major flat index  pos = i*K + j  (K = fan_in = weight.shape[1]).
Every read path in the port MUST use this same pos so the four kernels agree. pos fits
int32 for all SDXL layers (max layer << 2**31 elements).
"""
import math

import torch

# Salts (arbitrary 32-bit; the strong hash decorrelates them regardless of bit distance).
SUBSTRATE_SALT_A = 0x53554241  # 'SUBA'
SUBSTRATE_SALT_B = 0x9E3779B9  # golden-ratio constant, far from SALT_A

# Combine multipliers (large odd) + Murmur3 fmix32 finalizer constants.
_C_SEED = 0x9E3779B1
_C_POS = 0x85EBCA77
_F1 = 0x85EBCA6B
_F2 = 0xC2B2AE35
_M32 = 0xFFFFFFFF

_U_MIN = 1.0 / 16777216.0          # 2**-24 : the hash's min nonzero == log(0) guard
_TWO_PI = 2.0 * math.pi

SUBSTRATE_XAVIER = "xavier"
SUBSTRATE_KAIMING = "kaiming"


def _mix32(seed, pos, salt):
    """Strong 32-bit avalanche hash: combine (odd multipliers) + Murmur3 fmix32.
    Returns an int64 tensor in [0, 2**32). Logical right-shifts (positive int64 >> + the
    & _M32 masks); multiplies wrap mod 2**32 — the low 32 bits are exact regardless of
    int64 overflow, since (a*b) mod 2**32 depends only on the low 32 bits of a, b."""
    s = (torch.as_tensor(seed, dtype=torch.int64) & _M32)
    p = (pos.to(torch.int64) & _M32)
    h = ((s * _C_SEED) ^ (p * _C_POS) ^ int(salt)) & _M32
    h = (h ^ (h >> 16)) & _M32
    h = (h * _F1) & _M32
    h = (h ^ (h >> 13)) & _M32
    h = (h * _F2) & _M32
    h = (h ^ (h >> 16)) & _M32
    return h


def _hash_uniform(seed, pos, salt):
    """(seed, pos, salt) -> U[0,1) with 24-bit resolution, strong avalanche."""
    return (_mix32(seed, pos, salt) & 0xFFFFFF).to(torch.float32) * _U_MIN


def gauss_unit(seed, pos):
    """Per-element standard normal N(0,1) from (seed, pos) via Box-Muller (cos branch).

    seed : python int or int32 scalar tensor (the layer's substrate seed).
    pos  : int64 tensor of flat indices (i*K + j).
    returns float32 tensor, same shape as pos.
    """
    if not torch.is_tensor(seed):
        seed = torch.tensor(int(seed), dtype=torch.int32)
    u1 = _hash_uniform(seed, pos, SUBSTRATE_SALT_A).clamp_min(_U_MIN)  # log(0) guard
    u2 = _hash_uniform(seed, pos, SUBSTRATE_SALT_B)
    r = torch.sqrt(-2.0 * torch.log(u1))
    return r * torch.cos(_TWO_PI * u2)


def xavier_std(N, K, gain=1.0):
    """Glorot/Xavier-normal std for a [N=fan_out, K=fan_in] weight:
    gain * sqrt(2 / (fan_in + fan_out)). Matches prototype_packed_b._init_weight."""
    return gain * math.sqrt(2.0 / (N + K))


def kaiming_std(K, gain=math.sqrt(2.0)):
    """Kaiming/He-normal std for a [*, K=fan_in] weight: gain / sqrt(fan_in)
    (gain = sqrt(2) for a relu nonlinearity)."""
    return gain / math.sqrt(K)


def std_for(N, K, mode=SUBSTRATE_XAVIER, gain=None):
    if mode == SUBSTRATE_XAVIER:
        return xavier_std(N, K, 1.0 if gain is None else gain)
    if mode == SUBSTRATE_KAIMING:
        return kaiming_std(K) if gain is None else gain / math.sqrt(K)
    raise ValueError(f"unknown substrate mode {mode!r}")


def make_substrate(N, K, seed, mode=SUBSTRATE_XAVIER, gain=None, device=None):
    """Full [N, K] substrate tensor (weight units) — the EXPORT-bake / introspection /
    CPU-equivalence path ONLY. During training the kernel computes this per-element on
    the fly (no resident tensor). pos = i*K + j (row-major), matching the kernel."""
    pos = torch.arange(N * K, dtype=torch.int64).reshape(N, K)
    sub = gauss_unit(int(seed), pos) * std_for(N, K, mode, gain)
    if device is not None:
        sub = sub.to(device)
    return sub
