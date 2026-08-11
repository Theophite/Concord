"""Pure-torch CPU reference: CORRECTED dual-dissipation packed-B fine accumulator.

REWORK NOTE (this file supersedes its own prior denormal/init on the two reworked
mechanisms only). The AUTHORITATIVE spec is DUAL_DISSIPATION_ENCODING.md in this
directory. The co-equal CORE is kept verbatim; only the two mechanisms called out under
"REMOVE"/"ADD" change:

  ADD-1  SUBSTRATE + OFFSET.  weight = SUBSTRATE + OFFSET. The packed 32-bit word holds
         ONLY the offset. The substrate is a fixed, seed-derived (from-scratch) or
         base-supplied (finetune) random prior OUTSIDE the word — never stored in the
         word, never consolidated, never dissipated, never ledgered. Dissipation decays
         the OFFSET toward 0, i.e. the weight toward the substrate (the prior), NOT
         toward 0 -> the build-from-zero problem is GONE (no gap-fix, no ignition, no
         even-split-into-the-accumulator). load_weights ENABLED sets substrate := init/
         base and ZEROES the accumulator. DISABLED keeps the legacy even-split bit-exact
         (substrate = None -> the word holds the FULL weight, exactly as legacy).

  ADD-2  DENORMAL = per-element EXPONENT-CLAIM on e_H (replaces the removed linear
         fraction). For a denormal element (coarse word == 0) e_H's bits are read as a
         per-element block-float SUB-SCALE extension: 2 octave bits + 4 mantissa bits
         (implied leading 1) + the int8 sign, reaching 2^-1 .. 2^-4 of one mantissa unit
         BELOW the shared block scale. The HIGH-dissipation arm e_H is statistically
         drained near zero (by BOTH evap and consolidation) so its high bits are free to
         claim. A |e_H| guard (EH_VELO_CAP = 64) selects exponent-mode vs magnitude-mode:
         a large e_H is a real un-consolidated velocity -> DROP the claim, render PURE
         COARSE (it graduates via the normal chase). Octave range is capped at oct in
         0..3 so every exp-mode |e_H| <= 63 < 64 -> exp-mode and the guard are PROVABLY
         DISJOINT (spec HOLE-1). Graduation of a denormal that grows past one mantissa
         unit promotes whole units into e_L (a fine->fine move, booked as inflow) — NEVER
         a +128 credit into s_slow (the removed leak).

────────────────────────────────────────────────────────────────────────────────
WRONG PREDECESSORS (do NOT copy):
  #1  dither_accum_ref.py / DITHER_ACCUM_DESIGN.md  — int16 s_fast + fp32 SIDECARS
      (err_s, den_frac). FORBIDDEN. Here the ONLY fp state outside the 32-bit word is the
      seed-derived SUBSTRATE (ADD-1), which is conceptually seed->tensor, regenerable,
      and never part of the accumulator. The OFFSET is recoverable from the packed word
      alone; the live WEIGHT = substrate + offset (decode is a pure function of
      (packed, row_exp, col_exp, substrate)).
  #2  DUAL_DISSIPATION_DESIGN.md §1/§6  — packed the two int8 fine accumulators as the
      HIGH/LOW BYTES of one int16 (s_fast = e_H*256 + (e_L&0xFF)). 256x mass mismatch +
      a one-directional DC leak + self-nullifying denormal predicate. FATAL.
  the removed fractional denormal (THIS file, pre-rework): e_H_fraction / _denormal_build
      / the +128 promotion into s_slow. The +128 was an unbooked deploy credit (a 128x
      leak the conservation ledger caught). REPLACED by ADD-2's exponent-claim, whose
      graduation promotes into e_L (fine->fine, booked as inflow), never +128 to s_slow.

────────────────────────────────────────────────────────────────────────────────
THE CORRECTED CORE  (CO-EQUAL e_L / e_H — kept verbatim)
────────────────────────────────────────────────────────────────────────────────
WORD layout (32 bits, persistent, holds the OFFSET only when ENABLED):

    bits [31:24]  e_H   int8   x1     fine accumulator, HIGH dissipation (lambda_H)
    bits [23:16]  e_L   int8   x1     fine accumulator, LOW  dissipation (lambda_L)
    bits [15: 8]  s_slow int8  x128   deploy position   (legacy line 759)
    bits [ 7: 0]  v_slow int8  x128   deploy anchor      (legacy line 760)

    fine mantissa value = e_L + e_H          (CO-EQUAL, x1 each — NOT e_H*256 + e_L)

The legacy single-int16 quantity it replaces (DISABLED view):
    s_fast_legacy (one int16, x1, prototype_packed_b.py:758)  ==  e_L + e_H   (when ON)

RATES (point 2): ARITHMETIC bracket about the legacy nominal rate (mean = legacy lambda):
    lambda_legacy = lr_eff * gf_consol
    lambda_L = lambda_legacy * (1 - d)        (LOW  dissipation, retain arm)
    lambda_H = lambda_legacy * (1 + d)        (HIGH dissipation, boil  arm)

PER-ARM COHERENCE (point 3): sig SHARED (from d_sv); noise PER-arm (each arm's d_fs).
PRE-EVAP chase snapshot (point 4): consolidate coherent mass before dissipating any.
BLENDED-coh evaporation (point 5): (1-coh) uses the e-weighted blended raw coherence;
    require grad-accum M >= M_MIN=4, else collapse to legacy.
DISABLED == LEGACY (point 7): 16 fine bits read as ONE int16 s_fast; legacy chase/leak/
    evap/int16-clamp run VERBATIM; substrate = None; deploy = coarse. BIT-EXACT to legacy.

CONSERVATION INVARIANT (point 8, the gating test): every mantissa unit credited to the
deploy word (s_slow+v_slow)*128 is debited from the fine register (e_L+e_H), evap booked
as a sink. delta(deploy_mantissa) + delta(fine_mantissa) == inflow_int, EVERY step. The
substrate and the exponent-claim render are READ-SIDE addends OUTSIDE both ledger terms
(substrate is constant; the render reads e_H which is already on the fine side) so they
inject ZERO unbooked deploy mantissa.

LEGACY SOURCE OF TRUTH: prototype_packed_b.py (unpack 757-762, coherence 823-846, evap
936-952, SR-tick 988-992, chase 997-1042, leak 1044-1060, clamp/repack 1103-1110,
re-exponent 1132-1143, rebalance 1985-2058 [residual migration 2028/2037], load_weights
2645-2693, get_weight 2768-2781, consolidated_weight 2783-2800, _init_weight 2639-2643,
_hash_uniform 109-117).

CPU-ONLY. Assume CUDA_VISIBLE_DEVICES="". Run nothing here that needs a GPU.
"""

import torch

# ── format constants (shared with prototype_packed_b.py) ──
MANTISSA_BIAS = 15
INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
S_SLOW_FACTOR = 128
V_SLOW_FACTOR = 128
CARRY = 128                      # one s_slow/v_slow LSB == 128 mantissa units (= deploy LSB)
MAX_M = 24000                    # re-exponent trigger (prototype_packed_b.py:2816)

# ── dual-dissipation defaults ──
BRACKET_D = 0.5                  # arithmetic half-spread d: lambda_{L,H} = lambda*(1 -+ d)
M_MIN = 4                        # grad-accum floor; below this, collapse to legacy (point 5)

# ── ADD-2: denormal = per-element EXPONENT-CLAIM on e_H (reconciled with spec HOLE-1) ──
# e_H magnitude (1..127) of a denormal element splits as (oct << MANT_BITS) | mlow:
#   oct  in 0..3   (EXP_BITS effective octave bits — the high oct bit is reserved so the
#                   max exp-mode |e_H| stays below EH_VELO_CAP, making exp-mode and the
#                   magnitude-mode guard PROVABLY DISJOINT — spec §3.6 HOLE-1)
#   mlow in 0..15  (MANT_BITS significand bits, leading-1 IMPLIED: signif = 1 + mlow/16)
#   int8 sign      = the offset sign (no reserved sign bit)
EXP_BITS = 2                     # EFFECTIVE octave bits after reserving guard headroom
MANT_BITS = 4                    # significand bits (leading-1 implied)
OCT_MAX = (1 << EXP_BITS) - 1    # 3  -> octave in 0..3
MLOW_MASK = (1 << MANT_BITS) - 1 # 0xF
EH_VELO_CAP = 64                 # |e_H| >= 64 => MAGNITUDE MODE (pure coarse). PROVABLY
                                 #   disjoint from exp-mode: max exp-mode |e_H| =
                                 #   (OCT_MAX<<MANT_BITS)|MLOW_MASK = (3<<4)|15 = 63 < 64.
# Deepest reach = 2^-(OCT_MAX+1) = 2^-4 of one mantissa unit = deploy_LSB / 2^11.
# (No FRAC_BITS / FRAC_DEN / FRAC_CAPACITY — the linear-fraction field is REMOVED.)

# substrate init modes (ADD-1)
SUBSTRATE_XAVIER = "xavier"
SUBSTRATE_KAIMING = "kaiming"


# ============================================================================
# Bit layout: pack / unpack the int32 word
#   ENABLED view  : (e_H int8, e_L int8, s_slow int8, v_slow int8)  -- holds the OFFSET
#   DISABLED view : (s_fast int16, s_slow int8, v_slow int8)        -- reunified [31:16],
#                                                                       holds the WEIGHT
# Both views read the SAME 32 bits (legacy unpack 757-762, repack 1106-1110).
# ============================================================================
def unpack_dual(packed):
    """packed int32 -> (e_H, e_L, s_slow, v_slow) int32 (sign-extended), the ENABLED
    co-equal view. e_L + e_H is the fine mantissa value (x1 each) — NOT e_H*256 + e_L."""
    e_H = packed >> 24                       # bits 31:24, sign-extended int8
    e_L = (packed << 8) >> 24                # bits 23:16, sign-extended int8
    s_slow = (packed << 16) >> 24            # bits 15:8  (legacy 759)
    v_slow = (packed << 24) >> 24            # bits 7:0   (legacy 760)
    return e_H, e_L, s_slow, v_slow


def pack_dual(e_H, e_L, s_slow, v_slow):
    """(e_H int8, e_L int8, s_slow int8, v_slow int8) -> packed int32 (ENABLED layout,
    mirror of legacy repack 1106-1110)."""
    return (
        ((e_H & 0xFF) << 24)
        | ((e_L & 0xFF) << 16)
        | ((s_slow & 0xFF) << 8)
        | (v_slow & 0xFF)
    ).to(torch.int32)


def unpack_legacy(packed):
    """packed int32 -> (s_fast int16, s_slow int8, v_slow int8), the DISABLED view.
    s_fast = packed >> 16 is the EXACT legacy single int16 (prototype_packed_b.py:758)."""
    s_fast = packed >> 16                    # bits 31:16, sign-extended int16 (legacy 758)
    s_slow = (packed << 16) >> 24            # legacy 759
    v_slow = (packed << 24) >> 24            # legacy 760
    return s_fast, s_slow, v_slow


def pack_legacy(s_fast, s_slow, v_slow):
    """(s_fast int16, s_slow int8, v_slow int8) -> packed int32 (legacy repack 1106-1110)."""
    return (
        ((s_fast & 0xFFFF) << 16)
        | ((s_slow & 0xFF) << 8)
        | (v_slow & 0xFF)
    ).to(torch.int32)


def fine_value(e_H, e_L):
    """The fine mantissa value of the co-equal pair = e_L + e_H (x1 each)."""
    return e_L.to(torch.int32) + e_H.to(torch.int32)


# ============================================================================
# Substrate (ADD-1): seed-derived random prior, weight units, OUTSIDE the word.
# Never consolidated, dissipated, or written into any mantissa field ("neither believed
# nor disbelieved"). For a finetune it is the supplied pretrained base. It is a read-side
# addend in decode only and is CONSTANT across steps (delta_substrate == 0).
# ============================================================================
def make_substrate(N, K, seed, mode=SUBSTRATE_XAVIER, device=None):
    """Deterministic seed -> substrate tensor [N, K] in WEIGHT units. Regenerable from
    (seed, mode). Xavier std matches prototype_packed_b._init_weight:2640; Kaiming uses
    fan-in N. The substrate breaks symmetry + sets the per-row/col scale only. The draw is
    on a CPU generator (deterministic, seed-regenerable regardless of run device); the
    result is moved to `device` for the live weight's device."""
    g = torch.Generator().manual_seed(int(seed))           # CPU generator, deterministic
    if mode == SUBSTRATE_KAIMING:
        std = (2.0 / float(N)) ** 0.5
    else:  # SUBSTRATE_XAVIER
        std = (2.0 / float(N + K)) ** 0.5
    sub = torch.empty(N, K, dtype=torch.float32).normal_(0.0, std, generator=g)
    return sub if device is None else sub.to(device)


# ============================================================================
# ADD-2: per-element EXPONENT-CLAIM on e_H (denormal sub-scale extension).
# Replaces the REMOVED linear-fraction field. Reads e_H's bits as a per-element
# block-float octave/significand BELOW one mantissa unit. Pure functions of e_H.
# ============================================================================
def is_exp_mode(e_H):
    """Mode select (spec §2.4, HOLE-1). The exponent claim is valid only while e_H is
    drained near zero (the high-dissipation arm). |e_H| < EH_VELO_CAP -> EXPONENT MODE
    (read e_H as octave/significand). |e_H| >= EH_VELO_CAP -> MAGNITUDE MODE (a real
    un-consolidated velocity; DROP the claim, render pure coarse). PROVABLY disjoint from
    every valid exp-mode encoding (which has |e_H| <= 63 < 64)."""
    return e_H.abs() < EH_VELO_CAP


def decode_denormal_units(e_H):
    """EXPONENT-MODE decode (spec §2.3). e_H -> signed offset value in MANTISSA UNITS
    (|.| < 1), an IEEE-style implied-leading-1 sub-scale value.

    BUG-2 FIX (zero-collision). The naive packing a = (oct<<MANT_BITS)|mlow gives a == 0
    for the (oct=0, mlow=0) value 0.5, COLLIDING with true-zero (e_H == 0). To keep a == 0
    RESERVED for true-zero, the (oct, mlow) index is stored OFFSET BY ONE:

        SGN    = sign(e_H)                      (offset sign in the int8 sign)
        a      = |e_H|                          (1..63; a == 0 -> true zero -> 0)
        idx    = a - 1                          (0..62, the (oct,mlow) index)
        oct    = (idx >> MANT_BITS) & OCT_MAX   (0..3  downward octave)
        mlow   = idx & MLOW_MASK                (0..15 significand, leading-1 implied)
        signif = 1 + mlow/16                    (in [1.0, 1.9375))
        units  = SGN * signif * 2^-(oct+1)      (oct+1 -> at least 1 octave below the unit)

    Reach: oct 0..3 -> 2^-1 .. 2^-4 of one mantissa unit; the 63 nonzero codes a in 1..63
    span value 0.5 (a==1) down to 0.0625 (a==49, oct3 mlow0). max |a| = 63 < EH_VELO_CAP so
    exp-mode/magnitude-mode stay PROVABLY DISJOINT (HOLE-1). a == 0 -> exactly 0. Magnitude
    mode (|e_H| >= EH_VELO_CAP) is NOT decoded here — its units are 0 for the log path."""
    e_H = e_H.to(torch.int32)
    a = e_H.abs()
    sgn = torch.sign(e_H.to(torch.float32))
    idx = a - 1                                            # BUG-2: a in 1..63 -> idx in 0..62
    oct_ = (idx >> MANT_BITS) & OCT_MAX
    mlow = idx & MLOW_MASK
    signif = 1.0 + mlow.to(torch.float32) / float(1 << MANT_BITS)
    units = sgn * signif * torch.pow(2.0, -(oct_.to(torch.float32) + 1.0))
    # true-zero (a == 0) and magnitude-mode (|e_H| >= cap) contribute 0 to the log path.
    valid = (a != 0) & (a < EH_VELO_CAP)
    return torch.where(valid, units, torch.zeros_like(units))


def encode_denormal(v):
    """ENCODE inverse (spec §2.6). Quantize a sub-unit value v (|v| < 1 mantissa unit)
    into e_H = SGN * (1 + ((oct << MANT_BITS) | mlow)). Used for graduation re-encode and
    for an explicit sub-scale OFFSET write (NOT used at init under ADD-1 — a sub-scale
    WEIGHT lives in the substrate). DETERMINISTIC nearest rounding; see encode_denormal_sr
    for the DC-neutral stochastic variant (spec HOLE-2):

        SGN  = sign(v)
        av   = |v|  in (0, 1)
        oct  = clamp(floor(-log2(av)) - 1, 0, OCT_MAX)    (av in [2^-(oct+1), 2^-oct))
        mlow = clamp(round((av * 2^(oct+1) - 1) * 16), 0, 15)
        idx  = (oct << MANT_BITS) | mlow ;  a = idx + 1 ;  e_H = SGN * a   (BUG-2: a >= 1)
    av == 0 -> e_H = 0 (true zero). av < 2^-(OCT_MAX+1) floors to (oct=OCT_MAX, mlow=0)."""
    return _encode_denormal_impl(v, sr=None, pos=None, salt=None)


def encode_denormal_sr(v, hash_seed, pos, salt):
    """ENCODE with STOCHASTIC ROUNDING of mlow (spec HOLE-2). The sigma-delta carry of a
    sub-unit residual lives IN the e_H log field; re-deriving it each step is lossy, which
    would be a DC sink under nearest rounding. SR of the sub-unit residual is DC-unbiased
    in expectation (E[encode] == v to within the SR grain), keeping the linear->log carry
    DC-neutral with NO fp sidecar. Uses the SAME xorshift hash as every other SR tick."""
    return _encode_denormal_impl(v, sr=hash_seed, pos=pos, salt=salt)


def _encode_denormal_impl(v, sr, pos, salt):
    v = v.to(torch.float32)
    av = v.abs()
    sgn = torch.sign(v)
    nz = av > 0
    av_safe = torch.where(nz, av, torch.ones_like(av))      # avoid log2(0)
    # octave s.t. av in [2^-(oct+1), 2^-oct): then -log2(av) in [oct, oct+1), so
    # oct = ceil(-log2(av)) - 1 (boundary-exact: av == 2^-1 -> oct 0, av == 2^-2 -> oct 1).
    # (The prior "floor(.) - 1.0" was off by one — it mapped av in [0.25,0.5) to oct 0,
    # decoding back as [0.5,1): the round-trip then failed and graduation/relaxation
    # re-encodes drifted.  ceil(.) - 1 is the exact inverse of decode's idx partition.)
    oct_f = torch.ceil(-torch.log2(av_safe)) - 1.0
    oct_ = torch.clamp(oct_f, 0.0, float(OCT_MAX))
    # invert signif = 1 + mlow/16 at the chosen octave; mlow = (av*2^(oct+1) - 1)*16.
    mlow_real = (av_safe * torch.pow(2.0, oct_ + 1.0) - 1.0) * float(1 << MANT_BITS)
    mlow_real = torch.clamp(mlow_real, 0.0, float(MLOW_MASK))
    if sr is None:
        mlow = torch.round(mlow_real)
    else:
        mlow = _sr_round(mlow_real, sr, pos, salt).to(torch.float32)   # DC-neutral (HOLE-2)
    idx = (oct_.to(torch.int32) << MANT_BITS) | mlow.to(torch.int32)  # (oct,mlow) index
    idx = torch.clamp(idx, 0, (OCT_MAX << MANT_BITS) | MLOW_MASK)      # idx in 0..63
    a = torch.clamp(idx + 1, 1, (OCT_MAX << MANT_BITS) | MLOW_MASK)    # BUG-2: a = idx+1, a in 1..63
    e_H = (sgn * a.to(torch.float32)).to(torch.int32)
    # av == 0 -> true zero.
    e_H = torch.where(nz, e_H, torch.zeros_like(e_H))
    return e_H.clamp(INT8_MIN, INT8_MAX)


# ============================================================================
# Denormal detection (spec §2.1): COARSE WORD ONLY — never touches fine bits, so reading
# the offset value out of e_H cannot move the predicate (WRONG #2 BREAK A avoided).
# ============================================================================
def is_denormal(packed):
    """THE denormal predicate, a pure function of the COARSE deploy fields:
        is_denormal = (s_slow == 0) AND (v_slow == 0)
    Normal coords (coarse != 0) take pure coarse, bit-identical to legacy."""
    _, _, s_slow, v_slow = unpack_dual(packed)
    return (s_slow == 0) & (v_slow == 0)


# ============================================================================
# Scale helpers (block-float, mirror prototype_packed_b.py:769-771)
# ============================================================================
def _scale_fwd(row_exp, col_exp, mantissa_bias=MANTISSA_BIAS):
    exp = (row_exp[:, None].to(torch.float32)
           + col_exp[None, :].to(torch.float32) - float(mantissa_bias))
    return torch.pow(2.0, exp)


def _pos_hash(N, K, device=None):
    return ((torch.arange(N, device=device)[:, None] << 16)
            ^ torch.arange(K, device=device)[None, :]).to(torch.int32)


# ============================================================================
# Dither / stochastic rounding — xorshift hash IDENTICAL to the kernel
# (prototype_packed_b.py:109-117) so SR ticks match.
# ============================================================================
def _hash_uniform(x, pos, salt):
    """Deterministic xorshift dither (prototype_packed_b.py:109-117). int32 -> U[0,1)."""
    h = (x.to(torch.int64) ^ int(salt) ^ pos.to(torch.int64)) & 0xFFFFFFFF
    h = (h ^ (h << 13)) & 0xFFFFFFFF
    h = (h ^ (h >> 17)) & 0xFFFFFFFF
    h = (h ^ (h << 5)) & 0xFFFFFFFF
    h = (h ^ (h >> 7)) & 0xFFFFFFFF
    return (h & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)


def _sr_round(x, hash_seed, pos, salt):
    """Stochastic rounding: floor(x) + 1[u < frac(x)] (mirror 988-991)."""
    fl = torch.floor(x)
    u = _hash_uniform(hash_seed.to(torch.int32), pos, salt)
    return (fl + (u < (x - fl)).to(torch.float32)).to(torch.int32)


# ============================================================================
# Coherence — pure-torch mirror of the legacy USE_FIXED_COH / USE_COH_VHAT path
# (prototype_packed_b.py:823-846). sig SHARED across arms; only d_fs (noise) per arm.
# Returns (coh [chase/leak gate], coh_raw [dissipation]).
# ============================================================================
def compute_coherence(d_fs, d_sv, vhat, drift_cancel_C, coh_kappa,
                      sum_v_inv, N, K, scale_fwd, use_coh_vhat=True):
    """Byte-for-byte the legacy 823-846 (sig = C*d_sv*scale SHARED; coh_raw 826;
    cf 839; coh_n2 840; coh 841)."""
    sig_w = drift_cancel_C * d_sv * scale_fwd
    sig2 = sig_w * sig_w
    noise_w = (d_fs - drift_cancel_C * d_sv) * scale_fwd
    coh_n2 = noise_w * noise_w
    coh_raw = sig2 / (sig2 + coh_n2 + 1e-30)                          # 826
    if use_coh_vhat:
        vhat_fl = torch.clamp(vhat, min=0.03 / (sum_v_inv * N * K))   # 838
        cf = sig2 / (drift_cancel_C * drift_cancel_C * vhat_fl + 1e-30)  # 839
        coh_n2 = coh_n2 * coh_kappa / (cf + coh_kappa)               # 840
    coh = sig2 / (sig2 + coh_n2 + 1e-30)                             # 841
    return torch.clamp(coh, 0.0, 1.0), torch.clamp(coh_raw, 0.0, 1.0)


# ============================================================================
# Re-exponent safety on the REUNIFIED OFFSET fine value (mirror 1132-1143).
# Halves the OFFSET mantissa (e_H, e_L, s_slow, v_slow) and bumps row_exp. The substrate
# is in WEIGHT units (exponent-invariant) and is NOT touched here (spec §1.7):
#   substrate + (offset/2)*(2*scale) == substrate + offset*scale.
# DENORMAL-AWARE (spec §3.6 HOLE-3): a denormal element's e_H is an (oct, mlow) LOG field,
# NOT an integer — an integer halving would corrupt it. For denormal elements with a live
# exp-mode claim we shift the OCTAVE field instead (oct += net_right_shift), preserving the
# WEIGHT-units offset value across the scale doubling.
# ============================================================================
def renormalize_on_saturation(e_H, e_L, s_slow, v_slow, row_exp, packed_for_denorm=None,
                              max_m=MAX_M):
    """Value-preserving re-exponent (mirror 1132-1143, abs_eff = max(|full|, |fine|)).
    Returns (e_H, e_L, s_slow, v_slow, row_exp). When packed_for_denorm is given, denormal
    (coarse == 0) exp-mode elements re-encode their octave instead of integer-halving e_H
    (HOLE-3); pass None to use the plain integer halving on every element (legacy/normal)."""
    fine = fine_value(e_H, e_L)
    full = (s_slow.to(torch.int32) * 128 + fine + v_slow.to(torch.int32) * 128).abs()
    abs_eff = torch.maximum(full, fine.abs())                        # line 1143
    row_max = abs_eff.amax(dim=1)
    need = row_max >= max_m
    if bool(need.any()):
        sh = need[:, None]
        half = lambda t: torch.where(sh, torch.round(t.to(torch.float32) / 2.0).to(torch.int32), t)
        if packed_for_denorm is not None:
            denorm = (s_slow == 0) & (v_slow == 0)                   # COARSE-only predicate
            exp_live = denorm & is_exp_mode(e_H) & (e_H.abs() != 0)
            # normal/legacy halving for the non-denormal (and magnitude-mode) bytes
            e_H_norm = half(e_H)
            # denormal exp-mode: bump the octave by the row's right-shift count (here 1
            # octave per re-exponent), clamped — preserves the WEIGHT-units offset value
            # since the scale doubled (spec §3.6 HOLE-3). mlow/sign unchanged.
            a = e_H.abs()
            oct_ = (a >> MANT_BITS) & OCT_MAX
            mlow = a & MLOW_MASK
            oct_sh = torch.where(sh, torch.clamp(oct_ + 1, 0, OCT_MAX), oct_)
            a_new = (oct_sh << MANT_BITS) | mlow
            sgn = torch.sign(e_H.to(torch.float32))
            e_H_den = (sgn * a_new.to(torch.float32)).to(torch.int32)
            e_H = torch.where(exp_live, e_H_den, e_H_norm)
            e_L = half(e_L); s_slow = half(s_slow); v_slow = half(v_slow)
        else:
            e_H, e_L, s_slow, v_slow = half(e_H), half(e_L), half(s_slow), half(v_slow)
        row_exp = torch.where(need, row_exp + 1, row_exp)
    return e_H, e_L, s_slow, v_slow, row_exp


# ============================================================================
# Decode to weight.  weight = SUBSTRATE + OFFSET*scale  (ADD-1).
# Composition order (spec §1.4): substrate + (coarse_mantissa + denormal_extra)*scale —
# the denormal extra is added to the OFFSET mantissa BEFORE the scale multiply and BEFORE
# the substrate add, so it inherits the block-float scale and does NOT scale the substrate.
# substrate is None/0 when DISABLED (the word holds the full weight, byte-identical legacy).
# ============================================================================
def _substrate_addend(substrate, N, K, device=None):
    if substrate is None:
        return torch.zeros(N, K, dtype=torch.float32, device=device)
    return substrate.to(torch.float32)


def decode_to_live_weight(packed, row_exp, col_exp, substrate=None,
                          mantissa_bias=MANTISSA_BIAS):
    """LIVE weight = substrate + offset_mant_live * scale (analogue of get_weight:2775).
    offset_mant_live = s_slow*128 + (e_L + e_H) + v_slow*128. Pure function of
    (packed, row_exp, col_exp, substrate). DISABLED: substrate=None -> the offset IS the
    full legacy live mantissa (s_slow*128 + s_fast + v_slow*128)."""
    e_H, e_L, s_slow, v_slow = unpack_dual(packed)
    N, K = packed.shape
    offset_mant = (s_slow.to(torch.float32) * S_SLOW_FACTOR
                   + fine_value(e_H, e_L).to(torch.float32)
                   + v_slow.to(torch.float32) * V_SLOW_FACTOR)
    return _substrate_addend(substrate, N, K, device=packed.device) + offset_mant * _scale_fwd(
        row_exp, col_exp, mantissa_bias)


def decode_to_deploy_weight(packed, row_exp, col_exp, substrate=None, step_salt=0,
                            mantissa_bias=MANTISSA_BIAS, enabled=True,
                            deterministic=False):
    """DEPLOY weight = substrate + offset_mant_deploy * scale (analogue of
    consolidated_weight:2795). offset_mant_deploy = (s_slow + v_slow)*128 PLUS, ONLY where
    (is_denormal & is_exp_mode), the exponent-claim render of e_H (spec §2.3/§2.7). With
    enabled=False the denormal render is dropped (pure coarse, byte-identical legacy) and
    substrate is None.
        deterministic=False -> stochastic round of the sub-unit extra (E[render]==value).
        deterministic=True  -> round-to-nearest (reproducible shipping checkpoint)."""
    e_H, e_L, s_slow, v_slow = unpack_dual(packed)
    N, K = packed.shape
    offset_mant = (s_slow.to(torch.float32) * S_SLOW_FACTOR
                   + v_slow.to(torch.float32) * V_SLOW_FACTOR)
    if enabled:
        # exponent-claim extra ONLY on (is_denormal & is_exp_mode); magnitude-mode denormals
        # and all normal coords contribute 0 here (spec §2.4 deploy gate).
        gate = (is_denormal(packed) & is_exp_mode(e_H)).to(torch.float32)
        units = decode_denormal_units(e_H)                  # sub-unit offset value (|.| < 1)
        if deterministic:
            extra = units                                    # nearest: ship the exact value
        else:
            # DC-unbiased stochastic render into the (integer) offset mantissa grid, then
            # the residual sub-unit value rides at scale. We SR the sub-unit value to the
            # nearest representable render so E[render] == units (spec §2.7).
            frac = units
            fl = torch.floor(frac)
            u = _hash_uniform(e_H.to(torch.int32), _pos_hash(N, K, device=packed.device),
                              int(step_salt) ^ 0x0DEF0DEF)
            extra = fl + (u < (frac - fl)).to(torch.float32)
        offset_mant = offset_mant + gate * extra
    return _substrate_addend(substrate, N, K, device=packed.device) + offset_mant * _scale_fwd(
        row_exp, col_exp, mantissa_bias)


# ============================================================================
# Conservation invariant (point 8, the gating test). Pure functions of the OFFSET word.
# Substrate and the denormal render are READ-SIDE addends OUTSIDE both ledger terms.
# ============================================================================
def deploy_mantissa(packed):
    """The deploy-word integer mantissa (s_slow + v_slow) * 128."""
    _, _, s_slow, v_slow = unpack_dual(packed)
    return (s_slow.to(torch.int64) + v_slow.to(torch.int64)) * CARRY


def fine_mantissa(packed):
    """The fine-register INTEGER mantissa.

    BUG-1 (1a) — EXCLUDE the log field from the integer ledger. For a coord that is an
    exp-mode denormal ((s_slow==0 & v_slow==0) & |e_H| < EH_VELO_CAP), e_H is NOT a linear
    int8 velocity — it is a per-element block-float LOG field whose real value is
    decode_denormal_units(e_H), strictly sub-unit (|.| < 1) and therefore BELOW the integer
    ledger's resolution (rounds to 0). Counting raw e_H there as linear mass produces a
    phantom residual (e.g. e_H 0->19 encoding ~0.3 shows up as +19). So e_H contributes 0 to
    the ledger on exp-mode denormal coords; everywhere else it is linear as before. Whole-unit
    graduations of a denormal are promoted into e_L and booked as grad_inflow, so they ARE on
    the integer side — only the sub-unit log residual is excluded here."""
    e_H, e_L, s_slow, v_slow = unpack_dual(packed)
    exp_den = (s_slow == 0) & (v_slow == 0) & (e_H.abs() < EH_VELO_CAP)
    e_H_ledger = torch.where(exp_den, torch.zeros_like(e_H), e_H)
    return (e_L.to(torch.int64) + e_H_ledger.to(torch.int64))


def assert_conservation(packed_before, packed_after, delta_inflow_int, eps=0):
    """POINT 8 INVARIANT: delta(deploy) + delta(fine) == inflow_int (evap booked as a sink
    into inflow_int). The substrate (constant) and the denormal render (a read of e_H,
    already on the fine side) inject ZERO unbooked deploy mantissa. Returns (ok, residual)."""
    d_dep = (deploy_mantissa(packed_after) - deploy_mantissa(packed_before))
    d_fine = (fine_mantissa(packed_after) - fine_mantissa(packed_before))
    residual = (d_dep + d_fine - delta_inflow_int.to(torch.int64))
    rmax = int(residual.abs().max()) if residual.numel() else 0
    return rmax <= eps, rmax


# ============================================================================
# The dual-dissipation Layer.
# ============================================================================
class DualDissipationLayer:
    """CPU reference for ONE 2-D parameter (N x K), CORRECTED co-equal design with the
    SUBSTRATE + OFFSET structure (ADD-1) and the e_H exponent-claim denormal (ADD-2).

    Persistent in-word state: packed_w int32 [N,K] = (e_H, e_L, s_slow, v_slow) = the
    OFFSET (ENABLED) or the full weight (DISABLED). Out-of-word state: the seed-derived
    `substrate` (ENABLED only; weight units; never ledgered), the Adafactor rank-1 moments
    v_row/v_col, and the exponents. The live weight = substrate + decode_offset(packed).

    enabled=True  -> dual co-equal e_L/e_H + substrate/offset (this design).
    enabled=False -> legacy single-int16 path, substrate=None, BIT-EXACT to legacy.
    """

    def __init__(self, N, K, enabled=True, grad_accum_M=8, bracket_d=BRACKET_D, seed=0,
                 substrate_seed=None, substrate_mode=SUBSTRATE_XAVIER):
        self.N, self.K = N, K
        self.grad_accum_M = int(grad_accum_M)
        self.enabled = bool(enabled) and (self.grad_accum_M >= M_MIN)
        self.bracket_d = float(bracket_d)
        # Internal tensors live on this device. CPU until load_weights, then the
        # weight tensor's device (GPU-capable for the live run; this smoke is CPU).
        self.device = torch.device("cpu")
        self.packed_w = torch.zeros(N, K, dtype=torch.int32)
        self.row_exp = torch.zeros(N, dtype=torch.int32)
        self.col_exp = torch.zeros(K, dtype=torch.int32)
        self.v_row = torch.zeros(N, dtype=torch.float32)
        self.v_col = torch.zeros(K, dtype=torch.float32)
        self._salt = int(seed) & 0x7FFFFFFF
        self._step = 0
        # ADD-1: substrate (ENABLED-only). seed/mode used only when load_weights is given
        # no explicit base (from-scratch). substrate stays None until load_weights.
        self.substrate = None
        self.substrate_seed = int(seed if substrate_seed is None else substrate_seed)
        self.substrate_mode = substrate_mode

    # ---- ADD-1: load. ENABLED zeroes the offset + sets the substrate; DISABLED keeps
    #             the legacy even-split (bit-exact). Branches on self.enabled (the SAME
    #             flag step() keys on, so an M<M_MIN collapse takes the legacy branch). ----
    @torch.no_grad()
    def load_weights(self, W, base=None):
        """ENABLED (the rework): substrate := base (finetune: the pretrained W) or a fresh
        seeded draw (from-scratch); row/col exponents from |substrate|; the ACCUMULATOR is
        ZEROED (e_H = e_L = s_slow = v_slow = 0). At this point decode_live == decode_deploy
        == substrate exactly; the word carries NO copy of the weight. NO even-split, NO
        init-time denormal encoding (a sub-LSB WEIGHT lives in the substrate at offset 0).
        Dissipation later decays the offset -> 0 -> weight -> substrate (the prior).

        DISABLED (bit-exact legacy): the legacy even-split body — full weight quantized into
        s_slow/v_slow + an int16 s_fast residual, substrate = None (mirror 2645-2693)."""
        W = W.to(torch.float32)
        # All internal tensors live on the weight tensor's device (GPU-capable). The
        # seed-derived substrate is drawn on CPU (determinism) then moved here.
        self.device = W.device
        dev = self.device
        self.v_row = torch.zeros(self.N, dtype=torch.float32, device=dev)
        self.v_col = torch.zeros(self.K, dtype=torch.float32, device=dev)
        if self.enabled:
            # substrate := pretrained base (finetune) OR a fresh seeded draw (from-scratch).
            if base is not None:
                self.substrate = base.to(torch.float32).to(dev).clone()
            else:
                self.substrate = make_substrate(self.N, self.K, self.substrate_seed,
                                                self.substrate_mode, device=dev)
            # row/col exponents chosen from |substrate| (legacy exponent rule on the prior),
            # so one mantissa unit is a sensible sub-step of the substrate magnitude.
            max_abs = self.substrate.abs().amax(dim=1).clamp(min=1e-30)
            self.row_exp = torch.ceil(torch.log2(max_abs) + 1.0).clamp(-30, 30).to(torch.int32)
            self.col_exp = torch.zeros(self.K, dtype=torch.int32, device=dev)
            # ZERO the accumulator -> offset 0 -> weight == substrate.
            z = torch.zeros(self.N, self.K, dtype=torch.int32, device=dev)
            self.packed_w = pack_dual(z, z, z, z)
            self._step = 0
            return

        # ---- DISABLED: legacy even-split gap-zero init (mirror load_weights:2665-2692). ----
        self.substrate = None
        max_abs = W.abs().amax(dim=1).clamp(min=1e-30)
        self.row_exp = torch.ceil(torch.log2(max_abs) + 1.0).clamp(-30, 30).to(torch.int32)
        self.col_exp = torch.zeros(self.K, dtype=torch.int32, device=dev)
        scale = _scale_fwd(self.row_exp, self.col_exp)
        m_total = (W / scale).round().to(torch.int32).clamp(INT16_MIN, INT16_MAX)  # 2675
        coarse = (m_total.to(torch.float32) / 128.0).round().to(torch.int32).clamp(
            2 * INT8_MIN, 2 * INT8_MAX)                                            # 2677
        v_slow = (coarse.to(torch.float32) / 2.0).round().to(torch.int32).clamp(
            INT8_MIN, INT8_MAX)                                                    # 2682 (gap=0)
        s_slow = (coarse - v_slow).clamp(INT8_MIN, INT8_MAX)                       # 2684
        s_fast = (m_total - (s_slow + v_slow) * 128).clamp(INT16_MIN, INT16_MAX)   # 2686
        self.packed_w = pack_legacy(s_fast, s_slow, v_slow)
        self.v_row.zero_(); self.v_col.zero_(); self._step = 0

    # ---- one optimizer step ----
    @torch.no_grad()
    def step(self, grad_W, lr, *, grad_W_H=None, alpha=0.1, gf_consol=0.0, drift_cancel_C=0.02,
             alpha_v_fast=0.001, coh_kappa=1.0, v_scale=1.0, precond_p=0.5,
             eps=1.0, step_cap=10.0, min_leak=0.0, evap_build_min=128.0, beta1=0.0,
             mantissa_bias=MANTISSA_BIAS, beta2=0.999, use_coh_vhat=True,
             mass_preserve=True, chase_floor=0.05, leak_floor=0.05,
             consf=1.0, return_ledger=False):
        """One optimizer step. ENABLED: dual co-equal arms (arithmetic bracket, per-arm
        coherence, pre-evap chase, blended-coh evap) over the OFFSET — dissipation decays
        the offset toward 0 (weight toward the substrate). DISABLED: literal legacy single
        int16 path. Returns a diagnostics dict (+ the conservation ledger if return_ledger).

        TWO-GRADIENT step (the REAL mechanism). When `grad_W_H` is given (ENABLED only),
        `grad_W` is arm L's gradient (from data-half-1) and `grad_W_H` is arm H's gradient
        (from data-half-2): each arm integrates the FULL gradient from its OWN random half
        of the batch — NOT half of a shared gradient. This differentiates the two arms (the
        even-split stand-in below makes them identical, which is useless for real training).
        The shared preconditioner statistic (Adafactor vhat) uses the MEAN of the two halves'
        gradients (the settled batch gradient). `grad_W_H=None` -> the legacy even-split path,
        BIT-IDENTICAL to before (so the conservation/mechanics tests stay green)."""
        self._step += 1
        salt = (self._salt ^ (self._step * 0x9E3779B1)) & 0x7FFFFFFF
        N, K = self.N, self.K
        grad_W = grad_W.to(torch.float32).to(self.device)
        two_grad = grad_W_H is not None
        if two_grad:
            grad_W_H = grad_W_H.to(torch.float32).to(self.device)
        pos = _pos_hash(N, K, device=self.device)
        packed_before = self.packed_w.clone()

        # ── Adafactor rank-1 vhat (mirror 802-808). The preconditioner is a SHARED quantity;
        #    in the two-gradient path it tracks the MEAN gradient (the settled batch grad), so
        #    grad_W_H=None reproduces the single-grad statistic BIT-IDENTICALLY. ──
        g_stat = 0.5 * (grad_W + grad_W_H) if two_grad else grad_W
        g2 = g_stat * g_stat
        self.v_row = beta2 * self.v_row + (1 - beta2) * g2.mean(dim=1)
        self.v_col = beta2 * self.v_col + (1 - beta2) * (
            g2.mean(dim=0) / (self.v_row.mean() + 1e-30))
        sum_v_inv = 1.0 / (self.v_row.sum() + 1e-30)
        v_bc = 1.0 / (1.0 - beta2 ** self._step)
        vhat = (self.v_row[:, None] * self.v_col[None, :] * sum_v_inv) * v_bc

        scale_fwd = _scale_fwd(self.row_exp, self.col_exp, mantissa_bias)
        scale_inv = 1.0 / scale_fwd

        if not self.enabled:
            # DISABLED == legacy single-int16. No per-arm split exists; fold any second half
            # into the shared mean (g_stat) and run the literal legacy tick.
            out = self._legacy_tick(
                g_stat, lr, alpha, gf_consol, drift_cancel_C, alpha_v_fast, coh_kappa,
                v_scale, precond_p, eps, step_cap, min_leak, evap_build_min, beta1,
                use_coh_vhat, mass_preserve, chase_floor, leak_floor, consf,
                vhat, sum_v_inv, scale_fwd, scale_inv, pos, salt)
        else:
            out = self._dual_tick(
                grad_W, lr, alpha, gf_consol, drift_cancel_C, alpha_v_fast, coh_kappa,
                v_scale, precond_p, eps, step_cap, min_leak, evap_build_min, beta1,
                use_coh_vhat, mass_preserve, chase_floor, leak_floor, consf,
                vhat, sum_v_inv, scale_fwd, scale_inv, pos, salt, grad_W_H=grad_W_H)
        packed_after = self.packed_w
        if return_ledger:
            out["_ledger"] = (packed_before, packed_after, out.pop("_inflow_int"))
        else:
            out.pop("_inflow_int", None)
        return out

    # ------------------------------------------------------------------
    # DUAL co-equal tick (ENABLED). Operates on the OFFSET word.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _dual_tick(self, grad_W, lr, alpha, gf_consol, drift_cancel_C, alpha_v_fast,
                   coh_kappa, v_scale, precond_p, eps, step_cap, min_leak,
                   evap_build_min, beta1, use_coh_vhat, mass_preserve, chase_floor,
                   leak_floor, consf, vhat, sum_v_inv, scale_fwd, scale_inv, pos, salt,
                   grad_W_H=None):
        N, K = self.N, self.K
        e_H, e_L, s_slow, v_slow = unpack_dual(self.packed_w)
        e_H = e_H.to(torch.int32); e_L = e_L.to(torch.int32)
        s_slow = s_slow.to(torch.int32); v_slow = v_slow.to(torch.int32)

        # Denormal coords (coarse word == 0): their e_H is the exponent-claim LOG field, not
        # an integer velocity. We must NOT chase/evap that field as if it were a velocity.
        # exp-mode denormals (the default, |e_H| < cap) route through e_L and bank the
        # sub-unit carry in e_H's octave field; magnitude-mode denormals (|e_H| >= cap) are
        # bursting toward normal and take the ordinary integer path (graduate via the chase).
        denorm = is_denormal(self.packed_w)
        exp_den = denorm & is_exp_mode(e_H)                  # exp-mode denormal coords (START)

        # ── (A) per-arm velocity d_fs_X (point 3). SHARED gap d_sv (line 779). ──
        # For an exp-mode denormal, e_H holds a LOG value, not a velocity -> its arm velocity
        # is 0 (do not feed the log field into coherence/chase as an integer).
        d_fs_L = e_L.to(torch.float32)
        d_fs_H = torch.where(exp_den, torch.zeros_like(e_H, dtype=torch.float32),
                             e_H.to(torch.float32))
        d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY   # SHARED (779)

        # ── (B) per-arm coherence (shared sig, per-arm noise; legacy 823-846) ──
        coh_L, coh_raw_L = compute_coherence(d_fs_L, d_sv, vhat, drift_cancel_C,
                                             coh_kappa, sum_v_inv, N, K, scale_fwd, use_coh_vhat)
        coh_H, coh_raw_H = compute_coherence(d_fs_H, d_sv, vhat, drift_cancel_C,
                                             coh_kappa, sum_v_inv, N, K, scale_fwd, use_coh_vhat)

        # ── (C) e-weighted BLENDED raw coherence (point 5) ──
        wL = d_fs_L.abs(); wH = d_fs_H.abs()
        wsum = wL + wH
        coh_raw_blend = torch.where(
            wsum > 0, (coh_raw_L * wL + coh_raw_H * wH) / (wsum + 1e-30),
            0.5 * (coh_raw_L + coh_raw_H))

        # ── (D) AdamW preconditioned step. The fine VALUE (e_L+e_H) is the velocity proxy
        #        for the preconditioner noise; the SHARED preconditioner `denom` is the SAME
        #        for both arms (it is a settled second-moment statistic). For an exp-mode
        #        denormal e_H is NOT a velocity, so the noise proxy uses only e_L there. ──
        #
        #        INFLOW SPLIT — two regimes:
        #          * SINGLE-GRAD (grad_W_H is None): the legacy EVEN-SPLIT stand-in. Both arms
        #            get HALF the shared gradient's delta (inflow_L == inflow_H == 0.5*delta).
        #            BIT-IDENTICAL to before -> the conservation/mechanics tests stay green.
        #            (Useless for real training: with identical inflow the arms never diverge.)
        #          * TWO-GRAD (grad_W_H given): the REAL mechanism. Arm L integrates the FULL
        #            preconditioned gradient from data-half-1 (grad_W) and arm H the FULL
        #            preconditioned gradient from data-half-2 (grad_W_H) — each its OWN half's
        #            whole gradient, not half of a shared one. The random microbatch split
        #            makes grad_W != grad_W_H, so the arms diverge (the key real-training
        #            property). The shared preconditioner `denom` keeps step scaling matched.
        fine_v = torch.where(exp_den, e_L.to(torch.float32),
                             (e_L + e_H).to(torch.float32))
        noise_w = (fine_v - drift_cancel_C * d_sv) * scale_fwd
        v_proxy = noise_w * noise_w * v_scale
        denom = torch.pow(v_proxy + eps, precond_p)
        if grad_W_H is None:
            step_live = (grad_W / denom).clamp(-step_cap, step_cap)
            delta_grad = -lr * step_live * scale_inv               # mantissa units (970)
            inflow_L = 0.5 * delta_grad                            # even matched split
            inflow_H = inflow_L                                    # identical (stand-in only)
        else:
            step_live_L = (grad_W / denom).clamp(-step_cap, step_cap)
            step_live_H = (grad_W_H / denom).clamp(-step_cap, step_cap)
            inflow_L = -lr * step_live_L * scale_inv               # arm L: FULL half-1 grad
            inflow_H = -lr * step_live_H * scale_inv               # arm H: FULL half-2 grad

        # ── (E) PRE-EVAP chase snapshot (point 4). Add the gradient inflow to each arm
        #        FIRST, then chase off this pre-evap value, then evaporate the residual.
        #        For an exp-mode denormal the H-arm inflow is BANKED in the log field
        #        (ADD-2 sigma-delta in (I)), NOT SR-ticked into e_H as an integer. ──
        intent_L = d_fs_L + inflow_L
        intent_H = d_fs_H + torch.where(exp_den, torch.zeros_like(inflow_H), inflow_H)
        if beta1 != 0.0:
            intent_L = intent_L + consf * beta1 * coh_L * d_fs_L
            intent_H = intent_H + consf * beta1 * coh_H * d_fs_H
        # intent_{L,H} are the ABSOLUTE post-inflow accumulator values (d_fs + inflow), so the
        # SR-rounded intent IS the new e_{L,H}; the per-arm inflow is the DELTA round(intent) -
        # d_fs (the legacy path's `delta_t` is already a delta, so it adds; here intent is the
        # absolute value, so we must difference it — adding round(intent) on top of the old
        # e_L double-counts the prior accumulator, the BUG-1/BUG-3 over-credit).
        new_e_L = _sr_round(intent_L, e_L, pos, salt ^ 0x000000A1)
        new_e_H = _sr_round(intent_H, e_H, pos, salt ^ 0x000000B2)
        tickg_L = new_e_L - e_L
        tickg_H = new_e_H - e_H
        # do NOT write the integer H tick onto an exp-mode denormal's e_H (it is the log field).
        tickg_H = torch.where(exp_den, torch.zeros_like(tickg_H), tickg_H)
        e_L = e_L + tickg_L
        e_H = e_H + tickg_H
        inflow_int = (tickg_L + tickg_H).to(torch.int64)          # NEW mass into fine reg

        # PRE-EVAP snapshot for the chase (the value to consolidate).
        snap_L = e_L.to(torch.float32)
        snap_H = e_H.to(torch.float32)

        # ── (F) per-arm chase, gain 1, AFFINE ratio-coh gate (1010), consf-gated. The
        #        H-arm chase is suppressed on an exp-mode denormal (e_H is the log field). ──
        gate_L = chase_floor + (1.0 - chase_floor) * coh_L        # 1010
        gate_H = chase_floor + (1.0 - chase_floor) * coh_H
        chase_mant_L = alpha * gate_L * snap_L * consf            # gain 1 (1026)
        chase_mant_H = alpha * gate_H * snap_H * consf
        chase_mant_H = torch.where(exp_den, torch.zeros_like(chase_mant_H), chase_mant_H)
        tick_slow_L = _sr_round(chase_mant_L / float(CARRY), e_L, pos, salt ^ 0x5A5A5A5A)
        tick_slow_H = _sr_round(chase_mant_H / float(CARRY), e_H, pos, salt ^ 0xA5A5A5A5)
        tick_slow = tick_slow_L + tick_slow_H
        s_slow = s_slow + tick_slow                              # 1041
        e_L = e_L - tick_slow_L * CARRY                          # subtract EXACTLY carried (1042)
        e_H = e_H - tick_slow_H * CARRY

        # ── (G) PER-ARM evaporation off the BLENDED raw coherence (point 5), on the
        #        POST-CHASE residual. The H-arm evap is suppressed on an exp-mode denormal. ──
        if gf_consol > 0.0:
            lam = lr * gf_consol
            lam_L = lam * (1.0 - self.bracket_d)                 # arithmetic bracket (point 2)
            lam_H = lam * (1.0 + self.bracket_d)
            one_minus = (1.0 - coh_raw_blend)                    # BLENDED (point 5)
            evap_frac_L = torch.clamp(lam_L * one_minus, max=1.0 - min_leak)   # 936
            evap_frac_H = torch.clamp(lam_H * one_minus, max=1.0 - min_leak)
            res_L = e_L.to(torch.float32); res_H = e_H.to(torch.float32)
            p_build_L = torch.clamp(res_L.abs() / (evap_build_min + 1e-30), max=1.0)
            p_build_H = torch.clamp(res_H.abs() / (evap_build_min + 1e-30), max=1.0)
            r_build_L = _hash_uniform(e_L, pos, salt ^ 0x42420001)
            r_build_H = _hash_uniform(e_H, pos, salt ^ 0x42420002)
            ok_L = (r_build_L < p_build_L).to(torch.float32)
            ok_H = (r_build_H < p_build_H).to(torch.float32)
            evap_mant_L = evap_frac_L * res_L * ok_L * consf
            evap_mant_H = evap_frac_H * res_H * ok_H * consf
            evap_mant_H = torch.where(exp_den, torch.zeros_like(evap_mant_H), evap_mant_H)
            tick_ev_L = _sr_round(evap_mant_L, e_L, pos, salt ^ 0x0E0E0001)
            tick_ev_H = _sr_round(evap_mant_H, e_H, pos, salt ^ 0x0E0E0002)
            e_L = e_L - tick_ev_L
            e_H = e_H - tick_ev_H
            # Evaporation is a SINK: book it into inflow_int (as legacy folds evap into
            # delta_t) so the chase/leak transfer still conserves Δdeploy + Δfine == inflow.
            inflow_int = inflow_int - (tick_ev_L + tick_ev_H).to(torch.int64)

        # ── (H) leak -> v_slow (1044-1060), SHARED, blended coherence floor. ──
        coh_bar = 0.5 * (coh_L + coh_H)
        gap_v = s_slow.to(torch.float32) * 128 - v_slow.to(torch.float32) * 128
        delta_v8 = alpha_v_fast * gap_v / 128.0 * consf          # 1049
        delta_v8 = delta_v8 * (leak_floor + (1.0 - leak_floor) * coh_bar)   # 1051
        tick_v8 = _sr_round(delta_v8, e_L, pos, salt ^ 0x33335555)
        v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)
        actual_tick_v8 = v_slow_new - v_slow
        if mass_preserve:
            s_slow = s_slow - actual_tick_v8                     # 1060

        # ── (H.2) NO deploy dissipation (architect ruling). The consolidated theory
        #          (s_slow, v_slow) is EARNED knowledge and MUST NOT decay toward the
        #          substrate — only the HYPOTHESES (the fine register e_L/e_H) dissipate, via
        #          the step-(G) evap. Under zero gradient the fine drains but the integral
        #          (s_slow+v_slow) PERSISTS, so the weight settles at substrate + consolidated
        #          offset, NOT at the substrate. (A consolidated theory does not evaporate
        #          just because the gradient went quiet.)

        # ── (I) ADD-2 denormal sigma-delta + CONSERVING graduation (spec §2.5, HOLE-2). ──
        # An exp-mode denormal's gradient inflow (sub-LSB) is banked in the LOG field as a
        # fixed-point sigma-delta. When the running linear value grows past one mantissa
        # unit, PROMOTE the whole units into e_L (a fine->fine move, the same units e_L
        # holds — booked as inflow), and re-encode the sub-unit residual back into e_H's
        # octave field with SR (DC-neutral). NEVER a +128 credit into s_slow.
        if bool(exp_den.any()):
            # Bank the H-ARM's inflow into the log field (e_H is the high-dissipation arm's
            # sub-unit register). Single-grad: inflow_H == 0.5*delta (== the old half_inflow,
            # bit-identical). Two-grad: data-half-2's full preconditioned gradient.
            e_H, e_L, grad_inflow = self._denormal_graduate(
                exp_den, e_H, e_L, inflow_H, pos, salt)
            inflow_int = inflow_int + grad_inflow.to(torch.int64)

        # ── (I.5) BUG-1 (1b): denormal -> normal TRANSITION. The L-arm chase (F) / leak (H)
        #         may have ticked s_slow/v_slow on a coord that started this step as an
        #         exp-mode denormal. That coord is now NORMAL, so its e_H bits will be
        #         reinterpreted as a LINEAR int8 velocity from here on — both a phantom
        #         integer-ledger jump (fine_mantissa now counts the log bits as mass) AND a
        #         live-weight discontinuity (the garbage velocity). DROP the sub-unit log
        #         residual: clear e_H to 0 on coords that WERE exp-mode denormal and are now
        #         normal. The discarded value is < 1 mantissa unit (sub-deploy-LSB), so this
        #         injects no integer ledger mass (fine_mantissa already read it as 0 while it
        #         was a denormal, and reads e_H=0 as 0 now). ──
        now_normal = (s_slow != 0) | (v_slow_new != 0)
        transitioned = exp_den & now_normal
        if bool(transitioned.any()):
            e_H = torch.where(transitioned, torch.zeros_like(e_H), e_H)

        # ── (J) re-exponent safety (1132-1143), DENORMAL-AWARE (HOLE-3): the octave field
        #        of an exp-mode denormal shifts instead of integer-halving the log bits. ──
        packed_now = pack_dual(e_H.clamp(INT8_MIN, INT8_MAX), e_L.clamp(INT8_MIN, INT8_MAX),
                               s_slow.clamp(INT8_MIN, INT8_MAX),
                               v_slow_new.clamp(INT8_MIN, INT8_MAX))
        e_H, e_L, s_slow, v_slow_new, self.row_exp = renormalize_on_saturation(
            e_H, e_L, s_slow, v_slow_new, self.row_exp, packed_for_denorm=packed_now)

        # ── (K) clamp (int8 per byte) + repack ──
        # BUG-1 finalize SINK booking. The integer ledger is Δdeploy + Δ(ledger_fine) ==
        # inflow_int, ledger_fine = e_L + ledger_eH (ledger_eH EXCLUDES an exp-mode denormal's
        # sub-unit LOG field, 1a). The chase/leak deploy transfer conserves fine<->deploy by
        # construction. Two finalize effects remove integer mantissa from the fine ledger with
        # NO matching gradient/graduation credit, so each is booked as a SINK into inflow_int
        # (exactly as evaporation is) — derived from the REAL physical quantity, never from the
        # ledger identity (so the conservation test still has teeth):
        #   (i)   int8 CLAMP of the FINE bytes (a saturated e_L/e_H drops mantissa mass);
        #   (i.b) int8 CLAMP of the DEPLOY byte s_slow (a saturated s_slow cannot absorb the
        #         chase/leak credit -> the deploy word advances by LESS than the fine register
        #         was debited; the lost deploy mantissa is s_slow*128 units);
        #   (ii)  DENORMAL REINTERPRETATION of e_H: a coord NORMAL at step start (e_H a linear
        #         velocity, booked into inflow via tickg_H) that ends as an exp-mode denormal
        #         has its e_H re-read as a sub-unit LOG field -> dropped from the integer ledger.
        e_H_pre, e_L_pre, s_slow_pre = e_H, e_L, s_slow      # post-(J), pre-clamp values
        e_H = e_H.clamp(INT8_MIN, INT8_MAX)
        e_L = e_L.clamp(INT8_MIN, INT8_MAX)
        s_slow = s_slow.clamp(INT8_MIN, INT8_MAX)
        v_slow_new = v_slow_new.clamp(INT8_MIN, INT8_MAX)

        final_exp_den = (s_slow == 0) & (v_slow_new == 0) & is_exp_mode(e_H)
        # (i) clamp loss: ledger-visible change of the fine bytes from the int8 clamp. e_L is
        #     always linear; e_H is linear only where the FINAL coord is NOT an exp-mode
        #     denormal (an exp-mode denormal's e_H is the LOG field, out of the ledger).
        ledger_eH_postclamp = torch.where(final_exp_den, torch.zeros_like(e_H), e_H)
        ledger_eH_preclamp = torch.where(final_exp_den, torch.zeros_like(e_H_pre), e_H_pre)
        clamp_loss = ((e_L - e_L_pre).to(torch.int64)
                      + (ledger_eH_postclamp - ledger_eH_preclamp).to(torch.int64)
                      + (s_slow - s_slow_pre).to(torch.int64) * CARRY)   # (i.b) s_slow clamp
        # (ii) reinterpretation sink: for coords that were NOT exp-mode denormal at step START
        #      (e_H booked as linear) but ARE at step END, the final ledger e_H (now 0) drops
        #      the linear value that was carried — book -(that dropped linear e_H). For coords
        #      that were exp-mode denormal at start, e_H was already out of the ledger (no
        #      linear booking), so no sink. ledger_eH_preclamp is 0 on final_exp_den, so the
        #      reinterpreted drop is captured against the *pre-step linear* e_H booking via the
        #      start-vs-final ledger comparison below.
        reinterp_sink = torch.where(
            final_exp_den & (~exp_den), (-e_H).to(torch.int64), torch.zeros_like(e_H, dtype=torch.int64))
        inflow_int = inflow_int + clamp_loss + reinterp_sink
        self.packed_w = pack_dual(e_H, e_L, s_slow, v_slow_new)

        return {
            "coh_L": float(coh_L.mean()), "coh_H": float(coh_H.mean()),
            "coh_raw_blend": float(coh_raw_blend.mean()),
            "tick_slow_mean": float(tick_slow.to(torch.float32).mean()),
            "tick_v8_mean": float(tick_v8.to(torch.float32).mean()),
            "deploy_advanced": bool((tick_slow.abs() + actual_tick_v8.abs()).sum() > 0),
            "e_L_abs_max": int(e_L.abs().max()), "e_H_abs_max": int(e_H.abs().max()),
            "fine_abs_mean": float(fine_value(e_H, e_L).abs().float().mean()),
            "n_denormal": int(denorm.sum()), "n_exp_denormal": int(exp_den.sum()),
            "_inflow_int": inflow_int,
        }

    @torch.no_grad()
    def _denormal_graduate(self, exp_den, e_H, e_L, half_inflow, pos, salt):
        """ADD-2 conserving graduation (spec §2.5, HOLE-2). For exp-mode denormal coords:

          val_old = decode_denormal_units(e_H)          # signed, |.| < 1 mantissa unit
          val_new = val_old + inflow_into_denormal      # LINEAR add, mantissa units
          whole   = trunc(val_new)                      # whole units to promote (signed)
          resid   = val_new - whole                     # sub-unit residual, |resid| < 1
          e_L    += whole                               # PROMOTE into e_L (fine->fine)
          e_H     = encode_denormal_sr(resid)           # SR re-encode -> DC-neutral carry

        `whole` is the gradient inflow that arrived this step (sub-LSB banked until it
        crosses a unit), so it is booked as inflow_int (returned) — Δ(fine) = +whole is NEW
        mass on the fine side, NEVER a +128 deploy credit. Returns (e_H, e_L, grad_inflow)."""
        d = exp_den
        inflow = torch.where(d, half_inflow, torch.zeros_like(half_inflow))   # mantissa units
        val_old = decode_denormal_units(e_H)                                  # |.| < 1
        val_new = val_old + inflow
        whole = torch.trunc(val_new)                                          # signed whole units
        resid = val_new - whole                                              # |resid| < 1
        whole_i = whole.to(torch.int32)
        # (a) PROMOTE whole units into e_L (retain arm) — booked as inflow.
        e_L = torch.where(d, e_L + whole_i, e_L)
        # (b) re-encode the sub-unit residual back into e_H's octave/significand with SR.
        e_H_new = encode_denormal_sr(resid, e_H, pos, salt ^ 0x0DED0DED)
        e_H = torch.where(d, e_H_new, e_H)
        grad_inflow = torch.where(d, whole_i, torch.zeros_like(whole_i))
        return e_H, e_L, grad_inflow

    # ------------------------------------------------------------------
    # LEGACY single-int16 tick (DISABLED). BIT-EXACT to the legacy kernel (point 7).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _legacy_tick(self, grad_W, lr, alpha, gf_consol, drift_cancel_C, alpha_v_fast,
                     coh_kappa, v_scale, precond_p, eps, step_cap, min_leak,
                     evap_build_min, beta1, use_coh_vhat, mass_preserve, chase_floor,
                     leak_floor, consf, vhat, sum_v_inv, scale_fwd, scale_inv, pos, salt):
        N, K = self.N, self.K
        s_fast, s_slow, v_slow = unpack_legacy(self.packed_w)
        s_fast = s_fast.to(torch.int32); s_slow = s_slow.to(torch.int32); v_slow = v_slow.to(torch.int32)

        d_fs = s_fast.to(torch.float32)                          # 778
        d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY   # 779
        coh, coh_raw = compute_coherence(d_fs, d_sv, vhat, drift_cancel_C,
                                         coh_kappa, sum_v_inv, N, K, scale_fwd, use_coh_vhat)
        noise_w = (d_fs - drift_cancel_C * d_sv) * scale_fwd     # 787-788
        v_proxy = noise_w * noise_w * v_scale
        step_live = (grad_W / torch.pow(v_proxy + eps, precond_p)).clamp(-step_cap, step_cap)  # 869-871
        delta_grad = -lr * step_live * scale_inv                 # 970
        if gf_consol > 0.0:
            evap_frac = torch.clamp(lr * gf_consol * (1.0 - coh_raw), max=1.0 - min_leak)  # 936
            p_build = torch.clamp(d_fs.abs() / (evap_build_min + 1e-30), max=1.0)          # 948
            r_build = _hash_uniform(s_fast, pos, salt ^ 0x42424242)                        # 949
            build_ok = (r_build < p_build).to(torch.float32)                               # 951
            evap_mantissa = evap_frac * d_fs * build_ok                                     # 952
        else:
            evap_mantissa = torch.zeros_like(d_fs)
        delta_t = delta_grad + consf * (beta1 * coh * d_fs - evap_mantissa)                 # 984
        tick_fast = _sr_round(delta_t, s_fast, pos, salt)
        s_fast = s_fast + tick_fast                                                          # 992
        inflow_int = tick_fast.to(torch.int64)

        gate = chase_floor + (1.0 - chase_floor) * coh                                       # 1010
        chase_mantissa = alpha * gate * s_fast.to(torch.float32) * consf                     # 1026
        tick_slow = _sr_round(chase_mantissa / float(CARRY), s_fast, pos, salt ^ 0x5A5A5A5A)  # 1028-1031
        s_slow = s_slow + tick_slow                                                          # 1041
        s_fast = s_fast - tick_slow * CARRY                                                  # 1042

        gap_v = (s_slow.to(torch.float32) * 128 - v_slow.to(torch.float32) * 128)            # 1048
        delta_v8 = alpha_v_fast * gap_v / 128.0 * consf                                      # 1049
        delta_v8 = delta_v8 * (leak_floor + (1.0 - leak_floor) * coh)                        # 1051
        tick_v8 = _sr_round(delta_v8, s_fast, pos, salt ^ 0x33335555)                        # 1052-1055
        v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)                       # 1056
        if mass_preserve:
            s_slow = s_slow - (v_slow_new - v_slow)                                          # 1060

        # BUG-1 (legacy): the per-byte int16/int8 CLAMP of s_fast/s_slow can DESTROY deploy
        # mantissa — e.g. the chase debits s_fast by tick_slow*128 but a SATURATED s_slow
        # (pinned at INT8_MIN/MAX) cannot absorb the credit, so the deploy word advances by
        # LESS than the fine register was debited -> a 128-unit phantom leak. The clamp is a
        # SINK exactly like evaporation; book the ledger-visible clamp change into inflow_int
        # so Δ(s_slow+v_slow)*128 + Δs_fast == inflow_int closes EVERY step. (inflow_int is
        # ONLY the conservation-ledger return; the packed word is untouched, so the DISABLED
        # path stays BIT-EXACT to the literal legacy tick — point 7.)
        s_fast_pre, s_slow_pre = s_fast, s_slow
        s_fast = s_fast.clamp(INT16_MIN, INT16_MAX)                                          # 1104
        s_slow = s_slow.clamp(INT8_MIN, INT8_MAX)                                            # 1105
        clamp_loss = ((s_fast - s_fast_pre).to(torch.int64)
                      + (s_slow - s_slow_pre).to(torch.int64) * CARRY)
        inflow_int = inflow_int + clamp_loss
        self.packed_w = pack_legacy(s_fast, s_slow, v_slow_new)                              # 1106-1110

        return {
            "coh": float(coh.mean()), "coh_raw": float(coh_raw.mean()),
            "deploy_advanced": bool((tick_slow.abs() + (v_slow_new - v_slow).abs()).sum() > 0),
            "s_fast_abs_max": int(s_fast.abs().max()),
            "_inflow_int": inflow_int,
        }

    # ---- decode helpers ----
    @torch.no_grad()
    def live_weight(self):
        return decode_to_live_weight(self.packed_w, self.row_exp, self.col_exp,
                                     substrate=self.substrate)

    @torch.no_grad()
    def deploy_weight(self, step_salt=0, deterministic=False):
        return decode_to_deploy_weight(self.packed_w, self.row_exp, self.col_exp,
                                       substrate=self.substrate, step_salt=step_salt,
                                       enabled=self.enabled, deterministic=deterministic)


# ============================================================================
# Literal legacy single-int16 tick (free function) for the DISABLED==LEGACY
# bit-exactness test (point 7). Standalone re-implementation; identical hash/SR/clamp/order.
# ============================================================================
@torch.no_grad()
def legacy_single_int16_tick(packed, row_exp, col_exp, grad_W, lr, *, alpha, gf_consol,
                             drift_cancel_C, alpha_v_fast, coh_kappa, v_scale, precond_p,
                             eps, step_cap, min_leak, evap_build_min, beta1, beta2,
                             use_coh_vhat, mass_preserve, chase_floor, leak_floor, consf,
                             v_row, v_col, step, seed, mantissa_bias=MANTISSA_BIAS):
    """ONE step of the literal legacy single-int16-s_fast path. Returns
    (packed, v_row, v_col). Shares NO code with the Layer."""
    N, K = packed.shape
    salt = (int(seed) ^ (int(step) * 0x9E3779B1)) & 0x7FFFFFFF
    pos = _pos_hash(N, K)
    g2 = grad_W * grad_W
    v_row = beta2 * v_row + (1 - beta2) * g2.mean(dim=1)
    v_col = beta2 * v_col + (1 - beta2) * (g2.mean(dim=0) / (v_row.mean() + 1e-30))
    sum_v_inv = 1.0 / (v_row.sum() + 1e-30)
    v_bc = 1.0 / (1.0 - beta2 ** step)
    vhat = (v_row[:, None] * v_col[None, :] * sum_v_inv) * v_bc

    s_fast, s_slow, v_slow = unpack_legacy(packed)
    s_fast = s_fast.to(torch.int32); s_slow = s_slow.to(torch.int32); v_slow = v_slow.to(torch.int32)
    scale_fwd = _scale_fwd(row_exp, col_exp, mantissa_bias); scale_inv = 1.0 / scale_fwd
    d_fs = s_fast.to(torch.float32)
    d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY
    coh, coh_raw = compute_coherence(d_fs, d_sv, vhat, drift_cancel_C, coh_kappa,
                                     sum_v_inv, N, K, scale_fwd, use_coh_vhat)
    noise_w = (d_fs - drift_cancel_C * d_sv) * scale_fwd
    v_proxy = noise_w * noise_w * v_scale
    step_live = (grad_W / torch.pow(v_proxy + eps, precond_p)).clamp(-step_cap, step_cap)
    delta_grad = -lr * step_live * scale_inv
    if gf_consol > 0.0:
        evap_frac = torch.clamp(lr * gf_consol * (1.0 - coh_raw), max=1.0 - min_leak)
        p_build = torch.clamp(d_fs.abs() / (evap_build_min + 1e-30), max=1.0)
        r_build = _hash_uniform(s_fast, pos, salt ^ 0x42424242)
        build_ok = (r_build < p_build).to(torch.float32)
        evap_mantissa = evap_frac * d_fs * build_ok
    else:
        evap_mantissa = torch.zeros_like(d_fs)
    delta_t = delta_grad + consf * (beta1 * coh * d_fs - evap_mantissa)
    tick_fast = _sr_round(delta_t, s_fast, pos, salt)
    s_fast = s_fast + tick_fast
    gate = chase_floor + (1.0 - chase_floor) * coh
    chase_mantissa = alpha * gate * s_fast.to(torch.float32) * consf
    tick_slow = _sr_round(chase_mantissa / float(CARRY), s_fast, pos, salt ^ 0x5A5A5A5A)
    s_slow = s_slow + tick_slow
    s_fast = s_fast - tick_slow * CARRY
    gap_v = (s_slow.to(torch.float32) * 128 - v_slow.to(torch.float32) * 128)
    delta_v8 = alpha_v_fast * gap_v / 128.0 * consf
    delta_v8 = delta_v8 * (leak_floor + (1.0 - leak_floor) * coh)
    tick_v8 = _sr_round(delta_v8, s_fast, pos, salt ^ 0x33335555)
    v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)
    if mass_preserve:
        s_slow = s_slow - (v_slow_new - v_slow)
    s_fast = s_fast.clamp(INT16_MIN, INT16_MAX)
    s_slow = s_slow.clamp(INT8_MIN, INT8_MAX)
    packed = pack_legacy(s_fast, s_slow, v_slow_new)
    return packed, v_row, v_col


def assert_disabled_matches_legacy(seed=11, N=8, K=16, steps=60):
    """POINT 7 INVARIANT. The DISABLED Layer must be BIT-EXACT to the literal legacy
    single-int16 tick — packed word AND s_fast register — every step. Returns
    (ok, max_packed_gap, max_sfast_gap)."""
    torch.manual_seed(seed)
    W = torch.randn(N, K) * 0.05
    ref = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=seed)
    ref.load_weights(W)
    assert ref.substrate is None, "disabled load must leave substrate None"
    col_exp = ref.col_exp.clone()
    g = torch.randn(N, K) * 0.02
    ok = True; max_packed_gap = 0; max_sfast_gap = 0
    kw = dict(alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02, alpha_v_fast=0.001,
              coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
              min_leak=0.05, evap_build_min=128.0, beta1=0.0, beta2=0.999,
              use_coh_vhat=True, mass_preserve=True, chase_floor=0.1, leak_floor=0.05,
              consf=1.0)
    for t in range(steps):
        packed_in = ref.packed_w.clone()
        re_in = ref.row_exp.clone()
        vr_in = ref.v_row.clone(); vc_in = ref.v_col.clone()
        ref.step(g, lr=0.05, **kw)
        if bool((ref.row_exp != re_in).any()):
            continue                  # re-exponent fired: literal tick has no such path
        packed_leg, _, _ = legacy_single_int16_tick(
            packed_in, re_in, col_exp, g, 0.05, v_row=vr_in, v_col=vc_in,
            step=ref._step, seed=seed, **kw)
        packed_gap = int((ref.packed_w - packed_leg).abs().max())
        sf_a, _, _ = unpack_legacy(ref.packed_w)
        sf_b, _, _ = unpack_legacy(packed_leg)
        sfast_gap = int((sf_a - sf_b).abs().max())
        max_packed_gap = max(max_packed_gap, packed_gap)
        max_sfast_gap = max(max_sfast_gap, sfast_gap)
        if packed_gap != 0:
            ok = False
            break
    return ok, max_packed_gap, max_sfast_gap


# ============================================================================
# Self-test (CPU; `python dual_dissipation_ref.py`).  RUN NOTHING here that needs a GPU.
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    N, K = 8, 16

    # --- test 1: pack/unpack round-trip (BOTH views) + co-equal sum identity ---
    e_H = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
    e_L = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
    s_slow = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
    v_slow = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
    packed = pack_dual(e_H, e_L, s_slow, v_slow)
    e_H2, e_L2, s2, v2 = unpack_dual(packed)
    assert torch.equal(e_H, e_H2) and torch.equal(e_L, e_L2)
    assert torch.equal(s_slow, s2) and torch.equal(v_slow, v2)
    s_fast_view, _, _ = unpack_legacy(packed)
    assert torch.equal(s_fast_view, (packed >> 16))
    assert torch.equal(fine_value(e_H, e_L), (e_L + e_H))
    print("test 1 (pack/unpack both views + co-equal sum identity): PASS")

    # --- test 2: is_denormal reads COARSE ONLY (never moves with the fine bits) ---
    p_coarse0 = pack_dual(torch.full((N, K), 50, dtype=torch.int32),
                          torch.full((N, K), -30, dtype=torch.int32),
                          torch.zeros(N, K, dtype=torch.int32),
                          torch.zeros(N, K, dtype=torch.int32))
    assert bool(is_denormal(p_coarse0).all()), "s_slow==v_slow==0 must be denormal regardless of fine bits"
    p_coarse1 = pack_dual(torch.zeros(N, K, dtype=torch.int32),
                          torch.zeros(N, K, dtype=torch.int32),
                          torch.full((N, K), 3, dtype=torch.int32),
                          torch.zeros(N, K, dtype=torch.int32))
    assert not bool(is_denormal(p_coarse1).any()), "s_slow!=0 must NOT be denormal"
    print("test 2 (is_denormal from coarse word only): PASS")

    # --- test 3: ADD-2 exponent-claim decode + disjointness + sign continuity ---
    # (a) exp-mode decode value matches SGN*(1+mlow/16)*2^-(oct+1). BUG-2: the (oct,mlow)
    #     index is stored OFFSET BY ONE (a = idx + 1) so a == 0 stays RESERVED for true-zero
    #     (the (oct0,mlow0) value 0.5 is now code a == 1, no longer colliding with zero).
    for sgn in (+1, -1):
        for oct_ in range(OCT_MAX + 1):
            for mlow in range(0, 16, 5):
                idx = (oct_ << MANT_BITS) | mlow
                if idx > ((OCT_MAX << MANT_BITS) | MLOW_MASK) - 1:
                    continue                         # idx 63 is unrepresentable (a would be 64)
                a = idx + 1                          # BUG-2 offset-by-one encoding
                eh = torch.full((1, 1), sgn * a, dtype=torch.int32)
                got = float(decode_denormal_units(eh)[0, 0])
                want = sgn * (1.0 + mlow / 16.0) * 2.0 ** (-(oct_ + 1))
                assert abs(got - want) < 1e-6, (sgn, oct_, mlow, got, want)
    # (b) disjointness (HOLE-1): every exp-mode encoding has |e_H| < 64.
    max_a = (OCT_MAX << MANT_BITS) | MLOW_MASK
    assert max_a < EH_VELO_CAP, f"exp-mode max |e_H|={max_a} must be < {EH_VELO_CAP}"
    assert bool(is_exp_mode(torch.tensor([[max_a]], dtype=torch.int32))[0, 0])
    assert not bool(is_exp_mode(torch.tensor([[EH_VELO_CAP]], dtype=torch.int32))[0, 0])
    # (c) magnitude-mode denormal renders pure coarse (units 0, gate off in deploy).
    assert float(decode_denormal_units(torch.tensor([[100]], dtype=torch.int32))[0, 0]) == 0.0
    # (d) sign continuity: sweeping |e_H| up to 63 then 64 goes value -> 0 (no sign flip).
    v63 = float(decode_denormal_units(torch.tensor([[63]], dtype=torch.int32))[0, 0])
    v64 = float(decode_denormal_units(torch.tensor([[64]], dtype=torch.int32))[0, 0])
    assert v63 > 0.0 and v64 == 0.0
    print(f"test 3 (exp-claim decode + disjointness + continuity): "
          f"max exp-mode |e_H|={max_a}<64, smallest reach=2^-{OCT_MAX+1}: PASS")

    # --- test 4: PER-STEP CONSERVATION INVARIANT (point 8), incl. live denormals ---
    ref4 = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=2)
    W4 = torch.randn(N, K) * 0.05
    ref4.load_weights(W4)                      # from-scratch: substrate from seed
    g = -torch.sign(W4) * 0.02 + 0.02
    max_resid = 0; saw_denorm = 0
    for t in range(300):
        info = ref4.step(g, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                         coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0,
                         chase_floor=0.1, return_ledger=True)
        saw_denorm = max(saw_denorm, info["n_exp_denormal"])
        pb, pa, inflow = info["_ledger"]
        ok, resid = assert_conservation(pb, pa, inflow)
        max_resid = max(max_resid, resid)
        assert ok, f"CONSERVATION VIOLATED at step {t}: residual={resid}"
    print(f"test 4 (per-step conservation, incl. {saw_denorm} live exp-denormals/step): "
          f"max residual over 300 steps = {max_resid} (== 0)")
    assert max_resid == 0
    print("test 4: PASS")

    # --- test 5: DISABLED == LEGACY bit-exactness (point 7), many steps ---
    ok5, packed_gap5, sfast_gap5 = assert_disabled_matches_legacy(steps=60)
    print(f"test 5 (disabled == legacy, 60 steps): max packed gap={packed_gap5}, "
          f"max |s_fast diff|={sfast_gap5}")
    assert ok5 and packed_gap5 == 0 and sfast_gap5 == 0, \
        "disabled path must be BIT-EXACT to the literal legacy tick"
    ref5 = DualDissipationLayer(N, K, enabled=False, grad_accum_M=8, seed=1)
    ref5.load_weights(torch.randn(N, K) * 0.05)
    assert ref5.substrate is None, "disabled substrate must be None"
    g5 = torch.randn(N, K) * 0.02
    for t in range(40):
        ref5.step(g5, lr=0.05, gf_consol=0.0, drift_cancel_C=0.02)
    _, s5, v5 = unpack_legacy(ref5.packed_w)
    coarse_dep = ((s5 + v5).to(torch.float32) * 128 * _scale_fwd(ref5.row_exp, ref5.col_exp))
    assert torch.allclose(ref5.deploy_weight(), coarse_dep, atol=0), \
        "disabled deploy must be pure coarse (s_slow+v_slow)*128"
    print("test 5: PASS")

    # --- test 6: M-guard (point 5): M < M_MIN collapses to legacy (substrate None) ---
    ref6 = DualDissipationLayer(N, K, enabled=True, grad_accum_M=2, seed=3)
    assert ref6.enabled is False, "M < M_MIN must collapse the feature to legacy"
    ref6.load_weights(torch.randn(N, K) * 0.05)
    assert ref6.substrate is None, "collapsed layer takes the legacy branch (no substrate)"
    col_exp = ref6.col_exp.clone(); g6 = torch.randn(N, K) * 0.02
    kw = dict(alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02, alpha_v_fast=0.001,
              coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
              min_leak=0.05, evap_build_min=128.0, beta1=0.0, beta2=0.999,
              use_coh_vhat=True, mass_preserve=True, chase_floor=0.1, leak_floor=0.05,
              consf=1.0)
    pin = ref6.packed_w.clone(); rin = ref6.row_exp.clone()
    vr = ref6.v_row.clone(); vc = ref6.v_col.clone()
    ref6.step(g6, lr=0.05, **kw)
    if not bool((ref6.row_exp != rin).any()):
        pleg, _, _ = legacy_single_int16_tick(pin, rin, col_exp, g6, 0.05,
                                              v_row=vr, v_col=vc, step=ref6._step,
                                              seed=3, **kw)
        assert int((ref6.packed_w - pleg).abs().max()) == 0, \
            "M-guard collapsed layer must be bit-exact legacy"
    print("test 6 (M-guard collapse to legacy at M<4): PASS")

    # --- test 7: enabled deploy ratchets under a coherent drift + fine bytes bounded ---
    ref7 = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=4)
    W7 = torch.randn(N, K) * 0.05
    ref7.load_weights(W7)
    # train the OFFSET away from the substrate with a coherent drift.
    g7 = torch.ones(N, K) * 0.05
    dep0 = ref7.deploy_weight().clone()
    advanced = 0; max_eL = 0; max_eH = 0
    for t in range(300):
        info = ref7.step(g7, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                         coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
        advanced += int(info["deploy_advanced"])
        max_eL = max(max_eL, info["e_L_abs_max"]); max_eH = max(max_eH, info["e_H_abs_max"])
    moved = float((ref7.deploy_weight() - dep0).abs().sum())
    print(f"test 7 (enabled deploy ratchet): advanced {advanced}/300, moved {moved:.4e}, "
          f"max|e_L|={max_eL}, max|e_H|={max_eH} (both <=128)")
    assert advanced > 30 and moved > 0
    # int8-bounded means within [INT8_MIN, INT8_MAX] = [-128, 127]; the abs of the valid
    # floor INT8_MIN is 128 (> INT8_MAX=127), so bound the abs by -INT8_MIN, not INT8_MAX.
    assert max_eL <= -INT8_MIN and max_eH <= -INT8_MIN, "fine bytes must stay int8-bounded"
    print("test 7: PASS")

    # --- test 8: ADD-1 substrate. After load, live==deploy==substrate at step 0; decode
    #             is a pure function of (packed, row_exp, col_exp, substrate). ---
    ref8 = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=5)
    W8 = torch.randn(N, K) * 0.5
    ref8.load_weights(W8)                       # from-scratch: substrate from seed
    # offset is ZERO at step 0 -> weight == substrate exactly.
    assert int(ref8.packed_w.abs().max()) == 0, "accumulator must be ZERO after enabled load"
    live0 = ref8.live_weight(); dep0 = ref8.deploy_weight(deterministic=True)
    assert torch.equal(live0, ref8.substrate), "live weight must == substrate at step 0"
    assert torch.equal(dep0, ref8.substrate), "deploy weight must == substrate at step 0"
    # finetune variant: base supplied -> substrate IS the base, offset still zero.
    ref8b = DualDissipationLayer(N, K, enabled=True, grad_accum_M=8, seed=6)
    ref8b.load_weights(W8, base=W8)
    assert torch.equal(ref8b.substrate, W8), "finetune substrate must be the pretrained base"
    assert torch.equal(ref8b.live_weight(), W8), "finetune live weight == base at step 0"
    # decode is a pure function of (packed, row_exp, col_exp, substrate): re-decode a copy.
    g8 = torch.randn(N, K) * 0.1
    for t in range(150):
        ref8.step(g8, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                  coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
    fresh = ref8.packed_w.clone()
    w_a = decode_to_live_weight(ref8.packed_w, ref8.row_exp, ref8.col_exp, substrate=ref8.substrate)
    w_b = decode_to_live_weight(fresh, ref8.row_exp, ref8.col_exp, substrate=ref8.substrate)
    assert torch.equal(w_a, w_b), "live weight must be a pure function of (packed, exps, substrate)"
    # regenerating the substrate from the seed reproduces it exactly (checkpoint/relaunch).
    re_sub = make_substrate(N, K, ref8.substrate_seed, ref8.substrate_mode)
    assert torch.equal(re_sub, ref8.substrate), "substrate must regenerate from (seed, mode)"
    print("test 8 (substrate: live==deploy==substrate at step 0; decode pure fn; "
          "seed-regenerable): PASS")

    print("\nALL TESTS PASS")
