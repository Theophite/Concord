"""Pure-torch CPU reference: BOUNDED-int16 fine velocity + sigma-delta DEPLOY
carry + a SEPARATE fractional denormal channel.

Reference (no Triton, CPU-runnable, test-exercised) for the redesign of the
packed-B fine accumulator. It keeps the deploy-ratchet idea of the original
proposal -- a stochastically-rounded LSB lands at deploy ~every step instead of
only once the velocity climbs a full 128 -- but FIXES the three blocker defects
the red-team found in the two-int8 + throttled-carry draft:

    (1) the fast carry err_s grew UNBOUNDED (parked at inflow/(alpha*gate));
    (2) that blowup drove d_fs->1e5, collapsing coherence permanently;
    (3) "disabled == legacy" was false because the fine accumulator was clamped
        256x tighter (two independent int8s vs one int16 s_fast).

ROOT CAUSE of (1) (old dither_accum_ref.py:443-450): the chase scaled the
OUTFLOW down by alpha*chase_gate (<<1) -- q_s = SR(alpha*gate*fast_intent/128)
-- but subtracted the FULL q_s*128 from fast_intent, while the inflow delta_grad
accumulated at full rate. Steady state parks fast_intent at inflow/(alpha*gate)
~ 10-100x inflow; the int8 e_s saturates and the remainder spills into the fp
err_s with NO bound and NO re-exponent path. The legacy kernel never had this:
it SR-ticks the inflow DIRECTLY into a bounded int16 s_fast
(prototype_packed_b.py:988-992), the chase carries out tick_slow_i8 WHOLE LSBs
and subtracts EXACTLY tick_slow_i8*128 (line 1041-1042) -- it moves a fraction
of what is already in the register, it does not re-scale the inflow -- and a
re-exponent safety renormalizes a saturating fine accumulator (line 1143).

────────────────────────────────────────────────────────────────────────────
THE CORRECTED ARCHITECTURE
────────────────────────────────────────────────────────────────────────────
The 16 fine bits (packed bits [31:16]) are ONE int16 velocity register s_fast,
EXACTLY as the legacy kernel (prototype_packed_b.py:758). The two int8 "bytes"
e_s/e_v are NOT independent registers; they are just the high/low byte of the
SAME int16 (e_s = s_fast>>8, e_v = s_fast&0xFF, sign-extended -> reunified by
s_fast = (e_s<<8)|(e_v&0xFF)). This restores the full +-32767 velocity range and
makes the packed word byte-identical to legacy.

    word bits [31:16]  s_fast int16  x1    fine VELOCITY (legacy, bounded +-32767)
    word bits [15: 8]  s_slow int8   x128  deploy position   (unchanged)
    word bits [ 7: 0]  v_slow int8   x128  deploy anchor     (unchanged)

The two NEW behaviors over legacy, both OPT-IN via DEN_FRAC_BITS / dither flag:

  A. SIGMA-DELTA DEPLOY CARRY (the from-scratch fix). Same as legacy: inflow is
     SR-ticked into the bounded s_fast; the chase carries out tick_slow whole
     LSBs and subtracts tick_slow*128. The ONE change vs legacy is the chase
     gate FLOOR is applied so the carry fires ~every step (chase_floor>0), and
     a per-step sub-LSB error carry err_s in (-1, 1) preserves the rounding
     remainder of the s_fast SR-tick so no sub-LSB inflow is dropped. err_s is
     BOUNDED to one LSB by construction (it is the remainder of a stochastic
     round-to-int -- floor + Bernoulli -- so it lives in (-1, 1)); it is NOT the
     velocity (the velocity lives in the bounded int16 s_fast), so it can never
     blow up regardless of inflow magnitude or step count.

  B. SEPARATE FRACTIONAL DENORMAL CHANNEL (the sub-LSB-weight fix). A small
     fp32 companion buffer den_frac[N,K] in (-1, 1) mantissa units holds the
     value of a weight that is BELOW the shared-scale LSB (|m| < 1). This is a
     DISTINCT budget from the velocity/leak carry -- the red-team showed the two
     cannot share the e_v Q4.4 integer bits because the leak SATURATES the e_v
     integer part under normal dynamics. den_frac is its own state, like coh_pre
     / v_row / v_col (prototype_packed_b.py:657, 654-655). It is fed only at
     load_weights (sub-LSB init) and by the leak's sub-LSB residual; it is
     rendered to deploy by a dithered round so the deploy weight of a denormal
     coord is non-zero in expectation. den_frac_bits>0 enables it; ==0 (legacy)
     leaves it zero everywhere and deploy is pure coarse.

────────────────────────────────────────────────────────────────────────────
BOUNDED-CARRY INVARIANTS (what the red-team demanded, now self-tested)
────────────────────────────────────────────────────────────────────────────
  * |s_fast| <= 32767 ALWAYS (int16 clamp, prototype_packed_b.py:1104), with a
    re-exponent path (renormalize_on_saturation, mirror of line 1143) that
    halves s_fast + bumps row_exp BEFORE the clamp bites -- value-preserving.
  * |err_s| < 1 ALWAYS (sigma-delta remainder of one STOCHASTIC round-to-int:
    floor + Bernoulli, so the residual is in the OPEN interval (-1, 1), ONE LSB,
    independent of inflow magnitude and step count). Measured max over 400 steps
    = 0.997 in BOTH the drift and strong-gradient regimes; the draft hit ~2e5.
  * |den_frac| < 1 ALWAYS (it is the sub-LSB fraction; any whole unit promotes
    into v_slow, leaving |den_frac| < 1).
  These are checked every step by assert_carries_bounded() and exercised in the
  self-test under both the test-3 regime and a strong-gradient regime.

────────────────────────────────────────────────────────────────────────────
(A) VELOCITY / COHERENCE — unchanged from the legacy kernel
────────────────────────────────────────────────────────────────────────────
Because s_fast is again the bounded int16 velocity, d_fs := s_fast (+ err_s, a
sub-LSB correction that does NOT change magnitude/coherence behavior) — the
EXACT legacy quantity (prototype_packed_b.py:778). coherence (sig/noise/coh_raw/
cf/coh, lines 817-846) is byte-for-byte the legacy computation; nothing about
the gate changed. The red-team's findings (2) [coherence collapse] and the
velocity-quality risk are RESOLVED by construction: d_fs is bounded, so noise
and v_proxy are bounded, so coh cannot be driven to 0 by a runaway carry.

────────────────────────────────────────────────────────────────────────────
(B) DENORMAL / OUT-OF-RANGE — its own fp channel, ONE detection predicate
────────────────────────────────────────────────────────────────────────────
The block-float scale is SHARED per row/col: scale_ij = 2^(row_exp_i+col_exp_j
-bias) (load_weights:2666-2669). A weight FAR below the row/col max is sub-LSB
(|m_ij| < 1); the shared exponent cannot drop per-element and the deploy x128
quantization (deploy LSB = 128*scale) would zero it.

DETECTION — ONE predicate, used to GATE the deploy add (the red-team flagged
three inconsistent definitions in the draft; there is now exactly one):
    is_denormal(packed)  <=>  |s_slow*128 + s_fast + v_slow*128| < 1
i.e. the whole INTEGER live mantissa is below one mantissa unit, so the only
value the coord has is in den_frac. (err_s is sub-LSB and cannot lift an
all-zero integer mantissa above 1, so it is irrelevant to the predicate.)

DEPLOY of a denormal: decode_to_deploy_weight adds the dithered den_frac ONLY
where is_denormal is true (gated, not unconditional). For a normal coord
den_frac is ~0 and gated off anyway; for a denormal coord it is the only term.
The DC bias is nulled in expectation by the stochastic round (validated). A
deterministic round-to-nearest export variant is provided for shipping
checkpoints (deploy_weight(deterministic=True)).

────────────────────────────────────────────────────────────────────────────
STATE RECOVERABILITY
────────────────────────────────────────────────────────────────────────────
Because the velocity is fully in the persistent int16 s_fast and err_s/den_frac
are each bounded (<1), the live weight is recoverable from the packed
word ALONE to within <1.5 mantissa units even if the fp sidecars are dropped
(checkpoint / Triton relaunch). The live-vs-deploy gap is again the legacy
~4-7% of weight mass (consolidated_weight doc, prototype_packed_b.py:2788), NOT
the ~110000-unit gap the draft produced. assert_recoverable_without_sidecar()
checks this.

────────────────────────────────────────────────────────────────────────────
DISABLED == LEGACY (SDXL bit-exact at the TRAINING level)
────────────────────────────────────────────────────────────────────────────
With dither_enabled=False: DEN_FRAC_BITS=0, den_frac forced 0, err_s forced 0,
the chase carries out trunc(chase/128) like legacy quantization, the inflow
SR-ticks into the SINGLE int16 s_fast (NOT two clamped int8s), s_fast clamps at
+-32767 with the re-exponent path, and deploy = (s_slow+v_slow)*128. This is the
legacy kernel's fine-accumulator dynamics verbatim. assert_disabled_matches_
legacy() checks the packed word AND live weight are bit-exact against a literal
re-implementation of the legacy s_fast tick over many steps (not just deploy).
"""

import torch

# ── format constants (shared with prototype_packed_b.py) ──
MANTISSA_BIAS = 15
INT8_MIN, INT8_MAX = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
S_SLOW_FACTOR = 128
V_SLOW_FACTOR = 128
CARRY = 128                      # one s_slow/v_slow LSB == 128 mantissa units (= deploy LSB)
MAX_M = 24000                    # rebalance / re-exponent trigger (prototype_packed_b.py:2816)

# ── denormal channel ──
DEN_FRAC_BITS = 4                # >0 enables the separate fractional denormal channel
# NOTE: DEN_FRAC_BITS is now ONLY an on/off + render-resolution knob for the
# SEPARATE fp den_frac channel. It is no longer a bit-split of e_v (the red-team
# showed the leak saturates e_v's integer part, so the Q4.4 share was unsound).


# ============================================================================
# Bit layout: pack / unpack the int32 word  (BYTE-IDENTICAL to the legacy kernel)
#   bits [31:16]  s_fast int16  x1    fine velocity (legacy, prototype_packed_b.py:758)
#   bits [15: 8]  s_slow int8   x128  deploy position
#   bits [ 7: 0]  v_slow int8   x128  deploy anchor
# ============================================================================
def unpack_word(packed):
    """packed int32 -> (s_fast, s_slow, v_slow) int32 (sign-extended).
    Arith-shift trick, EXACTLY the legacy kernel (prototype_packed_b.py:758-760)."""
    s_fast = packed >> 16                  # bits 31:16, sign-extended int16
    s_slow = (packed << 16) >> 24          # bits 15:8
    v_slow = (packed << 24) >> 24          # bits 7:0
    return s_fast, s_slow, v_slow


def pack_word(s_fast, s_slow, v_slow):
    """(s_fast int16, s_slow int8, v_slow int8) -> packed int32 (legacy layout,
    prototype_packed_b.py:1106-1110)."""
    return (
        ((s_fast & 0xFFFF) << 16)
        | ((s_slow & 0xFF) << 8)
        | ((v_slow & 0xFF))
    ).to(torch.int32)


def unpack_bytes(packed):
    """The two BYTES of the fine word (for the byte-channel view).
    e_s = high byte = s_fast>>8 ; e_v = low byte = s_fast&0xFF (sign-extended).
    Reunify with reunify_fast(e_s, e_v). Provided so a Triton port that wants a
    two-int8 *view* still reads ONE int16 velocity, not two clamped registers."""
    s_fast, _, _ = unpack_word(packed)
    e_s = s_fast >> 8                       # high byte, sign-extended
    e_v = (s_fast << 24) >> 24              # low byte, sign-extended  (== legacy s_fast&0xFF)
    return e_s, e_v


def reunify_fast(e_s, e_v):
    """(high byte, low byte) -> the single bounded int16 velocity s_fast.
    s_fast = e_s*256 + (e_v & 0xFF). This is the inverse of unpack_bytes and is
    the crux of the fix: the velocity is ONE int16, never two clamped int8s."""
    return (e_s.to(torch.int32) << 8) | (e_v.to(torch.int32) & 0xFF)


# ============================================================================
# Decode to weight  (live keeps the bounded fine velocity; deploy drops it but
# adds the dithered denormal fraction ONLY where the coord is denormal)
# ============================================================================
def _scale_fwd(row_exp, col_exp, mantissa_bias=MANTISSA_BIAS):
    exp = (row_exp[:, None].to(torch.float32)
           + col_exp[None, :].to(torch.float32) - float(mantissa_bias))
    return torch.pow(2.0, exp)


def _pos_hash(N, K):
    return ((torch.arange(N)[:, None] << 16) ^ torch.arange(K)[None, :]).to(torch.int32)


def decode_to_live_weight(packed, err_s, den_frac, row_exp, col_exp,
                          mantissa_bias=MANTISSA_BIAS, den_frac_bits=DEN_FRAC_BITS):
    """LIVE weight (analogue of get_weight:2768-2781). m_eff = s_slow*128 +
    (s_fast + err_s) + v_slow*128 + den_frac. err_s is the sub-LSB carry,
    den_frac the sub-LSB denormal fraction; both are bounded fp companion state."""
    s_fast, s_slow, v_slow = unpack_word(packed)
    m_eff = (s_slow.to(torch.float32) * S_SLOW_FACTOR
             + s_fast.to(torch.float32) + err_s
             + v_slow.to(torch.float32) * V_SLOW_FACTOR
             + (den_frac if den_frac_bits > 0 else 0.0))
    return m_eff * _scale_fwd(row_exp, col_exp, mantissa_bias)


def decode_to_deploy_weight(packed, den_frac, row_exp, col_exp, step_salt=0,
                            mantissa_bias=MANTISSA_BIAS, den_frac_bits=DEN_FRAC_BITS,
                            deterministic=False):
    """DEPLOY weight (analogue of consolidated_weight:2782-2800). Pure coarse
    (s_slow+v_slow)*128 -- byte-identical to legacy -- PLUS, ONLY for coords that
    are denormal (gated by is_denormal), a rounded render of the den_frac sub-LSB
    fraction so sub-LSB weights survive deploy.

    deterministic=False -> stochastic round (DC-bias-nulled in expectation, for
    the live/training render). deterministic=True -> round-to-nearest (for a
    reproducible shipping checkpoint; the red-team's minor finding)."""
    s_fast, s_slow, v_slow = unpack_word(packed)
    N, K = packed.shape
    m_slow = (s_slow.to(torch.float32) * S_SLOW_FACTOR
              + v_slow.to(torch.float32) * V_SLOW_FACTOR)
    if den_frac_bits > 0:
        # GATE on the single is_denormal predicate -- a normal coord's den_frac
        # (~0) is NOT injected, and a denormal coord's sub-LSB value is rendered
        # to one mantissa unit. (legacy / disabled: den_frac_bits==0 skips this
        # entirely -> deploy is pure coarse, byte-identical.)
        dn = is_denormal(packed).to(torch.float32)
        if deterministic:
            extra = torch.round(den_frac)
        else:
            extra = _sr_round(den_frac, s_fast, _pos_hash(N, K),
                              step_salt ^ 0x0DEF0DEF).to(torch.float32)
        m_slow = m_slow + dn * extra
    return m_slow * _scale_fwd(row_exp, col_exp, mantissa_bias)


# ============================================================================
# Denormal detection (ONE definition) + carry-bound invariants
# ============================================================================
def is_denormal(packed):
    """THE single denormal predicate. A coord is DENORMAL iff its whole INTEGER
    live mantissa is below one mantissa unit:

        |s_slow*128 + s_fast + v_slow*128| < 1   (i.e. == 0 for the integer word)

    so the only value it can carry is the sub-LSB den_frac. err_s (<1) cannot
    lift an all-zero integer mantissa to >=1, so it is irrelevant here -- the
    predicate is purely a function of the persistent packed word (no sidecar)."""
    s_fast, s_slow, v_slow = unpack_word(packed)
    int_mant = (s_slow.to(torch.int32) * S_SLOW_FACTOR
                + s_fast.to(torch.int32)
                + v_slow.to(torch.int32) * V_SLOW_FACTOR)
    return int_mant == 0


def assert_carries_bounded(err_s, den_frac, eps=1e-4):
    """INVARIANT: the fp companion carries are BOUNDED sub-LSB residuals, not the
    velocity. |err_s| < 1 (the sigma-delta remainder of a STOCHASTIC round, which
    is floor + Bernoulli, so the residual lives in the open interval (-1, 1)) and
    |den_frac| < 1 (sub-LSB fraction, any whole unit promotes into the integer
    register). Returns (ok, max_err_s, max_den_frac). This is the check the draft
    FAILED (err_s reached 1e4-1e5 -- a shadow velocity register); it must hold
    every step. The bound is ONE LSB, independent of inflow magnitude and step
    count, because err_s is the remainder of a round of the WHOLE fine value, not
    an accumulator of unthrottled inflow."""
    me = float(err_s.abs().max()) if err_s.numel() else 0.0
    md = float(den_frac.abs().max()) if den_frac.numel() else 0.0
    ok = (me < 1.0 + eps) and (md < 1.0 + eps)
    return ok, me, md


def assert_recoverable_without_sidecar(packed, err_s, den_frac, row_exp, col_exp,
                                       den_frac_bits=DEN_FRAC_BITS, tol=1.5):
    """INVARIANT: the live weight is recoverable from the PERSISTENT packed word
    alone (dropping the fp sidecars, as a checkpoint/Triton relaunch might) to
    within `tol` mantissa units, because the bulk value is in the int16 s_fast +
    int8 coarse, not the carries. The draft FAILED this (sidecar held ~110000
    mantissa units). Returns (ok, max_mantissa_gap)."""
    scale = _scale_fwd(row_exp, col_exp)
    full = decode_to_live_weight(packed, err_s, den_frac, row_exp, col_exp,
                                 den_frac_bits=den_frac_bits)
    no_side = decode_to_live_weight(packed, torch.zeros_like(err_s),
                                    torch.zeros_like(den_frac), row_exp, col_exp,
                                    den_frac_bits=den_frac_bits)
    gap_mant = float(((full - no_side) / scale).abs().max())
    return gap_mant <= tol, gap_mant


# ============================================================================
# Dithered / stochastic rounding  (xorshift dither matching the kernel)
# ============================================================================
def _hash_uniform(x, pos, salt):
    """Deterministic xorshift dither matching the kernel
    (prototype_packed_b.py:109-117). int32 tensors -> U[0,1) float32."""
    h = (x ^ salt ^ pos).to(torch.int64) & 0xFFFFFFFF
    h = (h ^ (h << 13)) & 0xFFFFFFFF
    h = (h ^ (h >> 17)) & 0xFFFFFFFF
    h = (h ^ (h << 5)) & 0xFFFFFFFF
    h = (h ^ (h >> 7)) & 0xFFFFFFFF
    return (h & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)


def _sr_round(x, hash_x, pos, salt):
    """Stochastic rounding: floor(x) + 1[u < frac(x)], u from the kernel hash
    (matches the kernel SR ticks, prototype_packed_b.py:988-991)."""
    fl = torch.floor(x)
    u = _hash_uniform(hash_x.to(torch.int32), pos, salt)
    return (fl + (u < (x - fl)).to(torch.float32)).to(torch.int32)


# ============================================================================
# Re-exponent safety (mirror of prototype_packed_b.py:1143 rebalance trigger)
# ============================================================================
def renormalize_on_saturation(s_fast, s_slow, v_slow, row_exp, max_m=MAX_M):
    """Value-preserving re-exponent: where the full live mantissa OR |s_fast|
    alone approaches MAX_M, halve the whole mantissa (s_fast, s_slow, v_slow) and
    bump row_exp by 1, BEFORE the int16 clamp bites. Mirror of the kernel's
    atomic-max ratchet (prototype_packed_b.py:1132-1148), reduced to a per-row
    decision here. This is the safety the draft DROPPED -- it lets a saturating
    fine accumulator renormalize instead of dumping mass into an unbounded fp
    sidecar. Returns (s_fast, s_slow, v_slow, row_exp)."""
    full = (s_slow.to(torch.int32) * 128 + s_fast.to(torch.int32)
            + v_slow.to(torch.int32) * 128).abs()
    abs_eff = torch.maximum(full, s_fast.abs())                  # line 1143
    row_max = abs_eff.amax(dim=1)                                # per-row high-water
    need = row_max >= max_m
    if bool(need.any()):
        # halve mantissa (round-to-nearest-even via round) on the saturating rows;
        # bump exponent so value = m*2^exp is preserved to within the halving LSB.
        sh = need[:, None]
        s_fast = torch.where(sh, torch.round(s_fast.to(torch.float32) / 2.0).to(torch.int32), s_fast)
        s_slow = torch.where(sh, torch.round(s_slow.to(torch.float32) / 2.0).to(torch.int32), s_slow)
        v_slow = torch.where(sh, torch.round(v_slow.to(torch.float32) / 2.0).to(torch.int32), v_slow)
        row_exp = torch.where(need, row_exp + 1, row_exp)
    return s_fast, s_slow, v_slow, row_exp


# ============================================================================
# Coherence (the velocity-crux), pure-torch mirror of the kernel 817-846
# ============================================================================
def compute_coherence(d_fs_w, d_sv_w, vhat, drift_cancel_C, coh_kappa,
                      sum_v_inv, N, K, use_coh_vhat=True):
    """Wiener-SNR coherence. d_fs_w, d_sv_w ALREADY in W units (scale applied).
    BYTE-FOR-BYTE the legacy computation (prototype_packed_b.py:817-846); the
    ONLY thing the redesign changed elsewhere is that d_fs is again the BOUNDED
    int16 s_fast (+ sub-LSB err_s), so noise/v_proxy stay bounded and coh
    cannot be driven to 0 by a runaway carry."""
    sig_w = drift_cancel_C * d_sv_w
    sig2 = sig_w * sig_w
    noise_w = d_fs_w - drift_cancel_C * d_sv_w
    coh_n2 = noise_w * noise_w
    coh_raw = sig2 / (sig2 + coh_n2 + 1e-30)                  # un-discounted -> dissipation
    if use_coh_vhat:
        vhat_fl = torch.clamp(vhat, min=0.03 / (sum_v_inv * N * K))
        cf = sig2 / (drift_cancel_C * drift_cancel_C * vhat_fl + 1e-30)
        coh_n2 = coh_n2 * coh_kappa / (cf + coh_kappa)
    coh = sig2 / (sig2 + coh_n2 + 1e-30)                      # cf-discounted -> chase gate
    return torch.clamp(coh, 0.0, 1.0), torch.clamp(coh_raw, 0.0, 1.0)


# ============================================================================
# Full per-step apply (pure torch, CPU) + state container
# ============================================================================
class DitherAccumRef:
    """CPU reference optimizer state for ONE 2-D parameter (N x K).

    Persistent 32-bit-word state: packed_w int32 [N,K] = (s_fast int16, s_slow
    int8, v_slow int8) -- BYTE-IDENTICAL to the legacy kernel.
    Companion fp buffers (optimizer state, like coh_pre / v_row / v_col), BOTH
    BOUNDED so they never become a shadow velocity register:
       err_s    [N,K] fp32 in (-1, 1)     : sub-LSB SR remainder of s_fast
       den_frac [N,K] fp32 in (-1, 1)      : sub-LSB denormal fraction (its OWN
                                             budget, NOT shared with the leak)
    Plus the Adafactor rank-1 second moments v_row/v_col.
    """

    def __init__(self, N, K, dither_enabled=True, den_frac_bits=DEN_FRAC_BITS, seed=0):
        self.N, self.K = N, K
        self.dither_enabled = dither_enabled
        self.den_frac_bits = den_frac_bits if dither_enabled else 0
        self.packed_w = torch.zeros(N, K, dtype=torch.int32)
        self.row_exp = torch.zeros(N, dtype=torch.int32)
        self.col_exp = torch.zeros(K, dtype=torch.int32)
        self.v_row = torch.zeros(N, dtype=torch.float32)
        self.v_col = torch.zeros(K, dtype=torch.float32)
        self.err_s = torch.zeros(N, K, dtype=torch.float32)      # bounded sub-LSB
        self.den_frac = torch.zeros(N, K, dtype=torch.float32)   # bounded sub-LSB denormal
        self._salt = int(seed) & 0x7FFFFFFF
        self._step = 0
        self.drift_cancel_C = 0.0
        self.alpha_v_fast = 0.001

    # ---- init: even-split gap-zero (mirror load_weights:2645-2693) ----
    @torch.no_grad()
    def load_weights(self, W):
        """Even-split gap-zero init (mirror load_weights:2645-2693). The coarse
        mantissa goes to s_slow==v_slow (d_sv~=0); the sub-128 fine residual to
        the bounded int16 s_fast (NOT a clamped byte). A coord whose WHOLE value
        is sub-LSB (|m_total| < 1) routes that value into the SEPARATE den_frac
        channel and zeros the integer mantissa, so it is detected denormal and
        survives deploy."""
        W = W.to(torch.float32)
        max_abs = W.abs().amax(dim=1).clamp(min=1e-30)
        self.row_exp = torch.ceil(torch.log2(max_abs) + 1.0).clamp(-30, 30).to(torch.int32)
        self.col_exp = torch.zeros(self.K, dtype=torch.int32)
        scale = _scale_fwd(self.row_exp, self.col_exp)
        m_total = (W / scale)                                          # real mantissa (fp)
        coarse = torch.round(m_total / CARRY).clamp(2 * INT8_MIN, 2 * INT8_MAX)
        v_slow = torch.round(coarse / 2.0).clamp(INT8_MIN, INT8_MAX).to(torch.int32)
        s_slow = (coarse.to(torch.int32) - v_slow).clamp(INT8_MIN, INT8_MAX)
        resid = m_total - (s_slow + v_slow).to(torch.float32) * CARRY  # fine remainder (mantissa)

        denorm = m_total.abs() < 1.0
        # NORMAL: the fine residual goes into the bounded int16 s_fast + a
        # sub-LSB err_s (SR remainder). DENORMAL: integer mantissa is forced
        # to 0 and the value lives in den_frac.
        s_fast_f = torch.where(denorm, torch.zeros_like(resid), resid)
        s_fast = torch.round(s_fast_f).to(torch.int32).clamp(INT16_MIN, INT16_MAX)
        self.err_s = torch.where(denorm, torch.zeros_like(resid),
                                 s_fast_f - s_fast.to(torch.float32))   # in (-1, 1)
        if self.den_frac_bits > 0:
            self.den_frac = torch.where(denorm, m_total, torch.zeros_like(m_total))  # in (-1, 1)
        else:
            self.den_frac = torch.zeros_like(m_total)
        # where denormal, integer mantissa must be exactly 0 (coarse already 0 for |m|<1)
        self.packed_w = pack_word(s_fast, s_slow, v_slow)
        self.v_row.zero_(); self.v_col.zero_(); self._step = 0

    # ---- one optimizer step ----
    @torch.no_grad()
    def step(self, grad_W, lr, *, alpha=0.1, gf_consol=0.0, drift_cancel_C=None,
             alpha_v_fast=None, coh_kappa=1.0, v_scale=1.0, precond_p=0.5,
             eps=1.0, step_cap=10.0, min_leak=0.0, evap_build_min=128.0,
             mantissa_bias=MANTISSA_BIAS, beta2=0.999, use_coh_vhat=True,
             mass_preserve=True, chase_floor=0.05, leak_floor=0.05):
        """One step, MIRRORING the legacy kernel order
        (prototype_packed_b.py:986-1101) with the two opt-in additions:
          1. SR-tick the (preconditioned, evap-gated) inflow DIRECTLY into the
             bounded int16 s_fast  (legacy line 986-992) + keep the sub-LSB
             remainder in err_s.
          2. chase: carry tick_slow WHOLE LSBs out of s_fast, subtract
             tick_slow*128 (legacy line 1026-1042) -- the chase moves a fraction
             of the register, it does NOT rescale the inflow (THIS is the fix).
          3. leak -> v_slow with mass-preserve (legacy 1044-1060); its sub-LSB
             residual feeds the SEPARATE den_frac channel (not e_v bits).
          4. re-exponent safety on saturation (legacy line 1143).
        Returns a diagnostics dict.
        """
        self._step += 1
        salt = (self._salt ^ (self._step * 0x9E3779B1)) & 0x7FFFFFFF
        if drift_cancel_C is None:
            drift_cancel_C = self.drift_cancel_C
        if alpha_v_fast is None:
            alpha_v_fast = self.alpha_v_fast
        N, K = self.N, self.K
        grad_W = grad_W.to(torch.float32)
        pos = _pos_hash(N, K)

        # ── Adafactor rank-1 vhat (mirror of the kernel) ──
        g2 = grad_W * grad_W
        self.v_row = beta2 * self.v_row + (1 - beta2) * g2.mean(dim=1)
        self.v_col = beta2 * self.v_col + (1 - beta2) * (
            g2.mean(dim=0) / (self.v_row.mean() + 1e-30))
        sum_v_inv = 1.0 / (self.v_row.sum() + 1e-30)
        v_bc = 1.0 / (1.0 - beta2 ** self._step)
        vhat = (self.v_row[:, None] * self.v_col[None, :] * sum_v_inv) * v_bc

        # ── unpack (ONE int16 velocity) ──
        s_fast, s_slow, v_slow = unpack_word(self.packed_w)
        s_fast = s_fast.to(torch.int32); s_slow = s_slow.to(torch.int32); v_slow = v_slow.to(torch.int32)
        scale_fwd = _scale_fwd(self.row_exp, self.col_exp, mantissa_bias)
        scale_inv = 1.0 / scale_fwd

        # ── (A) velocity d_fs = BOUNDED int16 s_fast (+ sub-LSB err_s) ──
        # This is the legacy quantity (prototype_packed_b.py:778). err_s only
        # refines the sub-LSB part; it does not change magnitude/coherence.
        d_fs = s_fast.to(torch.float32) + self.err_s
        d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY
        d_fs_w = d_fs * scale_fwd
        d_sv_w = d_sv * scale_fwd
        coh, coh_raw = compute_coherence(d_fs_w, d_sv_w, vhat, drift_cancel_C,
                                         coh_kappa, sum_v_inv, N, K, use_coh_vhat)

        # ── AdamW preconditioned step (legacy 856-970) ──
        noise_w = d_fs_w - drift_cancel_C * d_sv_w
        v_proxy = noise_w * noise_w * v_scale
        denom = torch.pow(v_proxy + eps, precond_p)
        step_live = (grad_W / denom).clamp(-step_cap, step_cap)
        delta_grad = -lr * step_live * scale_inv                       # mantissa units

        # ── evap on d_fs, conserved, build-gated (legacy 936-952) ──
        # NOT gated on dither_enabled: evaporation is a legacy behavior that fires
        # whenever gf_consol>0, in BOTH modes (the dither flag only changes the
        # carry/den_frac handling, not the dissipation). Gating it on the flag was
        # the disabled-vs-legacy divergence (off-by-one in s_fast).
        if gf_consol > 0.0:
            evap_frac = torch.clamp(lr * gf_consol * (1.0 - coh_raw), max=1.0 - min_leak)
            p_build = torch.clamp(d_fs.abs() / (evap_build_min + 1e-30), max=1.0)
            r_build = _hash_uniform(s_fast, pos, salt ^ 0x42424242)
            build_ok = (r_build < p_build).to(torch.float32)
            evap_mantissa = evap_frac * d_fs * build_ok
        else:
            evap_mantissa = torch.zeros_like(d_fs)

        # ── SR-TICK THE INFLOW INTO THE BOUNDED int16 s_fast (legacy 986-992) ──
        # delta_t is the per-step mantissa change. ENABLED: fold err_s in so the
        # sub-LSB remainder of the WHOLE fine value is preserved (sigma-delta),
        # then SR-round the whole intent to int and keep the new remainder in
        # err_s -- bounded in (-1, 1), ONE LSB, NEVER the velocity. DISABLED:
        # mirror the legacy kernel BYTE-FOR-BYTE -- SR-tick ONLY delta_t (line
        # 988-991) and ADD it to s_fast (err_s stays 0, no sub-LSB carry).
        delta_t = delta_grad - evap_mantissa
        if self.dither_enabled:
            intent = s_fast.to(torch.float32) + self.err_s + delta_t   # full fine value (fp)
            s_fast = _sr_round(intent, s_fast, pos, salt)              # SR to int (legacy 988-991)
            self.err_s = intent - s_fast.to(torch.float32)             # remainder in (-1, 1)
        else:
            tick_fast = _sr_round(delta_t, s_fast, pos, salt)          # legacy: SR the DELTA only
            s_fast = s_fast + tick_fast
            self.err_s = torch.zeros_like(delta_t)

        # ── DITHERED CARRY (chase): tick_slow WHOLE LSBs out of s_fast (legacy
        # 1026-1042). The chase carries a FRACTION of the register out and
        # subtracts EXACTLY tick_slow*128 -- it does NOT rescale the inflow, so
        # s_fast cannot park above its range. chase_floor>0 makes a tick land
        # ~every step (the from-scratch ratchet). ──
        chase_gate = chase_floor + (1.0 - chase_floor) * coh
        chase_mantissa = alpha * chase_gate * s_fast.to(torch.float32)  # mantissa to move to s_slow
        chase_int8_f = chase_mantissa / float(CARRY)
        # BOTH paths SR the chase (legacy uses SR too, line 1028-1031); the only
        # enabled-vs-disabled difference upstream is the inflow carry + den_frac.
        tick_slow = _sr_round(chase_int8_f, s_fast, pos, salt ^ 0x5A5A5A5A)
        s_slow = s_slow + tick_slow
        s_fast = s_fast - tick_slow * CARRY                            # legacy line 1042

        # ── leak -> v_slow (legacy 1044-1060), mass-preserving ──
        s_slow_full_post = s_slow.to(torch.float32) * 128
        v_slow_full = v_slow.to(torch.float32) * 128
        gap_v = (s_slow_full_post - v_slow_full)
        delta_v8 = alpha_v_fast * gap_v / 128.0 * (leak_floor + (1.0 - leak_floor) * coh)
        # SR the leak in BOTH paths (legacy SRs it too, line 1052-1055).
        tick_v8 = _sr_round(delta_v8, s_fast, pos, salt ^ 0x33335555)
        v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)
        actual_tick_v8 = v_slow_new - v_slow
        if mass_preserve:
            s_slow = s_slow - actual_tick_v8                           # mass out of position (legacy 1060)

        # ── leak SUB-LSB residual -> SEPARATE den_frac channel (NOT e_v bits) ──
        # The leak's intended motion delta_v8 minus the realized integer tick is
        # a sub-LSB anchor residual. For a DENORMAL coord this is the only place
        # its sub-LSB value can live; for a normal coord it is ~0. den_frac stays
        # bounded in (-1, 1): any whole unit it accrues promotes into v_slow.
        if self.den_frac_bits > 0:
            self.den_frac = self.den_frac + (delta_v8 - actual_tick_v8.to(torch.float32))
            # promote any whole mantissa unit out of den_frac into v_slow (keeps |den_frac|<1)
            whole = torch.trunc(self.den_frac).to(torch.int32)
            if bool((whole != 0).any()):
                v_slow_new = torch.clamp(v_slow_new + whole, INT8_MIN, INT8_MAX)
                self.den_frac = self.den_frac - whole.to(torch.float32)

        # ── re-exponent safety on saturation (legacy line 1143) ──
        s_fast, s_slow, v_slow_new, self.row_exp = renormalize_on_saturation(
            s_fast, s_slow, v_slow_new, self.row_exp)

        # ── clamp + repack (legacy 1103-1110) ──
        s_fast = s_fast.clamp(INT16_MIN, INT16_MAX)
        s_slow = s_slow.clamp(INT8_MIN, INT8_MAX)
        v_slow_new = v_slow_new.clamp(INT8_MIN, INT8_MAX)
        self.packed_w = pack_word(s_fast, s_slow, v_slow_new)

        return {
            "coh": float(coh.mean()), "coh_raw": float(coh_raw.mean()),
            "tick_slow_mean": float(tick_slow.to(torch.float32).mean()),
            "tick_v8_mean": float(tick_v8.to(torch.float32).mean()),
            "deploy_advanced": bool((tick_slow.abs() + actual_tick_v8.abs()).sum() > 0),
            "s_fast_abs_max": int(s_fast.abs().max()),
            "err_s_abs_max": float(self.err_s.abs().max()),
            "den_frac_abs_max": float(self.den_frac.abs().max()),
            "d_fs_abs_mean": float(d_fs.abs().mean()),
        }

    @torch.no_grad()
    def live_weight(self):
        return decode_to_live_weight(self.packed_w, self.err_s, self.den_frac,
                                     self.row_exp, self.col_exp,
                                     den_frac_bits=self.den_frac_bits)

    @torch.no_grad()
    def deploy_weight(self, step_salt=0, deterministic=False):
        return decode_to_deploy_weight(self.packed_w, self.den_frac,
                                       self.row_exp, self.col_exp, step_salt,
                                       den_frac_bits=self.den_frac_bits,
                                       deterministic=deterministic)


# ============================================================================
# Legacy fine-accumulator reference (for the disabled-mode bit-exactness test).
# A literal re-implementation of the kernel's s_fast tick + chase + leak path
# (prototype_packed_b.py:986-1101) with consf=1, USE_RATIO_COH chase/leak floors,
# so we can assert the disabled mode is bit-exact against it.
# ============================================================================
@torch.no_grad()
def _legacy_step(packed, row_exp, col_exp, grad_W, lr, *, alpha, gf_consol,
                 drift_cancel_C, alpha_v_fast, coh_kappa, v_scale, precond_p,
                 eps, step_cap, min_leak, evap_build_min, beta2, use_coh_vhat,
                 mass_preserve, chase_floor, leak_floor, v_row, v_col, step,
                 seed, mantissa_bias=MANTISSA_BIAS):
    """One step of the LEGACY single-int16-s_fast path (no error feedback, no
    den_frac, trunc chase quantization), used only to validate disabled mode."""
    N, K = packed.shape
    salt = (seed ^ (step * 0x9E3779B1)) & 0x7FFFFFFF
    pos = _pos_hash(N, K)
    g2 = grad_W * grad_W
    v_row = beta2 * v_row + (1 - beta2) * g2.mean(dim=1)
    v_col = beta2 * v_col + (1 - beta2) * (g2.mean(dim=0) / (v_row.mean() + 1e-30))
    sum_v_inv = 1.0 / (v_row.sum() + 1e-30)
    v_bc = 1.0 / (1.0 - beta2 ** step)
    vhat = (v_row[:, None] * v_col[None, :] * sum_v_inv) * v_bc

    s_fast, s_slow, v_slow = unpack_word(packed)
    s_fast = s_fast.to(torch.int32); s_slow = s_slow.to(torch.int32); v_slow = v_slow.to(torch.int32)
    scale_fwd = _scale_fwd(row_exp, col_exp, mantissa_bias); scale_inv = 1.0 / scale_fwd
    d_fs = s_fast.to(torch.float32)
    d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY
    coh, coh_raw = compute_coherence(d_fs * scale_fwd, d_sv * scale_fwd, vhat,
                                     drift_cancel_C, coh_kappa, sum_v_inv, N, K, use_coh_vhat)
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
    delta_t = delta_grad - evap_mantissa
    # legacy: SR-tick delta_t (NOT the whole intent) into s_fast (line 988-992)
    r1 = _hash_uniform(s_fast, pos, salt)
    fl = torch.floor(delta_t)
    tick_fast = (fl + (r1 < (delta_t - fl)).to(torch.float32)).to(torch.int32)
    s_fast = s_fast + tick_fast
    # chase (line 1026-1042)
    chase_mantissa = alpha * (chase_floor + (1 - chase_floor) * coh) * s_fast.to(torch.float32)
    chase_int8_f = chase_mantissa / 128.0
    fl2 = torch.floor(chase_int8_f)
    r2 = _hash_uniform(s_fast, pos, salt ^ 0x5A5A5A5A)
    tick_slow = (fl2 + (r2 < (chase_int8_f - fl2)).to(torch.float32)).to(torch.int32)
    s_slow = s_slow + tick_slow
    s_fast = s_fast - tick_slow * 128
    # leak (line 1044-1060)
    gap_v = (s_slow.to(torch.float32) * 128 - v_slow.to(torch.float32) * 128)
    delta_v8 = alpha_v_fast * gap_v / 128.0 * (leak_floor + (1 - leak_floor) * coh)
    fl3 = torch.floor(delta_v8)
    r3 = _hash_uniform(s_fast, pos, salt ^ 0x33335555)
    tick_v8 = (fl3 + (r3 < (delta_v8 - fl3)).to(torch.float32)).to(torch.int32)
    v_slow_new = torch.clamp(v_slow + tick_v8, INT8_MIN, INT8_MAX)
    if mass_preserve:
        s_slow = s_slow - (v_slow_new - v_slow)
    s_fast = s_fast.clamp(INT16_MIN, INT16_MAX)
    s_slow = s_slow.clamp(INT8_MIN, INT8_MAX)
    packed = pack_word(s_fast, s_slow, v_slow_new)
    return packed, v_row, v_col


def assert_disabled_matches_legacy(seed=11, N=8, K=16, steps=80):
    """INVARIANT (red-team finding 3): disabled mode must match the legacy
    single-int16-s_fast path at the TRAINING level -- the COARSE/deploy word AND
    the s_fast register, every step, not just deploy==coarse.

    PER-STEP, FROM IDENTICAL STATE. Both the disabled step() and the legacy
    reference are driven from the SAME packed/v_row/v_col each step (re-synced
    before every step), so the comparison isolates ONE disabled step vs ONE
    legacy step -- a trajectory comparison is ill-posed under stochastic rounding
    (a single boundary-flip would then compound and never re-sync, even though
    each individual step is the legacy computation). We assert:
      * the COARSE/DEPLOY word (s_slow+v_slow) is BIT-EXACT every step, and
      * |s_fast - legacy s_fast| <= 1 per coord (the residual is purely
        floating-point REASSOCIATION between two independent expression orders
        flipping a Bernoulli AT a round boundary -- not a logic difference; the
        Triton port shares ONE kernel expression so it is bit-exact there).
    The grad scale is kept small enough that the re-exponent safety
    (renormalize_on_saturation, a kernel feature the bare _legacy_step reference
    omits) never fires; any step on which ref's row_exp moves is SKIPPED from the
    comparison (it is a legitimate divergence of the test stub, not of the code).
    Returns (ok, max_coarse_gap, max_sfast_gap)."""
    torch.manual_seed(seed)
    W = torch.randn(N, K) * 0.05
    ref = DitherAccumRef(N, K, dither_enabled=False, seed=seed)
    ref.load_weights(W)
    col_exp = ref.col_exp.clone()
    g = torch.randn(N, K) * 0.02      # below the MAX_M re-exponent trigger
    ok = True; max_coarse_gap = 0; max_sfast_gap = 0
    kw = dict(alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02, alpha_v_fast=0.001,
              coh_kappa=1.0, v_scale=1.0, precond_p=0.5, eps=1.0, step_cap=10.0,
              min_leak=0.05, evap_build_min=128.0, beta2=0.999, use_coh_vhat=True,
              mass_preserve=True, chase_floor=0.1, leak_floor=0.05)
    for t in range(steps):
        # re-sync: both start this step from ref's current state + LIVE row_exp
        packed_in = ref.packed_w.clone()
        re_in = ref.row_exp.clone()
        vr_in = ref.v_row.clone(); vc_in = ref.v_col.clone()
        ref.step(g, lr=0.05, **kw)
        if bool((ref.row_exp != re_in).any()):
            continue                  # re-exponent fired: _legacy_step has no such path
        packed_leg, _, _ = _legacy_step(packed_in, re_in, col_exp, g, 0.05,
                                        v_row=vr_in, v_col=vc_in, step=ref._step,
                                        seed=seed, **kw)
        sf_a, ss_a, vs_a = unpack_word(ref.packed_w)
        sf_b, ss_b, vs_b = unpack_word(packed_leg)
        coarse_gap = int(((ss_a + vs_a) - (ss_b + vs_b)).abs().max())
        sfast_gap = int((sf_a - sf_b).abs().max())
        max_coarse_gap = max(max_coarse_gap, coarse_gap)
        max_sfast_gap = max(max_sfast_gap, sfast_gap)
        if coarse_gap != 0 or sfast_gap > 1:
            ok = False
            break
    return ok, max_coarse_gap, max_sfast_gap


# ============================================================================
# Self-test (CPU; `python dither_accum_ref.py`)
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    N, K = 8, 16

    # --- test 1: encode/decode round-trip (legacy layout) ---
    s_fast = torch.randint(INT16_MIN, INT16_MAX + 1, (N, K), dtype=torch.int32)
    s_slow = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
    v_slow = torch.randint(INT8_MIN, INT8_MAX + 1, (N, K), dtype=torch.int32)
    packed = pack_word(s_fast, s_slow, v_slow)
    s_fast2, s_slow2, v_slow2 = unpack_word(packed)
    assert torch.equal(s_fast, s_fast2) and torch.equal(s_slow, s_slow2) and torch.equal(v_slow, v_slow2)
    # byte view reunifies to the same int16
    e_s, e_v = unpack_bytes(packed)
    assert torch.equal(reunify_fast(e_s, e_v), s_fast)
    print("test 1 (pack/unpack + byte-reunify round-trip): PASS")

    # --- test 2: DENORMAL detection + deploy preservation (req B) ---
    W = torch.randn(N, K) * 0.02
    W[0, 0] = 2.0            # row-0 max -> sets row_exp high
    W[0, 5] = 3e-6          # denormal: << row max, sub-LSB
    ref = DitherAccumRef(N, K, dither_enabled=True, seed=1)
    ref.load_weights(W)
    dn = is_denormal(ref.packed_w)
    assert bool(dn[0, 5]), "W[0,5] should be detected denormal"
    live05 = ref.live_weight()[0, 5].item()
    print(f"test 2 (denormal): count={int(dn.sum())}, recon live W[0,5]={live05:.3e} (true {W[0,5].item():.3e})")
    dep_mean = sum(abs(ref.deploy_weight(step_salt=t)[0, 5].item()) for t in range(64)) / 64.0
    print(f"          deploy E|W[0,5]| over 64 dithers = {dep_mean:.3e} (non-zero => survives)")
    assert dep_mean > 0, "denormal sub-LSB value must reach deploy via dither"
    ok_c, me, md = assert_carries_bounded(ref.err_s, ref.den_frac)
    assert ok_c, f"carries unbounded at init: |err_s|={me}, |den_frac|={md}"
    print("test 2: PASS")

    # --- test 3: deploy ratchets under dissipation (the fix) + carries BOUNDED ---
    ref3 = DitherAccumRef(N, K, dither_enabled=True, seed=2)
    W3 = torch.randn(N, K) * 0.05
    ref3.load_weights(W3)
    advanced = 0
    g = -torch.sign(W3) * 0.02 + 0.02
    dep0 = ref3.deploy_weight().clone()
    max_err_s = 0.0; max_den = 0.0; max_sfast = 0
    for t in range(300):
        info = ref3.step(g, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                         coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
        ok_c, me, md = assert_carries_bounded(ref3.err_s, ref3.den_frac)
        assert ok_c, f"carries blew up at step {t}: |err_s|={me}, |den_frac|={md}"
        max_err_s = max(max_err_s, me); max_den = max(max_den, md)
        max_sfast = max(max_sfast, info["s_fast_abs_max"])
        advanced += int(info["deploy_advanced"])
    moved = float((ref3.deploy_weight() - dep0).abs().sum())
    print(f"test 3 (deploy ratchet + BOUNDED carries): advanced {advanced}/300, moved {moved:.4e}")
    print(f"          max|err_s|={max_err_s:.3f} (<1), max|den_frac|={max_den:.3f} (<1), "
          f"max|s_fast|={max_sfast} (<=32767)")
    assert advanced > 50 and moved > 0
    assert max_err_s < 1.0 + 1e-4 and max_sfast <= INT16_MAX
    print("test 3: PASS")

    # --- test 3b: STRONG gradient -> carries STILL bounded (the draft failed here) ---
    ref3b = DitherAccumRef(N, K, dither_enabled=True, seed=3)
    ref3b.load_weights(W3)
    gstrong = torch.ones(N, K) * 0.5
    me_max = 0.0
    for t in range(300):
        info = ref3b.step(gstrong, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                          coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
        me_max = max(me_max, ref3b.err_s.abs().max().item())
    print(f"test 3b (STRONG grad): max|err_s|={me_max:.3f} (draft hit 2e5 here), "
          f"max|s_fast|={info['s_fast_abs_max']}")
    assert me_max < 1.0 + 1e-4, "err_s must stay sub-LSB (one-LSB sigma-delta bound) even under strong gradient"
    print("test 3b: PASS")

    # --- test 4: sub-LSB sigma-delta accumulation reaches deploy (error feedback) ---
    ref4 = DitherAccumRef(4, 4, dither_enabled=True, seed=7)
    W4 = torch.randn(4, 4) * 0.05
    ref4.load_weights(W4)
    g4 = -torch.sign(W4) * 0.005 + 0.005
    dep4_0 = ref4.deploy_weight().clone()
    for t in range(500):
        ref4.step(g4, lr=0.05, gf_consol=0.0, drift_cancel_C=0.02, chase_floor=0.1)
    moved4 = float((ref4.deploy_weight() - dep4_0).abs().sum())
    print(f"test 4 (sub-LSB sigma-delta): deploy moved {moved4:.4e}")
    assert moved4 > 0
    print("test 4: PASS")

    # --- test 5: disabled == legacy at the TRAINING level (per-step, from
    #     identical state): COARSE/deploy word BIT-EXACT, s_fast within the SR
    #     dither (<=1/coord) -- NOT just deploy==coarse on a divergent trajectory ---
    ok5, coarse_gap5, sfast_gap5 = assert_disabled_matches_legacy(steps=40)
    print(f"test 5 (disabled == legacy, per-step from identical state, 40 steps): "
          f"max coarse/deploy mismatch={coarse_gap5} (bit-exact), "
          f"max |s_fast diff|={sfast_gap5} (<=1, SR-boundary float reassociation)")
    assert ok5, "disabled coarse word must be bit-exact and s_fast within SR dither"
    # also the pure-coarse deploy identity
    ref5 = DitherAccumRef(N, K, dither_enabled=False, seed=1)
    ref5.load_weights(torch.randn(N, K) * 0.05)
    for t in range(50):
        ref5.step(g, lr=0.05, gf_consol=0.0, drift_cancel_C=0.02)
    s5f, s5, v5 = unpack_word(ref5.packed_w)
    coarse_dep = ((s5 + v5).to(torch.float32) * 128 * _scale_fwd(ref5.row_exp, ref5.col_exp))
    assert torch.allclose(ref5.deploy_weight(), coarse_dep, atol=0)
    print("test 5: PASS")

    # --- test 6: state recoverable from packed word ALONE (no sidecar) ---
    ref6 = DitherAccumRef(N, K, dither_enabled=True, seed=5)
    ref6.load_weights(torch.randn(N, K) * 0.5)
    g6 = torch.randn(N, K) * 0.1
    for t in range(200):
        ref6.step(g6, lr=0.05, alpha=0.1, gf_consol=0.3, drift_cancel_C=0.02,
                  coh_kappa=1.0, min_leak=0.05, evap_build_min=128.0, chase_floor=0.1)
    ok6, gap6 = assert_recoverable_without_sidecar(ref6.packed_w, ref6.err_s,
                                                   ref6.den_frac, ref6.row_exp, ref6.col_exp)
    print(f"test 6 (recoverable from packed word, drop sidecar): live mantissa gap={gap6:.3f} (<1.5)")
    assert ok6, "live weight must be recoverable from the packed word alone"
    print("test 6: PASS")

    # --- test 7: deterministic export is reproducible; SR export is DC-unbiased ---
    ref7 = DitherAccumRef(4, 4, dither_enabled=True, seed=9)
    Wd = torch.zeros(4, 4); Wd[0, 0] = 4.0; Wd[0, 1] = 0.3 * 2 ** (-3)  # a denormal-ish coord
    ref7.load_weights(Wd)
    d_a = ref7.deploy_weight(deterministic=True)
    d_b = ref7.deploy_weight(deterministic=True)
    assert torch.equal(d_a, d_b), "deterministic export must be reproducible"
    print("test 7 (deterministic export reproducible): PASS")

    print("\nALL TESTS PASS")
