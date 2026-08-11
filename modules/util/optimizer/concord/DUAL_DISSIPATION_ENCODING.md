# Dual-Dissipation Packed-B Encoding Spec (RECONCILED, AUTHORITATIVE)

Status: **authoritative** for the corrected dual-dissipation packed-B fine accumulator.
On any conflict this file supersedes `DUAL_DISSIPATION_DESIGN.md` (hi/lo bytes — WRONG),
`DITHER_ACCUM_DESIGN.md` (fp sidecars — WRONG), and the docstring/code currently in
`dual_dissipation_ref.py` *for the two reworked mechanisms only* (ADD-1 substrate, ADD-2
exponent-claim denormal). The co-equal core of `dual_dissipation_ref.py` is correct and is
kept verbatim — this spec changes only what is called out under "REMOVE" and "ADD".

Mental model: **weight = SUBSTRATE + OFFSET.** The packed 32-bit word holds ONLY the offset.
The substrate is a fixed, seed-derived random prior outside the word. The offset is an
epistemic hierarchy fast/tentative → slow/firm:

| field | scale | role | epistemic meaning |
|-------|-------|------|-------------------|
| `e_H` int8 | ×1 | fine accumulator, HIGH-dissipation arm | competing HYPOTHESIS (boil) |
| `e_L` int8 | ×1 | fine accumulator, LOW-dissipation arm  | competing HYPOTHESIS (retain) |
| `s_slow` int8 | ×128 | weakly-held theory / deploy position | weakly-held THEORY |
| `v_slow` int8 | ×128 | consolidated theory / deploy anchor | CONSOLIDATED THEORY |

`fine_value = e_L + e_H` (co-equal, ×1 each — NOT `e_H*256 + e_L`).
Deploy ships `substrate + (s_slow + v_slow)*128*scale` and DROPS the hypotheses
(the ~2–4% un-integrated offset residual).

CPU-only reference. Assume `CUDA_VISIBLE_DEVICES=""`. Nothing here runs on a GPU.

---

## 0. What carries over unchanged (the correct co-equal core)

KEEP from `dual_dissipation_ref.py` exactly as-is:

- Co-equal `e_L`/`e_H` (value = `e_L + e_H`, NOT hi/lo bytes). `unpack_dual`/`pack_dual`,
  `fine_value`, `unpack_legacy`/`pack_legacy`.
- Arithmetic bracket `λ_{L,H} = λ*(1 ∓ d)` (mean = legacy λ; no AM>GM drift).
- Per-arm coherence: SHARED `sig` from `d_sv`, PER-arm noise from `d_fs_X`, legacy formula
  (`compute_coherence`, mirrors `prototype_packed_b.py:823-846`).
- PRE-evap chase snapshot (consolidate coherent mass before dissipating any), gain 1,
  affine ratio-coh gate → sum of two gain-1 chases recenters on the legacy single chase.
- Blended-coh evaporation (e-weighted `coh_raw_blend`), `M >= M_MIN=4` guard.
- DISABLED == LEGACY bit-exact (single int16 `s_fast`, legacy chase/leak/evap/clamp).
- CONSERVATION ledger with EVAP already booked into `inflow_int` as a sink
  (`assert_conservation`, the gating test).

REMOVE entirely (the deploy-leak fractional denormal):
`e_H_fraction`, `_denormal_build`, `is_frac_free` as currently written, the `+128`
promotion into `s_slow`, the `FRAC_BITS`/`FRAC_DEN`/`FRAC_CAPACITY` *linear-fraction*
constants, and the init-time denormal fraction branch in `load_weights` (lines ~453-478).
The channel is currently gated off (`denorm` forced all-False); it is replaced by ADD-2.

---

## 1. ADD-1 — SUBSTRATE + OFFSET

### 1.1 New out-of-word state (ENABLED-only)

```
substrate       float32 [N, K]   WEIGHT UNITS   (seed-derived; regenerable)
substrate_seed  int                              (from-scratch: the RNG seed)
substrate_mode  str                              ("kaiming" | "xavier")
use_substrate   bool := self.enabled             (substrate is an ENABLED-only structure)
```

The substrate is **conceptually seed → tensor**. The CPU ref MAY materialize the tensor for
convenience, but nothing ever reads it from the packed word, and it is NEVER consolidated,
dissipated, or written into any mantissa field. For a finetune the substrate is the
pretrained `base` (NOT regenerable from a seed — it is re-supplied at load; see §1.6).

When disabled, `substrate = None` and is treated as `0` — the word holds the full weight
exactly as legacy.

### 1.2 Substrate generator (from-scratch)

```python
def make_substrate(N, K, seed, mode="xavier"):
    g = torch.Generator().manual_seed(int(seed))     # CPU generator, deterministic
    if mode == "xavier":
        std = (2.0 / (N + K)) ** 0.5                  # matches prototype _init_weight:2640
    else:  # "kaiming"
        std = (2.0 / N) ** 0.5
    return torch.empty(N, K).normal_(0.0, std, generator=g)
```

The substrate breaks symmetry and sets the per-row/col scale only ("neither believed nor
disbelieved").

### 1.3 `load_weights(W, base=None)` — branch on `self.enabled`

ENABLED path (the rework):
```
substrate := (base if base is not None else make_substrate(N, K, substrate_seed, mode))
            # finetune: base = the pretrained W; from-scratch: a fresh seeded draw.
            # (If the caller passes W as both the target and base for a finetune, that is
            #  the pretrained base — substrate := base.)
row_exp/col_exp := chosen from |substrate|.amax  (exactly the legacy exponent rule, so one
            mantissa unit is a sensible sub-step of the substrate magnitude)
packed_w := pack_dual(0, 0, 0, 0)                # e_H=e_L=s_slow=v_slow = 0
v_row, v_col := 0 ;  _step := 0
```
At this point `decode_to_live == decode_to_deploy == substrate` exactly. The accumulator
starts at ZERO; the word carries NO copy of the weight. **No even-split-into-the-accumulator,
no init-time denormal encoding** (a sub-LSB coordinate of the *weight* now lives entirely in
the substrate at offset 0 and deploys correctly with zero denormal machinery at init).

DISABLED path (kept bit-exact): the legacy even-split body — full weight quantized into
`s_slow`/`v_slow` + int16 `s_fast` residual, `substrate = None`. **The branch keys on the
SAME `self.enabled` that `step()` keys on** (so an M<M_MIN collapse takes the legacy branch).

### 1.4 Decode (ENABLED)

```
scale              = 2^(row_exp[:,None] + col_exp[None,:] - MANTISSA_BIAS)
offset_mant_live   = s_slow*128 + (e_L + e_H) + v_slow*128
offset_mant_deploy = (s_slow + v_slow)*128            [+ ADD-2 denormal_extra on gated coords]
live_weight   = substrate + offset_mant_live   * scale
deploy_weight = substrate + offset_mant_deploy * scale
```

Composition order is fixed: **`substrate + (coarse + denormal_extra) * scale`** — the
denormal extra is added to the offset mantissa BEFORE the scale multiply and BEFORE the
substrate add, so it inherits the block-float scale and does NOT scale the substrate (open
risk ADD-2-COMPOSITION; this is the canonical order).

Decode (DISABLED): `substrate = None/0`; `live = (s_slow*128 + s_fast + v_slow*128)*scale`,
`deploy = (s_slow+v_slow)*128*scale` — byte-identical legacy.

### 1.5 Dissipation now decays the OFFSET toward 0 ⇒ weight toward substrate

No formula change in `step()`. Evaporation pulls `e_L`/`e_H` → 0; the leak relaxes `s_slow`
→ `v_slow`. Because the OFFSET is the only thing in the word, offset → 0 means weight →
substrate (the random prior), NOT weight → 0. **The build-from-zero problem is gone**: no
gap-fix, no ignition, no even-split. "Offset 0" now means "weight = substrate".

### 1.6 Finetune variant & determinism

`load_weights(W, base=W)` sets `substrate := pretrained base`, zeroes the accumulator.
Training accumulates the DELTA in the offset; dissipation relaxes the delta toward 0 i.e.
toward the pretrained base. This replaces the gap-zero even-split finetune init AND the
`resplit_anchor` washout class entirely: there is no anchor stored in `v_slow` to wash out —
the base lives in the immutable substrate.

Decode is a pure function of `(packed_word, row_exp, col_exp, substrate)`, and substrate is a
pure function of `(seed, mode)` [from-scratch] or `(base)` [finetune]. Checkpoint/relaunch
must re-supply the from-scratch seed+mode (regenerate via `torch.Generator(seed)`) or the
finetune base. **Test 8's claim "live recoverable from the packed word ALONE" becomes
"recoverable from packed word + substrate"** — the test must decode against the known
substrate (or restrict the seed-only claim to disabled mode, where there is no substrate and
the word does hold the whole weight).

### 1.7 Re-exponent coupling (the one consistency point)

`renormalize_on_saturation` halves the OFFSET mantissa (`e_H, e_L, s_slow, v_slow`) and bumps
`row_exp`. The substrate is in WEIGHT units → exponent-invariant → **NOT halved, NOT
touched**: `substrate + (offset/2)*(2*scale_old) == substrate + offset*scale_old`. This holds
automatically because the substrate is never expressed in mantissa units. Document it: the
substrate tensor is weight-units, the offset is mantissa-units; re-exponent rescales only the
offset.

---

## 2. ADD-2 — DENORMAL = per-element EXPONENT-CLAIM on `e_H`

The denormal mechanism is a per-element block-float SUB-SCALE extension consulted ONLY when
the element's coarse word is zero. It reallocates `e_H`'s bits from UPWARD-linear momentum
range (which the small residual does not need) to DOWNWARD-log sub-scale reach (which it does).

### 2.1 Gating predicate (coarse-only — unchanged)

```python
def is_denormal(packed):           # reads COARSE word ONLY — never fine bits
    _, _, s_slow, v_slow = unpack_dual(packed)
    return (s_slow == 0) & (v_slow == 0)
```
This is THE invariant that avoids WRONG-#2 BREAK A (self-nullification): the predicate never
reads `e_H`, so reading the offset *value* from `e_H` cannot move the *predicate*. Normal
coords (coarse ≠ 0) take pure coarse, bit-identical to legacy.

### 2.2 Bit split of `e_H` for a denormal element (EXPONENT MODE, the default)

```
u8  = e_H as raw two's-complement int8
SGN = sign(e_H)            # the offset sign lives in the int8 sign — no reserved sign bit
a   = |e_H|                # 1..127 ; a == 0 ⇒ true zero ⇒ offset 0

# split a's 7 magnitude bits: EXP_BITS = 3 high, MANT_BITS = 4 low
oct  = (a >> MANT_BITS) & 0x7     # 0..7   downward octave count k
mlow = a & 0xF                    # 0..15  4-bit mantissa, leading-1 IMPLIED
```

### 2.3 Decode (offset value, weight units)

One mantissa unit at the shared scale = `1*scale`; deploy LSB = `128*scale`. The denormal
lives BELOW one mantissa unit, with an IEEE-style implied leading 1 so 4 stored bits buy a
full extra bit:

```
signif    = 1 + mlow/16                         # in [1.0, 1.9375)
units     = SGN * signif * 2^-(oct+1)           # oct+1 ⇒ at least 1 octave below the unit
offset_wt = units * scale = SGN*(1+mlow/16) * 2^-(oct+1) * scale
```

Reach: `oct ∈ 0..7` → `2^-1 .. 2^-8` of one mantissa unit. Smallest representable nonzero
`|offset| = 2^-8 * scale = deploy_LSB / 2^15`, far below the deploy LSB. `a == 0` → exactly 0.

This is a self-contained per-element block-float: `row_exp+col_exp` = block exponent
(rank-1), `e_H`'s 3 high bits = per-element downward octave, 4 low bits = per-element
significand, int8 sign = offset sign. `is_denormal == True` selects this decode; `== False`
ignores `e_H` for deploy.

### 2.4 Mode selection — the `|e_H|` guard ("something is very wrong", rare)

The exponent claim is valid only while `e_H` is drained near zero (it is the high-dissipation
arm — drained by BOTH evap and consolidation, so its high bits are statistically free).

```python
EH_VELO_CAP = 64            # half-scale; == the |resid| bound a real fine residual produces
def is_exp_mode(e_H):
    return e_H.abs() < EH_VELO_CAP
```

- `is_denormal & is_exp_mode` → EXPONENT MODE (the default): deploy adds `offset_wt` (§2.3).
- `is_denormal & ~is_exp_mode` → MAGNITUDE MODE: a large `e_H` is a big un-consolidated
  velocity bursting toward normal. **DROP the exponent claim, render that element as PURE
  COARSE** (offset contribution 0; coarse word is 0 so it ships 0 that step). It will
  graduate via the normal chase within a few steps; losing the extension costs nothing.
- Normal coords (`~is_denormal`) → pure coarse, bit-identical legacy.

**Deploy gate: add the denormal extra ONLY where `(is_denormal & is_exp_mode)`.**

`EH_VELO_CAP = 64` coincides with the `oct`-field aliasing boundary: `oct` is bits 4..6, so
`|e_H| >= 64` sets bit 6 (`oct >= 4`); a true velocity sets bit 6/bit 5 → the guard fires
before the field aliases. See §3.3 for the disjointness proof.

### 2.5 Sigma-delta accumulate + conserving graduation

A denormal element's gradient inflow is sub-LSB (< 1 mantissa unit/step). It is banked in the
LOG field as a fixed-point sigma-delta, and when it grows past one mantissa unit it
**graduates CONSERVINGLY into the FINE register `e_L`, NEVER `+128` into `s_slow`** (the
removed leak). The linear→log accumulate-and-carry must be specified as fixed-point (§3.4 is
the conservation-critical part — this is exactly where the old `_denormal_build`
double-counted/leaked):

```
# inflow_into_denormal is in mantissa units (linear). Accumulate it against the current
# log-decoded offset value (also mantissa units) in a LINEAR fixed-point register, NOT in
# the log field directly:
val_old = is_exp_mode ? decode_units(e_H)        # §2.3 units (signed, |.| < 1)
                      : 0                          # magnitude mode contributes 0 to the log path
val_new = val_old + inflow_into_denormal          # linear add, mantissa units

whole   = trunc(val_new)                          # whole mantissa units to promote (signed)
resid   = val_new - whole                         # sub-unit residual, |resid| < 1

# (a) PROMOTE whole units into e_L via an SR tick (retain arm; inside fine_mantissa):
e_L    += whole                                   # SR-rounded; whole is the same units e_L holds
# (b) re-encode resid back into e_H's exponent field (§2.6 inverse):
e_H     = encode_denormal(SGN(resid), resid)      # writes (sign, oct, mlow) into e_H
```

Promotion is a **fine→fine move**: `whole` is the integer mantissa the element now
legitimately holds, and it is the gradient inflow that was already SR-ticked this step
(counted in `inflow_int`). `e_L` is inside `fine_mantissa`, so `Δ(fine) = +whole` exactly
accounts for the new mass — never a fresh `+128`. From there `e_L != 0` chases into `s_slow`
via the standard mass-preserving chase (which conserves). This is the SAME residual-migration
discipline the rebalance uses at `prototype_packed_b.py:2028/:2037` (coarse-quant residual
migrates into the fine register, not credited as deploy mass); graduation rides on that path.

### 2.6 Encode (inverse, for LOAD of a sub-scale coord and for graduation re-encode)

A coord whose whole value is sub-scale (`coarse == 0`) is encoded by quantizing its sub-unit
value `v` (`|v| < 1` mantissa unit) into `(SGN, oct, mlow)`:

```
SGN  = sign(v)
av   = |v|                                   # in (0, 1)
oct  = clamp(floor(-log2(av)) - 1, 0, 7)     # octave index s.t. av ∈ [2^-(oct+1), 2^-oct)
mlow = round((av * 2^(oct+1) - 1) * 16)      # invert signif = 1 + mlow/16
mlow = clamp(mlow, 0, 15)
a    = (oct << 4) | mlow                      # 7-bit magnitude (a >= 1 since signif >= 1)
e_H  = SGN * a                                # int8 with sign; e_L = 0; s_slow = v_slow = 0
```
For `av < 2^-8` the value floors to the smallest representable offset (`oct=7, mlow=0`); for
`av == 0` set `e_H = 0` (true zero). Under ADD-1 this encode is NOT used at init (a sub-scale
*weight* lives in the substrate); it is used only when a sub-scale *offset* must be
materialized (graduation re-encode, or an explicit sub-scale offset write).

### 2.7 Deploy render (DC-unbiased)

The denormal extra is stochastically rounded into the bf16 deploy weight the same way the old
fraction was, so `E[render] == offset value` (no DC leak). Deterministic mode (shipping
checkpoints) rounds to nearest. The extra is in mantissa units; it is added to the offset
mantissa, then scaled (§1.4 order).

---

## 3. ADVERSARIAL CONSERVATION AUDIT

The gating invariant, every step:
`Δ(deploy_mantissa) + Δ(fine_mantissa) == inflow_int` (evap booked as a sink), where
`deploy_mantissa = (s_slow+v_slow)*128` and `fine_mantissa = e_L+e_H`, both pure functions of
the packed OFFSET word. The question: does the exponent-claim RENDER, the conserving
PROMOTION, or the SUBSTRATE inject any UNbooked deploy mass (the failure that produced the
prior `+128` leak)?

### 3.1 Substrate — CLEAN (no unbooked mass)

- The ledger terms are pure functions of the OFFSET word. The substrate is never written into
  `s_slow/v_slow/e_L/e_H`, so it appears in NEITHER ledger term. It is a read-side addend in
  decode only.
- The invariant is a DELTA invariant. The substrate is CONSTANT every step (`Δsubstrate = 0`
  by definition — `step()` never updates it), so it contributes 0 to every delta and cannot
  move the residual.
- Contrast with the removed `+128` leak, which wrote `+128` into `s_slow` with no fine debit
  → unbooked deploy mantissa → nonzero residual (the test that caught it). The substrate is
  never credited into any mantissa field at all — there is nothing to book.
- Re-exponent preserves this: it halves the offset mantissa and bumps `row_exp`
  (value-preserving on the offset) and leaves the weight-unit substrate untouched.
- **Verdict: substrate injects ZERO unbooked deploy mass. Ledger unchanged.**

### 3.2 Exponent-claim RENDER — CLEAN (no unbooked mass)

- The exponent claim is a pure READ re-interpretation of `e_H`'s existing bits for the DEPLOY
  RENDER. It does NOT credit `s_slow`/`v_slow` at all. `e_H` is already counted in
  `fine_mantissa`, so the bits it reads are already on the fine side of the ledger.
- A step's transfer accounting (chase + leak) touches NONE of the denormal render: the render
  is a decode-time read, not a state write. `Δ(deploy_mantissa)` and `Δ(fine_mantissa)` are
  computed from the word, and the render adds nothing to either.
- The render adds `offset_wt` to the *deploy WEIGHT* (a float, weight units), NOT to
  `deploy_mantissa` (the integer `(s_slow+v_slow)*128`). The ledger governs only the integer
  `deploy_mantissa`; the denormal extra is outside it, exactly like the substrate.
- **Verdict: the render injects ZERO unbooked deploy mantissa. It changes the deploy WEIGHT
  (a read), never the deploy MANTISSA (the ledgered integer).**

> NOTE (important framing, and a hole — see §3.6): because the denormal extra is NOT in
> `deploy_mantissa`, the deploy weight now has a component (`offset_wt`) that the integer
> ledger does not track. This is *sound for conservation of the ledger* (the ledger is about
> the integer transfer between fine and deploy registers, and the extra is a read of `e_H`
> which IS ledgered on the fine side). But it means "deploy weight" is no longer exactly
> `deploy_mantissa * scale + substrate`; it is that PLUS a bounded sub-scale read of `e_H`.
> The conservation test asserts the INTEGER ledger, which stays green; a SEPARATE test must
> assert that the denormal extra is bounded by `< 1 mantissa unit * scale` and is a pure
> function of `e_H` (so nothing escaped into a sidecar).

### 3.3 The `|e_H|` guard — disjointness of exp-mode vs magnitude-mode

Claim to verify: no legitimate exponent-mode encoding ever produces `|e_H| >= 64` (which the
guard would wrongly drop), AND the guard fires before the `oct` field aliases.

- Exponent-mode encoding: `a = (oct<<4) | mlow`, `oct ∈ 0..7`, `mlow ∈ 0..15`. Max `a` =
  `(7<<4)|15 = 127`. So an exponent-mode `|e_H|` CAN exceed 64 (e.g. `oct=7` → `a >= 112`).
  **This is a real overlap, not benign as stated in the open risk.** A genuine smallest-reach
  denormal (`oct=7`) has `|e_H| >= 112 >= 64` → the guard fires and drops it to magnitude
  mode → renders pure coarse (0). See §3.6 HOLE-1 for the fix.
- The guard DOES fire before catastrophic field aliasing in the sense that `|e_H| >= 64` is
  detectable (bit 6 set), but `EH_VELO_CAP = 64` does NOT cleanly separate "real velocity"
  from "deep-octave denormal" — both can set bit 6. The two regimes are NOT disjoint at
  cap 64.
- **Verdict: the guard threshold as specified has a real overlap with deep-octave denormals.
  Fixed in §3.6 HOLE-1 by restricting the usable octave range so exp-mode `|e_H| < 64`
  always, making the two regimes provably disjoint.**

### 3.4 Graduation / linear→log carry — the conservation-critical path

This is exactly where the old `_denormal_build` leaked. Audit of §2.5:

- Inflow is LINEAR (mantissa units). The accumulate is done in a LINEAR fixed-point register
  (`val_new = val_old + inflow`, both mantissa units), NOT in the log field. The log field is
  only the STORAGE of the sub-unit residual `resid` (`|resid| < 1`). This avoids the old bug
  of accumulating a linear inflow directly into a log field (which double-counted).
- Promotion `e_L += whole` is SR-ticked and `whole` is the same integer mantissa unit `e_L`
  holds. The promoted `whole` is part of `inflow_int` for that step (it is the gradient that
  arrived), so `Δ(fine) = +whole` is booked as inflow, NOT as an internal transfer — the
  ledger reads it on the fine side with no deploy credit.
- **No `+128` ever touches `s_slow` from this path.** `d_sv` opens later when `e_L` chases
  into `s_slow` via the standard conserving chase. (Open risk: verify a freshly-graduated
  denormal actually OPENS the chase — at `d_sv = 0`, `coh = 0`, so the chase fires on its
  floor term `chase_floor` only; `e_L != 0 × chase_floor > 0` does tick. It does not stall as
  a permanent `e_L` resident. This must be a test, §4.)
- **DC bias of the linear→log round-trip:** the decode→encode of `resid` (§2.6/§2.3) is a
  quantization with `<= 2^-8` granularity at the deepest octave; the residual `resid` is
  carried in the LINEAR `val` register across steps (sigma-delta), so the quantization error
  does NOT accumulate as DC — it is dithered out provided the carry is kept in the linear
  register and only the *display* is re-encoded each step. **HOLE-2 (§3.6): if `val_old` is
  re-derived from the log field each step (lossy) instead of carried in a linear register,
  the encode quantization becomes a DC sink. The spec REQUIRES the carry live in a linear
  fixed-point register; but that register must itself live inside the 32-bit word (no fp
  sidecars). See §3.6 HOLE-2 for the resolution.**

### 3.5 Disabled == legacy interaction — CLEAN

- `use_substrate := self.enabled`. Disabled (feature-OFF or M<M_MIN collapse) → `substrate =
  None`, the accumulator holds the FULL weight (legacy even-split), single int16 `s_fast`,
  deploy = coarse. The exponent claim is an ENABLED-only read of `e_H`; in disabled mode
  `e_H`/`e_L` reunify as one int16 and the denormal decode is never invoked.
- `load_weights` branches on `self.enabled` (the SAME flag `step()` keys on). A slip that ran
  the substrate branch under an M<M_MIN collapse would break bit-exactness — the spec
  mandates the single-flag branch and the existing `test_dualdis_disabled_legacy` /
  `assert_disabled_matches_legacy` guards it.
- **Verdict: disabled stays bit-exact legacy. The substrate and the exponent claim are both
  enabled-only structures, never consulted in the disabled path.**

### 3.6 CONSERVATION HOLES FOUND (with fixes)

**HOLE-1 — guard/octave overlap is NOT benign (contradicts open-risk #2/#5).**
As specified, exponent-mode `|e_H|` ranges up to 127 (`oct=7`), which collides with the
`EH_VELO_CAP = 64` magnitude-mode guard: a legitimately deep denormal (`oct >= 4`, i.e.
`|e_H| >= 64`) is wrongly dropped to magnitude mode and rendered as 0. It also breaks sign
continuity at the boundary (open-risk #5): a denormal whose `|e_H|` crosses 64 jumps from its
sub-scale value to 0.
*Fix:* restrict the usable octave to `EXP_BITS_EFF` such that the max exp-mode `|e_H| < 64`.
With `EXP_BITS = 3` but the high bit of `oct` reserved as the mode flag, use `oct ∈ 0..3`
(2 effective octave bits) and keep `MANT_BITS = 4`: max `a = (3<<4)|15 = 63 < 64`. Then
`is_exp_mode(e_H) = |e_H| < 64` is PROVABLY disjoint from any exp-mode encoding (all exp-mode
encodings have `|e_H| <= 63`), and `|e_H| >= 64` is unambiguously real velocity. Reach is now
`2^-1 .. 2^-4` of one mantissa unit (`oct ∈ 0..3`), i.e. smallest `|offset| = 2^-4 * scale =
deploy_LSB / 2^11`. If deeper reach is needed, spend a MANT bit instead of an octave bit, or
accept that anything below `2^-4*scale` graduates via promotion rather than living in the log
field. **This is the binding fix: EXPONENT range and the guard MUST be disjoint, and the only
way to guarantee it with a shared `|e_H|` is to cap the encoded magnitude below the guard.**
(Document the reach reduction as the cost of provable disjointness; an ablation, open-risk #1,
confirms `2^-4*scale` is below the gradient-noise scale for the target SDXL layers.)

**HOLE-2 — the linear sigma-delta carry needs a home inside the word (no fp sidecars).**
§2.5 requires the linear carry `val` to persist across steps to avoid a DC sink, but the only
per-element state is `e_H` (the log field) and `e_L` (integer). Re-deriving `val_old` from the
log field each step is lossy (the encode quantizes to `<= 2^-4` units after HOLE-1's fix),
which is a DC sink if the dropped fraction is discarded.
*Fix (conserving, sidecar-free):* the carry IS the log field, and the dropped sub-`2^-4`
fraction is handled by STOCHASTIC ROUNDING of the encode (§2.6 `mlow = SR_round(...)` instead
of `round(...)`), seeded by the same xorshift hash as every other SR tick. SR makes the
re-encode DC-unbiased in expectation: `E[encode(decode(e_H) + inflow)] == decode(e_H) +
inflow` to within the SR grain, so no systematic mass is created or destroyed across steps.
The whole-unit part is exact (promoted to `e_L`); only the sub-unit residual is SR-encoded,
and SR of a sub-unit value has zero DC bias. **This keeps everything in the 32-bit word and
makes the linear→log carry DC-neutral.** The conservation ledger is unaffected because the
sub-unit residual is NOT in either ledger term (it is a sub-mantissa read of `e_H`, like the
render); only the promoted whole units cross into `e_L` and those are booked as inflow.

**HOLE-3 — rebalance right-shift corrupts the log field (open-risk #7 is real).**
The per-row/col rebalance (`prototype_packed_b.py:2018`) SR-right-shifts `s_fast` as an
INTEGER when the block exponent ticks up. In the dual layout the top 16 bits are
`(e_H, e_L)`; right-shifting `e_H` as an integer corrupts the `(oct, mlow)` log encoding of a
denormal element (the octave field is positional, not linear). A naive port that shifts the
reunified 16 bits would scramble every denormal's exponent.
*Fix:* the rebalance MUST skip the integer right-shift of `e_H` on denormal (`coarse == 0`)
elements and instead RE-ENCODE: when `row_exp += 1`, a denormal's effective scale doubles, so
its stored offset value should be unchanged in WEIGHT units → its `(oct, mlow)` must shift one
octave deeper (`oct += 1`, clamped) to compensate, leaving `mlow` and `SGN` intact. Concretely
for denormal elements: `e_H := encode(decode(e_H) ...)` is *automatically* correct if the
decode/encode are done in WEIGHT units and the scale used is the NEW scale — but the cheaper
exact operation is `oct := min(oct + net_right_shift, 3)` (HOLE-1 cap) on the log field, NOT
an integer shift. Normal elements keep the legacy integer shift on `e_L`+`e_H`-as-int16. This
requires the rebalance to be denormal-aware (branch on `coarse == 0`), which the legacy
rebalance is not — it is a REQUIRED change to the rebalance kernel for the GPU port, and a
CPU-ref test must cover a re-exponent step with live denormals.

**HOLE-4 — `e_L` graduation residency (open-risk #4) is a stall risk, not a leak.**
Not a conservation hole (no unbooked mass), but a correctness risk: a promoted denormal sits
in `e_L` until the chase moves it. At `d_sv = 0` the chase fires only on `chase_floor`, so it
DOES advance (`tick_slow = SR(alpha * chase_floor * e_L / 128)`), but slowly. This is the
intended behavior (the old design forced `+128` into `s_slow` specifically to open `d_sv`,
which we now forbid). *Resolution:* accept the slow floor-driven graduation; verify by test
(§4) that a graduated denormal's deploy advances within a bounded horizon and `e_L` does not
grow unboundedly (it is int8-clamped and the chase drains it).

---

## 4. REQUIRED TESTS (CPU, no GPU) — what each must assert

Existing tests to KEEP green (they already encode the core invariants):
`test_dualdis_conservation.py` (the integer ledger — the gating test),
`test_dualdis_disabled_legacy.py` (bit-exact legacy),
`test_dualdis_recenter.py`, `test_dualdis_from_scratch.py`.

`test_dualdis_denormal.py` must be REWORKED for the exponent-claim (the old linear-fraction
assertions are obsolete). New/changed assertions:

1. `is_denormal` coarse-only (unchanged) — toggling `e_H`/`e_L` never moves the predicate.
2. EXPONENT-MODE decode: a denormal coord with `(SGN, oct, mlow)` deploys
   `SGN*(1+mlow/16)*2^-(oct+1)*scale` in expectation (DC-unbiased over salts); deterministic
   render reproducible.
3. Disjointness (HOLE-1): every exp-mode encoding has `|e_H| < 64`; every `|e_H| >= 64`
   denormal renders pure coarse (0). No exp-mode encoding is ever dropped by the guard.
4. Sign continuity across the mode boundary (HOLE-1): a denormal whose `|e_H|` sweeps up
   through 63→64 does not flip sign or jump value discontinuously (it goes value→0 at the
   boundary, monotonically, never sign-flip).
5. Graduation conservation (HOLE-2, §2.5): drive sub-LSB inflow into a denormal; assert the
   integer ledger stays green every step (the existing conservation test, extended with live
   denormals), AND that promoted whole units land in `e_L` (NOT `s_slow`), AND `E[render]`
   over the run tracks the true accumulated value (no DC sink — SR re-encode).
6. Re-exponent with live denormals (HOLE-3): a re-exponent step preserves each denormal's
   deploy value in WEIGHT units (the `oct` shift compensates the scale doubling); the integer
   ledger is skipped on that step (as today) but the denormal weight is value-preserving.
7. Graduation opens the chase (HOLE-4): a freshly-graduated denormal advances deploy within a
   bounded horizon under a coherent drift; `e_L` stays int8-bounded and drains.
8. Substrate (ADD-1): after `load_weights`, `live == deploy == substrate` exactly; decode is
   a pure function of `(packed, row_exp, col_exp, substrate)`; re-exponent leaves the
   substrate untouched and the live weight value-preserving to within SR noise; disabled load
   takes the legacy even-split branch (bit-exact) and `substrate is None`.

---

## 5. CONSTANTS (reconciled)

```
MANTISSA_BIAS = 15
INT8_MIN, INT8_MAX   = -128, 127
INT16_MIN, INT16_MAX = -32768, 32767
S_SLOW_FACTOR = V_SLOW_FACTOR = CARRY = 128     # one coarse LSB = 128 mantissa units
MAX_M = 24000                                    # re-exponent trigger
BRACKET_D = 0.5                                  # arithmetic half-spread d
M_MIN = 4                                        # grad-accum floor; below ⇒ collapse to legacy

# ADD-2 (exponent-claim denormal), reconciled with HOLE-1:
EXP_BITS    = 2     # EFFECTIVE octave bits after reserving headroom for the guard (oct ∈ 0..3)
MANT_BITS   = 4     # significand bits (leading-1 implied), mlow ∈ 0..15
EH_VELO_CAP = 64    # |e_H| >= 64 ⇒ magnitude mode (pure coarse). PROVABLY disjoint from
                    #   exp-mode (max exp-mode |e_H| = (3<<4)|15 = 63 < 64).
# Deepest reach = 2^-(3+1) = 2^-4 of one mantissa unit = deploy_LSB / 2^11.
# REMOVED: FRAC_BITS / FRAC_DEN / FRAC_CAPACITY (the linear-fraction field).
```

(If the deepest octave is widened back to `oct ∈ 0..7` for more reach, `EH_VELO_CAP` can NOT
be 64 — the guard and the encoding would overlap, HOLE-1. Reach vs guard-disjointness is the
trade; this spec chooses provable disjointness at `oct ∈ 0..3`.)

---

## 6. SUMMARY OF THE REWORK (delta vs current `dual_dissipation_ref.py`)

- ADD `substrate` (float32, weight units, ENABLED-only, seed/base-derived, never ledgered);
  decode = `substrate + offset*scale`; dissipation decays offset → 0 → weight → substrate.
- ADD-side: `load_weights` branches on `self.enabled`; ENABLED zeroes the accumulator and
  sets the substrate; DISABLED keeps the legacy even-split (bit-exact).
- REPLACE the linear-fraction denormal (`e_H_fraction`/`_denormal_build`/`+128`) with the
  per-element EXPONENT-CLAIM on `e_H` (`(SGN, oct, mlow)`, `oct ∈ 0..3`, implied leading 1),
  gated by `is_denormal & is_exp_mode(|e_H| < 64)`; graduation promotes whole units into
  `e_L` (NOT `s_slow`) with an SR-encoded sub-unit carry.
- KEEP the co-equal core, the ledger, and disabled==legacy bit-exactness verbatim.

CONSERVATION VERDICT: substrate and the exponent-claim RENDER inject ZERO unbooked deploy
mantissa (both are read-side addends outside the integer ledger; the render reads `e_H` which
is already on the fine side). Graduation is fine→fine (`e_L`), booked as inflow, never `+128`.
Three holes were found and fixed: (1) the `|e_H|` guard/octave OVERLAP — fixed by capping
exp-mode magnitude below the guard (`oct ∈ 0..3`, max `|e_H| = 63 < 64`), making the modes
provably disjoint and continuous; (2) the linear→log carry DC sink — fixed by carrying the
residual in the log field with an SR-encoded re-write (DC-neutral, sidecar-free); (3) the
rebalance right-shift corrupting the log field — fixed by making the rebalance denormal-aware
(shift the `oct` field, not the integer bits, on `coarse == 0` elements). The integer
conservation ledger remains green; HOLE-1 and HOLE-3 are binding REQUIREMENTS for the GPU port.
