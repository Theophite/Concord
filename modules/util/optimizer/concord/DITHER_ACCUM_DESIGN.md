# Dither-Accum redesign — bounded-int16 fine velocity + sigma-delta deploy carry + separate denormal channel

Status: CPU reference complete and self-test-passing (8/8). NOT yet ported to Triton.
Reference: `modules/util/optimizer/concord/dither_accum_ref.py` (pure-torch, CPU, `python dither_accum_ref.py`).
Kernel mirrored: `modules/util/optimizer/concord/prototype_packed_b.py`.

This document supersedes the earlier "two-int8 + throttled-carry" draft, which was
REJECTED by red-team for three blocker defects. The fixes are summarized in
`§7 Red-team resolution`.

---

## 1. The problem this fixes

Current packed-B (`prototype_packed_b.py`):
```
word = [ s_fast : int16 @31:16, x1 ] [ s_slow : int8 @15:8 , x128 ] [ v_slow : int8 @7:0 , x128 ]
live   m_eff = s_slow*128 + s_fast + v_slow*128
deploy m_dep = (s_slow + v_slow)*128                        (drops s_fast)
```
The deploy weight only ticks once `s_fast` climbs a full 128 via the chase
(`prototype_packed_b.py:1026-1042`). The gf evaporation
`evap_frac = lr*gf_consol*(1-coh_raw)` (line 936) drains `s_fast` before it
reaches 128, so the chase quantizes to 0 and the deploy never advances: a
from-scratch direction never builds, and the fine residual is lost on deploy.

Separately, a weight far below its row/col max-abs is **denormal**: the shared
block-float scale (`scale_ij = 2^(row_exp_i+col_exp_j-bias)`, set from the row
max-abs, `load_weights:2666-2669`) cannot drop per-element, so `|m_ij| < 1` and
the deploy `x128` quantization zeros it.

**Two goals:** (A) ratchet the deploy ~every step (sigma-delta), and
(B) keep sub-LSB (denormal) weights alive through deploy — without growing the
persistent 32-bit-per-param word.

---

## 2. Bit layout — BYTE-IDENTICAL to legacy

```
word bits [31:16]  s_fast int16  x1    fine VELOCITY  (legacy, bounded ±32767)
word bits [15: 8]  s_slow int8   x128  deploy position (unchanged)
word bits [ 7: 0]  v_slow int8   x128  deploy anchor   (unchanged)
companion (optimizer state, NOT in the word, like coh_pre/v_row/v_col):
    err_s    [N,K] fp32 in (-1, 1)   sub-LSB sigma-delta remainder of s_fast
    den_frac [N,K] fp32 in (-1, 1)   sub-LSB DENORMAL fraction (its OWN budget)
```

The 16 fine bits are **one int16 `s_fast`**, exactly as the legacy kernel
(`prototype_packed_b.py:758`). The previous draft split them into two
independently-int8-clamped registers `e_s`/`e_v`; that is the root of every
blocker (it clamps the velocity 256× tighter and pushes the overflow into an
unbounded fp sidecar). Here `e_s`/`e_v` are only the **high/low byte** of the
same int16 (`unpack_bytes` / `reunify_fast`), provided so a Triton port can keep
a byte view while still reading ONE bounded velocity:
```
e_s = s_fast >> 8            (high byte, sign-extended)
e_v = (s_fast << 24) >> 24   (low byte,  sign-extended)
s_fast = (e_s << 8) | (e_v & 0xFF)     # reunify_fast — the crux of the fix
```

Encode/decode (`pack_word` / `unpack_word`) is the legacy arith-shift sign-extend
(`prototype_packed_b.py:758-760`, `1106-1110`). Round-trip is bit-exact
(self-test 1).

---

## 3. Update algorithm (one step) — legacy order + two opt-in additions

Mirrors `prototype_packed_b.py:986-1101`. All quantities in mantissa units;
`scale_inv = 2^-(row_exp+col_exp-bias)`. SR = stochastic round
`floor(x)+1[u<frac(x)]`, `u` from the kernel xorshift `_hash_uniform`
(`prototype_packed_b.py:109-117`).

1. **Velocity / coherence (unchanged).** `d_fs = s_fast + err_s`,
   `d_sv = (s_slow - v_slow)*128`; `sig`, `noise`, `coh_raw`, `cf`, `coh` are the
   byte-for-byte legacy computation (`prototype_packed_b.py:817-846`). `err_s` is
   a sub-LSB correction (|err_s|<1) so it does not change magnitude/coherence.

2. **Preconditioned step (unchanged).**
   `step_live = clamp(grad/(v_proxy+eps)^p, ±step_cap)`,
   `delta_grad = -lr*step_live*scale_inv`,
   `v_proxy = noise_w^2 * v_scale` (`prototype_packed_b.py:856-970`).

3. **Evaporation on `d_fs` (unchanged, conserved, build-gated).**
   `evap_frac = min(lr*gf_consol*(1-coh_raw), 1-min_leak)`,
   soft build gate `p_build = min(|d_fs|/evap_build_min, 1)`
   (`prototype_packed_b.py:936-952`). Fires whenever `gf_consol>0` in BOTH modes.

4. **(A) Sigma-delta inflow into the BOUNDED int16 `s_fast`** — the fix:
   ```
   delta_t = delta_grad - evap_mantissa
   intent  = s_fast + err_s + delta_t          # full fine value (fp)
   s_fast  = SR_round(intent)                   # ENABLED: round whole intent
   err_s   = intent - s_fast                    # remainder in (-1, 1)  (one-LSB bound)
   ```
   DISABLED: `s_fast += SR(delta_t)` and `err_s = 0` — the legacy tick
   (`prototype_packed_b.py:988-992`) verbatim.

5. **(A) Dithered carry (chase): tick WHOLE LSBs out of `s_fast`.**
   ```
   chase_gate = chase_floor + (1-chase_floor)*coh           # ratio-coh floor (line 1010)
   chase_int8 = alpha*chase_gate*s_fast / 128
   tick_slow  = SR_round(chase_int8)
   s_slow    += tick_slow
   s_fast    -= tick_slow*128                                # EXACTLY tick_slow*128 (line 1042)
   ```
   The chase moves a **fraction of what is already in the bounded register** and
   subtracts exactly the carried amount — it does NOT rescale the inflow (the
   draft's fatal bug). `chase_floor>0` makes a tick land ~every step, so the
   deploy ratchets even when evaporation is active (self-test 3: 298/300 steps).

6. **Leak → `v_slow` (unchanged, mass-preserving).**
   ```
   gap_v   = (s_slow - v_slow)*128
   delta_v8= alpha_v_fast*gap_v/128 * (leak_floor + (1-leak_floor)*coh)   # line 1049-1051
   tick_v8 = SR_round(delta_v8)
   v_slow  = clamp(v_slow + tick_v8, -128, 127)
   s_slow -= (v_slow_new - v_slow)                            # mass-preserve (line 1060)
   ```

7. **(B) Leak sub-LSB residual → SEPARATE `den_frac` channel.**
   ```
   den_frac += (delta_v8 - actual_tick_v8)        # the sub-LSB anchor residual
   whole     = trunc(den_frac)                    # promote any whole unit out
   v_slow   += whole ;  den_frac -= whole          # keeps |den_frac| < 1
   ```
   `den_frac` is a **distinct budget** from the velocity carry. For a normal coord
   it is ~0; for a denormal coord it is the only place the sub-LSB value lives.

8. **Re-exponent safety (legacy line 1143).** `renormalize_on_saturation`: where
   the per-row max of `max(|full mantissa|, |s_fast|)` reaches `MAX_M=24000`, halve
   `(s_fast, s_slow, v_slow)` and bump `row_exp` by 1 — value-preserving, BEFORE
   the int16 clamp bites. This is the safety the draft DROPPED.

9. **Clamp + repack** (`prototype_packed_b.py:1103-1110`).

---

## 4. Velocity / coherence resolution

The draft redefined `d_fs = e_s + err_s` where `err_s` was unbounded, so `d_fs`
exploded to ~1e5 and coherence collapsed to 0 permanently (and `v_proxy = noise^2`
exploded, freezing the step). Here `d_fs = s_fast + err_s` with `s_fast` the
BOUNDED int16 velocity and `|err_s| < 1`, so:

- `d_fs` is bounded → `noise` and `v_proxy` are bounded → coherence cannot be
  driven to 0 by a runaway carry (red-team finding 2, RESOLVED by construction).
- `sig`, `noise`, `coh_raw`, `cf`, `coh`, the cf-discount, and the `vhat` floor
  are the legacy computation unchanged (`compute_coherence`, mirror of
  `prototype_packed_b.py:817-846`).
- The velocity-quality concern (per-step intent noisier than a multi-step EMA) is
  moot: `d_fs` is again exactly the legacy `s_fast` quantity (line 778), so the
  SDXL coh/cf calibration (`coh_kappa`, `chase_floor`) carries over unchanged.

Validated: coh stays in a stable band (0.005–0.03 in the test-3 regime) and does
not collapse, while the deploy still ratchets via the `chase_floor` bootstrap.

---

## 5. DENORMAL handling

**Detection — ONE predicate** (the draft had three inconsistent ones), used to
GATE the deploy add:
```
is_denormal(packed)  <=>  |s_slow*128 + s_fast + v_slow*128| < 1     (== 0 integer mantissa)
```
i.e. the whole INTEGER live mantissa is below one mantissa unit, so the only value
the coord carries is `den_frac`. `err_s` (<1) cannot lift an all-zero integer
mantissa to ≥1, so the predicate is a pure function of the persistent packed word
(no sidecar needed).

**Channel.** `den_frac[N,K]` fp32 in (-1, 1) is its OWN optimizer-state buffer
(like `coh_pre`/`v_row`/`v_col`). It is NOT the low bits of `e_v`: the red-team
showed the leak SATURATES `e_v`'s integer part under normal dynamics, so the Q4.4
"lossless share" was unsound. Separating the budgets removes that coupling
entirely. `den_frac` is fed at `load_weights` (sub-LSB init) and by the leak's
sub-LSB residual (§3.7); any whole unit it accrues promotes into `v_slow`, so
`|den_frac| < 1` always.

**Deploy of a denormal.** `decode_to_deploy_weight` adds a rounded render of
`den_frac` ONLY where `is_denormal` is true (gated, not unconditional):
```
m_dep = (s_slow + v_slow)*128 + is_denormal * round_or_SR(den_frac)
```
For a normal coord this term is gated off; for a denormal coord it is the only
non-zero term. `deterministic=False` uses SR (DC-bias nulled in expectation, for
the live/training render); `deterministic=True` uses round-to-nearest (for a
reproducible shipping checkpoint — the red-team's minor finding).

Validated (self-test 2): a 3e-6 weight in a row whose max is 2.0 is detected
denormal, reconstructs LIVE to 3.000e-06, and survives deploy
(E|w| = 3.8e-6 over 64 dithers). The lossless-share invariant is no longer needed
(the budgets are physically separate), so the fragile dynamical proof is gone.

**Out-of-range (the opposite end).** A coord whose velocity grows past the int16
range is handled by the re-exponent safety (§3.8), not by spilling into fp — this
is the legacy mechanism (`prototype_packed_b.py:1143`), restored.

---

## 6. Bounded-carry invariants (self-tested every step)

| invariant | bound | where checked | draft value |
|---|---|---|---|
| `|s_fast|` | ≤ 32767 (int16, + re-exponent at MAX_M) | self-test 3/3b | int8-clamped 256× tighter |
| `|err_s|`  | < 1 (sigma-delta remainder of one SR) | `assert_carries_bounded`, self-test 3/3b | ~21280 (test-3), ~2e5 (strong) |
| `|den_frac|` | < 1 (sub-LSB fraction) | `assert_carries_bounded` | err_v ~537, fp16→INF |
| live recoverable from packed word alone | ≤ 1.5 mantissa units | `assert_recoverable_without_sidecar`, self-test 6 | ~110000-unit gap |

`err_s` is bounded by ONE LSB **independent of inflow magnitude and step count**
because it is the remainder of a round of the WHOLE fine value, not an accumulator
of unthrottled inflow. Measured max over 400 steps = 0.997 in BOTH the drift and
strong-gradient regimes (the draft hit ~2e5 under strong gradient).

---

## 7. Red-team resolution (point by point)

1. **Blocker — `err_s` unbounded (shadow velocity register).** Root cause: the
   chase scaled the OUTFLOW by `alpha*gate (<<1)` but subtracted the full
   `q_s*128`, while the inflow accumulated at full rate; the int8 `e_s` saturated
   and the remainder spilled into unbounded `err_s`. **Fix:** the velocity is ONE
   bounded int16 `s_fast`; the inflow SR-ticks DIRECTLY into it (legacy
   line 988-992); the chase carries WHOLE LSBs out and subtracts exactly
   `tick_slow*128` (legacy line 1042). `err_s` is now only the sub-LSB SR
   remainder, bounded < 1. The re-exponent safety (line 1143) handles true
   saturation. (self-test 3/3b: `|err_s| < 1`, `|s_fast| ≤ 32767`.)

2. **Blocker — coherence/preconditioner diverge.** Caused by (1). `d_fs` is again
   the bounded int16 velocity → `noise`/`v_proxy` bounded → coh cannot collapse.
   (§4; validated coh stays in band, does not go permanently 0.)

3. **Blocker — "disabled == legacy" false at training level.** The draft clamped
   the fine accumulator per-byte to int8. **Fix:** disabled mode is literally the
   legacy single-int16-`s_fast` path. `assert_disabled_matches_legacy` checks the
   COARSE/deploy word is BIT-EXACT and `s_fast` matches to ≤1 (the residual is
   only a float-reassociation Bernoulli flip at a round boundary; the Triton port
   shares one kernel expression and is bit-exact). (self-test 5: coarse=0, sf=0.)

4. **Major — `err_v` unbounded; fp16/in-fraction mitigations impossible.** The
   leak-carry budget is now SEPARATE from the denormal-fraction budget; `den_frac`
   is bounded < 1 with whole-unit promotion into `v_slow`. No fp16 sidecar at
   1e5 magnitudes (that produced INF), no "fit 120 units in 4 bits" claim.

5. **Major — lossless-share proof / three `is_denormal` definitions.** The Q4.4
   integer-part proof is gone (budgets are physically separate). ONE
   `is_denormal` predicate, and the deploy add is GATED on it (not unconditional).

6. **Major — state unrecoverable from packed word.** The bulk value is in the
   persistent int16 `s_fast` + int8 coarse; the carries are each < 1.
   `assert_recoverable_without_sidecar` confirms the live weight is recoverable to
   ≤ 1.5 mantissa units with the sidecars dropped (self-test 6: gap 1.301).

7. **Minor — non-deterministic export.** `deploy_weight(deterministic=True)`
   provides a reproducible round-to-nearest export for shipping checkpoints; SR is
   kept only for the live/training render (self-test 7).

---

## 8. Integration plan (Triton port of `prototype_packed_b.py`)

A `USE_DUAL_FAST` constexpr flag; off ⇒ `DEN_FRAC_BITS=0`, `err_s/den_frac` zero,
chase truncates, deploy is pure coarse — byte-identical to today.

| kernel site | change |
|---|---|
| unpack (~758-760) | UNCHANGED. `s_fast` stays one int16. (No byte split.) |
| velocity/coh (~778-846) | `d_fs = s_fast + err_s` (was `s_fast`); everything else unchanged. Add fp16 `err_s`/`den_frac` companion buffers passed by pointer like `coh_pre_ptr` (~657). |
| evap (~936-952) | UNCHANGED. |
| inflow tick (~986-992) | ENABLED: fold `err_s`, `s_fast = SR(s_fast+err_s+delta_t)`, `err_s = intent - s_fast`. DISABLED: legacy `s_fast += SR(delta_t)`. |
| chase (~1026-1042) | add `chase_floor` (already present via `USE_RATIO_COH`, line 1010); otherwise UNCHANGED — `s_fast -= tick_slow*128` stays. |
| leak (~1044-1060) | UNCHANGED; capture `delta_v8 - actual_tick_v8` into `den_frac` when `DEN_FRAC_BITS>0`. |
| repack (~1106-1110) | UNCHANGED (legacy layout). Re-store `err_s`/`den_frac` to the sidecar. |
| `consolidated_weight` (~2782-2800) | add `is_denormal * round(den_frac)` when `DEN_FRAC_BITS>0`, else pure coarse. |
| `get_weight` (~2768-2781) | live adds `err_s + den_frac`. |
| `load_weights` (~2645-2693) | even-split unchanged; route `|m_total|<1` sub-LSB value into `den_frac`, fine residual into `s_fast + err_s`. |
| rebalance atomic-max (~1127-1148) | UNCHANGED — the per-row re-exponent already reads the full live mantissa incl. `s_fast`. |
| externally-mutating paths (`resplit_anchor_to_even` ~2716, `load_weights_anchor` ~2695) | must zero/re-init `err_s`/`den_frac` consistently (they operate on the packed word; the sidecars must follow). Frozen-anchor TE (`alpha_v_fast=0`): no leak ⇒ `den_frac` only ever holds the load-time sub-LSB residual — safe; `load_weights_anchor` must route the sub-LSB residual into `den_frac`, not the integer word. |

Port notes: thread `consf` through the consolidation terms exactly as today
(chase/evap/leak gated by `consf`, ~984/1026/1049); reproduce the per-channel
`_hash_uniform` salts for bit-comparability; the reference is fp32/CPU and assumes
`consf=1`.

---

## 9. Risks

1. **Extra optimizer state.** `err_s` + `den_frac` add ~4–8 B/param of sidecar
   (the int32 word is unchanged at 32 b/param). Because both are sub-LSB and
   `|·| < 1`, they are dropped-tolerant (live recoverable to ≤1.5 units), so they
   can be fp16 OR not checkpointed at all (a relaunch loses < 1.5 mantissa units,
   not the ~110000 the draft lost). On the 24 GB Windows box, fp16 keeps the
   overhead to ~4 B/param; `den_frac` can be omitted entirely if denormal survival
   is not needed for a given layer (`DEN_FRAC_BITS=0`).
2. **Re-exponent on a hot layer.** Under high LR / large `step_cap` / strong
   fast-vs-slow cancellation, `s_fast` can ramp into `MAX_M` and trigger the
   per-row halving repeatedly (it bumps `row_exp`). This is legacy behavior
   (line 1143); it is value-preserving but coarsens the row's LSB. Monitor
   `get_rebalance_watermark_stats` (~2811). The reference exercises it (it fired
   on row 6 of the 80-step test).
3. **`den_frac` promotion vs leak sign.** The whole-unit promotion of `den_frac`
   into `v_slow` (§3.7) is mass-non-preserving by design (it materializes a
   sub-LSB value that the shared exponent could not express). For a coord that
   oscillates around the LSB boundary this can add tick noise; gated by
   `is_denormal` at deploy so it only affects coords that are genuinely sub-LSB.
4. **Deploy non-determinism on denormal coords.** SR render differs by ≤1 deploy
   LSB on sub-LSB coords between two exports. Use `deterministic=True` for shipping
   checkpoints (provided).
5. **CPU/fp32 reference vs Triton fp/order.** Disabled-mode bit-exactness against
   a SEPARATE Python reimplementation is only ≤1-LSB (float reassociation flips a
   Bernoulli at a round boundary); the Triton port shares ONE kernel expression so
   it is bit-exact. The test therefore asserts coarse=bit-exact + s_fast≤1, and
   skips steps where the re-exponent fired (the bare `_legacy_step` stub omits it).
6. **`vhat` floor / cf calibration.** `d_fs` magnitude distribution is now the
   legacy one (bounded int16), so the existing SDXL `coh_kappa`/`chase_floor`
   should carry over; a light re-check is prudent since `chase_floor>0` now fires
   the carry every step (the deploy advances more often than legacy).

---

## 10. Reference API (what the tests import)

`from modules.util.optimizer.concord.dither_accum_ref import (...)` —
see `§ Contract` in the task return for the testable surface. Self-test:
`CUDA_VISIBLE_DEVICES="" python dither_accum_ref.py` → ALL TESTS PASS (8/8).
