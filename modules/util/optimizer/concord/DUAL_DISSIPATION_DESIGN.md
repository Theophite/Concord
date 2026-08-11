# Dual-Dissipation Fine Accumulator — bracketed two-int8 `e_L`/`e_H`, int-only storage

Status: PAPER DESIGN (architect-confirmed 7-point spec). No code, no Triton port yet.
Kernel mirrored / to be mirrored: `modules/util/optimizer/concord/prototype_packed_b.py`.

This document SUPERSEDES `DITHER_ACCUM_DESIGN.md` / `dither_accum_ref.py`. That prior
design kept the int16 `s_fast` and bolted on **fp32 sidecar buffers** (`err_s`,
`den_frac`), violating the int-only-storage premise and producing a one-directional
DC-bias leak into the deploy word. **This design has NO fp sidecars of any kind** —
every bit of state lives inside the existing 32-bit-per-param word.

The core idea: split the 16 fine bits `[31:16]` (legacy single int16 `s_fast`,
`prototype_packed_b.py:758`) into **two independent int8 fine accumulators** `e_H` and
`e_L` that run the EXACT legacy per-step machinery at **two bracketed dissipation
rates** (`λ_L < λ_legacy < λ_H`). The gradient-accumulation microbatches (which ARE the
fine register's inflow under the fused step) are split into two timestep-matched halves
so the per-microbatch losses already computed give a FREE prequential A/B signal. Each
accumulator coherence-rates itself and soft-consolidates its coherent part into the
SHARED `s_slow`/`v_slow`; the combine gains are calibrated so the blend recenters EXACTLY
on the legacy single-int16 magnitude. A denormal coord (integer mantissa == 0) encodes
its sub-LSB value in the reliably-free leading bits of the aggressively-dissipated `e_H`.
With the feature OFF the two int8s reunify to one int16 and the kernel is **bit-exact to
legacy**.

---

## 1. Word layout & bit map

The persistent word is the SAME 32 bits as legacy. Total width unchanged, **no companion
buffers**.

### Feature ON

```
 bit  31 30 29 28 27 26 25 24 | 23 22 21 20 19 18 17 16 | 15 14 .. 9 8 | 7 6 .. 1 0
 field|<------ e_H int8 ----->|<------ e_L int8 ------->|<- s_slow i8 ->|<- v_slow i8 ->
 scale|        x1 (mantissa)  |        x1 (mantissa)    |     x128      |     x128
 rate |  HIGH dissipation λ_H |  LOW dissipation  λ_L   |   shared      |   shared
 role |  noise-boil + denormal|  momentum (full range)  |  position     |   anchor
      |  channel (held small) |                         |               |
```

Unpack (mirrors legacy `758-762`, the only change is splitting the top 16 bits into two
sign-extended int8 instead of one int16):

```
e_H        = packed >> 24                 # bits [31:24], arith shift sign-extends, int8 [-128,127]
e_L        = (packed << 8)  >> 24         # bits [23:16], arith shift sign-extends, int8 [-128,127]
s_slow_i8  = (packed << 16) >> 24         # bits [15: 8]  UNCHANGED (legacy line 759)
v_slow_i8  = (packed << 24) >> 24         # bits [ 7: 0]  UNCHANGED (legacy line 760)
s_slow_full = s_slow_i8 * 128             # (761 verbatim)
v_slow_full = v_slow_i8 * 128             # (762 verbatim)
```

Repack (mirrors legacy `1106-1110`, two int8 segments in `[31:16]` instead of one int16):

```
packed = ((e_H_c & 0xFF) << 24) | ((e_L_c & 0xFF) << 16)
       | ((s_slow_c & 0xFF) << 8) | (new_v_int8 & 0xFF)
```

with `e_H_c, e_L_c` each clamped to `[-128, 127]` (the int8 analogue of the legacy int16
clamp at `1104`).

### The reunify identity (the crux that makes disabled-mode bit-exact)

`e_H` is the **high** byte and `e_L` the **low** byte of the same signed 16-bit field:

```
s_fast_legacy  ==  (e_H << 8) | (e_L & 0xFF)  ==  e_H*256 + (e_L & 0xFF)  ==  packed >> 16
```

This is an algebraic identity, not a data change: the 16 bits `[31:16]` are one
contiguous int16 regardless of whether we read them as one int16 or two int8. Disabled
mode reads them as the single legacy `s_fast`; enabled mode reads them as `e_H`,`e_L`.

### Live weight and velocities

```
m_eff = s_slow_full + (e_H*256 + (e_L & 0xFF)) + v_slow_full    # == legacy line 774 with the reunified fine value
d_fs_H = e_H        d_fs_L = e_L                                 # per-accumulator velocity (analogue of line 778)
d_sv   = s_slow_full - v_slow_full                              # SHARED coarse gap (line 779) — ONE value, both arms use it
```

**One live weight, not two.** Both halves of the microbatch split forward through the
SAME live weight `m_eff` (the reunified fine value). The A/B split is only in WHERE the
gradient tick is deposited (into `e_L` vs `e_H`), NOT in two separate rendered weights.
This is decision **D3** below — it keeps one bf16 weight buffer per layer (no 2× weight
cost) and keeps the prequential comparison fair (both halves judged against the same
current parameter).

---

## 2. The two accumulators + bracketed rates

`e_L` and `e_H` are **genuine, full-state int8 velocity registers** — each runs the
complete legacy per-step update applied to its own byte. They are NOT the high/low byte
of one jointly-updated int16 (that was the prior ref's "reunify" cop-out; here the bytes
diverge during the accumulation window because they receive different microbatches and
dissipate at different rates).

Legacy dissipation is a single rate (line 936):
`evap_frac = min(lr_eff * gf_consol * (1 - coh_raw), 1 - min_leak)`. Define the legacy
nominal rate `λ_legacy = lr_eff * gf_consol` (the un-clamped slope). Bracket it
**geometrically** with one dimensionless spread `ρ_br > 1` (default ~2.0):

```
λ_L = λ_legacy / ρ_br      (LOW  dissipation → e_L retains momentum)
λ_H = λ_legacy * ρ_br      (HIGH dissipation → e_H boils noise fast, leading bits free)
```

Geometric bracketing guarantees `λ_L < λ_legacy < λ_H` AND
`sqrt(λ_L · λ_H) = λ_legacy` — the bracket is **log-centered** on legacy, which is the
natural metric for a survival factor (a per-step multiplier). The symmetric
`coh_L == coh_H` case then recenters cleanly (Section 6).

Each arm's evaporation uses its OWN rate against its OWN velocity and its OWN raw
coherence (mirror of `936`/`952`, applied per byte):

```
evap_frac_X     = min(λ_X * (1 - coh_raw_X), 1 - min_leak)            # X in {L, H}
evap_mantissa_X = evap_frac_X * d_fs_X * g_active * build_ok_X        # mirror of line 952
```

with `min_leak` slam-shut floor (929-936), per-arm soft build gate `p_build`
(941-952), and `g_active` (lazy/sighting gate, 899-915) all applied per arm.

**Why this is the from-scratch fix.** The legacy single rate evaporates `s_fast` at full
`λ_legacy` before the chase can carry a whole 128-LSB into `s_slow`, so the deploy never
ratchets from a zero/denormal start (the DITHER doc §1 failure). With bracketing, `e_L`
dissipates at `λ_L < λ_legacy`, so under a persistent gradient it RETAINS nascent
momentum long enough for `d_sv` to grow, which raises `coh`, which opens the chase gate
— bootstrapping the deploy from zero. `e_H` runs hot to shed genuine noise so it does not
pollute `s_slow`, and its aggressive decay keeps its high bits free for the denormal
channel (Section 8).

`ρ_br` is FIXED per this paper design. The free prequential loss gap (Section 3) is the
recorded signal that could later servo-recenter `ρ_br` across steps, but the consolidation
does NOT act on it (soft only).

---

## 3. Matched-timestep microbatch split (the free prequential test)

The fused step makes the fine register the gradient accumulator: `delta_grad` ALWAYS
applies and accumulates into the fine register (line 981/984), while the consolidation
terms are `consf`-gated; there is **NO separate `.grad` buffer**. So the
gradient-accumulation loop is the only place to inject an A/B contrast — and it is free.

A gradient-accumulation cycle has `M` microbatches; the trainer already calls
`set_consolidate(False)` for the first `M-1` and `True` on the last
(`prototype_packed_b.py:502-507`). The dual path additionally:

1. **Splits the `M` microbatches into two equal halves** `H_L` and `H_H` with **MATCHED
   timestep distribution**. The diffusion timestep `t` dominates gradient
   magnitude/character, so an unmatched split would confound the dissipation A/B with a
   timestep-difficulty difference. Matching is done by **antithetic / matched-pairs
   assignment**: draw `t` in pairs (the same sampled `t`, or an antithetic pair around
   the schedule mean) and deal one of each pair to `H_L` and one to `H_H` (equivalently,
   stratify `t` into bins and deal strata alternately `L,H,L,H,…`). This guarantees
   identical mean `t` and, to first order, identical `E[g²]` per half.

2. **Routes each half's gradient tick to its own accumulator.** An `H_L` microbatch's
   backward deposits its `delta_grad` tick (line 991 logic) ONLY into `e_L`; an `H_H`
   microbatch ONLY into `e_H`. Because `delta_grad` always applies regardless of `consf`
   (line 981), `e_L` ends the cycle holding the summed gradient of its `M/2`
   microbatches and `e_H` its `M/2` — each fine register becomes the half-cycle gradient
   accumulator for its arm. Both halves forward through the SAME `m_eff` (decision D3).

3. **The per-microbatch losses already computed for accumulation ARE the prequential A/B
   signal.** Sum the `H_L` losses vs the `H_H` losses: the difference is an unbiased
   online (sequential-predictive) comparison of `λ_L` vs `λ_H` on matched-difficulty
   data, at **ZERO extra forward passes and ZERO extra backward passes**. This is
   recorded as a global cross-check (and a future `ρ_br`-recentering signal) but is NOT
   acted on — consolidation is soft and per-coord (Sections 4–5).

`consf` gating is unchanged: chase/leak/evap consolidation fires once per optimizer step
(on `consf=1`) from both accumulators' settled state, exactly as legacy gates
consolidation.

> **Note on half-rate inflow.** Each arm sees `M/2` microbatches, so in expectation each
> carries ~half the per-step accumulated velocity that the single `s_fast` would carry:
> `E[e_L] + E[e_H] = E[s_fast_legacy]`. This **mass-split** (not rate-split) identity is
> the foundation of the combine calibration (Section 6) — the gradient mass is PARTITIONED
> between the two bytes, not duplicated.

> **⚠ OPEN GAP — SMALL-`M` ESTIMATOR VARIANCE + A DISSIPATION-RECENTER HOLE (red-team:
> "Matched-mean-timestep + half-batch" = UNCERTAIN, major). The core invariants survive
> (recentering, mass-split, disabled==legacy are NOT broken), but two under-documented
> effects emerge, both worst exactly where the design is most fragile (the from-scratch
> bootstrap):**
>
> 1. **Convex over-dissipation bias the affine-chase identity does NOT cover.** Matched MEAN
>    timestep equalizes `E[noise_X]` across arms (zero inter-arm bias in expectation) but,
>    by Jensen, does NOT equalize `Var[noise_X]`: each arm carries residual within-half
>    variance `~Var/(M/2)`. The §6 affine-combine identity protects the CHASE, but
>    DISSIPATION is driven by the per-arm, un-blended `coh_raw_X` (line 826/936), which is a
>    concave-saturating function of `noise²`. Inflating `E[noise²]` by the within-half
>    variance LOWERS mean `coh_raw` (Jensen) — a SYSTEMATIC, non-canceling over-dissipation
>    bias at small `M`, on BOTH arms, landing on the bootstrap.
> 2. **Per-arm estimator variance ~2×; degenerate at `M=2`.** Each arm builds `noise²`
>    (hence the `v_proxy` preconditioner and `coh`) from `M/2` samples, ~doubling per-arm
>    estimator variance. At `M=2` (common with grad-accum) there is ONE microbatch per arm,
>    zero within-arm averaging — coherence rated on a single velocity increment, and the
>    matched-histogram guarantee is vacuous.
>
> **RESOLUTION (adopted):**
> - **R-MSPLIT-1 (close the dissipation-recenter hole).** Drive each arm's evaporation off a
>   BLENDED `coh_raw_blend` (analogous to `coh_blend`), NOT the per-arm `coh_raw_X`, so the
>   convex Jensen over-dissipation bias cancels the SAME way the chase does — making BOTH
>   consolidation channels (chase AND evap) recenter on the legacy single-rate operating
>   point. (Note this also interacts with R-COMBINE-1/2 in §6; verify jointly.)
> - **R-MSPLIT-2 (M-guard, hard).** Define a minimum grad-accum `M_min ≥ 4` (i.e. ≥2
>   microbatches/arm) below which `USE_DUAL_FAST` COLLAPSES TO LEGACY (single int16, no
>   split). At `M=2` the per-arm coherence is a one-sample estimate and the matched-timestep
>   guarantee is vacuous — the feature must not engage.
> - **R-MSPLIT-3 (validate at the smallest M).** The from-scratch CPU fixed-point bootstrap
>   analysis (Risk 6 / R-DEN-4) MUST be run at the SMALLEST intended `M`, not an idealized
>   large-`M` limit, before trusting the mechanism. Document the ~2× per-arm estimator
>   variance and the small-`M` over-dissipation bias in §10 as the dominant calibration risk
>   on the bootstrap.

---

## 4. Per-accumulator coherence rating

Each accumulator is rated by its OWN per-weight coherence using the EXISTING legacy
machinery (`compute_coherence` path, `prototype_packed_b.py:817-846`) — **no new
estimator**. The signal is SHARED (same coarse gap, same `C*`); only the noise differs
(each arm's own velocity):

```
sig_w     = drift_cancel_C * d_sv * scale_fwd                   # SHARED (line 823) — same target for both arms
sig2      = sig_w * sig_w
For X in {L, H}:
  noise_X   = d_fs_X - drift_cancel_C * d_sv                    # per-arm (line 787)
  noise_w_X = noise_X * scale_fwd
  coh_raw_X = sig2 / (sig2 + noise_w_X^2 + 1e-30)               # (826) → drives e_X's OWN dissipation
  # cf-discount (838-841), per arm, with d_fs_X:
  vhat_fl   = max(v_hat, 0.03 / (sum_v_inv * N * K))            # (838) shared v_hat, per-arm unaffected
  cf_X      = sig2 / (C^2 * vhat_fl + 1e-30)                    # (839)
  coh_n2_X  = noise_w_X^2 * coh_kappa / (cf_X + coh_kappa)      # (840)
  coh_X     = sig2 / (sig2 + coh_n2_X + 1e-30)                  # (841) → drives e_X's OWN chase/leak gate
```

`drift_cancel_C` is computed once (`compute_drift_cancel_C`, lines 52-102, mass-preserve
2× form): the shared `d_sv` and shared leak rate `alpha_v_fast` mean `C*` is a property
of the shared coarse dynamics; each arm's own `d_fs` lag is what differs.

Because the two arms share `sig` and differ only through their own noise term, coherence
rates HOW WELL each arm's retained velocity aligns with the shared established drift. The
LOW arm, retaining more momentum, reads HIGHER `coh` once a real direction forms; the
HIGH arm reads high `coh` only on a strongly-coherent burst and otherwise ~0 (its noise
dominates because dissipation kept it small relative to `sig`). `coh_raw_X` drives
dissipation (un-cf'd, line 826); `coh_X` drives the chase/leak gate (cf-discounted, line
841) — the same two-coherence split legacy maintains, per arm.

---

## 5. Soft consolidate / dissipate

**No hard winner-take-all.** Each accumulator soft-consolidates its COHERENT part into
the SHARED `s_slow`/`v_slow` via the legacy coh-gated chase/leak, and dissipates its
INCOHERENT part at its OWN bracketed rate. Both arms drain into BOTH `s_slow` (via chase,
1041) and `v_slow` (indirectly, via the single shared leak that moves `s_slow → v_slow`,
1044-1060).

Per-arm chase (mirror of `1026-1042`, gain derived in Section 6):

```
gate_X            = chase_floor + (1 - chase_floor) * coh_X          # affine ratio-coh gate (line 1010)
chase_mantissa_X  = alpha * gate_X * gate_gain * e_X * consf         # gain = 1 per arm (Section 6)
chase_int8_f_X    = chase_mantissa_X / 128.0
tick_slow_X       = SR(chase_int8_f_X)                               # stochastic round, own salt
s_slow_i8        += tick_slow_X
e_X              -= tick_slow_X * 128                                # subtract EXACTLY the carried amount (line 1042)
```

Each arm subtracts exactly its realized carry from its OWN byte (mass bookkeeping per
arm). The two arms use **decorrelated SR salts** (e.g. `step_salt ^ 0x5A5A5A5A` for the
L-chase, `step_salt ^ 0xA5A5A5A5` for the H-chase) so the dithers are independent.

Per-arm dissipation: each arm's `evap_mantissa_X` (Section 2) drains its incoherent
velocity into the `delta_t_X` fed to its SR-tick — exactly as legacy folds
`evap_mantissa` into `delta_t` (line 984), per arm.

**Leak is shared and fires ONCE.** The leak rate `alpha_v_fast` (line 1049) is a property
of the SHARED `s_slow → v_slow` gap, NOT of the fine bytes. After BOTH arms have chased
into `s_slow`, the leak runs once on the post-chase shared gap (lines 1047-1060),
unchanged from legacy, using a blended coherence `coh_bar = (coh_L + coh_H)/2` for its
ratio floor (1051). There is exactly ONE leak gain → no double-leak, no per-byte `v_slow`
bias is possible by construction. The downstream `wd_sv`/`wd_sf`/`wd_anchor` ticks
(1062-1101) likewise operate on the shared `s_slow`/`v_slow` gap and fire once; the
`wd_sf` and `wd_anchor` ticks that target the fine register are split across the two arms
in proportion to each arm's share (or, simplest and equivalent in expectation, applied to
the reunified fine value and re-split — see decision D5).

---

## 6. Combine & recenter math

**Goal.** The coherence-weighted blend of the two bracketed accumulators must deposit into
`s_slow`/`v_slow` the SAME expected magnitude per step as the legacy single-int16
`s_fast`, so effective LR is unchanged — NO factor-of-2 over-consolidation, and NO bias
when `coh_L ≠ coh_H`.

**Legacy chase** (line 1026), the consolidation that sets effective LR:

```
chase_mant_legacy = alpha * gate(coh) * gate_gain * s_fast * consf,   gate(coh) = chase_floor + (1-chase_floor)*coh
```

**The mass identity** (from the half-rate inflow, Section 3): every gradient tick legacy
put into one int16 is now put into exactly one of the two bytes, and the reunified value
`e_H*256 + e_L` is the same live fine value, so

```
e_L + e_H  =  s_fast_legacy      (mass split, not duplicated)        (★)
```

**Per-arm chase with gain = 1** (NOT 1/2 — see below), each using its OWN affine gate:

```
chase_mant_L + chase_mant_H
  = alpha*gate_gain*[ gate(coh_L)*e_L + gate(coh_H)*e_H ]
  = alpha*gate_gain*[ chase_floor*(e_L+e_H) + (1-chase_floor)*(coh_L*e_L + coh_H*e_H) ]
  = alpha*gate_gain*(e_L+e_H)*[ chase_floor + (1-chase_floor)*coh_blend ]
  = alpha*gate(coh_blend)*gate_gain*(e_L+e_H)
where  coh_blend := (coh_L*e_L + coh_H*e_H)/(e_L+e_H)     # the e-WEIGHTED coherence
```

By **(★)**, `e_L + e_H = s_fast_legacy`, so the dual chase deposits EXACTLY the legacy
chase magnitude, automatically weighted by each arm's own coherence. **This is the
recentering.** It works because the ratio-coh gate is **AFFINE in `coh`** (line 1010): the
affine gate linearizes the blend, so the sum of two per-arm chases equals one chase
evaluated at the blended coherence. The result is unbiased for any `coh_L ≠ coh_H` (the
more-coherent arm dominates `coh_blend`), and at `coh_L = coh_H` it is the plain average
gate (no factor-of-2).

> **⚠ OPEN GAP — RECENTER PROOF IS EVALUATED ON THE WRONG QUANTITY (red-team verdict
> "COMBINE CALIBRATION" = BREAKS, major; and "MASS-CONSERVATION" BREAK 2 = blocker).**
> The proof above evaluates **(★)** on the GRADIENT mass `e_X^grad`, but the legacy chase
> does NOT chase the gradient mass — it chases `s_fast` AFTER the per-step momentum
> reinforcement and evaporation have been folded in (verified op-order: `delta_t`,
> `prototype_packed_b.py:984` → commit `:992` → chase reads the UPDATED value `:1026`).
> Because the dual path runs momentum/evap per-arm at its OWN bracketed `λ_X` and OWN
> coherence, the post-evap chased mass is `e_X^grad·(1 + β1·coh_X − λ_X·(1−coh_X))`, which
> differs per arm. **(★)** then holds for the gradient component but is BROKEN for the value
> actually consolidated. Two concrete failures:
>
> 1. **Symmetric under-consolidation (geometric-bracket AM>GM drift).** At `coh_L=coh_H`
>    and even split, the dual path removes EXTRA incoherent mass
>    `s·(1−coh)·[(λ_L+λ_H)/2 − λ_legacy]`. The geometric bracket log-centers the per-step
>    SURVIVAL multiplier, but the chase consolidates an arithmetic SUM, and
>    `(λ_L+λ_H)/2 = λ_legacy·(ρ_br+1/ρ_br)/2 > λ_legacy` for any `ρ_br>1` (AM>GM). At the
>    default `ρ_br=2` this is **1.25× extra dissipation** of the incoherent fraction — a
>    ~25% downward effective-LR drift on `(1−coh)`-mass per step. This is the OPPOSITE sign
>    of the 2× over-consolidation fear the original §6 rebutted; the under-consolidation
>    was never considered.
> 2. **`coh_L ≠ coh_H` quadratic bias.** The affine-gate identity holds for the gate ALONE,
>    but the chase multiplies `gate(coh_X)` against the coherence-dependent pre-multiplier
>    `(1 + β1·coh_X − λ_X·(1−coh_X))`. The product is QUADRATIC in `coh_X`, so it does NOT
>    collapse to `gate(coh_blend)·Σe_X^grad`; the residual depends on the coherence SPLIT,
>    re-introducing exactly the `coh_L≠coh_H` bias §6 claims to eliminate.
>
>    The MASS-CONSERVATION blocker adds a second, independent break on this same algebra:
>    **(★)** is written as plain integers `e_L + e_H = s_fast`, but the reunify identity
>    (§1, line 70) is `s_fast = e_H*256 + (e_L & 0xFF)` — `e_H` is the HIGH byte. As plain
>    integers `e_L + e_H ≠ s_fast`. If the H-arm chase reads the BYTE value `e_H` (not the
>    mantissa value `e_H*256`) it credits deploy in `s_slow` LSBs (×128) while debiting a
>    byte worth ×256 in the live weight — a **256× ledger mismatch** and a one-directional
>    DC leak whenever `coh_H` opens the gate.
>
> **RESOLUTION (adopted into the spec — to be implemented and CPU-verified before TDD):**
> - **R-COMBINE-1 (op-order fix).** Chase off the PRE-momentum/PRE-evap gradient snapshot:
>   capture `e_X^grad` before folding momentum/evap into the byte and consolidate THAT, then
>   apply per-arm evaporation to the post-chase residual. This makes the chased quantity
>   linear in `e_X^grad`, restores **(★)** exactly, and removes BOTH the symmetric drift and
>   the `coh_L≠coh_H` quadratic bias. (Equivalent reorder: chase reads pre-evap `s_fast`,
>   evap fires after.)
> - **R-COMBINE-2 (mantissa-correct ledger).** The mass identity is
>   `e_H*256 + (e_L & 0xFF) = s_fast`, NOT `e_L + e_H`. The H-arm chase MUST operate on the
>   MANTISSA value `e_H*256` (`chase_mant_H = alpha·gate·gate_gain·(e_H*256)`) with its
>   byte subtraction removing the matching mantissa amount — OR, cleaner and adopted as the
>   canonical implementation, run the chase on the **reunified int16** `s_fast = packed>>16`
>   and re-split the residual to bytes (the D5 pattern), so the `s_slow += tick` /
>   `s_fast -= tick*128` conservation is byte-agnostic and bit-identical to legacy. Update
>   **(★)** in this section to the mantissa form before re-deriving.
> - **R-COMBINE-3 (CPU gate, blocking).** A CPU fixed-point check at `ρ_br ∈ {1.5, 2}` MUST
>   compare per-step `Δs_slow` against the legacy single-int16 path under BOTH `coh_L=coh_H`
>   and `coh_L≠coh_H` and confirm ZERO effective-LR drift. A per-step ledger assertion
>   `Δ(deploy_word) == −Δ(fine-register mantissa value)` across BOTH chases and every
>   `is_denormal` toggle is REQUIRED (this is the exact assertion that catches both breaks).
> - **Fallback (does NOT fully fix):** an ARITHMETIC-matched bracket
>   `λ_L = λ_legacy·(1−d)`, `λ_H = λ_legacy·(1+d)` kills the symmetric drift but NOT the
>   `coh_L≠coh_H` bias; do not rely on it alone.

**Why gain = 1, not 1/2.** The naive fear "two accumulators → 2× consolidation" is wrong
precisely because the gradient MASS was split, not the rate: each `e_X ~ s_fast/2`, so
summing two gain-1 chases at half magnitude already reproduces the full legacy chase. A
1/2 gain would DOUBLE-COUNT the split and halve the effective LR. The correct calibration
is **gain 1 per arm + the affine-gate identity**.

**Subtraction bookkeeping preserves the fine register**: each arm subtracts exactly its
realized carry from its own byte (`e_X -= tick_slow_X * 128`), so the sum of subtractions
equals the total added to `s_slow` — `s_slow` gains exactly the blended legacy amount, the
fine registers lose exactly that.

**Leak** is shared, fires once, depends only on the `s_slow/v_slow` gap (Section 5), so it
is automatically un-doubled — no per-arm leak gain exists.

> **Decision D2 (chase gate must be affine).** The recentering proof relies on the gate
> being affine in `coh` — the `USE_RATIO_COH` path (line 1010). Under `USE_COHPRE`
> (1011-1024) or `USE_GAP_FEEDBACK` (`c_pass`, 1007-1008) the gate is no longer affine in
> the per-arm `coh`, so `Σ gate(coh_X)*e_X ≠ gate(coh_blend)*(e_L+e_H)` and a residual
> consolidation bias appears. **Dual-dissipation is therefore restricted to the
> `USE_RATIO_COH` gate.** This is the production gate (the live SDXL path uses ratio-coh
> floors, lines 740-743), so the restriction is free in practice; under cohpre/gap-feedback
> the feature must be disabled (collapse to legacy).

---

## 7. Denormal — steal from `e_H`'s leading zeros

> **⚠ OPEN GAP — THE WHOLE DENORMAL CHANNEL HAS THREE INDEPENDENT BREAKS (red-team:
> "MASS-CONSERVATION" BREAK 1 = blocker; "Denormal steal-from-e_H" = breaks, major;
> "From-scratch denormal build" = breaks, major). This section is the highest-risk part of
> the design and is NOT cleared for implementation as written.** The three breaks and the
> adopted resolutions are detailed at the end of this section under "RESOLUTION — denormal
> channel redesign". Read that BEFORE coding §7.

**Detection — ONE predicate**, a pure function of the persistent word (no sidecar):

```
int_mant    = s_slow_full + (e_H*256 + (e_L & 0xFF)) + v_slow_full     # the whole integer live mantissa
is_denormal = (int_mant == 0)
```

i.e. the coord carries no representable integer mantissa at the shared block-float scale
`2^(row_exp+col_exp-bias)` (set from the row max-abs, `load_weights:2666-2674`). The
shared exponent cannot drop per-element, so a weight far below its row max is sub-LSB and
the deploy `×128` (consolidated_weight, 2793-2799) quantizes it to zero — the legacy
denormal hole.

**Encode the sub-LSB value in `e_H`'s leading bits.** On a denormal coord the integer
mantissa is 0, so `e_H`'s NORMAL job (hold a small high-dissipation velocity) is vacuous —
there is no integer value to accumulate. The high dissipation `λ_H` is precisely what
makes `e_H`'s high bits **reliably free**: under any non-coherent signal `e_H` decays
toward 0 within a few steps (survival `(1 - λ_H·(1-coh))^k`), so for an active coord
`|e_H| << 127` and its leading bits sit at the sign-extension of a small magnitude.
`e_L` is NOT touched — it runs at `λ_L < λ_legacy` precisely to RETAIN the from-scratch
momentum and needs its full `[-128,127]` range; stealing its bits would cap the mechanism
the design depends on.

**Mechanism (all in-word, no fp).** When `is_denormal`, `e_H`'s 8 bits are reinterpreted
as a signed sub-LSB fixed-point fraction `den = e_H / 128` in `(-1, 1)` of one mantissa
unit (a Q0.7-style value: sign + 7 fractional bits, ~1/128 mantissa-unit resolution =
matches the deploy LSB granularity). On a denormal coord the inflow tick is accumulated
into the `e_H` byte interpreted as `frac × 128` (a fixed-point sigma-delta in the SAME 8
bits — floor/Bernoulli on the ×128-scaled value), so the sub-LSB state lives entirely in
the `e_H` byte of the int32 word — **no fp sidecar**. The only inflow path for a denormal
coord is this `e_H` fraction (its chase/leak whole-unit ticks are all zero); it also banks
the leak's sub-LSB residual (the `delta_v8 - actual_tick_v8` remainder legacy DISCARDS at
1055-1060) so a coord climbing out of true denormal does not lose its sub-unit progress to
rounding.

**Promotion.** When the `e_H` fraction accrues a whole mantissa unit (the byte would
overflow ±128 at the ×128 interpretation), it PROMOTES one whole unit into the integer
field (into `v_slow` via the leak path, or `e_L`), the coord becomes non-denormal, and
`e_H` reverts to its normal high-dissipation integer role.

**Render to deploy, GATED by `is_denormal`** (mirror of consolidated_weight 2793-2799):

```
m_dep = (s_slow_i8 + v_slow_i8) * 128 + is_denormal * SR_or_round(den)
```

For a normal coord the term is gated OFF (pure legacy coarse); for a denormal coord it is
the only nonzero term. SR for the live/training render (DC-bias nulled in expectation);
round-to-nearest for a reproducible shipping checkpoint. `get_weight` (live, 2768-2781)
adds the same fraction. This is strictly sub-LSB so it cannot lift a nonzero-mantissa
coord (`is_denormal` is false there anyway).

> **Decision D4 (hysteresis on the denormal boundary).** A coord oscillating across
> `int_mant == 0 ↔ ≠0` within a few steps would churn `e_H` between fraction-mode and
> integer-mode, injecting tick noise. Require `int_mant == 0` for `K` consecutive
> consolidate steps before entering fraction-mode, and promote-on-overflow only (latch
> denormal until the integer mantissa reaches `≥ 1`). This rules out boundary DC-bias into
> the deploy word.

---

### ⚠ RESOLUTION — denormal channel redesign (three breaks, all closed on paper, all CPU-gated)

The red-team found three independent failures in the denormal channel as drafted above.
The byte-overloading approach (reinterpret the whole `e_H` byte as both a ×256 integer
velocity AND a `/128` sub-LSB fraction) is the common root cause and is **abandoned**.

**BREAK A — `is_denormal` self-nullifies (blocker, "MASS-CONSERVATION" BREAK 1).**
`int_mant` (line 334) INCLUDES `e_H` at ×256, so `int_mant == 0` forces `e_H == 0`
(barring measure-zero cancellation), which forces `den = e_H/128 = 0`. The render term
`is_denormal · SR(den)` is therefore ALWAYS `True · SR(0) = 0` — **the channel can never
deploy a nonzero sub-LSB value; it is inert.** The only way to make it fire is to exclude
`e_H` from `int_mant`, but then the velocity machinery still credits `s_slow` while
debiting an `e_H` that is no longer in the integer mantissa — a pure one-directional DC
leak into the deploy word (the exact dither_accum_ref failure, relocated to the H-arm
chase ledger).

**BREAK B — `e_H` high bits are NOT reliably free on a COHERENT denormal inflow
(major, "steal-from-e_H").** Dissipation is `evap_frac ∝ (1 − coh_raw)` (line 936). On a
coherent sub-LSB burst `coh_raw_H → 1`, so `evap_frac_H → 0` and `e_H` ACCUMULATES the
coherent inflow at full mantissa-unit scale instead of boiling it (e.g. `e_H ≈ 30` over a
window). The SAME 8 bits are simultaneously read as `den = e_H/128 ≈ 0.23` and rendered to
deploy — the steal collides with real velocity content. "Reliably free" holds ONLY when
the inflow is INCOHERENT, i.e. exactly when there is nothing worth encoding. D4 hysteresis
PROLONGS the collision window (it latches fraction-mode while `int_mant==0`, which is
precisely when the coherent burst inflates `e_H`), it does not prevent it.

**BREAK C — the denormal-from-scratch build is routed through the WRONG arm and cannot
ignite (major, "From-scratch denormal build").** §7 routes all denormal inflow into `e_H`
(the HIGHEST-dissipation arm, `λ_H = λ_legacy·ρ_br`) and starves `e_L` (the LOW-dissipation
retain arm whose whole purpose is the from-scratch bootstrap). At the all-zero start
`d_sv = s_slow − v_slow = 0`, so `coh_raw = coh = 0` and the bracket cannot lift it; the
only thing preventing `e_H` evaporation is the PRE-EXISTING legacy `p_build` gate, not any
new dual mechanism. Promotion into `v_slow` recreates the counterfeit-anchor washout
(`d_sv = −128`, spurious `coh ≈ 0.5` — the pathology `resplit_anchor_to_even` exists to
cure); promotion into `e_L` leaves `d_sv = 0`, no chase into `s_slow`, deploy never
ratchets. The matched-timestep prequential split (§3) is also moot on denormal coords:
since only `e_H` is live there, `e_L` receives no microbatches and the A/B loss-gap cannot
even be measured for the coords whose build is in question.

**ADOPTED RESOLUTION (replaces the byte-overloading scheme; all four parts required):**

- **R-DEN-1 — dedicated fraction sub-field, never overload `e_H`.** Do NOT reinterpret the
  whole `e_H` byte as the fraction. Reserve `e_H` exclusively as a ×256 integer velocity
  register (it stays in `m_eff`, `int_mant`, and the re-exponent). Carve the sub-LSB
  fraction out of a DIFFERENT, declared bit slice that is provably zero for the integer
  machinery — e.g. repartition the low byte to `e_L` = 10–13 integer bits + a 2–3-bit
  fraction tag, OR steal only the low `k` bits of `e_L`. The predicate and the render MUST
  read the SAME bits at the SAME scale.
- **R-DEN-2 — define `is_denormal` from the DEPLOY-word integer fields only.**
  `is_denormal = (s_slow_i8 == 0 and v_slow_i8 == 0)` — i.e. the deployed coarse is zero —
  NOT from an `int_mant` that includes `e_H*256`. The render fraction is the bits this
  predicate leaves free. This kills BREAK A (the predicate no longer forces the fraction to
  zero) and makes the ledger conservative.
- **R-DEN-3 — route the denormal BUILD through the LOW-dissipation arm `e_L`, with a
  coh-independent ignition.** On a denormal coord, deposit inflow into `e_L` (the retain
  arm) so the §2 bootstrap loop (e_L retains → fraction promotes → `s_slow` LEADS) can
  actually run, and make promotion deposit the whole unit into `s_slow` (so `d_sv > 0`,
  opening the chase) — NOT into `v_slow` (washout) and NOT into `e_L`-only (`d_sv=0`
  stall). Because `d_sv=0` forces `coh=0` at the all-zero start, the chase into `s_slow`
  MUST rely solely on the `chase_floor` term (production `chase_floor ≥ 0.1`, never 0).
  This kills BREAK C (build runs through the correct arm and ignites on the floor) and
  resolves the §3-vs-§7 routing contradiction (state explicitly in §3: denormal coords
  build through `e_L`; the prequential split applies only to non-denormal coords).
- **R-DEN-4 — guard the render against a busy fraction register, and CPU-prove ignition.**
  Gate the render on `is_denormal AND is_frac_free` where `is_frac_free` requires the
  guard bits of the fraction slice to be sign-extension-of-zero (or `|fraction| <
  frac_capacity`); if the slice carries real velocity, render PURE COARSE that step
  (graceful skip, not a silent collision) — this closes BREAK B. A CPU fixed-point
  recursion MUST show the fraction promotes to a whole `s_slow` unit BEFORE dissipation
  drains it at `coh=0` (promotion rate > drain rate at the denormal magnitude); until a
  stable promotion fixed point above the deploy-tick threshold is demonstrated, the
  **denormal-from-scratch claim is DOWNGRADED from a feature to an OPEN RISK** and the
  channel ships disabled.

**Frozen anchor (`alpha_v_fast = 0`) remains safe:** no leak, so a denormal coord's
fraction is only ever the load-time residual.

**Init/load.** `load_weights` (2645-2693): the even-split routes the integer mantissa
across `s_slow`/`v_slow`/`e_L`; a `|m_total| < 1` sub-LSB residual is routed into `e_H`'s
fraction field. `load_weights_anchor` (2695-2714) and `resplit_anchor_to_even` (2716-2754)
operate on the packed word and must preserve/re-split `e_H`'s fraction consistently.
Frozen anchor (`alpha_v_fast = 0`): no leak, so a denormal coord's fraction is only ever
the load-time residual — safe.

---

## 8. Disabled == legacy (bit-exact)

DISABLED (a `USE_DUAL_FAST` constexpr = False) collapses BIT-EXACT to the legacy kernel,
by construction, because the enabled and disabled paths share ONE kernel expression set
selected by the constexpr (NOT a separate file) — there is no fp sidecar arithmetic to
reorder, so no float-reassociation `≤1-LSB` slack.

1. **WORD.** Bits `[31:16]` are the same 16 bits. Reunify
   `s_fast = (e_H << 8) | (e_L & 0xFF) = packed >> 16` (legacy line 758 verbatim) — a
   no-op reinterpretation, not a data change. Repack stores the same 16 bits. The
   persistent word is byte-identical to legacy.

2. **STEP.** Disabled mode does NOT split microbatches (no A/B): all `M` microbatches
   deposit into ONE shared inflow stream. It does NOT run two coherences, does NOT
   bracket (`ρ_br = 1` ⇒ `λ_L = λ_H = λ_legacy`, the geometric bracket degenerates). It
   runs the SINGLE legacy update on the reunified int16 `s_fast`:
   `d_fs = s_fast` (778), ONE coherence, ONE evap at `λ_legacy` (936), ONE chase
   `chase_mantissa = alpha*gate*gate_gain*s_fast` (1026, with the partition collapsing to
   `coh_L/(coh_L+0) = 1` so the single-arm gain is the legacy gain verbatim), `s_fast -=
   tick_slow*128` (1042), ONE leak (1044-1060). The clamp is the **int16** clamp (1104)
   applied to the reunified value (then re-split to the two bytes — a pure relabel, no
   value change), NOT the per-byte int8 clamp.

3. **DENORMAL.** Extension OFF: `e_H` is never reinterpreted, `int_mant`/`is_denormal`
   unused (forced false), no bits stolen. Deploy = `(s_slow + v_slow)*128 * 2^exp` — pure
   coarse (consolidated_weight 2793-2799), byte-identical.

4. **RE-EXPONENT.** Reads `max(|full mantissa|, |s_fast|)` on the reunified value (1143)
   unchanged.

**Verification pattern** (the dither ref's `assert_disabled_matches_legacy`): the
coarse/deploy word is bit-exact and the reunified `s_fast` register is bit-exact to a
literal legacy `s_fast` tick. Because disabled mode branches to the SINGLE int16 path (not
a degenerate two-int8 path with two SR ticks), there is no SR-boundary reassociation slack
— the claim is HARD bit-exactness, not `≤1-LSB`. **Decision D6**: the OFF path MUST use
the literal legacy single-register expression, not a `ρ_br = 1` two-register path, to
guarantee this.

> **✅ HOLDS (minor) — but make it self-enforcing, not a discipline (red-team:
> "DISABLED==LEGACY bit-exactness" = HOLDS, minor).** The OFF-path bit-exactness could not
> be broken on paper: it rests on a true algebraic identity (`(e_H<<8)|(e_L&0xFF) ==
> packed>>16`) and the design has explicitly closed the three traps that would otherwise
> break it — (1) the SR hash MUST be seeded from the FULL reunified int16 `s_fast` (a
> `ρ_br=1` two-int8 path seeding from 8-bit bytes would produce DIFFERENT SR ticks → not
> bit-exact; D6 forbids it); (2) the clamp MUST be the int16 clamp on the reunified value,
> NOT per-byte int8 clamps (a per-byte clamp corrupts any `|s_fast|>127`, which trained
> values routinely exceed); (3) deploy gates `is_denormal` false → the term is
> constexpr-eliminated, and with NO fp sidecars there is no float-reassociation path. The
> ONLY residual fragility is that the guarantee is CONTINGENT on implementation discipline
> (D6 plus the trainer-side split also being flag-gated off — the kernel cannot self-prove
> the latter). Convert discipline into invariants:
>
> - **R-OFF-1 (CI gate).** Run the OFF kernel and a literal legacy kernel on a random packed
>   word + grad and assert the packed words AND the full SR-tick stream are byte-identical
>   (both are deterministic given `step_salt`).
> - **R-OFF-2 (shared source).** Factor the legacy `s_fast` tick/chase/leak/clamp into a
>   helper called IDENTICALLY by the OFF branch and by the legacy file, so the int16-clamp
>   and the `s_fast` hash seed cannot drift.
> - **R-OFF-3 (split flag-gated).** Gate the trainer-side microbatch A/B split on the SAME
>   `USE_DUAL_FAST` flag; test that with the flag off the routing is the single legacy
>   inflow stream.
> - **R-OFF-4 (clamp regression).** Unit test that `|s_fast| ∈ (127, 32767]` survives OFF
>   mode (catches an accidental int8 clamp).

---

## 9. Key design decisions (where the three proposals disagreed)

- **D1 — Byte order: `e_H` = high byte `[31:24]`, `e_L` = low byte `[23:16]`.**
  (Proposals 2 & 3; Proposal 1 had them swapped.) Chosen so the disabled-mode reunify
  `s_fast = (e_H<<8)|(e_L&0xFF) = packed>>16` is the legacy line 758 with `e_H` as the
  high-order byte — the natural, contiguous int16 reading. The denormal steal is within
  `e_H`'s own 8 bits regardless of which word-position they occupy.

- **D2 — Combine math: affine-gate identity, per-arm gain = 1 (NOT explicit
  convex-weight normalization).** (Proposal 2's derivation.) Proposal 1 used convex
  weights `coh_X/(coh_L+coh_H+eps)` to force a partition of unity, which is correct only
  when both gates share a gain and introduces an `eps`/cold-start `50/50`-of-noise
  degeneracy when both `coh ≈ 0`. Proposal 2 proves that because the gate is affine in
  `coh` AND the mass is split (`e_L+e_H = s_fast`, not duplicated), the SUM of two gain-1
  per-arm chases is EXACTLY the legacy chase at the e-weighted blended coherence — no
  division, no `eps`, no degeneracy, unbiased under `coh_L ≠ coh_H`. This is the soundest
  option and is adopted as canonical. It costs the restriction in D2'.

- **D2' — Restrict to the `USE_RATIO_COH` (affine) gate.** The recentering proof needs an
  affine gate; cohpre/gap-feedback break it. Production uses ratio-coh, so this is free;
  under the other gates the feature disables to legacy.

- **D3 — One live weight, not two.** (Proposal 2's correction of its own draft; Proposals
  1 & 3 left the per-arm forward weight ambiguous / implied two weights.) Both halves
  forward through the SAME `m_eff` (reunified fine value); the split is only in tick
  ROUTING. This avoids a 2× weight-buffer cost and keeps the prequential comparison fair
  (same parameter under test). Proposal 3 flagged the 2-weight materialization as a real
  cost risk; D3 removes it entirely.

- **D4 — Hysteresis on the denormal boundary** (latch `K` consecutive `int_mant==0`
  steps; promote-on-overflow only). All three proposals flagged the boundary-churn risk;
  this is the agreed mitigation.

- **D5 — Shared-gap ticks (leak, `wd_sv`) fire once on the shared state; fine-targeting
  ticks (`wd_sf`, `wd_anchor`) apply to the reunified fine value and re-split.** Keeps the
  WD dynamics identical to legacy in magnitude and avoids per-arm WD double-counting.

- **D6 — Disabled path branches to the literal legacy single-int16 expression** (not a
  `ρ_br=1` two-register degenerate path), so disabled == legacy is HARD bit-exact, not
  `≤1-LSB`.

---

## 10. Risks (paper-level; require empirical / CPU-fixed-point validation before trusting)

1. **Int8 range is 256× tighter than legacy int16 for the MOMENTUM register.** `e_L` is
   `[-128,127]` vs legacy `s_fast` `[-32768,32767]`. The legacy fine residual rides ~±64
   (load_weights:2686) so headroom exists, but a high-LR / large-`step_cap` (Muon-style)
   regime that legacy absorbs in the int16 could saturate `e_L` and fire the re-exponent
   ratchet (1143) far more often, coarsening the row LSB. **MITIGATION:** the re-exponent
   must max in `max(|e_H*256+e_L|, …)` on the REUNIFIED value (not per byte) so a hot
   layer halves-and-renormalizes the whole fine value before either int8 clamp bites; the
   half-rate inflow (D3/§3) keeps each byte ~half legacy magnitude. **This is the failure
   mode that sank earlier two-int8 drafts** — it MUST be checked empirically on the fine
   convs (the known overcook layers, `up_blocks.2`/`down_blocks.0`). An asymmetric bit
   budget (`e_L` 9-10 bits, `e_H` 6-7 bits) is a fallback if `e_L` pins.

2. **`e_H` double-duty (velocity vs denormal fraction).** `e_H` is the high-dissipation
   noise accumulator for normal coords AND the sub-LSB fraction holder for denormal coords;
   the `is_denormal` gate switches interpretation. The boundary-churn risk is mitigated by
   D4 hysteresis, but it needs a clean proof (or self-test) that `e_H ≈ 0` in BOTH
   interpretations at the boundary so the switch is continuous.

3. **Matched-timestep split feasibility.** Antithetic/matched-pairs `t` assignment assumes
   the OneTrainer fork's dataloader/timestep sampler can be paired without perturbing the
   loss scale, and that the trainer exposes per-microbatch `t` to stratify the deal. If
   microbatches are pre-batched opaquely, the guarantee weakens to stratification only and
   the prequential A/B is confounded by timestep difficulty — needs a trainer hook to
   assign microbatch → {L,H} by timestep bin. (`set_consolidate` already exists as a
   per-micro-step hook, 502-507, so the plumbing precedent is there.)

4. **`ρ_br` is uncalibrated.** `λ_L = λ_legacy/ρ_br`, `λ_H = λ_legacy·ρ_br` is asserted,
   not derived. Too wide → `λ_H` boils everything / `λ_L` never dissipates noise; too narrow
   → the arms are indistinguishable (no prequential signal, no benefit over legacy). The
   free loss-gap signal could auto-tune `ρ_br`, but the control law is unspecified (and
   would need the paired-difference variance, not the unpaired one, because antithetic
   pairing makes the halves correlated).

5. **Two decorrelated SR streams into a shared `s_slow`.** Two independent SR inflow ticks
   double the dither entropy vs one int16 tick. The partition-of-unity chase should
   preserve the deploy DC-bias null in expectation, but two SR streams into a shared
   `s_slow` want a joint-unbiasedness check (CPU, like `compute_drift_cancel_C`'s
   fixed-point analysis).

6. **From-scratch bootstrap is qualitative.** The positive-feedback argument (`e_L`
   retains → `d_sv` grows → `coh` rises → chase opens) is plausible but not shown to have
   a stable fixed point above the deploy-tick threshold under realistic noise. It could
   stall in a low-coh basin where `chase_floor` ticks in and the leak ticks back out (the
   exact legacy failure, slower). A CPU fixed-point analysis of the dual-arm recursion is
   needed before trusting the from-scratch claim.

7. **Shared `sig` across arms.** Both arms use the same `d_sv` and `C*` for `sig`, so they
   cannot disagree on the coherent TARGET, only on noise. If the real benefit requires the
   arms to track DIFFERENT directions (not just different retention of the same direction),
   this construction cannot express it — the soft blend may collapse to "whichever arm has
   less noise," a noisier version of the legacy single rate. (Acceptable for the stated
   goal — bracketed retention of ONE drift — but bounds the upside.)

8. **Recenter proof assumes equal inflow split.** **(★)** `e_L ~ e_H ~ s_fast/2` holds
   only if the matched-timestep split is exactly even (equal count, matched histograms).
   Odd `M` or un-pairable timestep histograms leave the halves with unequal velocity and a
   residual effective-LR bias; the bias bound as a function of split imbalance should be
   derived.

---

## 11. Build map (kernel sites to touch, all in `prototype_packed_b.py`)

| site | change |
|---|---|
| unpack (758-762) | split `[31:16]` into `e_H = packed>>24`, `e_L = (packed<<8)>>24` (ON); reunify `s_fast = packed>>16` (OFF). |
| velocity/coh (778-846) | call the coherence path TWICE with `d_fs_L=e_L`, `d_fs_H=e_H`, shared `d_sv`/`sig`/`C*`. |
| evap (936-952) | per-arm `evap_frac_X = min(λ_X*(1-coh_raw_X), 1-min_leak)`, bracketed `λ_X`. |
| inflow tick (986-992) | two SR ticks (decorrelated salts), one per arm, into the routed byte; denormal coords accumulate into `e_H`'s fraction. |
| chase (997-1042) | per-arm chase, gain 1, affine gate `gate(coh_X)` (restricted to `USE_RATIO_COH`); `e_X -= tick_slow_X*128`. |
| leak (1044-1060) | UNCHANGED — fires once on the shared gap, blended `coh_bar`. |
| wd_sv/wd_sf/wd_anchor (1062-1101) | shared-gap ticks once; fine-targeting ticks on the reunified value, re-split (D5). |
| clamp/repack (1104-1110) | two int8 clamps `[-128,127]` (ON); int16 clamp on reunified value (OFF, D6). |
| re-exponent (1127-1148) | max in `|e_H*256+e_L|` (reunified) so saturation halves the whole fine value. |
| consolidated_weight (2783-2800) | add `is_denormal * SR_or_round(den)`; pure coarse when OFF. |
| get_weight (2768-2781) | live adds the denormal fraction. |
| load_weights / _anchor / resplit (2645-2754) | route sub-LSB residual into `e_H`'s fraction; preserve it across re-split. |

OFF path: `USE_DUAL_FAST=False` ⇒ single int16 `s_fast`, no split, no bracket, no denormal
— byte-identical to today.

---

## 12. Red-team verdicts

Six independent adversarial reviews of this design against the legacy kernel
(`prototype_packed_b.py`). Each verdict's resolution or OPEN GAP marker has been folded
into the relevant section above; this section is the consolidated ledger. Line references
are to `prototype_packed_b.py` unless noted.

| # | Risk area | Verdict | Severity | Folded into |
|---|---|---|---|---|
| 1 | DISABLED==LEGACY bit-exactness (two int8 reunify to single int16, deploy = pure coarse) | **HOLDS** | minor | §8 (R-OFF-1..4) |
| 2 | COMBINE calibration — coherence-weighted blend recenters on legacy magnitude? | **BREAKS** | major | §6 (R-COMBINE-1..3) |
| 3 | Denormal steal-from-`e_H` — high bits reliably free? | **BREAKS** | major | §7 (R-DEN-1,2,4 / BREAK B) |
| 4 | From-scratch deploy build from zero/denormal start | **BREAKS** | major | §7 (R-DEN-3 / BREAK C) |
| 5 | Matched-mean-timestep + half-batch under small grad-accum `M` | **UNCERTAIN** | major | §3 (R-MSPLIT-1..3) |
| 6 | Mass-conservation / DC-bias via denormal render + H-arm consolidation | **BREAKS** | **blocker** | §6 (R-COMBINE-2) + §7 (R-DEN-1,2 / BREAK A) |

### 12.1 — Verdict 1 (HOLDS, minor): DISABLED==LEGACY bit-exactness

Could not be broken on paper. Rests on a true algebraic identity
(`(e_H<<8)|(e_L&0xFF) == packed>>16`, a bit-pattern relabel over int32, not a data change)
and on three traps the design explicitly closed: the SR hash MUST be seeded from the full
reunified int16 (line 110-112; an 8-bit-seeded `ρ_br=1` path produces different ticks); the
clamp MUST be the int16 clamp on the reunified value (line 1104), NOT per-byte int8 (a
per-byte clamp corrupts any `|s_fast|>127`); deploy gates `is_denormal` false →
constexpr-eliminated (2793-2799). NO fp sidecars ⇒ no float-reassociation, so HARD
bit-exact (not `≤1-LSB`). Residual: the guarantee is contingent on discipline (D6 + the
trainer-side split also being flag-gated, which the kernel cannot self-prove). **Resolution
in §8: R-OFF-1 (CI bit-exact gate on packed word AND SR-tick stream), R-OFF-2 (shared
legacy source helper), R-OFF-3 (split flag-gated + test), R-OFF-4 (`|s_fast|∈(127,32767]`
survives OFF).**

### 12.2 — Verdict 2 (BREAKS, major): COMBINE calibration

The §6 recenter proof evaluates the mass identity **(★)** and the affine-gate linearization
on the GRADIENT mass `e_X^grad`, but the legacy kernel chases `s_fast` AFTER momentum
reinforcement and evaporation are folded in (op-order: `delta_t` 984 → commit 992 → chase
reads updated value 1026; no pre-evap snapshot). Per-arm post-evap mass is
`e_X^grad·(1 + β1·coh_X − λ_X·(1−coh_X))`, differing per arm. Two failures:
**(1) symmetric under-consolidation** — geometric bracket makes
`(λ_L+λ_H)/2 = λ_legacy·(ρ_br+1/ρ_br)/2 > λ_legacy` (AM>GM), removing
`s·(1−coh)·[(λ_L+λ_H)/2 − λ_legacy]` extra incoherent mass (~25% at `ρ_br=2`); log-centering
the survival multiplier does NOT arithmetic-center the consolidated sum. **(2) `coh_L≠coh_H`
bias** — the chase multiplies `gate(coh_X)` against a `coh_X`-dependent pre-multiplier →
QUADRATIC in `coh_X`, defeating the affine-gate identity, residual depends on the coherence
SPLIT. **Resolution in §6: R-COMBINE-1 (chase the pre-momentum/pre-evap snapshot, evap
after), R-COMBINE-2 (mantissa-correct ledger), R-COMBINE-3 (CPU drift check at
`ρ_br∈{1.5,2}` under both coherence regimes). Arithmetic bracket is a partial fallback.**

### 12.3 — Verdict 3 (BREAKS, major): denormal steal-from-`e_H`

Dissipation is `evap_frac ∝ (1−coh_raw)` (936); on a COHERENT sub-LSB inflow `coh_raw_H→1`
so `evap_frac_H→0` and `e_H` accumulates real velocity (`e_H≈30` over a window) — the same
8 bits read as `den=e_H/128≈0.23` collide with that velocity. "Reliably free" holds only
when the inflow is INCOHERENT, i.e. when there is nothing worth encoding. D4 hysteresis
PROLONGS the collision. **Resolution in §7: R-DEN-1 (dedicated fraction sub-field, never
overload `e_H`), R-DEN-4 (`is_frac_free` guard → render pure coarse if the slice carries
velocity).**

### 12.4 — Verdict 4 (BREAKS, major): from-scratch deploy build

NORMAL zero start already works in legacy via production `chase_floor ≥ 0.1` (never 0) +
the `_EVAP_BUILD_MIN` gate — the dual bracket is over-credited there. DENORMAL start FAILS:
inflow is routed into `e_H` (HIGHEST dissipation) while `e_L` (the retain arm, the whole
point) is starved; `d_sv=0 ⇒ coh=0` and the bracket cannot lift it; promotion into `v_slow`
recreates the counterfeit-anchor washout (`resplit_anchor_to_even` pathology), promotion
into `e_L` leaves `d_sv=0` and the deploy never ratchets. The §3 prequential split is moot
on denormal coords (only `e_H` live → no A/B). **Resolution in §7: R-DEN-3 (route the
denormal BUILD through `e_L`; promote the whole unit into `s_slow` so `d_sv>0` opens the
chase; ignite on `chase_floor`), plus the §3-vs-§7 routing contradiction stated explicitly.
Until CPU fixed-point shows a stable promotion fixed point above the deploy-tick threshold
at `coh=0`, the denormal-from-scratch claim is DOWNGRADED to an OPEN RISK and ships
disabled.**

### 12.5 — Verdict 5 (UNCERTAIN, major): matched-mean-timestep + small `M`

Core invariants (recentering, mass-split, disabled==legacy) survive — NOT broken. But
matched MEAN `t` equalizes `E[noise_X]`, not `Var[noise_X]` (Jensen): residual within-half
variance `~Var/(M/2)` feeds the per-arm, un-blended `coh_raw_X` that drives DISSIPATION
(826/936) — which the affine-CHASE identity does NOT cover. Concave `coh_raw` ⇒ systematic
over-dissipation bias at small `M`, on both arms, landing on the bootstrap; plus ~2× per-arm
estimator variance, degenerate at `M=2` (one microbatch/arm). Could not be bounded on paper
as small-enough-to-survive or large-enough-to-kill — hence uncertain. **Resolution in §3:
R-MSPLIT-1 (drive evap off BLENDED `coh_raw`), R-MSPLIT-2 (`M_min≥4` else collapse to
legacy), R-MSPLIT-3 (run the bootstrap CPU analysis at the SMALLEST `M`).**

### 12.6 — Verdict 6 (BREAKS, BLOCKER): mass-conservation / DC-bias

The blocker. The same 8 bits of `e_H` carry two interpretations differing by `256·128 =
32768×`, and the seam re-creates the prior design's one-directional deploy leak.
**BREAK 1 (self-nullify):** `int_mant` (334) includes `e_H` at ×256, so `int_mant==0` forces
`e_H==0` forces `den=0` — the denormal render is ALWAYS `True·SR(0)=0`, INERT; the only
"fix" (exclude `e_H` from `int_mant`) credits `s_slow` while debiting an out-of-mantissa
`e_H` → pure DC leak. **BREAK 2 (recenter algebra drops the ×256):** **(★)** is written
`e_L+e_H=s_fast` as plain integers, but `s_fast = e_H*256 + (e_L&0xFF)` — `e_H` is the HIGH
byte. The H-arm chase credits deploy in `s_slow` LSBs (×128) while a unit of `e_H` is worth
×256 in the live weight → 256× ledger mismatch, DC bias following `e_H`'s sign whenever
`coh_H` opens the gate. **Resolution in §6 R-COMBINE-2 (chase on the mantissa value
`e_H*256`, or on the reunified int16 with byte re-split — D5 pattern) and §7 R-DEN-1/R-DEN-2
(dedicated fraction slice never on `e_H`; `is_denormal` from the deploy-word integer fields
`s_slow_i8==0 && v_slow_i8==0` only). Mandatory per-step ledger assertion
`Δ(deploy_word) == −Δ(fine-register mantissa value)` across BOTH chases and every
`is_denormal` toggle — the exact test that catches both breaks.**

### 12.7 — Disposition

- **Bit-exact OFF path (Verdict 1): cleared**, pending the R-OFF CI gates.
- **Combine/recenter (Verdicts 2, 6): redesign required** — op-order snapshot
  (R-COMBINE-1) + mantissa-correct ledger (R-COMBINE-2), both CPU-gated.
- **Denormal channel (Verdicts 3, 4, 6): byte-overloading scheme abandoned** — replaced by
  a dedicated fraction sub-field, deploy-word `is_denormal`, build routed through `e_L`
  (R-DEN-1..4). Ships DISABLED until the from-scratch ignition fixed point is demonstrated.
- **Small-`M` calibration (Verdict 5): mitigated** by blended-`coh` evap + an `M_min≥4`
  guard (R-MSPLIT-1..2).

**No part of the design is cleared to TDD without first passing its CPU fixed-point gate.**
The mandatory CPU checks before any Triton port: (a) the OFF bit-exact + SR-stream identity
(R-OFF-1); (b) the per-step conservation ledger `Δdeploy == −Δfine-mantissa` (R-COMBINE-3 /
Verdict 6); (c) the recenter drift check at `ρ_br∈{1.5,2}` under `coh_L=coh_H` AND
`coh_L≠coh_H` (R-COMBINE-3); (d) the from-scratch / denormal-ignition fixed-point recursion
at the SMALLEST intended `M` (R-DEN-4 / R-MSPLIT-3).
