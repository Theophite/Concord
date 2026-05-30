# Session notes — 2026-05-29

Headline CIFAR-10 result: **90.21% best / 90.15% final** on WiderConvNet
(3.19M params), 80 epochs, with the unified clean recipe documented below.

This file is for the next Claude (or human) picking this up later. It
captures what was learned today, the final clean recipe, and what's left
on the table.

---

## The clean recipe (canonical command)

```
python cifar_concord_packed_b.py --epochs 80 --batch_size 16 \
    --weight_decay 0.1 --gf_trust_radius 10.0 \
    --bn_lr 0.1 --lr_min_frac 0.001
```

Key hyperparameters (everything else is script defaults):
- `lr = 0.1`, `wd = 0.1`, `alpha = 0.1`, `alpha_v_fast = 0.001`
- `drift_cancel_C = None` (auto-computed via `compute_drift_cancel_C`)
- `gf_trust_radius = 10.0`, `step_cap = 10.0`
- `bn_lr = 0.1`, `lr_min_frac = 0.001`
- AdamW three_accum packed-B on all 4 Conv2d + 3 Linear (~3.19M params)
- Biases on their own SGD group at the main lr (NOT v_lr_scale'd)
- BN params on a separate SGD group at `bn_lr * (cur_lr / args.lr)` —
  cosine-scaled like the main lr but anchored at `bn_lr`

Run takes ~8 min wall on the dev box.

---

## What was learned today

### 1. The "OLD vs NEW bias path" 0.45% gap was a BN bug

The original bias-path code had `pg['params'] is bn_params` to identify
the BN parameter group. This always returned False, because PyTorch's
`Optimizer.__init__` copies the params list internally. Net effect: BN
was training at the cosine-scheduled main lr (~0.1) the whole time,
not at `args.bn_lr=0.01` as intended.

Fixing the bug (use explicit `BN_GROUP=0`, `BIAS_GROUP=1` indexing)
exposed that the OLD baseline was getting accuracy from bn_lr being
effectively 0.1. At bn_lr=0.01, accuracy dropped to 89.82%.

Sweep at {0.01, 0.03, 0.05, 0.1}: monotone increasing — **0.1 wins
(90.17%)**, matching the OLD baseline's accidental-correct setting.

### 2. `lr_min_frac = 0.001` (not 0.01)

Default 0.01 had the cosine bottoming at `lr = lr_init * 0.01 = 0.001`,
which let the model drift 0.29% past its peak by epoch 80 (90.15%
at ep 73 → 89.86% at ep 80).

Sweep at {0.001, 0.0001, 0.0}: **0.001 best** — peak-to-end drift
collapsed to 0.06%, headline 90.21%/90.15%.

0.0001 underperformed (89.73%/89.37%), but 0.0 was fine (90.20%/89.95%),
so it's not a regime collapse — the 0.0001 result is probably a bad
seed within the ±0.2-0.3% run-to-run variance we observed throughout.

### 3. Cautious weight decay = "always decay toward slow"

WD now decays `s_fast` toward 0 (equivalently, decays the live weight
W toward `s_slow_full + v_slow_full`), instead of decaying W toward 0.
Kernel change in `prototype_packed_b.py`: use
`step_live += wd * (d_fs * scale_fwd)` instead of `wd * current_weight`.

This is the shape that supports `wd = 0.1` cleanly. With the old
"decay W toward 0" formulation, wd=0.1 would pull the weights too hard.

### 4. β1 is damping, not Adam-style momentum

packed-B's `beta1` knob subtracts `β1 · s_fast` from each step. Sweep:
- `β1 = 0` (default): baseline 90.17%
- `β1 = +0.1`: 88.90%/88.61% — less responsive, ~1.3% worse
- `β1 = -0.1`: 35.32%/10.01% — **catastrophic blow-up**, saturates
  every row/col_exp to MAX (7), tr_loss locks at ln(10) ≈ 2.3026

The α=0.1 chase IS already β1=0.9-style momentum in Adam terms (s_slow
is an EMA of s_fast with EMA rate α). There's no headroom to add "more"
momentum on top of the geometry.

### 5. **`v_slow` IS the architecture's Polyak average**

This was the most interesting finding. Per-accumulator ablation at
end of training (magnitude-aware fp32 reconstruction):

| accumulator combo | val_acc |
|---|---|
| all (1,1,1) — baseline | 90.10% |
| s_slow + v_slow (1,0,1) | 90.06% |
| **2×v_slow only** | **90.11%** |
| 2×s_slow only | 89.90% |
| s_fast contribution (50× to match magnitude) | 10.00% (chance) |

So:
- `s_fast` carries **zero positional signal** — it's purely the update
  vehicle (gradient + drift-cancel residue), mean ≈ 0 across weights.
- `s_slow + v_slow` is the entire predictive model.
- `2×v_slow alone` (at correct magnitude) slightly exceeds the live
  weight — v_slow is a Polyak average of s_slow over the α_v_fast=0.001
  time constant (~1000 steps), and at convergence the smoothed iterate
  is slightly better than the instantaneous one.

**Implication**: at inference, you can drop both `s_fast` (16 bits) and
`s_slow_i8` (8 bits) and just use `v_slow_i8 * 256 * envelope` (= 8
bits per element). 4× storage compression for free, possibly slight
accuracy bump.

### 6. gf-weighted blends don't help

Tested: `(1 - gf_ij) · 2·s_slow + gf_ij · 2·v_slow` per element, plus
hard-threshold variants. All underperformed the flat `s_slow + v_slow`
by 0.1-0.3%.

Reason: when gf is high (noise-dominated), the weight is already
converged so `s_slow ≈ v_slow` at that element — the blend is a no-op.
When gf is low (signal-dominated), the blend gates to `s_slow`, which
alone is the worst single-accumulator (89.76%-89.90%). Net: the
gf-weighted blend ≈ s_slow most places ≈ worst.

The right inference recipe isn't per-element-gated — it's just
`s_slow + v_slow` (or equivalently `2 × v_slow`) globally.

### 7. External CPU Polyak EMA is redundant with v_slow

Tested: CPU fp32 EMA at β=0.9 per epoch on `(s_slow + v_slow)` over
the last 10 epochs. Result: 89.73% — **exactly matches `2 × v_slow`
alone**, and slightly under the static `(1,0,1)` baseline.

Algebra: in a stationary regime, `s_slow ≈ v_slow`, and EMA-of-EMA
is the same EMA, so `Polyak(s_slow + v_slow) ≈ 2 × v_slow`.

To get a true Polyak average that captures earlier-epoch weights, you'd
need either a longer external Polyak window (last 30+ epochs at β=0.95)
that outlives v_slow's intrinsic memory, OR a slower α_v_fast in the
optimizer itself.

---

## Code modifications in this session

`cifar_concord_packed_b.py`:
- New CLI flags: `--polyak_window`, `--polyak_beta`
- Per-epoch CPU fp32 Polyak EMA accumulator (`polyak_cpu` dict)
- End-of-training Polyak evaluation block
- End-of-training per-accumulator ablation (magnitude-aware fp32 recon)
- End-of-training gf-weighted blend ablations (linear + threshold)
- Fixed bias/BN scheduler with explicit `BN_GROUP=0` / `BIAS_GROUP=1`
  indexing (the old `pg['params'] is bn_params` was always False)
- Helper `_layer_fp32_slow_weight(m)` returns the
  `(s_slow_full + v_slow_full) * envelope` weight for a layer

`prototype_packed_b.py` (touched in earlier session, relevant context):
- Cautious WD kernel: `step_live += weight_decay * (d_fs * scale_fwd)`
- gf trust region: `v_proxy += gf_trust_delta_sq * v_hat`
- Rebalance trigger includes `v_slow`: `abs_eff = |s_slow*128 +
  s_fast + v_slow*128|`
- `_row_max_hwm` / `_col_max_hwm` high-watermark buffers
- `get_garbage_fraction_stats()` and `get_rebalance_watermark_stats()`
- Constants: `S_SLOW_FACTOR = V_SLOW_FACTOR = 128`, `MANTISSA_BIAS = 15`

---

## State stats baseline for sanity-checking future runs

At end of clean recipe (lr_min_frac=0.001, bn_lr=0.1):
- `|s_slow_full|` ≈ 1500–2600 per layer
- `|v_slow_full|` ≈ same as `|s_slow_full|` (locked by drift-cancel)
- `|s_fast|` ≈ 40 per layer (active, not collapsed — BN trains slowly so
  concord layers carry feature-scale drift)
- `row_exp.max ∈ [-1, 2]`, `col_exp.max ∈ [1, 2]` (well below EXP_MAX=7)
- Rebalance high-watermark median 18-24k (just below MAX_M=24000)
- Garbage fraction `signal<0.1` at 45-65% per layer, mean Var/E[g²] 0.15-0.30

**Failure modes to watch for** in a future run:
- `|s_slow_full|` > 4000 → saturation, optimizer broken (check β1 sign,
  check that drift_cancel_C is auto-computed not hard-coded)
- `row_exp.max = 7` → exponent rail, layer has saturated rebalance
- `signal<0.1` at 99%+ everywhere → tr_loss likely stuck at ln(N_classes),
  network is diverged

---

## Open threads / suggested next experiments

1. **Inference-only `2 × v_slow` recipe**: build a checkpoint format that
   stores only `v_slow_i8` + `row_exp` + `col_exp` (no s_fast, no
   s_slow_i8). Compress live storage from 32 bits/element to ~8.

2. **Slower α_v_fast sweep**: at α_v_fast=1e-4 (vs current 1e-3) the
   v_slow Polyak window grows from ~1000 steps to ~10000 steps — about
   3 epochs at bsz=16, possibly capturing earlier peak weights. Risk:
   v_slow lags too much in mid-training and hurts accuracy more than it
   helps at the end.

3. **BEST-epoch checkpointing**: orthogonal to the optimizer changes —
   just save weights when val_acc hits a new high, restore at end. Would
   directly recover the headline 90.21% as the inference number with no
   averaging gymnastics.

4. **Re-run minfrac=0.0001 on a different seed**: confirm it was bad
   luck and not a real cliff.

5. **Larger batch sizes**: recipe is tuned at bsz=16. Scaling to
   bsz=128 may need re-tuning of bn_lr (probably bn_lr ≈ 0.05 at
   bsz=128 to compensate for 8× fewer steps per epoch).

6. **Bigger architecture**: 3.19M-param WiderConvNet at 90.21% is
   probably near the ceiling for this net. Most remaining loss to
   modern ResNet recipes is the architecture itself. Worth porting
   the recipe to a small ResNet-18 to see if the packed-B optimizer
   keeps up with torch.optim.AdamW on a stronger backbone.

---

## Run log (this session, for cross-referencing)

| tag | bn_lr | min_frac | recipe extras | best | final | notes |
|---|---|---|---|---|---|---|
| bn-fix | 0.01 | 0.01 | wd=0.1, gf=10 | 89.82 | 89.59 | first run with BN bug fixed |
| bnsweep_0.03 | 0.03 | 0.01 | wd=0.1, gf=10 | 89.94 | 89.77 | |
| bnsweep_0.05 | 0.05 | 0.01 | wd=0.1, gf=10 | 89.99 | 89.88 | |
| bnsweep_0.1 | 0.10 | 0.01 | wd=0.1, gf=10 | 90.17 | 90.11 | matches OLD baseline |
| ablation2 | 0.10 | 0.01 | + per-accumulator ablation | 90.44 | 90.10 | first ablation run; 2×v_slow=90.11 |
| gf_blend2 | 0.10 | 0.01 | + gf-blend ablation | 90.31 | 90.00 | gf-blends underperform flat (1,0,1) |
| polyak10 | 0.10 | 0.01 | + CPU Polyak β=0.9 last 10 | 90.15 | 89.86 | Polyak ≈ 2×v_slow, redundant |
| minfrac_0.001 | 0.10 | 0.001 | full recipe | **90.21** | **90.15** | **headline** |
| minfrac_0.0001 | 0.10 | 0.0001 | full recipe | 89.73 | 89.37 | likely bad seed |
| minfrac_0.0 | 0.10 | 0.0 | full recipe | 90.20 | 89.95 | also fine |

---

# Part 2 — Optimizer-mechanism investigation (what is concord *actually*?)

After the 90.21 headline, a long thread dissected what the packed-B
optimizer is really doing. Findings, in order, with the dead-ends kept so
they're not re-run.

## 2.1 The adaptive machinery is (numerically) inert; eps=1 is the real knob
Instrumented the AdamW step denominator `step = grad / (noise²·v_scale +
δ²·v̂ + eps)^p`. At the shipped settings (v_scale=1, gf_trust_radius=10 →
δ²=0.01, eps=1.0, precond_p=0.5) the measured term magnitudes are:
`drift(noise²) ~1e-4`, `gf(δ²·v̂) ~1e-9`, `eps = 1.0`. **eps dominates by
4–9 orders**, so `denom≈1`, `step≈grad`. So the drift-cancel preconditioner
AND the gf trust region are numerically dead; step_cap=10 also never binds
(real step_raw ~1e-2). Confirmed by A/B: gf_trust_radius=0 ≡ baseline;
eps=1e-8 degrades (step_cap then catches it, ~graceful not blow-up).
Root cause = **units mismatch**: `noise`/`v̂` are mantissa- vs gradient-
scaled, so the ratio never reaches O(1).

## 2.2 precond_p (Padam exponent) sweep — no interior win
Added `precond_p` (power on the denominator: 0=SGD, 0.5=sqrt/Adam-like).
Sweep p∈{0,0.15,0.25,0.35,0.5} @ v_scale=16, eps-warm: monotone-ish,
**p=0 (90.25) is best**, higher p underfits (p=0.5 ~87). The drift
preconditioner, when actually engaged (v_scale↑ / eps↓), is *worse* than
flat SGD on CIFAR (per-element noise² scaling is badly conditioned).

## 2.3 IMPORTANT correction — concord is NOT SGD
Earlier I called the headline "effectively SGD." Wrong, and the user was
right. `step≈grad` only means the **per-coordinate scaling is uniform**
(no Adam √v̂). But the injection feeds the **accumulator cascade**:
s_fast (velocity, ~10-step EMA of grad) → chase α → s_slow (position) →
v_slow (~1000-step Polyak average). That cascade IS momentum + Polyak
averaging. So concord = **mantissa-quantized momentum-SGD + Polyak avg +
cautious-wd, lacking only Adam's per-coord preconditioning.** Vanilla SGD
does NOT reach 90.25; the chase (= the momentum) is why it does.

## 2.4 Coherence-gating saga (all variants ≤ chase on CIFAR)
Idea: gate the optimizer by per-coord gradient *coherence* to keep signal,
reject noise. Tested, in escalating sophistication, ALL on the clean recipe
(p=0, wd=0, bn_lr=0.1, lr_min_frac=0.001):
- **gf-evaporation** (drain incoherent s_fast, κ·(1-coh)·s_fast, replaces
  cautious-wd): constant κ → 89.6 (monotone worse with κ); lr-annealed κ →
  peak **89.84** @ gf_consol=0.5. Both < chase 90.25.
- **coh_pre gated acceptance** (gate the chase by coh+coh_pre·(1-coh),
  coh_pre = per-coord EMA of coh init 1, λ=α_v): **failed hard, 71% @ ep40,
  tr_loss stuck 0.94.** Over-throttles.
- Full kernel impl of coh_pre exists behind `--cohpre` (per-elem fp32
  `_coh_pre` buffer, USE_COHPRE constexpr). Stable, healthy state, but
  underfits — leave OFF.

## 2.5 WHY it fails — and the epoch-noise caveat (the live thread)
1-D sims (`exp1_walk.py`, `exp2_selectivity.py`, `exp3_vslow_gate.py`,
`exp4_epoch_noise.py`, `exp5_gated_leak.py`) settled the mechanism:
- The gate's coherence `coh = (α_v·d_sv)²/v̂` = `SNR²/(1+SNR²)` — only
  "opens" at per-coord SNR≳1. **CIFAR per-coord SNR ≪ 1** (signal is
  sub-noise per step; it only emerges via temporal averaging), so coh≈0.01
  → gate throttles the very coords that carry signal → underfit. Exp 2
  should have been run BEFORE the kernel build (lesson).
- **The chase EMA is already the (near-optimal) noise filter** for the
  minibatch regime — gating per-step coherence fights the averaging that
  works. That's why every gate variant ≤ chase.
- BUT (user's key insight): minibatch noise is **not Brownian** — with
  shuffled epochs it **sums to ~0 over one epoch** (a bridge, not a walk).
  So noise accumulation is hard-bounded at 1 epoch; signal accumulates
  across epochs. With epoch-structured noise + a **sign-persistence**
  coherence `|EMA(d_sv)|²/EMA(d_sv²)` + window **≥ 1 epoch**, 1-D
  separation is clean (**9.6×**, exp4). Gating the v_slow *leak* by it
  keeps signal (track ~1.0) and denoises the anchor — but only **~2.6×**
  (exp5), because v_slow's EMA *already* exploits epoch-cancellation by
  averaging. Most of the gain is just **window length**, which is FREE.
- **Mistuning found:** α_v=0.001 → v_slow window ~1000 steps < 1 CIFAR
  epoch (~3125 @ bsz=16). So v_slow doesn't span an epoch → leaves
  epoch-averaging gain on the table.

## 2.6 RESOLVED — coherence-gating is a CIFAR dead-end (don't build the gate)
- α_v sweep {1e-3, 3e-4, 1e-4} on the chase baseline (free lever): the
  SHORT window WINS. α_v=1e-3 (<1 epoch, the current setting) → **90.30**
  best / 89.93 2×v_slow; α_v=3e-4 (~1 ep) → 89.77/89.26; α_v=1e-4 (~3 ep)
  → 90.06/89.55. Lengthening v_slow to span an epoch HURTS — it's a
  bias-variance optimum: longer window averages more noise but LAGS the
  moving weights more, and inference = v_slow, so lag costs accuracy. The
  current α_v is at the optimum, not mistuned.
- Verdict: the strictly-more-powerful full-averaging lever failed to beat
  the chase, so the gate's 2.6× denoise (exp5) on the same residual won't
  either. **Do NOT build the sign-persistence v_slow gate for CIFAR.** The
  chase + short v_slow is at the CIFAR noise-handling ceiling.
- The epoch-noise insight is still TRUE (noise is separable, bounded by 1
  epoch — overturned the earlier false "spectrally irreducible" claim); it
  just nets negative on CIFAR (small separable residual × lag tradeoff).
  Revisit only where the noise/lag balance differs from CIFAR.

## 2.6c 150-epoch run — MORE EPOCHS is the biggest lever; window gap closes
Ran the epoch-length window (av3e4) vs control (av1e3) at 150 ep:
|              | 80 ep | 150 ep (best/final) |
|--------------|-------|---------------------|
| av1e3 short  | 90.30 | **90.75** / 90.34   |
| av3e4 1-epoch| 89.77 | 90.65 / 90.29       |
- **No flip:** short window still wins at 150 ep (90.75 vs 90.65, ~noise).
  But the gap closed −0.53→−0.10 — confirms the lag-amortization intuition
  (longer anchor's fixed ~1-epoch lag shrinks as a fraction of training);
  it converges TO the control, not past it. Short v_slow window stays the
  recommended setting.
- **Real finding:** 150 ep lifts BOTH ~+0.5–0.9. Control **90.75 best** is
  the new session high (+0.45 over the 80-ep 90.21 headline). The 3.2M net
  was NOT at its epoch-ceiling at 80 ep. Cheapest real gain = just train
  longer. (Caveat: still drifts past peak in the long tail, 90.75@125 →
  90.34@150, even at lr_min_frac=0.001.)

## 2.6d Vanilla AdamW reference at 150 ep — concord trails by ~1.4, √v̂-shaped
Built `cifar_vanilla_adamw.py` (SAME WiderConvNet, fp32, plain nn layers,
real torch.optim.AdamW, 96 bits/param state; same loader/cosine/bsz=16).
150 ep:
| optimizer | state | precision | 150-ep best |
|---|---|---|---|
| AdamW lr=1e-3 wd=0.05 | 96 bit | fp32 | **92.19** / 92.09 final |
| AdamW lr=3e-3 | 96 bit | fp32 | ~80 @ ep55 (too high, killed) |
| concord packed-B | 32 bit | bf16 | 90.75 |
- **fp32 AdamW beats concord by ~1.4 pts** — the expected price of 3× state
  compression + bf16. BUT the gap is **concentrated in the low-lr
  fine-tuning tail**: neck-and-neck through ~ep80, then AdamW gains +3.8 in
  its last 60 epochs (88.1→92.2) while concord gains ~+0.4 (90.3→90.75).
- That tail divergence is the **√v̂ signature**: near convergence, Adam's
  per-coordinate step sizes fine-tune precisely; concord's UNIFORM scaling
  (the inert drift/gf denominator, §2.1) can't. This is the most direct
  evidence in the whole session that **per-coordinate `√v̂` preconditioning
  is concord's one real missing piece** — not coherence, not the chase.

## 2.6e NEXT (clear priority now)
- **Build factored `√v̂` preconditioning** (Adafactor v_row/v_col already
  half-built): `step = grad / (√v̂ + ε)` with the rank-1 row/col estimate,
  uniform→per-coordinate. This is the lever that should close the
  fine-tuning-tail gap to AdamW. Test on CIFAR-150 first (cheap, has the
  gap), THEN nanoGPT vs AdamW at matched compute (where per-coord scale
  varies more and the gap is bigger).
- Session highs: concord **90.75** (av1e3, 150 ep) vs AdamW-fp32 **92.19**.

## 2.7 Free deviation-preconditioner — state-less partial √v̂ (PROMISING, testing)
Key realization (user): because concord is UNwhitened (§2.1), the slow
deviation `D = s_slow − v_slow` carries the per-element gradient 2nd moment.
`D` is a stationary AR(1) on the increments (`D_t = β(D_{t-1} − U_t)`), so
`E[D²] = β²/(1−β²)·s² ≈ s²/(2(1−β))` — and for unwhitened `U≈ηG`,
`E[D_ij²] ∝ E[G_ij²]`. So rank-1 factoring `D²` gives a FREE per-element
`√v̂` estimate, zero persistent state, from the `s_slow`/`v_slow` already held.
- **exp6** (1-D): factored-D² recovers σ² at **0.984 log-corr over 390,000×
  range**; scale matches the predicted η²/(2(1−β)) (confirms the AR(1)
  factor-of-2, not 1/(1−β)). Factoring de-noises the single-sample D² ~3×
  (CoV 1.43→0.49); sqrt-precond CoV 0.24.
- **exp7** (1-D closed loop): preconditioning by `(est/mean)^p` is a STABLE
  PARTIAL whitener — measured heterogeneity exponent `k = 2/(1+2p)` exactly
  (p=0→2 SGD, 0.5→1, 1→0.67, 2→0.40), stable at all scales. CANNOT reach
  full Adam (k=0) at finite p: the est is built from the same D it whitens
  (self-limiting). That's precisely why Adam STORES v̂ — a gradient-derived
  2nd moment is decoupled from the weight trajectory and escapes the loop.
- **Build (zero kernel change):** `_deviation_precondition()` in the autograd
  backward (prototype_packed_b.py) divides grad_W by `(factored
  (s_slow−v_slow)² / mean + eps)^p` BEFORE the inert apply kernel (run with
  `precond_p=0`). Module global `_DEV_PRECOND["p"]`, set by `--dev_precond_p`.
  Branchless init-safe (D²+1e-12 floor → no-op when s_slow≈v_slow at init).
  Smoke: stable, learns, +25% backward overhead. Graph-compatible.
- **RUNNING:** CIFAR-150 `dev_precond_p ∈ {0.5,1,2}` (k=1.0,0.67,0.40) vs
  concord-90.75 (no precond) and AdamW-92.19. Q: how much of the 1.4-pt
  √v̂ tail gap does a state-less partial whitener recover? If p=2 claws back
  a meaningful chunk, this is per-coord preconditioning for free.
- **RESULT — it hurts, monotonically (don't ship it).** CIFAR-150
  dev_precond_p {0.5,1,2} → best {90.42, 90.28, 90.08} vs no-precond 90.75
  (and AdamW 92.19). Monotone worse with p, and even p=0.5 (k=1, ~no
  whitening) is below baseline.
- **Diagnosis (re-derives why Adam STORES v̂):** the tell is that even the
  mildest setting hurts — so it's not the whitening that's wrong, it's the
  NOISE. The single-sample rank-1 `est` has CoV 0.24 (exp6); preconditioning
  by a per-step-noisy scale injects step variance that costs more than the
  partial whitening buys. The deviation proxy recovers σ² in EXPECTATION
  (0.98 corr) but is too noisy PER STEP to precondition with. A stationary
  EMA estimate (= Adam's decoupled stored v̂) wouldn't have this noise.
- **Verdict on the √v̂ thread:** the 1.4-pt AdamW gap is real, tail-shaped,
  and IS per-coordinate preconditioning — but closing it needs the
  DECOUPLED STORED 2nd moment, not the free weight-deviation. The free proxy
  is a lovely 1-D result (exp6/7) that doesn't survive contact with per-step
  noise. NEXT (if continued): make the existing inert v_row/v_col Adafactor
  EMA the actual preconditioner (low-noise, k=0), fixing the §2.1 scale
  mismatch so it engages — NOT the deviation. (`--dev_precond_p` left in,
  defaults off.)
- **Bigger picture:** the orthogonal axis worth carrying is `√v̂` factored
  (Adafactor v_row/v_col, already half-built) — 10–100× per-coord spread,
  SNR-independent, the real SGD↔Adam gap. The decisive test for the whole
  optimizer is **nanoGPT vs AdamW at matched compute**, NOT more CIFAR
  (CIFAR's ceiling here is the 3.2M-param architecture, not noise handling).

## 2.8 SINGLE-SAMPLE factored g² — the fix the deviation needed (user, RUNNING)
The §2.7 verdict conflated two failure modes. User caught it: the deviation
doesn't just have noise, its estimate is SELF-DEGRADED. Two distinct sources:
- **Deviation D²** is shaped BY the precond it feeds → fixed-point algebra
  closes at `precond ∝ σ^(2p/(1+2p))`; it reads a degraded `σ^(2/(1+2p))`,
  so k = 2/(1+2p) NEVER reaches 0 for finite p. Self-limiting.
- **Single-sample g²** has `E[g²]=σ²` UNdegraded — gradient noise is set by
  minibatch sampling, precond-independent — so factoring IT reaches FULL
  whitening at finite p. Cost is per-step noise; the rank-1 factor (averaging
  a whole row+col) is exactly what knocks that down.
- **exp8** (1-D, both side by side): whitening law
  `E[U²] ∝ σ^k`, k_dev = 2/(1+2p) vs **k_grad = 2−4p**. Confirmed: at
  realistic spread (1e2 = 100× σ² ratio) single-sample p=0.5 → **k=0.21 ≈
  full Adam**, deviation stuck at k=1.01. Single-sample stable everywhere
  (even over-whitening k<0 at p>0.5, no blow-up). At extreme spread (1e4–1e6)
  the eps=0.01 floor caps it — but CIFAR's per-row/col spread is modest, in
  the regime where it reaches k≈0.
- **Build:** same `_deviation_precondition()`, new `src` mode. `src="grad"`
  (default): `base2 = grad_W²` (no unpacking — simpler & cheaper than dev).
  `--dev_precond_src grad|dev`. Smoke (3 ep): grad p=0.5 stable under graph,
  and AHEAD of off (ep3 73.87 vs 72.71) where the deviation was BEHIND.
- **CONFOUND (caught via banner):** first run (ba3lazv30) dropped the
  explicit `--precond_p 0` the baseline/dev-sweep used; `--precond_p`
  DEFAULTS to 0.5, so the apply-kernel's own sqrt-precond stacked on top of
  dev_precond = DOUBLE whitening (effectively over-whitened, ~p=1). p=0.5
  arm sank to 89.05, p=0.75 thrashed, p=0.25 segfaulted. Discarded. (User:
  "double-applying the same value" — for the p=0.5 arm it was literally 0.5
  in both knobs; proven independent because the p=0.75 arm's banner still
  read precond_p=0.5 not 0.75.) Relaunched clean (bx29a8s12) with
  `--precond_p 0`, p ∈ {0.5,0.25,0.125}; the precond_p=0 path also cleared
  the segfault.
- **RESULT — clean, monotone, decisive (state-less √v̂ is IMPOSSIBLE):**
  | p | k=2-4p | BEST | FINAL | tr_loss@150 |
  | none | 2 | **90.75** | 90.34 | 0.018 |
  | 0.125 | 1.5 | 90.24 | 89.88 | 0.025 |
  | 0.25 | 1.0 | 89.64 | 88.85 | 0.074 |
  | 0.5 | 0 | 88.75 | 87.32 | 0.181 |
  | AdamW | full | 92.19 | — | low |
  Monotone on BOTH axes: every increment of whitening costs accuracy AND
  raises train loss (underfit climbs 10x, 0.018->0.181, as k:2->0). No sweet
  spot; best family member is the degenerate no-whiten = baseline. p->0
  recovers 90.75 exactly.
- **Mechanism (airtight, bracketed both sides):** the whitening EXPONENT is
  not the lever (exp8: any k reachable, incl. k=0). The estimator NOISE is.
  Single-sample reaches k=0 but the per-step est noise scales WITH the
  whitening (the whitening IS the noisy division g/est^p) -> monotone
  underfit. AdamW whitens fully and fits to low train loss (92.19); the ONLY
  difference from gsc-p0.5 is v̂ = EMA over ~1000 steps. Same exponent, +3.4,
  entirely variance reduction. Deviation failed by UNDER-whitening (self-
  limiting), single-sample by UNDERFITTING (noise) — two different reasons,
  together proving the stored decoupled EMA is necessary.
- **Verdict:** the value of v̂ is its LOW VARIANCE via temporal averaging,
  NOT the whitening op. No eps rescues single-sample (raising eps cuts noise
  AND whitening in lockstep — same quantity; only the EMA decouples them).
  The single-sample IS the β2=0 limit of the stored v_row/v_col EMA: it
  fails on noise, and adding the EMA (β2->0.999) is literally the denoising
  fix = the half-built Adafactor estimator. So the NEXT build is confirmed
  correct AND motivated: make v_row/v_col (decoupled, low-noise) the
  preconditioner, fixing the §2.1 eps=1 scale-swamp so it engages. That is
  the only un-exhausted route to the 1.4-pt gap. `--dev_precond_src grad|dev`
  both left in, default off.

## 2.7 New CLI flags / knobs added this thread (cifar_concord_packed_b.py)
`--eps`, `--v_scale`, `--precond_p`, `--gf_consol`, `--cohpre`,
`--eps_warm_steps/--eps_final` (eps schedule, graph-compatible: eps is now
a device tensor `_eps_buf` like lr), `--diag_steps` (denominator-term
instrumentation). All default to the headline behaviour when unset.

## 2.9 WHERE does v-hat change the weights? + is the difference LOW-RANK?
Mechanism-localization study (user-designed). Build: `cifar_vmode_fork.py`
(clean fp32 WiderConvNet) + `analyze_vmode.py`. Three optimizers IDENTICAL
except the 2nd-moment denominator: none=`m_hat` (momentum SGD), rank1=
`m_hat/sqrt(v_row⊗v_col)` (Adafactor), full=`m_hat/sqrt(v_elem)` (Adam).
Same init -> FIRST 10 ep run in none mode (shared trajectory) -> common W10
-> fork each (fresh opt state, own peak LR: lr_none=0.02, lr_adam=1e-3) to
150 ep. full mode = AdamW validator (should hit ~92). Saves per-layer final
W + coh(=EMA[g]^2/EMA[g^2], the garbage factor) + mode-independent coh_warm
from the shared warmup. RESUMABLE: loads W10 from {out}_warm.pt (skips
warmup), skips any already-saved arm -> a crash costs only the in-flight arm
(machine rebooted mid-run once; recovered with zero lost progress).
- **User hypothesis (the sharp one):** the full-vs-none weight difference
  dW is PRIMARILY LOW-RANK and DETECTABLE -> if dW is low-rank AND sits in
  W's observable top-singular subspace, the v-hat correction is a cheap
  low-rank add-on you could bolt onto SGD/concord (below even Adafactor's
  rank-1-per-step cost). analyze_vmode tests it directly: stable-rank /
  erank / r90 of dW vs full rank, top-k energy fraction, and dW's alignment
  with W's top-m subspace vs the random-matrix baseline (m/R).
- **Three questions answered per pair:** are the most-moved weights big/small
  |W|? signal/garbage coh? loading onto important (big-s) singular modes?
- **Preliminary (THROWAWAY 5-ep SMOKE only -- weights immature, treat as
  direction not result):** full-none pooled already shows dW stable-rank
  ~11% of full, top-5 modes = 39% of the difference, 27% of dW in W's top-5%
  subspace vs 5% random (5.4x), corr(a_k^2, s_k)=+0.49 (loads on big-s modes),
  while |W| and coh are FLAT. I.e. both halves of the hypothesis (low-rank +
  aligned/detectable) register even before divergence matures. Real 150-ep
  run (b9uvu5vx6) pending; analyzer + 5-panel figs validated and waiting.
- If the real run confirms: the actionable read is a LOW-RANK correction in
  W's top-singular subspace recovers most of Adam's edge -> concord carries a
  rank-k (not full, not even rank-1-per-coord) v-hat surrogate. Follow-ups:
  (a) align dW's top dirs with the gradient-2nd-moment eigenvectors (stronger
  "detectable from grad stats"); (b) checkpoint W over training to see if
  low-rank holds throughout or only at convergence.

## 2.10 SECOND miser axis: TEMPORAL -- freeze settled weights, reclaim v-hat
User idea: you don't have to hold all the variance at once. Freeze a weight
once it's resolved ("all signal, no noise"), reclaim its v-hat memory, "burn
down the stuck pieces one by one." Limit = coordinate descent / working-set:
peak optimizer memory = active-fraction*params, trading passes for peak mem
(exactly the deal to fit a model sideways into a card).
- **Detector nuance (important):** "settled" is NOT high coh. coh=EMA[g]^2/
  EMA[g^2] COLLAPSES to ~0 at convergence (settled weight's gradient is pure
  noise about zero). The right freeze signal is WEIGHT-TRAJECTORY STABILITY --
  the value stopped moving -- = the accumulator deviation D=(s_slow-v_slow)
  going quiet. The deviation we built (too noisy to PRECONDITION with, 2.7/2.8)
  is exactly the right SETTLE detector -- a role it actually fits.
- **Two axes compose (multiplicatively):** spatial low-rank v-hat (2.9, few
  directions) x temporal freezing (few weights active at once) -> peak v-hat
  memory ~ active_fraction * rank, not params * full.
- **UNIFICATION hypothesis (the prize):** if the LATE-settling (always-active)
  weights are the SAME as the top-singular-subspace where v-hat matters, the
  two axes are one structure: carry precise low-rank v-hat only for the
  persistently-active top subspace, freeze the settled tail. Test =
  corr(settle_epoch, SVD-leverage) > 0.
- **Built + validated (all waiting on the run):**
  - `cifar_vmode_fork.py`: + trajectory snapshots (ANALYZE weights every
    `--ckpt_every` 15 ep per fork) -> settling timeline; + full state_dict per
    arm -> reconstruction eval. RESUMABLE (W10 + skip-done-arms).
  - `reconstruct_sweep.py`: ACCURACY-rank curve -- smallest rank-k of dW that
    buys back >=90% of the Adam gap -> the bits answer (rank-k v-hat = Nx
    cheaper than full per-element). [validated on synthetic ckpts]
  - `settle_analysis.py`: f(e) settle curve, mean active-fraction (= time-avg
    variance budget), per-mode (does Adam settle faster than SGD?), and
    corr(settle_epoch, SVD-leverage) for the unification test. [validated]
- **Run status:** `b485ckptc` (150 ep, none->full->rank1, ckpt_every 15).
  Survived TWO recoveries: a machine reboot mid-run, and a post-kill CUDA-init
  segfault (exit 139, transient GPU state after TaskStop -> cleared, retried).
  W10 + resumability meant zero lost progress both times.
- Morning deliverable: run `analyze_vmode`, `reconstruct_sweep` (none->full,
  none->rank1, rank1->full), `settle_analysis`; write the two-axis verdict +
  the unification test; figures in vmode_fig_*.png + vmode_fig_settle.png.
- **ROBUSTNESS (box is flaky -- soft-crashed at none ep13, ep47):** harness
  now does INTRA-ARM resume -- atomically checkpoints model+opt+traj every
  `--resume_every` (10) ep to {out}_{mode}_resume.pt; on restart, resumes the
  in-flight arm from its last ckpt (skips warmup via W10, skips done arms).
  Launch via an AUTO-RESTART WRAPPER that re-runs through crashes (incl. the
  post-crash CUDA-init segfault) with a 20s cooldown until all 3 arms saved.
  RECOVERY (cold, after a hard reboot) -- just re-run this exact line:
  ```
  cd /c/concord && for i in $(seq 1 80); do \
    python cifar_vmode_fork.py --epochs 150 --warmup_epochs 10 \
      --modes none,full,rank1 --out vmode --seed 0 --ckpt_every 15 \
      --resume_every 10; \
    [ -f vmode_none.pt ] && [ -f vmode_full.pt ] && [ -f vmode_rank1.pt ] \
      && break; sleep 20; done
  ```
  It picks up from W10 + the last per-arm resume ckpt with zero/minimal loss.

## 2.11 ROOT CAUSE (user): CIFAR is ZERO-BAYES-ERROR -> v-hat is IDLE here
The fork run's first arm settled it: `none` (clean fp32 momentum-SGD, NO v-hat)
hit **92.89 best / 92.55 final** -- BEATING AdamW-fp32 (92.19) and crushing
concord-packed (90.75), with no second moment at all.
- **The reframing:** v-hat exists to handle IRREDUCIBLE, per-coordinate-
  HETEROGENEOUS gradient noise. CIFAR-10 (clean labels) is ~realizable / zero
  Bayes error: there's a W* with train loss ~0, gradients become consistent
  near it, no persistent noise structure for v-hat to adapt to. So v-hat idles
  and well-tuned momentum-SGD wins. This RETROACTIVELY EXPLAINS the whole v-hat
  thread: coh-gating (2.4), deviation-precond (2.7), single-sample (2.8), the
  fork (2.9) ALL found v-hat neutral/harmful -- not bad ideas, WRONG REGIME.
  The 1.4-pt concord->AdamW gap was implementation (quantization + eps=1 inert
  precond + the chase), NOT the missing 2nd moment.
- **It's the textbook split:** SGD>=Adam on clean vision; Adam>>SGD on LMs.
  Axis = Bayes error / non-realizability + heavy-tailed (heterogeneous) grads.
  LM next-token is stochastic (Bayes error) AND heavy-tailed (rare tokens spike
  specific coords = the per-coord heterogeneity v-hat exploits). CIFAR has
  neither.
- **Miser's corollary:** zero-Bayes-error task -> don't pay for v-hat at all
  (concord = int-stored momentum-SGD, done). The low-rank/freezing variance-
  budget question is only LIVE where v-hat matters (nonzero Bayes error).
- **Tests built/planned:** (a) `--label_noise q` knob added to
  cifar_vmode_fork.py (per-batch random TRAIN label flip -> injects Bayes
  error; test/val clean; default 0 = no-op). Cheap probe: does `none`-vs-`full`
  cross over as q rises? CAVEAT: label noise is ~uniform across coords, may not
  make strong per-coord heterogeneity -> partial signal. (b) DEFINITIVE:
  nanoGPT vs AdamW -- both Bayes error AND heavy-tailed heterogeneity; the
  regime v-hat was built for and where concord's whole premise gets stress-
  tested. The decisive test is NOT more CIFAR.
- Current control run (bz9he8pnc, zero noise): none done 92.89; full + rank1
  finishing -> locks the in-run "v-hat idle at zero Bayes error" anchor + the
  descriptive low-rank/settling data.

## 2.12 ROOT-of-root (user): MEMORIZATION is strictly dominant when it fits
The deeper reason beneath 2.11. Overparam (3.2M params) + realizable (clean
CIFAR) => a MANIFOLD of zero-train-loss interpolators, and the loss is
INDIFFERENT between the memorizing and generalizing ones. The optimizer is
never FORCED to extract signal -- it breaks a tie by implicit bias. So
optimizer choice (v-hat, variance budget) washes into idiosyncrasy. Proof: the
fork clusters none 92.89 ~ full ~91 ~ concord 90.75, all within ~2 pts; a task
where the optimizer mattered would show a chasm.
- **Consequence 1 (changes the noise test):** in an overparam net, FIXED label
  noise does NOT escape memorization-dominance -- 3.2M params >> 50k labels, so
  the net memorizes the flipped labels (fits the noise, still dominant). The
  `--label_noise` knob is valid ONLY because it RESAMPLES the flip per batch ->
  the noisy fraction is a MOVING TARGET -> non-memorizable -> genuine
  irreducible gradient noise -> forces v-hat to work. per-batch-resample (not
  fixed-flip) is the whole point.
- **Consequence 2 (the coupling):** truest regime = data >> capacity (LMs),
  where even the SIGNAL is non-memorizable (can't store the corpus in weights)
  -> forced to compress/generalize everything. AND: under-memory-pressure
  (model too big for card) <=> task-too-big-for-model (forced generalization)
  <=> v-hat load-bearing. ONE axis. nanoGPT is the ONLY regime where concord's
  premise (save optimizer memory WHILE generalizing) is even coherent. Not "a
  harder benchmark" -- the only meaningful one.
- **Consequence 3 (caveat on 2.9 analysis):** in the memorization regime, dW
  between optimizers = "which interpolator did each implicit bias pick," NOT
  "what did v-hat correct." So the CIFAR low-rank/settling structure measures
  memorization idiosyncrasy, not signal-extraction dynamics. Complete it as the
  negative anchor; read with that asterisk.
- **Cheap CIFAR knobs to force generalization (escape "fits in the space"):**
  (a) per-batch-resampled label noise (built, the right probe for the noisy
  fraction); (b) UNDERPARAMETERIZE (model < data information -> must compress) --
  but CIFAR's signal is easy, may plateau without stressing v-hat. The real
  answer stays nanoGPT.

## 2.13 REHABILITATION (user): the "underfit" washers are NOT useless
Corollary of 2.12 + a walk-back. "We cannot discard the underfit version that
washes out incoherent gradients." A coordinate with an INCOHERENT gradient
(high per-step variance) is one that requires MEMORIZATION to fit; a whitener
that shrinks its step DECLINES to memorize it. So the whole washing family --
coherence-gating (2.4), deviation-precond (2.7), single-sample (2.8) -- are
MEMORIZATION FILTERS, not broken optimizers. They "underfit" on clean CIFAR
because there memorization == generalization (zero Bayes err), so refusing to
memorize only costs. SAME mechanism, SAME metric (clean val acc), OPPOSITE
verdict by regime. I condemned a regularizer on the one task where
regularization can't help.
- **Implicates concord itself:** its 90.75 vs SGD 92.89 gap is partly
  quantization washing fine gradient detail = ALSO a memorization-resistor.
  So "make concord fit CIFAR like SGD" may be the WRONG goal. SUCCESS METRIC
  must move: from "matches SGD on clean CIFAR" to "holds clean val under
  NON-memorizable load (resampled noise / data>>capacity)".
- **Decisive test (no new metric needed -- clean val acc under resampled
  noise):** `single` mode added to cifar_vmode_fork.py (= the 2.8 per-step
  rank-1 g^2 whitener, the aggressive washer). Sweep `none` (no wash) vs `full`
  (gentle EMA wash) vs `single` (aggressive wash) at label_noise q in
  {0,.2,.4}. PREDICTION: q=0 -> none wins (memorize=win); as q rises, none
  chases resampled noise and its CLEAN val degrades while the washers refuse it
  and hold -> washing modes overtake. If so, the "underfit" IS the
  generalization win, and the §2.8 verdict is reversed. (`single` additive;
  none/full/rank1 untouched -> running control safe.)

## 2.14 nanoGPT: RAW CONCORD MATCHES AdamW (and self-regularizes) -- the pivot
First test in the regime that matters (LM = non-realizable, next-char
stochastic). Built `src/nanogpt.py` (10.8M char GPT: n_layer=n_head=6,
n_embd=384, block=256, bias=False Linears, untied lm_head) + `src/train_nanogpt.py`
(every nn.Linear -> ConcordLinearPackedB, eps=1 shipped recipe, step fused in
backward, per-step rebalance; aux AdamW on embeddings+LN; warmup+cosine). Char
tiny-shakespeare (1.1M chars), 5000 iters, bsz=64, seed 0, NO dropout.
- **Controlled head-to-head (identical model/data/schedule):**
  | optimizer | best val | final val | final train |
  | Concord (raw, int-packed) | **1.5364** | 1.544 | 1.18 |
  | AdamW (fp32) | **1.5334** | 4.361 | 0.087 |
- **(1) Best-val TIE** (1.5364 vs 1.5334, ~noise): raw Concord with NO per-coord
  v-hat matches AdamW's PEAK generalization on a from-scratch transformer LM.
  Refutes "Concord trails AdamW" -- in the v-hat-relevant regime it keeps up.
- **(2) AdamW MEMORIZED, Concord didn't:** AdamW train->0.087 (memorized 1M
  chars), val EXPLODED to 4.36 (worse than random ~4.17). Concord held: train
  1.18, val 1.54. Concord's int-storage+chase+washing = FREE regularization
  that resists memorization. §2.13 made real: the "underfit washer" is the
  ROBUST one where memorization is the failure mode. (Concord uses the full
  schedule, val improves to ~iter4250; AdamW peaks ~iter1000 then overfits.)
  ~110ms/iter; 2.3GB Concord vs 2.7GB AdamW.
- **Caveats:** NO dropout on either (standard char-nanoGPT uses 0.2 -> stops
  AdamW's blowup, gets ~1.47). char-shakespeare is MEMORIZABLE (AdamW proved
  it). So this is "memorization possible + AdamW does it," not yet the pure
  data>>capacity regime. NEXT rungs: (a) dropout-matched both -> AdamW's tuned
  best vs Concord; (b) DATA>>CAPACITY (bigger corpus, neither memorizes) = the
  pure optimization-quality test of Concord-without-v-hat; (c) washing-spectrum
  (none/full/single) here.
- VERDICT: first contact in the right regime is a WIN -- raw Concord ties
  AdamW's best AND is far more overfit-robust. (src/nanogpt.py +
  src/train_nanogpt.py committed; nanogpt_data/ gitignored.)
- **Rung #1 -- DROPOUT-MATCHED (user). Grid (5000 iter, seed 0):**
  | opt | dropout | best val | final | train |
  | Concord | 0.0 | **1.5364** | 1.544 | 1.18 |  (stable, final~best)
  | Concord | 0.2 | 1.81  | 1.80 | -- |   (dropout HURTS: over-regularized)
  | AdamW | 0.0 | 1.5334 | 4.36 | 0.087 | (memorized)
  | AdamW | 0.2 | **1.4713** | 1.70 | 0.63 | (best@~it1750, still tail-overfits)
  EACH AT ITS BEST: **AdamW 1.4713 vs Concord 1.5364 -> AdamW +0.065 nats** =
  the honest v-hat cost in the LM regime. But: AdamW's 1.47 needs dropout 0.2
  AND early-stopping (two regularizers, must catch the min); Concord's 1.5364
  needs NEITHER (final~best, stable) and REJECTS dropout (self-regularizes ->
  double-damp underfits to 1.81). Dropout helped only AdamW. Concord trades
  ~0.065 for tuning-free + overfit-robust + ~1/3 optimizer-state bits.
  (Caveat: Concord's chase lr=0.05 NOT swept for nanoGPT -- a sweep may close
  some of the 0.065.) NEXT: rung #2 data>>capacity (neither memorizes = pure
  optimization-quality test); rung #3 washing-spectrum (none/full/single).
