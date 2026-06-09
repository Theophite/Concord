# CPU dynamics experiments: Concord vs AdamW

Small-scale experiments probing the dynamics of the Concord winner rule against AdamW,
run on CPU via `concord_ref.py` — a real-valued reference of the update rule in
`concord/packed_b.py` (stochastic rounding makes the integer kernel equal this rule in
expectation; order of operations, gating, schedules, and init convention mirror the
kernel line-by-line). **Caveats up front**: fp32 reference (no SR/integer effects, no
bf16), tiny MLPs, 3 seeds, no per-arm lr tuning. These probe *dynamics*, not SDXL-scale
behavior.

Scripts: `exp1_lr_sweep.py`, `exp2_signal_noise.py`, `exp3_mnist.py`,
`exp4_label_noise.py`. MNIST IDX files go in `data/` (ossci mirror).

## Exp 1 — how much it moves weights, by lr (`exp1_lr_sweep.png`)

Noisy teacher-student regression (Bayes floor 0.25), 800 steps, relative Frobenius
displacement of the 2D weights from init.

| peak lr | AdamW disp | Concord live | Concord deploy | tail loss A / C |
|---|---|---|---|---|
| 1e-4 | 0.107 | 0.155 | 0.155 | 0.266 / 0.258 |
| 1e-3 | 0.243 | 0.356 | 0.357 | 0.254 / 0.254 |
| 1e-2 | 1.009 | 0.713 | 0.711 | 0.254 / 0.254 |
| 5e-2 | 3.091 | **NaN** | **NaN** | 0.260 / NaN |

- At small/moderate lr Concord moves weights **~1.5× more** than AdamW for the same
  loss: β1 = 0 (no heavy-ball averaging) and the step cap 10 admit larger effective
  per-step motion than Adam's bias-corrected step.
- At high lr the friction bites and Concord moves *less* (0.71 vs 1.01).
- At lr = 5e-2 Concord **diverges while AdamW survives**: the evaporation term
  `u ← u − lr·κ·(1−coh)·u` has a hard linear-stability ceiling **lr·κ ≲ 2**
  (κ = 50 → lr < ~4e-2). Verified: κ=0 at the same lr is stable; lr=3e-2 (lr·κ=1.5)
  trains; lr=4.5e-2 (2.25) is marginal; 5e-2 (2.5) blows up. The validated configs sit
  two orders inside the bound (nanoGPT 5e-4·50 = 0.025; SDXL 7.5e-5·50 = 0.004). In the
  integer kernel the int16 clamp would saturate instead of NaN-ing, but the dynamics
  would be equally broken.
- The init convention (all mass in the velocity, `load_weights`) makes **warmup
  load-bearing**: friction is lr-proportional, and the warmup keeps it off while the
  chase consolidates the initial weight into `P` over the first ~1/α steps.

## Exp 2 — response to controlled gradient streams (`exp2_signal_noise.png`)

One 64×64 weight, hand-fed gradients `g = m·G + s·ξ`, constant lr 1e-3, Concord's own
noise injection off. Displacement from the post-consolidation reference (step 100):

| stream | AdamW | Concord live | Concord deploy | mean coh |
|---|---|---|---|---|
| pure noise (0, 1) | 2.83 | 2.41 | 2.40 | 0.07 |
| buried signal (0.3, 1) | 2.89 | 2.46 | 2.45 | 0.07 |
| pure drift (1, 0) | 121.5 | 108.3 | 108.7 | **0.26** |

Selectivity is directional but **mild at shipped settings**: ~15% less random-walk
displacement under pure noise, equal drift tracking. The gate is the reason it is not
stronger — see below.

## The drift-cancel coefficient is ~2× too small for the mass-preserving leak

Under pure drift the gate should read coh → 1; it reads **0.26–0.29** (and 0.55 even
with gates held open and κ=0, i.e. under the exact assumptions of the C\* derivation).
Root cause, visible in the recursion: the leak is mass-preserving (`A += l` **and**
`S −= l`), so the telescope `d = S − A` drains at **2·α_v**, but
`compute_drift_cancel_C` derives with ρ = α_v. The shipped C\* therefore under-predicts
the drift, the residual swallows ~half the signal, and drift coherence saturates near
0.5 instead of 1 (lower still once the gate floors and friction shorten the effective
lags). Steady-state analysis of the recursion predicts coh ≈ 0.42 at open gates;
measured 0.55 (v̂ rank-1 inexactness accounts for the gap).

Empirical check (pure drift, full winner config): shipped C\* → coh 0.29; **2×C\* →
coh 0.96**, while the pure-noise control stays at 0.12. The exact fixed-point value is
`C* = L·2α_v/(1 − 2α_v) ≈ 0.018036`.

**The fix is applied on this branch**: `compute_drift_cancel_C` in `concord/packed_b.py`,
`dist/concord_winner/concord/packed_b.py`, and `notebook/src/prototype_packed_b.py` now takes
`mass_preserve=True` (matching the layer default `mass_preserve_v=True`) and returns the
corrected value; the legacy formula is preserved under `mass_preserve=False`, and the
wrapper call sites pass their own `mass_preserve` flag through. (The older
`notebook/src/concord_triton_fused.py` lineage is untouched — different codepath, leak semantics
not re-verified.) Same shape of fix as the one already recorded in that docstring's
history (the previous constant was 11× off in the other direction). Porting to
`concord-integration` is the same edit in
`modules/util/optimizer/concord/prototype_packed_b.py`.

## After the fix: re-run results

Exp 2 rerun with the honest gate (figure regenerated): pure-drift coherence 0.84 in the
full winner config, and the gate's character changes — **drift tracking is now faster
than AdamW** (157 vs 122 displacement; friction shuts off on coherent motion) at the
cost of slightly more noise walk (2.69 vs 2.41). Displacement selectivity (drift/noise
ratio): AdamW 43, shipped gate 45, fixed gate **58**.

MNIST, matched κ = 50, deploy accuracy (3 seeds):

| regime | legacy C\* | fixed C\* |
|---|---|---|
| clean (exp 3) — dissip | 95.84 ± 0.08 | **96.02 ± 0.10** |
| clean (exp 3) — winner | 95.79 ± 0.12 | **95.99 ± 0.12** |
| label noise, no-overfit budget (exp 4) — dissip | 93.20 ± 0.36 | **93.34 ± 0.42** |
| label noise, overfitting regime — dissip | **90.06 ± 0.33** (24.6% mem.) | 89.66 ± 0.39 (27.8% mem.) |
| label noise, overfitting regime — winner | **90.00 ± 0.27** (24.2% mem.) | 89.74 ± 0.43 (26.5% mem.) |

And the κ frontier in the two regimes (dissip arm, deploy):

| κ | clean, legacy | clean, fixed | noisy-overfit, legacy | noisy-overfit, fixed |
|---|---|---|---|---|
| 50 | 95.84 | **96.02** | 90.06 (24.6%) | 89.66 (27.8%) |
| 150 | 94.71 | 95.11 | **90.81 (—)** | 90.53 (17.5%) |

The pattern is consistent and instructive:

1. **At matched κ, the honest gate wins wherever coherent = generalizing** (clean task,
   short-budget noisy task) and **loses where coherent = memorizing** (the overfitting
   regime): memorization gradients are per-weight coherent over many epochs, so a
   sharper drift detector waves them through. The miscalibrated gate's half-blindness
   was acting as accidental extra skepticism — the κ = 50 anti-memorization result of
   exp 4 was partly *tuned around the bug*.
2. **κ re-tuning moves both gates up in the noisy regime** (κ = 150: legacy 90.81,
   fixed 90.53 — both beating every κ = 50 cell) and costs the clean task. The
   dissipation strength, not the gate calibration, is the regime knob; the gate
   calibration decides how cleanly friction exempts whatever the gate calls signal.
3. Practical conclusion for the repo: the fix makes the gate measure what its derivation
   says it measures (and what the variance-map diagnostic assumes). Whether it improves
   the *validated* tasks is a different question — nanoGPT/SDXL sit in a different
   regime (real Bayes error, heavy tails, no label-noise memorization pressure), the
   winner's κ/floors were tuned against the dull meter, and adopting the fix should be
   gated on the same-seed nanoGPT A/B in the real harness (GPU), ideally with a small κ
   sweep. The CPU evidence says: expect the meter to read correctly, expect the tuning
   optimum to shift.

## Exp 3 — clean MNIST ablation (784–256–10 MLP, 2 epochs, lr 1e-3, 3 seeds)

| arm | test acc (live) | test acc (deploy) |
|---|---|---|
| AdamW | 95.08 ± 0.08 | — |
| Concord bare | **96.55 ± 0.19** | **96.56 ± 0.16** |
| + dissipation | 95.86 ± 0.10 | 95.84 ± 0.08 |
| + noise (winner) | 95.81 ± 0.11 | 95.79 ± 0.12 |
| winner, 2×C\* | 96.00 ± 0.07 | 96.01 ± 0.11 |

Every Concord arm beats AdamW at matched lr/schedule — but the ablation ordering is
**reversed** relative to nanoGPT: on a clean task the dissipation costs ~0.7% and the
noise adds nothing. This is exactly the repo's own regime note
(`notebook/notes/SESSION_NOTES_2026-05-29.md`): near-zero-Bayes-error tasks don't pay for the
gate; memorization *is* generalization there.

## Exp 4 — MNIST + 30% label noise, overfitting regime (4k subset, 25 epochs)

Clean-test accuracy; "memorized" = train accuracy on the *wrongly-labeled* examples
(chance = 10%, higher = fitting noise):

| arm | live | deploy | noise memorized |
|---|---|---|---|
| AdamW | 89.15 ± 0.13 | — | 19.3% |
| Concord bare | 87.99 ± 0.38 | 88.15 ± 0.43 | **42.9%** |
| + dissipation | 89.87 ± 0.51 | **90.06 ± 0.33** | 24.6% |
| + noise (winner) | 89.83 ± 0.43 | 90.00 ± 0.27 | 24.2% |
| winner, 2×C\* | 89.58 ± 0.53 | 89.58 ± 0.31 | 26.2% |

With genuine overfitting pressure the ordering **flips exactly as designed**: bare
memorizes 43% of the wrong labels and falls below AdamW; the dissipation cuts
memorization to 24% and takes the lead; deploy beats live consistently (+0.1–0.2). At
the shorter exp-3-style budget (10k × 6 epochs) nobody memorizes (~10% = chance) and
the orderings match exp 3 — the dissipation only pays once there is noise to reject
*and time to memorize it*. The fluctuation half adds nothing beyond the dissipation
here (winner ≈ dissip) — consistent with the repo's own caveats that the σ gain is
small, single-seed, and possibly BatchNorm-mediated on CIFAR (this MLP has no BN).

## Summary

1. **Same loss, different path**: Concord moves weights more than AdamW at fine-tune
   lrs (no first moment, cap 10), less at high lr (friction), and has a hard stability
   ceiling **lr·κ ≲ 2** that AdamW doesn't share.
2. **The gate works but is half-blind as shipped**: drift coherence saturates ~0.3
   because C\* misses the factor 2 from the mass-preserving leak; refitting it makes
   the meter read correctly (0.96 drift / 0.12 noise).
3. **The fluctuation–dissipation pair is regime-dependent, exactly as the repo's notes
   claim**: it costs accuracy on clean tasks (bare wins clean MNIST by +0.7%) and earns
   it back with interest under label noise + overfitting pressure (+1.9% over bare,
   +0.9% over AdamW, with 43% → 24% noise memorization).
4. **Deploy ≥ live** in every regime tested, with the margin appearing exactly when
   gradients are noisy — consistent with the "ship the posterior mean" story.

## Exp 5 — the dissipation curve: κ\*(noise) (`exp5_kappa_noise_curve.py`)

Full grid: label-noise fraction ρ ∈ {0, 10, 20, 30, 45%} × κ ∈ {0…800}, overfitting
regime (4k × 25 epochs), fixed-C\* gate, fluctuation off (σ = 0) so the curve isolates
the dissipation. Deploy clean-test accuracy, 3 seeds (`exp5_results.json` has every cell;
figure `exp5_kappa_noise_curve.png`).

| label noise ρ | κ\* | acc at κ\* | acc at κ=0 | Δ |
|---|---|---|---|---|
| 0% | **0** | 93.66 | 93.66 | — |
| 10% | **100** | 92.72 | 92.42 | +0.30 |
| 20% | **200** | 91.72 | 90.53 | +1.19 |
| 30% | **400** | 90.77 | 88.27 | +2.50 |
| 45% | **400** | 89.53 | 83.01 | +6.52 |

Findings:

1. **κ\* rises roughly linearly with noise, then saturates**: κ\* ≈ 1000·ρ up to
   ρ ≈ 30%, flat at ≈ 400 beyond. An odds-law extrapolation (κ\* ∝ ρ/(1−ρ), predicting
   ~740 at 45%) was tested and falsified: κ = 600 and 800 at 45% both score below
   κ = 400. There is a maximum useful friction — past it, draining signal costs more
   than the marginal noise protection buys, even with nearly half the labels wrong.
   The plateau sits well inside the stability ceiling (lr·κ < 2 → κ < 2000 at this lr):
   the optimum is loss-driven, not stability-driven.
2. **At fixed κ, memorization is nearly noise-level-independent** (κ = 100 → ~19–23%
   of wrong labels memorized at every ρ; κ = 400 → ~12%): the dissipation sets a
   *memorization-rate budget* — how fast slow coherent drift may consolidate — rather
   than responding to how much noise there is.
3. **The risk is asymmetric.** Over-damping on clean data is cheap (κ = 400 costs
   −1.5%); under-damping on noisy data is expensive (κ = 0 at 45% costs −6.5%). With
   unknown noise levels, err high.
4. **κ\* = 0 exactly at zero noise** — every κ > 0 monotonically hurts a clean task in
   this regime. The dissipation is pure insurance; its premium is only worth paying
   when there is noise to reject. (The winner's κ = 50 reads as a mild-noise prior —
   sensible for the heavy-tailed LM/diffusion streams it was tuned on.)

## Exp 6 — autotuning the dissipation from an estimated noise level (`exp6_autotune.py`)

Closing the loop on exp 5: κ\* indexed by a *measurable* statistic instead of
ground-truth ρ. Three iterations, each informative (`exp6_results.json`,
`exp6_ramp_results.json`, `exp6_v2_results.json`, figure `exp6_autotune.png`):

1. **v1 — continuous control from raw-gradient coherence: the meter fails, the loop
   "works" anyway.** A per-layer EMA-gradient coherence statistic saturates at
   η ≈ 0.995–0.999 for every ρ (at batch 128 the raw gradient stream is almost entirely
   minibatch noise), yet closed-loop accuracy matched or beat the oracle for ρ ≤ 30%.
   The mechanism was not noise tracking but the emergent *time profile*: κ low early,
   high late.
2. **The schedule alone is not enough.** Meter-free linear κ ramps (0 → K over
   training) reproduce the low-noise wins but lose badly at high noise (45%: best ramp
   87.55 vs oracle 89.53) — memorization starts early, so friction must arrive early,
   which requires sensing. And the late-κ observation stands on its own: κ = 400
   applied after epoch 3 costs only −0.2 on a clean task vs −1.5 applied from step 0 —
   most of the dissipation's clean-task tax is paid early in training.
3. **v2 — the right meter is the gate itself.** Mid-training *gate coherence*
   (`mean coh`, velocity-side — after the telescope has integrated out minibatch noise)
   discriminates label-noise levels cleanly and monotonically: coh(epochs 3–8, κ=50) =
   0.387 / 0.314 / 0.288 / 0.274 / 0.256 for ρ = 0/10/20/30/45%, spreads ~10–50×
   smaller than the separations. **Probe-then-commit**: train at the default κ = 50 for
   epochs 0–8, read coh over epochs 3–8, commit κ from the piecewise-linear
   (coh → κ\*) table for the rest:

| ρ | committed κ (oracle κ\*) | autotuned deploy acc | oracle fixed | κ=0 |
|---|---|---|---|---|
| 0% | 2 (0) | 93.52 ± 0.05 | 93.66 | 93.66 |
| 10% | 103 (100) | 92.71 ± 0.19 | 92.72 | 92.42 |
| 15% (held out) | 151 (—) | 92.11 ± 0.10 | — | — |
| 20% | 205 (200) | 91.64 ± 0.45 | 91.72 | 90.53 |
| 30% | 381 (400) | 90.54 ± 0.24 | 90.77 | 88.27 |
| 38% (held out) | 400 (—) | 89.59 ± 0.51 | — | — |
| 45% | 400 (400) | 87.99 ± 0.68 | 89.53 | 83.01 |

The meter recovers oracle κ\* almost exactly (2/103/205/381/400 vs 0/100/200/400/400)
and interpolates sensibly at held-out noise levels, with no knowledge of ρ. Accuracy is
within ~0.2% of the oracle through ρ = 30% while paying only −0.14 on the clean task.
The remaining gap at extreme noise (−1.5 at 45%) is the probe's price: eight epochs at
κ = 50 lock in early memorization that the late commit can't undo — the same
"friction must arrive early under heavy noise" constraint the ramp test exposed. A
shorter probe, a higher default-κ probe, or a continuous controller on the gate-coh
meter (with its κ-feedback compensated) are the obvious next iterations.

Caveats: the (coh → κ\*) table is calibrated on this task/architecture/schedule — the
transferable object is the *procedure* (the gate's own mid-training coherence is a
reliable, nearly-free noise meter; map it through a κ\* curve measured once per domain).
In the real optimizer, mean gate coherence is available per layer at zero extra cost —
the kernel already computes coh per weight; aggregating it is one reduction.

## Rolled into the package (with the C\* rescale)

The autotuner now ships in `concord/packed_b.py` alongside the recalibrated gate it
depends on:

- `gate_coherence_from_fields(...)` / `measure_coherence(layer)` — the kernel's Wiener
  gain computed host-side from the packed state (scale-invariant, so the exponents
  cancel and it never touches the kernel; call it occasionally, zero per-step cost).
- `DissipationAutoTuner(layers, probe_start, probe_end, table)` — probe-then-commit:
  train at the default κ through the probe window, read the mean gate coherence, commit
  κ once from a calibrated piecewise-linear (coh → κ) table.

Validated by `test_autotuner_parity.py` (CPU, execs the shipped source): the package
formula equals the reference gate including scale invariance under random exponents,
`measure_coherence` round-trips the packing, and the tuner's probe/commit/interpolation
match the exp-6 logic that produced the v2.1 table above. Caveats are in the class
docstring: the table is task-calibrated; the probe window must match the calibration
window and sit after the init-consolidation transient; under a captured CUDA graph,
gf_consol is baked at capture time (probe eagerly, then capture — or port κ to a device
tensor like lr/σ/floors when wiring into the Stage-3 graph).

## Exp 7 — coherence-gated momentum (β1) under autotuned dissipation (`exp7_beta1_sweep.py`)

The winner ships β1 = 0 (the kernel notes ungated momentum diverges; the gated term was
left off). Both decisions predate the C\* rescale, so: sweep β1 ∈ {0…0.8} with the v2.1
autotuner active, clean and 30%-label-noise regimes, 3 seeds
(`exp7_results.json`, `exp7_joint_results.json`).

| β1 (coh-gated) | clean deploy | 30%-noise deploy | noise memorized |
|---|---|---|---|
| 0.00 | 93.53 ± 0.05 | **90.53 ± 0.21** | 19.1% |
| 0.05 | 93.52 ± 0.06 | 90.09 ± 0.25 | 21.5% |
| **0.10** | **93.68 ± 0.04** | 88.83 ± 0.60 | 28.6% |
| 0.20 | 93.27 ± 0.08 | 87.30 ± 0.96 | 32.6% |
| 0.40 | 92.98 ± 0.12 | 84.32 ± 1.45 | 38.9% |
| 0.80 | 92.83 ± 0.06 | 79.54 ± 1.32 | 45.1% |

1. **β1 = 0.10 works on clean streams**: +0.15 over β1 = 0 (outside seed spread), the
   best clean result in this whole series — and it sits exactly at the linear
   critical-damping boundary `(1+β1)(1−α) ≈ 1`: coherent velocity is sustained, not
   amplified.
2. **No β1 > 0 survives label noise.** Even 0.05 is negative at ρ = 10% (92.47 vs
   92.71), and 0.10 drives memorization 23% → 43%. The mechanism is the gate's known
   blind spot: memorization drift is coherent, and momentum is a coherence
   *amplifier*. A sharper gate (β1·coh², tested) softens but does not fix it
   (89.44 vs 88.83 vs 90.53-at-β1=0 at ρ = 30%).
3. **Nothing diverged, even at β1 = 0.8** — not only the gate's self-limiting, but the
   autotuner acting as a stability governor: excess momentum reads as velocity
   incoherence at the probe, so the tuner commits high κ and the friction contains it
   (committed κ on clean: 0 → 41 → 400 as β1 rises past 0.1).
4. **The probe can pick β1 too.** Joint rule — one probe commits κ from the table *and*
   β1 = 0.1 iff probe coh ≥ 0.35 — gets the best of both with zero regressions:
   ρ = 0: 93.66 (β1 on); ρ = 5% (held out): 93.11 (correctly off, κ → 67);
   ρ = 10%: 92.72; ρ = 30%: 90.53 (both identical to β1 = 0 baselines). Momentum only
   where the stream is clean, never under noise — selected by measurement, not by the
   user.

The β1 = 0 default is vindicated for Concord's target regimes (noisy LM/diffusion
streams); the measured exception (β1 = 0.1 on clean streams, probe-selected) is one
task deep — gate any adoption on the nanoGPT A/B, like the rest.

## Exp 8 — the head-to-head: fully autotuned Concord vs AdamW, variable noise (`exp8_vs_adamw.py`)

The complete package as now shipped — fixed-C\* gate, one probe committing both κ
(exp-6 table) and β1 (0.1 iff probe coh ≥ 0.35) — against AdamW at wd = 0 and wd = 0.01,
same peak lr and schedule, 4k × 25 epochs, 3 seeds (`exp8_results.json`,
`exp8_vs_adamw.png`):

| ρ | AdamW | AdamW wd=0.01 | Concord autotuned (deploy) | margin | committed |
|---|---|---|---|---|---|
| 0% | 92.78 ± 0.16 | 92.76 | **93.66 ± 0.10** | +0.88 | κ=3, β1=0.1 |
| 5% | 92.14 ± 0.22 | 92.19 | **93.11 ± 0.11** | +0.92 | κ=67 |
| 10% | 91.97 ± 0.29 | 91.94 | **92.72 ± 0.21** | +0.75 | κ=101 |
| 20% | 90.73 ± 0.29 | 90.72 | **91.63 ± 0.44** | +0.90 | κ=202 |
| 30% | 89.15 ± 0.13 | 89.19 | **90.53 ± 0.21** | +1.34 | κ=381 |
| 45% | 86.25 ± 0.95 | 86.24 | **87.99 ± 0.68** | +1.74 | κ=400 |

1. **Concord wins every cell**, +0.75 to +1.74, with the margin *growing* with noise —
   the dissipation pays exactly where AdamW has nothing to answer with (decoupled
   wd = 0.01 is indistinguishable from wd = 0 here).
2. The autotuner committed sensible knobs at every level without being told ρ:
   κ tracking the noise, momentum on only for the clean stream.
3. The advantage is not only memorization suppression: at 5–10% noise Concord
   *memorizes more* than AdamW (25% vs 19%) yet generalizes better by ~+0.9 — the
   preconditioner/deploy-weight machinery contributes independently of the friction.
4. Same caveats as the whole series: CPU reference, one task/architecture, 3 seeds,
   task-calibrated probe table; the GPU nanoGPT A/B remains the adoption gate.
