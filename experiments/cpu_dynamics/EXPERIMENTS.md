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

## Exp 9 — Muon from the packed state (`exp9_muon.py`)

The hypothesis (the manifold-Lookahead argument): the velocity is already the momentum
buffer, so the spectral preconditioner comes free — orthogonalize via NS5 (a retraction
onto the polar manifold), let the chase do the averaging. Arms at each regime's own
κ\* (σ off everywhere; lr untuned per arm; AdamW reference from exp 8):

| arm | clean (own κ\*) | 30% noise (own κ\*) | noise memorized |
|---|---|---|---|
| AdamW | 92.78 | 89.15 | 19.3% |
| Concord, v̂ drive | 93.66 (κ=0) | 90.77 (κ=400) | 12.4% |
| **Concord, NS drive (c=0)** | **94.88 ± 0.01 (κ=0)** | **92.02 ± 0.28 (κ=100)** | 12.2% |
| native Muon (Euclidean, β=0.95) | **95.40 ± 0.10** | 82.32 ± 0.60 | **100.0%** |

1. **Native Muon is a memorization machine under label noise**: 100.0% of wrong labels
   fit, clean-test collapse to 82.3. Spectral whitening democratizes directions — and
   wrong-label gradients live in exactly the rare directions it amplifies. (It is also
   the best clean-data optimizer at this protocol, consistent with its LM reputation.)
2. **The Concord cascade completely tames that pathology**: same NS5 drive inside the
   gate/friction/deploy machinery memorizes 10–14% instead of 100%, and at its own
   κ\* = 100 **beats every arm tested under noise** (+1.25 over the v̂ drive's oracle,
   +2.9 over AdamW) while also beating the v̂ drive on clean (+1.2). Zero additional
   persistent state — the NS5 pass is transient (and per-element scale-free after
   dequant; the only resident cost remains the 32-bit word). v̂, its O(N+K) vectors,
   and its EMA pass are deleted in this arm.
3. **The momentum blend was wrong; the Lookahead idiom was right.** Blending the
   velocity into the NS input (NS(c·û + ĝ), c > 0) is a self-reinforcing direction
   loop — |u| grows ~20×, telescope coherence falls 0.48 → 0.37, consolidation
   throttles, clean accuracy falls monotonically with c (94.88 / 93.54 / 90.93 at
   c = 0/1/3). The resolution: the chase already IS the EMA. Positions integrate
   orthogonalized directions; retraction is needed on *directions*, not positions; so
   per-step NS(ĝ) + chase is the correct manifold-Lookahead composition, and no
   Euclidean momentum belongs in the drive.
4. **κ\* is drive-dependent** (NS: 100; v̂: 400 at this noise level) — the NS tick is
   unit-RMS by construction, so friction is more effective per unit κ. Any autotune
   table is per-drive, like everything else about it.

Open before kernel work: the σ/fluctuation interaction (off here; injected noise would
be re-amplified to unit spectral weight by NS), Muon's clean-data edge over the NS-c0
arm (+0.5 — native's Euclidean β=0.95 momentum; a *pre*-NS gradient EMA would need
state, which is the one thing this design refuses), Conv2d flattening, and the standard
gates: multi-seed, the real bench, the GPU A/B.

## Exp 9b — the fluctuation for the Muon drive: after NS, not before (`exp9b_muon_noise.py`)

Exp 9 flagged the σ×NS interaction; this resolves the placement. MuonConcord (c=0) at
each regime's Muon-arm κ\*, rising-late σ schedule as in the winner:

| arm | clean (κ=0) | 30% noise (κ=100) |
|---|---|---|
| no noise (exp 9 ref) | 94.88 ± 0.01 | 92.02 ± 0.28 |
| post-NS σ=0.3 | 94.87 ± 0.05 | 91.99 ± 0.22 |
| post-NS σ=0.6 | 94.88 ± 0.06 | 91.98 ± 0.25 |
| **pre-NS σ=0.6 (control)** | **94.61 ± 0.12** | **91.71 ± 0.28** |

1. **Placement confirmed: pre-NS is the wrong place** — consistently worse in both
   regimes (−0.27 / −0.31, ~2× seed spread). Noise before NS isn't a fluctuation; it
   rotates the input direction and NS re-amplifies the rotation to unit spectral
   weight. Post-NS (a perturbation of the step, σ in step-norm units) is the faithful
   transplant of the winner's fluctuation.
2. **Magnitude inert on this task family**: post-NS at σ ∈ {0.3, 0.6} is exactly
   neutral — consistent with every MNIST test in this campaign (σ added nothing beyond
   the dissipation in exps 3/4/8) and with the repo's own caveats about where σ earned
   its keep (the nanoGPT regime: real Bayes error, heavy tails). A structural reading:
   the cascade is noise-cancelling by design — isotropic post-step noise lands in `u`,
   reads as incoherent, and is evaporated or averaged away; injected fluctuation only
   pays when it does exploration work the task actually rewards.
3. Design decision for the Muon arm: **σ goes after NS if it goes anywhere; whether it
   goes anywhere is a nanoGPT-bench question.** No divergence at any setting.

## Exp 9c — the Muon drive obsoletes the trust region and step cap (`exp9c_muon_unbounded.py`)

The exp-9 Muon arm already ran with neither (the v̂ denominator IS the winner's trust
region; the NS drive replaced it wholesale, unclamped). This supplies the receipts.

**(a) Per-element |step| tails over training** (seed 0):

| drive | rms | p99.9 | max | cap=10 binds |
|---|---|---|---|---|
| NS, clean | 0.66 | 2.88 | **6.7** | — (no cap) |
| NS, 30% noise | 0.75 | 3.11 | **6.6** | — (no cap) |
| v̂ pre-clamp, clean | 1.51 | 9.12 | **323** | 1.58% of elements |
| v̂ pre-clamp, 30% noise | 2.22 | 12.95 | **406** | 1.73% of elements |

The NS step is self-bounded exactly as the spectral argument predicts: its all-training
max (6.7) never reaches the old cap, and is noise-level-independent. The v̂ drive's cap
is meanwhile doing constant, real work — binding on ~1.6% of elements every step, with
pre-clamp tails 30–40× past it. **Cap and trust region: required by v̂, dead weight for
NS.**

**(b) lr stability sweep** (κ = 0, clean, deploy acc; v̂ keeps its cap, NS has none):

| lr | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|
| v̂ (capped) | 93.66 | 94.59 | 94.68 | 93.06 | 78.05 ± 10.31 |
| **NS (capless)** | 94.84 | 95.89 | **96.07 ± 0.05** | 95.38 | **95.14 ± 0.30** |

1. **The spectral bound is a better trust region than the trust region**: the capless
   NS drive is stable and strong across two orders of magnitude of lr (95.1–96.1 from
   1e-3 to 1e-1), while the *capped* v̂ drive degrades past 1e-2 and effectively breaks
   at 1e-1 (±10.3 seed spread — the cap prevents NaN but not collapse).
2. **New best on this protocol**: NS at its own lr (1e-2) reaches 96.07 ± 0.05 — above
   native Muon's 95.40 (lr 1e-3; not lr-swept — caveat) and +1.4 over the v̂ drive's
   own lr-swept best. Exp 9's Muon-arm numbers were an lr handicap, not a ceiling.
3. **Kernel implication**: the Muon-arm apply kernel needs no `v_row`/`v_col`/
   `sum_v_inv`, no `eps`, no `step_cap` — tick + gate + friction + chase + leak only.
   The lr·κ < 2 friction ceiling is drive-independent and still applies when κ > 0
   (at lr = 0.1 it caps κ at 20 — high-lr + heavy-noise needs that arithmetic done).
4. Caveats: sweep is κ = 0 / clean only, 3 seeds, this protocol; the noisy-regime lr
   envelope (where κ > 0 re-enters) is unswept.

## Exp 10 — 80 epochs + augmentation on the small set: the ablation (`exp10_aug_ablation.py`)

4k subset, 80 epochs, pad-2 random-crop augmentation on the train subset only,
aug × optimizer × regime, each arm at its best-known settings (3 seeds; same-seed arms
see identical augmented streams). The v̂ noisy cells as originally specified **diverged
— by our own law**: the spec combined v̂'s lr\* (1e-2) with its 25-ep κ\* (400),
lr·κ = 4 > 2, exactly the exp-1 ceiling (the fork guard of `INSTALL_SDXL.md` §2 would
have refused the config at startup). Repaired cells (lr·κ ≤ 1.5) marked †; the Muon
noisy no-aug cell got the symmetric κ repair (its 25-ep κ\* = 100 was equally stale) ‡.

**Clean:**

| arm | 25ep no-aug (ref) | 80ep no-aug | 80ep + aug |
|---|---|---|---|
| AdamW | 92.78 | 93.49 | 96.60 |
| Concord v̂ | 94.68 | 94.68 | 96.43 |
| Concord NS | 96.07 | 94.75 | **97.58 ± 0.10** |
| native Muon | 95.40 | 95.53 | 97.46 |

**30% label noise** (memorized % in parens):

| arm | 80ep no-aug | 80ep + aug |
|---|---|---|
| AdamW | 81.17 (86.1) | 95.06 (11.2) |
| Concord v̂ † | **90.04 (12.6)** | 94.62 (10.3) |
| Concord NS ‡ | 86.28 (47.8) | **96.31 ± 0.22 (10.6)** |
| native Muon | 77.17 (100.0) | 86.89 (29.0) |

† best of {lr 1e-3/κ 400, lr 1e-2/κ 150}; the no-aug winner is lr 1e-2/κ 150 (lr·κ = 1.5,
near-ceiling friction). ‡ κ = 150 at lr 1e-2; the stale κ = 100 gave 78.86 (70.8%).

Findings:

1. **Augmentation dominates epochs.** The pure-compute control *hurt* the strong arms
   on clean data (NS: 96.07 @ 25ep → 94.75 @ 80ep — classic small-set overfit; the
   25-ep schedule was ending near the sweet spot), while aug + 80ep lifts every arm.
   The NS drive reaches **97.58 from 4,000 examples** — approaching the full-data
   no-aug MLP ballpark (~98–98.5).
2. **Best noisy result of the campaign: NS + aug = 96.31 at 30% label noise** — only
   1.3 below its own clean number. Cascade and augmentation compose: aug slows
   memorization (each wrong label must be fit across 25 crop variants), the cascade
   blocks what leaks through (10.6% memorized).
3. **Augmentation alone does not rescue native Muon**: it breaks the 100%
   memorization (→ 29%) but accuracy stops at 86.89 — nearly 10 points below the same
   drive inside the cascade. Aug is a memorization *slower*; the cascade is a
   memorization *barrier*; they are different mechanisms and they stack.
4. **A real regime split for the drives**: in the *extended noisy grind without data
   diversity*, v̂ + near-ceiling friction defends best (90.04/12.6% vs NS 86.28/47.8%
   at matched lr·κ — spectral democratization keeps re-funding wrong-label directions
   over a 3× longer horizon). With augmentation — i.e., any realistic training diet —
   the NS drive wins everything. Drive choice is regime-dependent at the margin;
   NS remains the default where data diversity exists.
5. **κ\* depends on the horizon as well as the drive and the noise** (NS noisy κ\*:
   100 @ 25ep → ≥150 @ 80ep): the autotune table is (drive, horizon, aug)-conditioned,
   which is one more argument for probe-based tuning over static tables — and the
   probe-then-commit design re-measures per run by construction.

## Exp 11 — computing the dissipation curve LIVE (`exp11_live_kappa.py`, `exp11d_reprobe.py`)

Can κ be controlled live, instead of (or beyond) probe-then-commit? Four
iterations, each informative (`exp11{,b,c,d}_results.json`):

1. **Naive continuous laws fail on meter conditioning.** Table-interp,
   self-referenced integral servo, and trend laws — engaged from κ=0 in an
   early window — all read coherence ABOVE the table's whole range and sat at
   κ≈0 across every static noise level. The table's (coh → κ) range is
   conditioned on the κ=50 probe over epochs 3–8; a meter read under other
   (κ, phase) conditions is on a different scale. Corollary: a self-referenced
   servo can detect *change* but not *level* — under static noise the run's
   own reference IS the noisy stream. Calibration-free level inference needs
   an absolute anchor; that is what the table is.
2. **Continuous tracking through the calibrated table is self-defeating for
   level** (11b). Probe-conditioned one-commit lands near-oracle (κ_post
   89/291 vs κ\* 100/400). Re-evaluating the same table every epoch UNDOES
   it: friction drains the incoherent velocity, the velocity-side meter reads
   clean, the tracker relaxes (κ 89→7, 291→26; deploy −0.2/−0.8; memorized
   +7/+10). The exp-6 v1 "controller chases its own tail" warning, measured
   on the v2 meter.
3. **Two-sided change detection fires on the benign side** (11c). Post-commit
   coherence RISES as friction works; a ±band watchdog re-probes on that rise
   and recommits low (κ 287→5).
4. **The validated live law: commit + ONE-SIDED event-driven re-probe**
   (11d). Hold the committed κ; windowed meter vs a slow-EMA baseline;
   re-probe only on a DROP (m < base − 0.08); rises update the baseline.
   Static ρ ∈ {0, 10%, 30%}: zero events, numerically identical to the
   shipped tuner. Mid-run regime change (clean → 30% label noise at epoch
   12, where one-commit is wrong by construction): one re-probe, recommit
   κ≈115 — deploy 92.76 vs 92.43, memorized 12.2% vs 13.4% (2 seeds).

Takeaway: "live" done right is **event-driven re-calibration, not continuous
feedback** — the meter's absolute level belongs to calibrated probe
conditions; its deviations belong to the watchdog. The calibration burden
(the per-domain table) is unchanged.


## Exp 12 — the muon dissipation curve: high-λ gates, prediction refuted (`exp12_muon_lambda.py`)

Tests MUON_DRIVE.md §12: λ* should scale with the drive's noise-energy
injection rate, so the NS5 drive (every singular direction written at equal
magnitude) should want λ 10–100× above the v̂ winner — plausibly near the
Wiener point λ=1. Exp-5 protocol exactly (4k×25ep, σ=0, fixed C*, seeds
0/1/2), extended along λ with the min-leak servo floor in `concord_ref`
(`evap_term`, min_leak=0.1 — kernel parity; bit-compatible no-op on the old
grid, verified against `exp5_results.json`). λ quoted at peak lr (κ·1e-3).

**Result: refuted — λ*(muon) ≈ 0 at every noise level.** Deploy accuracy is
monotone-DECREASING in λ for the muon drive everywhere (clean 94.86 → 86.10
over λ 0 → 1.5; ρ45% 90.64 → 83.95, with a within-spread +0.07 blip at
λ=0.1). The servo never slammed shut (final coh 0.42–0.56 across the grid —
the floor works); the decline is signal loss, not gate death.

The two columns that localize the mechanism:

- **The clean column is the smoking gun.** At ρ=0 there is no label noise to
  scrub, yet λ=1.5 costs muon −8.8% (v̂: only −3.5%). For the muon drive the
  evaporation is a near-uniform SIGNAL tax, λ-proportional.
- **The memorization column shows the scrub itself works**: ρ30% wrong-label
  fit falls 14.8% → 10.2% (≈chance) by λ=1 — but deploy falls faster than the
  scrub pays. The friction drains muon's signal and noise at nearly the same
  rate.

Mechanism (replaces §11's injection-balance story): **whitening destroys the
per-element SNR contrast the gate keys on.** The Wiener meter
coh = μ²/(μ²+ν²) is per-ELEMENT; the v̂ drive writes signal coordinates
harder than noise coordinates, so coh separates them and the evaporation is
selective. NS5 writes everything at the same magnitude — signal and noise
arrive element-wise indistinguishable, coh carries no contrast, and
λ(1−coh)·u taxes both equally. "Indiscriminate about the directions it
writes to" was the right premise with the inverted consequence: an
indiscriminate drive makes the dissipation indiscriminate too.

And the reason muon doesn't NEED the friction: **whitening is itself the
noise control.** At λ=0 the muon drive beats the v̂ drive at its per-regime
oracle κ at EVERY noise level — clean 94.86 vs 93.66, ρ10% 94.61 vs 92.72,
ρ30% 92.85 vs 90.77, ρ45% 90.64 vs 89.53 — and the gap is largest exactly
where v̂ needs κ most (v̂ at λ=0, ρ45%: 83.01; muon: 90.64). Equal-magnitude
writes cap any one sample's influence (sign-SGD-style robustness), so the
defense the dissipation provides is already built into the drive.
(Supersedes exp 9's stored muon rows, which were the c=3 blend arm; the c=0
λ=0 numbers here match the exp-9 c-sweep notes, 94.88/92.0x.)

v̂ high-λ extension (κ ∈ {500, 1000, 1500}): the exp-5 grid edge was at the
peak after all, not below it — λ*(v̂) ≈ 0.4–0.5 at ρ≥30% (κ500 ties κ400
within spread: 90.63 vs 90.77 at ρ30%) and everything declines by λ=1. On
THIS task the curve does not continue to the Wiener point; the SDXL
quality-monotone-in-λ observation is a domain property (diffusion gradient
noise, deploy-sample metric), not a universal law of the update rule.

Gate-1 re-read: the nanoGPT κ-flat plateau was not "under-damped
everywhere" — there is no high-λ regime where muon improves. Flatness is
what a near-uniform λ-proportional tax looks like at small tested λ (the
MNIST slope ≈ −5.8%/unit-λ predicts ≪0.01 loss across gate-1's swept range).

What survives for the muon line: a SPECTRAL gate. The per-element meter is
the wrong basis for a whitened drive; coherence measured in the singular
basis (the `wiener` rank mode — implemented, unrun) would restore the
contrast the evaporation needs, and is now the single reopen point that
addresses both this result and the emergent-rank starvation of exp 10/§11.


## Exp 13 — the spectral gate inside Newton-Schulz: energy is not SNR (`exp13_spectral_gate.py`)

Exp 12's reopen point, implemented the zero-cost way: X@(X^T X) = U diag(σ³) V^T,
so cube-and-renormalize c times before NS5 is a smooth RELATIVE spectral
threshold (gap → gap^(3^c)) in 2c matmuls — no SVD, basis-preserving, and
nominally "emergent-annealed" (a flat spectrum should pass unchanged).

**Negative at both sharpnesses, λ ∈ {0, 0.1}, all noise levels** (NS5
controls from exp 12, same seeds):

    deploy %, λ=0      clean   ρ10%    ρ30%    ρ45%
    ns5 (control)      94.86   94.61   92.85   90.64
    wiener-NS c=1      92.41   92.01   90.55   88.88
    wiener-NS c=2      88.28   87.47   85.65   84.04

- **The clean column kills it**: −2.45 (c=1) and −6.58 (c=2) with zero label
  noise. The minibatch gradient spectrum is not spike-plus-bulk — the small-σ
  directions carry real signal, so a relative-energy threshold is a soft rank
  cut applied every step: exp 10's starvation in smooth clothing. The
  "emergent annealing" hope fails because the early spectrum is never flat
  ENOUGH; cubing always reweights toward the top directions.
- **The suppression works perfectly and loses anyway**: wns2 holds
  memorization at chance even at ρ=10% (9.3%) while deploy is the worst in
  the table — the purest form of the exp-12 lesson.
- **λ=0.1 still hurts every wns cell**: cleaning the drive's spectrum by
  energy does not restore the per-element meter's contrast either.

Combined exp 12+13 statement: **every suppressor keyed on the gradient's own
statistics — per-element friction on a whitened drive, energy-thresholded
spectral shrinkage — loses to NS5's bare equal-magnitude write.** The
suppressors that DO work in this codebase (the v̂-drive's gate) key on a
SIGNAL REFERENCE: the telescope drift C*(S−A). The recorded reopen point is
therefore the drift-referenced spectral gate — per-direction coherence
diag(U^T D V) of the step's singular pairs against the drift matrix D, i.e.
the existing Wiener meter transported to the singular basis with the same
external reference. That needs the basis (SVD or a subspace sketch, the cost
exp 10 measured at ~4×); the cube trick was worth one experiment precisely
because it was free, and what it bought is the clean negative: energy is not
SNR.


## Exp 14 — the drift-referenced spectral gate: four versions, a measured bind (`exp14_rank_deficient.py`)

The rank-deficient test of exp 12+13's reopen point. Task: teacher
y = W2·tanh(W1·x) with W1, W2 both RANK 4 (64-dim inputs), infinite fresh
data, ±0.5σ target-noise arm; metrics at the deploy weights: clean-test MSE
and **complement leakage** — the learned delta's energy outside the
teacher's rank-4 input row space (the dead directions an indiscriminate
drive writes into). 3 seeds, λ=0.

**The mechanism is confirmed at the weight level** (3-seed means):

    drive          clean MSE   noisy MSE   leak (clean)  leak (noisy)
    vhat           0.0015      0.0037      25.8%         23.6%
    muon (NS5)     0.0007      0.0030      37.3%         44.4%
    spec v4        0.0035      0.0088      41.9%         44.0%

muon writes ~2× the dead-direction mass — "indiscriminate about the
directions it writes to", quantified. **But on this task it does not pay a
function-level price**: muon's deploy MSE beats v̂ in both arms anyway. The
leaked mass is real and harmless here (the downstream layer nulls it; the
absolute magnitudes are small). The task-level muon failure on
rank-deficiency remains an EMERGENT-rank / long-horizon phenomenon (nanoGPT
char-LM, gate 1) that this 3k-step synthetic does not reproduce. The leak
metric is the mechanism probe, not the damage.

**The retrofit gate fails in four escalating versions:**

- **v1 — matched filter diag(UᵀDV)**: D and g share the live subspace but
  not singular PAIRINGS; the diagonal scatters, the gate sits at its floor.
  Leak 44.6% = ungated.
- **v2 — Wiener subspace projectors from the drift D = C*(S−A)**: the drift
  is a VELOCITY meter and the chase floor passes noise into it — measured
  effective rank ~13/64. Leak 44.4 → 40.5%.
- **v3 — projectors from the net learned structure R = (S+A) − W0**: the
  right kind of reference, but SELF-CONTAMINATED — R's dead-direction
  content IS the leakage the gate should have prevented. Leak 42.0%.
- **v4 — calibrated knee (6× mean energy, from the measured spectrum: rank
  spike 3.4–5.4× mean over a 1.9× bulk), smoothstep-sharpened projectors,
  discovery-timescale floor**: the gate finally gates (mean pass 6.5%) — and
  noisy MSE TRIPLES (0.0088) with leak unchanged (44.0%).

The time-resolved diagnostic explains all four at once: **the leak is baked
in early** — 65.5% at t=500 under the open bootstrap floor (φ=0.85),
decaying to 44% only by dilution — and the reference spectrum only becomes
readable at t≳1500. So the bind: an open early floor lets the dead mass
consolidate before any internal reference can know the subspace (v1–v3); a
floor tight enough to block it starves discovery and continued learning
without recovering the already-consolidated leak (v4, exp 13, exp 10).

**The law, final form: selectivity cannot be retrofitted downstream of an
indiscriminate write — by the time any state-internal reference knows the
live subspace, the dead mass is already in the state, and the state is the
only reference there is. Selectivity must act AT the write** (the v̂ drive's
per-element SNR scaling is exactly that). Candidate future probe, unrun:
pre-NS v̂ scaling (orthogonalize the SNR-scaled gradient — NS re-whitens
magnitudes but the DOMINANT SUBSPACE it preserves shifts toward signal;
cousin of §8's pre-NS EMA). Until then: muon stays opt-in at λ≈0 for
clean/high-rank tasks, v̂ + the element gate everywhere else.


## Exp 15 — pre-NS v̂ scaling: erased by NS; the line closes (`exp15_prens.py`)

The last candidate from exp 14: orthogonalize the SNR-scaled gradient
h = g/√(v̂+ε), betting that NS re-whitens magnitudes but preserves the
dominant subspace of its INPUT. Both predictions fail:

- **Rank-4 synthetic**: prens leak 37.3% / 43.4% (clean/noisy) — identical
  to plain muon (37.3 / 44.4), nowhere near v̂ (25.8 / 23.6). MSE identical
  to muon. The written subspace did not move.
- **MNIST λ-response**: prens λ=0 ≈ ns5 (slightly worse clean: 94.46 vs
  94.86), and the λ 0→0.1 penalty is IDENTICAL (−1.08 vs −1.06 clean) —
  the element gate is exactly as blind downstream of prens as of NS5. No
  legibility was restored.

Why — and this is the arc's deepest finding: **the reference meters are
axis-aligned; task subspaces are rotated.** The teacher's rank-4 subspace is
a random rotation, so per-element statistics (the element coh) and rank-1
factored statistics (v̂ = v_row ⊗ v_col) are blind to it BY CONSTRUCTION —
there is nothing axis-aligned for the pre-NS weighting to encode, so NS has
nothing to preserve. Re-read in this light, the v̂ drive's leak advantage
(23.6 vs 44.4) was never subspace-selectivity: it is MAGNITUDE-selectivity
(noise writes are small in W units), and magnitude is exactly the channel NS
erases. All five failures (exp 12-15) are one sentence: every meter in this
system lives in the element basis, magnitude is the only selectivity it can
express, and orthogonalization deletes magnitude.

Closure: selectivity-at-the-write for an orthogonalized drive requires
second-moment statistics in a ROTATED basis — full-matrix / Kronecker
(Shampoo-family) preconditioning or a maintained spectral sketch. That is a
research program, not a patch; the muon line CLOSES here. Standing verdict:
muon opt-in at λ≈0 for clean/high-rank tasks (where it beats v̂ outright —
exp 12), v̂ + the element gate everywhere else; nothing about the winner
configuration changes.

## Exp 16 — fine lr ablation for the NS drive (`exp16_ns_lr.py`)

4k × 25ep, κ = 0, σ off, deploy acc, 3 seeds; pad-2 random-crop aug off/on.

| lr | 3e-3 | 5e-3 | 7e-3 | 1e-2 | 1.5e-2 | 2e-2 | 3e-2 | 5e-2 |
|---|---|---|---|---|---|---|---|---|
| no aug | 95.89 | 96.02 | **96.11 ± 0.08** | 96.07 | 95.57 | 95.45 | 95.38 | 95.27 |
| aug | 97.30 | 97.37 | 97.43 | **97.51 ± 0.10** | 97.36 | 97.36 | 97.19 | 96.52 |

1. **lr\* ≈ 7e-3–1e-2 in both conditions** (no-aug peak 7e-3, aug peak 1e-2 — the
   coarse 9c grid's 1e-2 was already essentially at-peak). Augmentation nudges the
   optimum slightly up and **widens the plateau**: under aug, everything in
   [3e-3, 3e-2] — a full decade — is within 0.32 of the peak.
2. **The NS drive is lr-insensitive across a decade**, now with a fine grid behind the
   claim (±0.2% over [3e-3, 3e-2] under aug). Compare the v̂ drive (±1.0 swing over
   the same span, 9c) and AdamW (collapse at high lr). For practitioners: any lr in
   [5e-3, 2e-2] is within noise of optimal — precision lr tuning buys essentially
   nothing on this drive.
3. **25 epochs at lr\* under aug ≈ converged**: 97.51 here vs exp 10's 97.58 at 80
   epochs — the extra 55 epochs bought +0.07. The compute-efficient recipe for this
   protocol is 25ep + aug + lr 1e-2, and exp 10's headline number had no lr headroom
   left in it.


## Exp 16b — apportioning the lr flatness: native Muon on the same grid (`exp16b_native_lr.py`)

Native Muon has the NS normalization and none of the cascade, so its fine-grid lr curve
(same protocol as exp 16; live weights) splits the credit for the lr insensitivity:

| lr | 3e-3 | 5e-3 | 7e-3 | 1e-2 | 1.5e-2 | 2e-2 | 3e-2 | 5e-2 |
|---|---|---|---|---|---|---|---|---|
| native, no aug | 93.89 | 93.18 | 93.07 | 93.10 | 93.24 | 92.96 | 93.15 | 92.68 |
| native, aug | 96.56 | 95.43 | 94.71 | 93.81 | 91.54 | 87.48 | 70.89 | **29.20 ± 5.37** |
| Concord-NS, aug (exp 16) | 97.30 | 97.37 | 97.43 | 97.51 | 97.36 | 97.36 | 97.19 | 96.52 |

1. **The lr flatness belongs to the cascade, almost entirely.** Under augmentation,
   native Muon collapses 67 points across one decade — the sharpest lr curve measured
   in this campaign — while Concord-NS drifts 0.8 over the same span *with friction
   off* (exp 16 was κ = 0). The pre-registered prediction ("flatter than v̂, sharper
   than Concord-NS, ugly at 5e-2") was right on shape and wrong on attribution: it
   credited NS normalization with most of the robustness; the data gives it to the
   regulation.
2. **Division of labor, now clean**: NS's spectral bound funds the *tail* robustness
   (cap/trust-region deletion, exp 9c); the cascade — gate-throttled consolidation +
   chase averaging + shipping `P` instead of the live endpoint — funds the *lr*
   robustness. Constant-magnitude steps are exactly what spectral normalization
   produces; without an averaging/regulating layer, the endpoint of that walk is
   lr-critical, and augmentation (faster-rotating momentum) makes it worse, not
   better.
3. Native's lr\* sits at or below the grid edge (≤ 3e-3; exps 9/10 ran it at 1e-3,
   which the curve retroactively justifies), so its earlier clean numbers were not
   handicapped.
4. Product implication: the lr-insensitivity does **not** transfer to bare NS
   optimizers — it is a Concord-cascade property, i.e., a real differentiator rather
   than inherited Muon credit.

## Exp 17 — noise with the character of augmentation: the hierarchy test (`exp17_noise_character.py`)

Arena: the exp-10 corner where augmentation mattered most (4k, 30% label noise, 80ep,
NS drive, κ=150 @ lr 1e-2 — stale-high κ biases *against* the diversity arms, so the
ordering is conservative). References on the books: none = 86.28 (47.8% mem),
crop-aug = 96.31 (10.6% mem).

| arm | level | deploy acc | memorized |
|---|---|---|---|
| none (ref) | — | 86.28 ± 0.77 | 47.8% |
| iso, post-NS σ=0.6 | L0 | 86.79 ± 0.69 | 46.2% |
| Σ_g-shaped, pre-NS | L1 | 87.20 ± 0.85 | 46.1% |
| vicinal (chord jitter, labels fixed) | L2a | 89.14 ± 0.62 | 38.6% |
| **mixup (chords + label interp.)** | L2b | **92.52 ± 0.35** | 25.8% |
| small-batch (32) control | temp. | 89.56 ± 0.27 | **22.6%** |
| crop-aug (ref) | L3 | **96.31 ± 0.22** | 10.6% |

The ladder is monotone in the hierarchy (L0 ≈ none < L1 < L2a < L2b < L3), with four
refinements to the model:

1. **Σ_g is nearly inert (+0.9), and the reason refines the theory**: its covariance is
   right but its *support is static* — it re-injects directions already present in the
   same 4k gradients every epoch (including the wrong-label directions themselves!),
   so it adds no per-visit decorrelation beyond what SGD already provides. The
   operative augmentation property is not the covariance; it is **fresh support beyond
   the empirical points, resampled per visit**. (This also retro-explains the original
   nanoGPT ablation's isotropic ≥ Σ_g verdict more deeply than "BN-mediated.")
2. **Label interpolation is load-bearing at Level 2**: chord geometry alone (vicinal)
   buys +2.9; adding target mixing (mixup) buys +6.2 — under label noise, mixing
   dilutes every wrong label so the model is never trained on a pure corrupted target.
   Level 2 splits into L2a (vicinity) and L2b (vicinity + target dilution), and most
   of the practical power is in the dilution.
3. **The remaining gap to crop (+3.8 over mixup) is the on-manifold premium**: chords
   between digits are off-manifold blends; shifts are on-manifold orbits. Domain
   knowledge buys exactly that.
4. **Small batch is a different axis, not a rung**: temperature, not information. It
   suppresses memorization hardest of all injectables (22.6%) but converts little of
   it into accuracy (89.56) — more SGD noise blurs signal and noise alike, where
   diversity arms replace noise-fitting with signal.

Refined principle: *the character of augmentation = per-visit-decorrelated vicinity
with support beyond the empirical sample, ideally on-manifold, plus target dilution
where labels can be wrong.* Predicted next rung (untested): k-NN/local-PCA-directed
jitter — on-manifold chords — should land between mixup and crop without domain
knowledge. For label-free domains (diffusion), the mixup analogue is target-space
mixing, not class interpolation.

## Exp 18 — the telescope window in epoch units (`exp18_telescope_window.py`)

α_v had never been swept (0.001 in every experiment and, per the records, the repo's
history) — an *absolute* 500-step window meaning 16 epochs on this protocol and a
quarter-epoch on a 2k-image bs1 SDXL run. Sweep: window = 1/(2·α_v) in epoch units,
80ep protocol, NS drive, lr 1e-2, κ: clean 0 / 30%-noise 150 (κ not retuned per
window — caveat), 3 seeds. Boundary cells and a gate-closure control added after the
grid optimum landed on the edge.

| window (ep) | clean deploy | noisy deploy | noisy memorized | noisy late coh |
|---|---|---|---|---|
| 1 | 94.58 | **80.21 ± 0.93** | **72.5%** | 0.296 |
| 4 | 94.80 | 82.52 | 64.1% | 0.282 |
| 16 (historical) | 94.85 | 86.68 | 48.2% | 0.246 |
| 64 | 94.73 | 91.29 | 31.5% | 0.166 |
| 256 | — | 92.50 | 27.2% | 0.101 |
| 1024 | — | 92.40 | 25.9% | 0.074 |
| 4096 | — | 92.56 | 26.0% | 0.063 |
| **control: gate closed (C\*=0), same F** | — | **92.56 ± 0.36** | **25.8%** | 0 |

1. **The control collapses the mechanism.** The long-window plateau is *exactly* the
   gateless limit — at 256+ epoch windows the gate is nearly shut (coh 0.06–0.10) and
   the residual telescope signal adds nothing. The initially attractive
   "temporal-prior / early-phase-anchoring" interpretation is **not supported**: the
   monotone improvement from 1 → 64 epochs is progressively *less gate-approval of
   memorization drift*, and the asymptote is no approval at all.
2. **Short windows are actively dangerous under noise**: at a 1-epoch window the gate
   re-anchors on the most recent motion — including the memorization phase — and
   exempts it from friction: 72.5% memorized at the very same F that achieves 25.8%
   with the gate shut. **The window is the gate's trust timescale**: short = trusts
   the present; long = trusts almost nothing.
3. **Regime conclusion, consistent with exps 10/12**: in the heavy-memorization,
   no-diversity corner the gate is a liability (its blind spot is the regime's
   dominant failure), and the optimum is strong *ungated* friction — reachable either
   by window → ∞ or directly by closing the gate. 92.56/25.8% **ties** (does not beat —
   an earlier draft of this entry miscredited it) the standing no-aug record, exp 17's
   mixup at 92.52/25.8%: two unrelated mechanisms — ungated weight-space friction
   (= EMA-teacher distillation, per the κ-identity) and data-space target dilution —
   converging on the same number, both 3.8 below crop-aug + *gated* cascade (96.31). In the regimes the product targets (real data diversity), the gated
   arms have consistently won — the conclusion is not "delete the gate" but "gate
   trust is a regime knob, and the noisy-static corner wants it at zero."
4. **Clean regime: window-insensitive** (94.6–94.9 across 64×; mild dip only at 1ep) —
   one more entry in the cascade's insensitivity ledger.
5. **Parametrization stands regardless of mechanism**: α_v belongs in epoch units,
   derived per-run like `total_steps` (the current absolute constant silently spans
   ¼-epoch to 16-epoch windows across real regimes), and the SDXL fork's ¼-epoch
   window sits on the *dangerous* short side for memorization-pressured fine-tunes.
   The telescope amplitude grew sublinearly (|d|max ×30 over a ×64 window range, no
   int8 pressure at these scales).


### Exp 18 addendum — do the two 92.5s stack? (suppression bound)

| arm (30% noise, 80ep, no crop-aug) | deploy | memorized |
|---|---|---|
| gateless friction F=1.5 alone | 92.56 ± 0.36 | 25.8% |
| mixup, gated, F=1.5 | 92.52 ± 0.35 | 25.8% |
| **gateless F=1.5 + mixup** | **93.15 ± 0.10** | **18.0%** |
| gateless F=0.5 + mixup | 87.35 ± 0.89 | 74.2% |
| crop-aug + gated (exp 10 ref) | **96.31 ± 0.22** | 10.6% |

1. **Mostly the same 92.5**: stacking buys only +0.6 accuracy (though memorization
   composes better, 25.8 → 18.0). Consistent with a **suppression bound**: friction
   and dilution can stop wrong labels from being fitted but cannot recover the
   information the corrupted 30% would have carried — while crops *add* information
   (orbit structure) and raise the ceiling itself. Optimizer-side and loss-side
   defenses plateau ~92.5–93.2 in this corner; the data-side fix reaches 96.3 — and
   does so with the gate ON, because augmentation repairs the coherence signal the
   gate needs.
2. **Friction is load-bearing, mixup is an adjuvant**: at F=0.5 ungated, mixup alone
   collapses (74.2% memorized) — dilution *slows* memorization per step but over 80
   epochs the diluted wrong labels still get fitted unless friction deletes the drift.
   Exp 17's mixup number was partly riding on its F=1.5.
3. Standing conclusion for the corner: best-known no-aug = gateless F=1.5 + mixup
   (93.15); best-known overall = restore data diversity and keep the gate (96.31).

## Exp 19 — augmentation × long horizon (`exp19_aug_long.py`)

160ep (2× exp 10), 30% noise + crop-aug, NS drive, lr 1e-2, F=1.5, 3 seeds. The
pre-registered prediction (with aug repairing the coherence signal, the gated default
should beat the corner's gate-disabled winners) **failed**:

| arm (noisy+aug, 160ep) | deploy | memorized |
|---|---|---|
| gated, default window | 95.88 ± 0.23 | 11.2% |
| long window (64ep) | 96.07 ± 0.37 | 10.9% |
| **gateless F=1.5** | **96.26 ± 0.06** | 10.7% |
| full stack (crop+mixup+gated) | 96.10 ± 0.36 | 10.5% |
| clean+aug gated (ceiling) | 97.54 ± 0.10 | — |

Longer horizon helped nothing (80ep was already past optimum; clean flat at 97.54),
eroded the gated arm most (−0.43; aug decorrelates pixel-keyed memorization but a
wrong label still pushes a coherent class-level direction across crops, which the
gate slowly approves), and mixup is redundant once crops are present.

### Exp 19b — the gate ablation at matched F (the missing cells)

Every prior κ sweep confounded gate with friction. Measured exemption value at
matched F:

| regime | gated | gateless | gate Δ |
|---|---|---|---|
| clean, F=1.5, 80ep | **96.25 ± 0.06** | 95.74 ± 0.02 | **+0.51** |
| 10% noise, F=1.0, 80ep | 92.07 (40.1%) | **94.85 ± 0.15 (22.1%)** | **−2.78** |
| 30% noise no-aug, F=1.5 (exp 18) | 86.28 (47.8%) | 92.56 (25.8%) | −6.28 |
| 30% noise + aug, F=1.5, 80ep | 96.31 (10.6%) | 96.15 (10.2%) | +0.16 |
| 30% + aug, 160ep (exp 19) | 95.88 | 96.26 | −0.38 |

1. **The "mild-noise home turf" hypothesis is inverted**: at 10% noise the exemption
   posts its worst score — few wrong labels, but perfectly coherent drift, exempted
   from friction (memorization 40% vs 22%). Coherent wrong-label drift is the typical
   case at every noise level; **the exemption is net-negative wherever labels can be
   wrong**.
2. **The gate's one clear win is clean data under high friction** (+0.51) — and that
   cell also revises exp 5: clean gated F=1.5 @ 80ep (96.25) ≫ clean κ=0 @ 80ep
   (94.85). **κ\* ≠ 0 on clean data at long horizons** — friction is a general
   anti-overfit regularizer; the horizon-dependence of κ\* (exps 5/10/13) reduces to
   this.
3. **Scope caveat, load-bearing**: all cells are the label-corruption family
   (adversarially coherent noise). The LM/diffusion regimes have real Bayes error —
   incoherent by nature — and the original nanoGPT validation never gate-ablated at
   matched F either (the "split" = gate+friction vs neither). **The matched-F gate
   ablation is now the top-priority cell for the GPU bench.** The gate's *meter* role
   (the probe; exp 6) is unaffected — only the exemption is in question.
4. Design implication: split the roles — keep the meter; make the exemption a
   probe-committed dial (0 = gateless … 1 = full), alongside F and β1.


## Exp 20 — best-of-both synthesis: the floored gate at the epoch window (`exp20_window_floor_synthesis.py`)

Crosses the two campaigns' winners: NS drive at its own lr (1e-2, exp 16),
crop-aug, F = 1.5 WITH the min-leak floor (every F=1.5 cell in exps 18/19
ran unfloored — survival factor 1 − 1.5(1−coh) < 0 wherever coh < 1/3:
sign-flip ringing), the window swept in epoch units, and the F=0 control
exp 12 taught us to always run. 4k×25ep, 3 seeds, deploy acc.

    gated, F=1.5        W=1ep    W=4     W=16    W=64    W=256
    clean               96.93    96.64   95.96   95.35   95.13
    30% noise           95.60    95.07   94.24   93.73   93.49

    controls (W=16 unless noted)   clean          30% noise
    gateless F=1.5                 95.13          93.48
    gated    F=0                   97.46          94.69
    gated    F=0, W=1ep            97.45          94.42

Three reversals:

1. **The window optimum is ONE EPOCH — and shorter beats longer everywhere.**
   Monotone-decreasing in W in both regimes; W=1ep memorizes at chance
   (10.1%). Exp 18's "1ep windows are catastrophic (72.5% memorized)" was
   measured unfloored: at short windows coh reads low, the unfloored F=1.5
   survival factor went deeply negative, and the resulting ringing — not the
   window — destroyed the run. The trust-timescale rule lands exactly at the
   dataset revisit period: every example votes once, then motion counts.

2. **The floor flips the gate-ablation verdict.** Gated beats gateless at
   matched F=1.5 in BOTH regimes now (+1.80 clean, +2.12 noisy at W=16) —
   exp 19b's "exemption net-negative under any label noise" (−2.78 to −6.28)
   was the ringing artifact, not the gate.

3. **Friction's value is set by reference freshness (F × W interaction).**
   Noisy: F=1.5@W=1 (95.60) > F=0 at any window (94.4–94.7) > F=1.5@W=16
   (94.24). Friction with a FRESH trust reference is selective and wins;
   friction against a stale reference taxes signal and loses; with no
   friction the window barely matters. Clean: F=0 stays champion (97.46 —
   reproducing exp 16's 97.51); friction only costs where there is nothing
   to scrub.

The freshness law retroactively unifies the campaigns' λ disagreement:
exp 12's λ*≈0 was measured at the CPU default window (≈16ep — stale), while
the SDXL fork's monotone-quality-in-λ observation runs at alpha_v=0.001 ≈ a
ONE-EPOCH window at that dataset size (len~1885, bs4) — the fresh-reference
regime where this grid says friction pays. The CPU default was the mis-set
one, not the SDXL default.

Recipe (this protocol): NS @ lr 1e-2 + aug + gate on; clean → F=0;
noisy → F=1.5 (floored) with alpha_v = 1/(2·steps_per_epoch). SDXL
translation: keep alpha_v pinned to the epoch (scale it if the dataset
grows); the floored high-λ sweep has CPU support under noise for the first
time.


## Exp 21 — the F sweep at the champion configuration (`exp21_f_sweep_optimal.py`)

Exp 20's configuration (NS @ lr 1e-2, crop-aug, gate on, window = 1 epoch,
min-leak floor) with the F axis filled in, including the classically
forbidden F >= 2 zone (runnable only because the floor clamps survival —
no ringing, no divergence). 4k×25ep, 3 seeds; F = 0/1.5 anchors from exp 20.

    F           0      0.1    0.25   0.5    1.0    1.5    2.5    4.0
    clean       97.45  97.51  97.42  97.37  97.21  96.93  96.60  96.31
    30% noise   94.42  95.00  95.60  96.03  96.00  95.60  94.90  94.48
    (memorized) 12.0   11.5   10.7   10.7   10.3   10.1   10.1   10.0

1. **Noisy F\* is interior: ~0.5–1.0, peak 96.03.** A genuine optimum, not
   exp 12's λ\*=0 (stale window) and not the SDXL monotone-past-the-ceiling
   pattern: at the fresh window, friction works — then over-friction taxes
   signal faster than the scrub pays (memorization keeps falling past the
   peak while deploy drops: the kills stay "good" but become too many).
2. **Clean has a free shoulder to F ≈ 0.5** (everything in [0, 0.5] within
   ~0.1; F=0.1 nominally tops the table at 97.51). "Friction only costs on
   clean data" was an artifact of the stale-window regimes; at the fresh
   window, light friction is free insurance. Decline is real past F ≈ 1.
3. **The floor makes over-friction graceful**: F = 4 (2× the classical
   stability ceiling) costs only −1.1/−1.5 from peak instead of diverging.
   The lam < 2 ceiling is now advisory in every sense.
4. **Universal default: F ≈ 0.5** — at the clean shoulder's edge AND the
   noisy peak simultaneously (clean −0.08, noisy +1.61 vs F=0). One number,
   both regimes, no per-dataset tuning at this protocol.

SDXL implication, stated carefully: the freshness-repaired CPU now supports
moderate friction with an INTERIOR peak — it does NOT reproduce unbounded
monotone improvement in λ. The SDXL sweep (different noise character,
deploy-sample metric) may still peak elsewhere, but this predicts it HAS a
peak, plausibly at lam below where the sweep was heading. The fork's
λ ∈ {0.25, 0.5, 1.0} bracket at the epoch window is the decisive next GPU
run.

## Exp 22 — the s_slow chase window: the telescope's missing middle rung (`exp22_chase_window.py`)

The cascade jumps from a ~5-step chase (α=0.1) straight to the epoch anchor,
so the drift numerator C\*(S−A) compares ~5 steps of commits against the
epoch integral. Question (user proposal): does a fraction-of-epoch s_slow —
v_slow at epoch, s_slow at epoch/k — buy better coherence (trend agreement
between two well-sampled integrals) and a Polyak-averaged deploy? Constraint:
ρ = α_v/α < ½ or C\* poles, so the ladder tops at epoch/2.5.

Protocol: v̂ drive (the fork's), lr 1e-3, σ OFF (the fork's operating
point), gate + min-leak, pad-2 crop-aug, 4k × 25 ep, **bs 32** (SPE = 125,
so the ladder spans 10×; exp-20/21 anchors at bs 128 do NOT transfer — the
α=0.1 arms are the internal baseline), anchor = 1 epoch, 3 seeds. Slow arms
use SPLIT-INIT packing (S = A = W₀/2, u = 0 — the telescope's zero-input
fixed point, so drift starts exactly 0; the α=0.1 packing pair is the
neutrality control).

    chase window      ep/25(legacy) ep/16   ep/8    ep/4    ep/2.5
    clean   deploy    95.07         94.30   93.16   91.81   91.00
    30%     deploy    93.40         92.47   90.77   88.98   87.77
    mean coh (cl/no)  .25/.19       .29/.22 .36/.28 .41/.33 .45/.37

1. **Monotone-down, both regimes — the middle rung is refuted at fixed
   budget.** And the diagnostics show *why* cleanly: coherence does exactly
   what the proposal predicted — it nearly doubles up the ladder (the
   trend-agreement signal is real and better-sampled) — but the better meter
   never pays for the slower consolidation. Commit speed is the binding
   constraint, not measurement quality.
2. **The Polyak hope is flat**: deploy−live ≈ +0.3 at every window (noisy);
   the averaging benefit does not grow with the window.
3. **Split-init is neutral** (95.07 vs 94.96 clean, 93.40 vs 93.45 noisy —
   within spread). Adopt it for its own reasons: it starts the system at the
   telescope's static fixed point, killing the init-residue artifact family
   (probe floor, early coh inflation, the boil washout) at the source — a
   fine-tune benefit MNIST-from-scratch cannot exhibit.
4. **Incidental headline — F is monotone-DOWN on this protocol** (phase 2,
   legacy window, v̂):

       F           0      0.25   0.5    1.0
       clean       96.63  95.89  95.07  93.19
       30% noise   94.86  94.43  93.40  91.26
       (memorized) 10.9   10.1   10.0    9.8

   λ\*(v̂ | aug + 1-ep anchor + σ-off + floor) = 0. Contrast exp 21 (NS
   drive, bs128: interior F\* ≈ 0.5–1.0) and the exp-5-era v̂ λ\* ≈ 0.4–0.5
   (stale 16-ep window, no aug). Pattern across the campaign: every repair
   that adds regularization or freshens the trust reference (floor, epoch
   window, aug) shrinks friction's constituency; on this fully-repaired v̂
   protocol it reaches zero. Memorization falls only ~1 point over the whole
   F range — the scrub buys almost nothing the aug didn't already buy.

Caveats: fixed 25-ep budget (slow arms are rate-limited, not given longer —
but fixed budget is the operationally relevant comparison); single protocol;
bs 32; one task.

SDXL implication: CPU no longer offers independent support for λ > 0 on the
v̂ fork — the SDXL monotone-λ quality observation now stands as a pure
domain property (fine-tune on real data, deploy-sample metric, conditioning
noise with different character). That makes the GPU λ bracket the decisive
arbiter, and the running λ=0.5 + γ-SNR job is effectively one of its arms.
Do not spend GPU on the chase window; consider porting split-init packing to
the fork for the artifact kill alone.
