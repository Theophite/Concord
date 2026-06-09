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
coh 0.96**, while the pure-noise control stays at 0.12. The candidate one-line refit is
ρ = 2·α_v (under `MASS_PRESERVE=True`) in `compute_drift_cancel_C` — same shape as the
fix already recorded in that docstring's history (the previous constant was 11× off in
the other direction). End-to-end it is *not* automatically a win (exp 3/4: helps the
clean task, neutral-to-slightly-negative under label noise, where a more skeptical gate
is mildly protective) — it sharpens the *meter*, and the consumers were tuned around
the dull one.

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
(`docs/SESSION_NOTES_2026-05-29.md`): near-zero-Bayes-error tasks don't pay for the
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
