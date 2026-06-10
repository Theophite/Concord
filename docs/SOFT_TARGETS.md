# Soft targets from the in-word teacher

`target_t = λ·onehot(y_t) + (1−λ)·softmax(f_P(context)/τ)` with **f_P = the
model's own deploy (sv) weights** — the slow position field teaching the live
field. The teacher rides inside the int32 word: zero extra state, a temporal
self-consistency regularizer. Implemented in the bench
(`notebook/src/train_nanogpt.py --target_lambda --target_tau`; eager-only):
the teacher pass is dropout-free under no_grad via the eval machinery's
`_bf16_weight_buf` reference swap, with live aux params — the teacher is
exactly the object the deployed-sv metric measures. CE is linear in the
target, so the loss decomposes exactly as λ·CE_onehot + (1−λ)·CE_soft
(teacher-only temperature). Cost ≈ +55%/step (one extra forward).

## Bench results (same-seed char-nanoGPT, deployed-sv, 3000 iters, fixed C\*)

| arm | deployed-sv | Δ vs own control |
|---|---|---|
| v̂ winner control (κ50, σ0.6) | 1.5157 | — |
| **v̂ + soft λ=0.9, τ=1** | **1.5065** | **−0.009** |
| v̂ + soft λ=0.8, τ=2 | 1.5598 | +0.044 |
| NS-drive control (κ50) | 1.5385 | — |
| NS + soft λ=0.9, τ=1 | 1.5484 | +0.010 |
| NS + soft λ=0.8, τ=2 | 1.5745 | +0.036 |

Findings:
1. **A gentle blend improves the winner** (−0.009, the session best), and the
   λ0.9 run was still descending at iter 3000 — longer-horizon headroom.
2. **Sharply dose-dependent**: λ0.8τ2 is harmful. (λ and τ were moved
   together — the disentangling λ0.8τ1 cell is unrun.)
3. **Teacher quality decides the sign**: the same recipe hurts the NS drive
   at every blend — its slow field carries whitened noise (MUON_DRIVE.md §9)
   and self-consistency against a noisy teacher compounds. Soft targets are
   a *clean-slow-field* amplifier, not a general regularizer.
4. Early training the slow field ≈ init, so the soft term begins as label
   smoothing; a **λ-ramp** (1.0 → λ as P matures) is the natural v2, as is
   re-checking κ/floors under the softened gradient statistics.

Status: single-seed, bench-only; opt-in (`--target_lambda`, default 1.0 =
off). Promising enough to gate-test properly (multi-seed, then SDXL) if
adopted into the recipe.
