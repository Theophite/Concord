# Mixup, and the noise model that generalizes augmentation

Results of exp 17 (`experiments/cpu_dynamics/exp12_noise_character.py`), the hierarchy
test for "noise with the character of augmentation." Same validity bounds as the rest
of the CPU campaign (`RESULTS.md` §"Threats to validity"); mixup's α untuned
(Beta(1,1)); the arena's κ was tuned for the no-diversity corner, which biases
*against* the diversity arms — orderings below are conservative.

## TL;DR

In the campaign's hardest corner (4k examples, 30% label noise, 80 epochs, NS drive),
**mixup recovers 62% of the crop-augmentation gap with zero domain knowledge** —
+6.2 accuracy and memorization halved — and the decomposition shows **the label
interpolation, not the input geometry, carries most of it**:

| arm | level | deploy acc | wrong labels memorized |
|---|---|---|---|
| no noise | — | 86.28 ± 0.77 | 47.8% |
| isotropic σ (post-NS) | L0 | 86.79 ± 0.69 | 46.2% |
| Σ_g-shaped (pre-NS) | L1 | 87.20 ± 0.85 | 46.1% |
| vicinal — chord jitter, labels fixed | L2a | 89.14 ± 0.62 | 38.6% |
| **mixup — chords + label interpolation** | **L2b** | **92.52 ± 0.35** | **25.8%** |
| small-batch control (B=32) | temp. | 89.56 ± 0.27 | 22.6% |
| crop augmentation | L3 | 96.31 ± 0.22 | 10.6% |

## 1. The question

Augmentation was the most powerful single intervention in this campaign (exp 10), while
injected gradient noise was inert in every test (exps 3/4/8/9b). Both perturb the
gradients; only one helps. So: *what is the noise model of which augmentation is an
instance* — noise **with the character of augmentation** — and how far up that ladder
can you climb without domain knowledge?

## 2. The formalization

Augmentation replaces each example's gradient with a draw from a per-example gradient
distribution: `g_i → ∇L(f(T x_i), y_i)`, `T ~ τ`. This is vicinal risk minimization
(Chapelle–Weston–Bottou–Vapnik 2000): replace the empirical point masses with local
vicinity distributions. Decomposed, the draw has a **mean shift** (the orbit-averaged
gradient — invariance regularization; Bishop's noise-as-Tikhonov to second order) and a
**fluctuation** whose character, going in, we took to be: (a) per-example anchoring,
(b) manifold-tangent covariance, (c) per-visit decorrelation.

The Concord-native statement of why this matters: property (c) means augmentation
**converts memorization from temporally coherent to temporally incoherent** — to fit a
wrong label, a weight direction must survive the whole orbit, and the orbit fluctuation
erases the off-orbit solutions visit after visit. Augmentation is a coherence filter
implemented in data space; the dissipation is one implemented in weight space; they
target the same quantity from opposite sides, which is why exp 10 found them composing.
It also explains every inert σ result at once: isotropic noise has amplitude but no
data character, and the cascade — correctly — deletes it.

## 3. What the experiment taught us (two corrections to the model)

1. **Σ_g failed, informatively (L1, +0.9).** Its covariance is "right" (the
   per-example gradient scatter — simulated batch resampling, the repo's original
   `_SIGMAG` design) but its **support is static**: it re-injects directions already
   present in the same 4k gradients every epoch, *including the wrong-label directions
   themselves*. So the operative property is not the covariance — it is **fresh
   support beyond the empirical sample, resampled per visit**. A noise source cannot
   decorrelate memorization if it is drawn from the distribution that contains the
   memorization. (This also explains the original nanoGPT isotropic-≥-Σ_g ablation at
   the mechanism level.)
2. **At Level 2, the label side dominates.** Chord geometry alone (L2a, labels
   untouched) is real but modest (+2.9). Adding target interpolation (mixup) more than
   doubles it (+6.2): under label noise, every corrupted target arrives **diluted** —
   the model is never trained on a pure wrong label. Mixup's own paper documents this
   robustness; the decomposition here shows how much of "Level 2" it is: most.

Refined principle: **the character of augmentation = per-visit-decorrelated vicinity
with support beyond the empirical sample, ideally on-manifold, plus target dilution
wherever targets can be wrong.** The residual +3.8 from mixup to crop is the
on-manifold premium (chords between digits are off-manifold blends; a shifted digit is
a real digit) — that number is the measured price of domain knowledge. The small-batch
control sits on a different axis entirely: temperature suppresses memorization hardest
of the injectables (22.6%) but converts little into accuracy — noise blurs signal and
noise alike, where vicinity *replaces* noise-fitting with signal.

## 4. Mixup, operationally

```python
j   = torch.randperm(B)
lam = float(torch.rand(()))                    # Beta(1,1); higher α → stronger
out  = net(lam * x + (1 - lam) * x[j])
loss = lam * ce(out, y) + (1 - lam) * ce(out, y[j])
```

- Trainer-level, two lines, domain-agnostic (no assumption beyond "inputs live in a
  vector space"), batch-internal (no extra data motion).
- Composes with the cascade rather than substituting for it (exp 10's aug × cascade
  result; exp 17 ran *inside* the cascade at κ=150 and still gained +6.2).
- λ untuned here; the mixup literature uses stronger interpolation (higher α) against
  heavier label noise — a free knob we did not sweep.

## 5. Guidance and next rungs

Decision rule, by what you have:

| you have | use | level |
|---|---|---|
| domain transforms (crops, flips, EXIF-safe jitter) | them | L3 |
| only vectors + labels | **mixup** | L2b |
| only vectors, no labels to mix | chord/k-NN jitter (modest) | L2a |
| nothing (optimizer-only) | σ/Σ_g — measured not worth it | L0/L1 |

- **For the SDXL fork**: the caption IS the label (corrected from an earlier draft
  that claimed otherwise). Miscaptioned data is label noise in exactly the exp-12
  sense, and captions enter as text-encoder embeddings — a continuous space where
  interpolation is well-defined. Three observations make caption mixup concrete:
  (a) with shared ε and t, the v-prediction target is affine in z₀, so latent mixup
  mixes targets **exactly** (`v(λz+(1−λ)z′) = λv+(1−λ)v′`) — an identity where
  classification's mixed CE is a heuristic; (b) **CFG caption dropout is already the
  Bernoulli endpoint of caption mixing** (dilution toward the null embedding at
  λ ∈ {0,1}) — caption mixup is the continuous interior of standard practice;
  (c) both ingredients are cached in the fork (latents, text embeds), so mixing is
  two lerps per batch, same-shape within aspect buckets, graph-compatible via the
  existing inject-per-replay pattern. Two variants to test, conservative first:
  **caption-side-only dilution** (mix embeddings, keep latent/target pure — pure
  anti-caption-noise, no off-manifold latents) and **full mixup** (the exact identity
  above; open distributional question: whether training on chord blends biases
  generations toward blends — hedge with small-α Beta). Untested in diffusion;
  flagged as the highest-value transfer experiment from this series. The L3 path
  (real image augmentation) remains strictly preferable where semantically safe.
- **Predicted next rung (untested)**: k-NN/local-PCA-directed jitter — *on-manifold*
  chords, still domain-agnostic — should land between mixup and crop. The test is the
  same arena; the prediction is on the books.
## 6. Target interpolation without a teacher forward (the κ identity)

For MSE losses (diffusion), interpolating the target toward the deploy-weight teacher
needs no teacher evaluation — it linearizes into the optimizer:

```text
L  = ½‖f_W − (λ·v + (1−λ)·f_P)‖²
∇L = λ·g_plain + (1−λ)·Jᵀ(f_W − f_P)
   ≈ λ·g_plain + (1−λ)·JᵀJ·(W − P)        [f_W − f_P = J·u + O(‖u‖²); |u| measured ~0.006]
```

The teacher term is a Gauss-Newton-weighted pull of `W` toward `P` — and its diagonal
approximation **is the evaporation term**: `u ← u − lr·κ(1−coh)·u`, with the coherence
gate as a per-weight adaptive λ (distill where the deviation reads as noise, exempt
where it reads as learning). **Concord's dissipation is first-order self-distillation
from the Polyak teacher, at zero extra forwards and zero extra memory** — which
retrodicts κ's anti-memorization record (exps 4–12) as the known distillation-from-EMA
label-noise remedy, reads the κ tradeoff (exp 5) as the textbook
regularization-vs-underfitting tradeoff with an adaptive λ, and makes the autotuner a
device that tunes distillation strength to the measured noise level. The exact teacher
differs only by the off-diagonal metric (function-space vs weight-space pull); the
intermediate rung, if ever needed, is `JᵀJ·u` via one JVP on the existing graph — still
no teacher forward. The open empirical question is therefore not "does teacher
distillation help" (κ answered it) but **what the off-diagonal Fisher buys over the
free diagonal** — the explicit-teacher arm vs κ-matched friction, same arena.

- **For the noise machinery in the optimizer**: exp 17 closes the question the σ
  ablations kept reopening, with one amendment from §6: the optimizer *can*
  synthesize the teacher-shaped form of target dilution — it already does, as κ. The fluctuation half of the design was reaching for
  augmentation-character noise, and the measurement says that character requires
  support beyond the empirical sample — which an optimizer alone cannot synthesize,
  and a two-line trainer change can. σ stays default-off; the cascade keeps the
  weight-space side of the filter; vicinity belongs to the data loader.
