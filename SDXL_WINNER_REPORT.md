# Concord: what it does, centered on the SDXL winning configuration

A condensed map of this repository. Two branches, one idea:

- **`main`** — the research repo where the Concord optimizer was developed and validated:
  closed-loop simulations (`sims/`), CIFAR/nanoGPT/T5 ablations (`src/`), the distilled
  winner package (`dist/concord_winner/`, `concord/`), and `WINNING_CONFIG.md` (the exact
  validated configuration, with provenance).
- **`concord-integration`** — a full fork of [OneTrainer](https://github.com/Nerogar/OneTrainer)
  with that winner wired into the SDXL fine-tuning path. The deliverable: a **full SDXL UNet
  fine-tune (not LoRA) in ~15 GB**, fitting a 24 GB card with headroom.

The full mechanism deep-dive — storage format, kernel pipeline, gating, integer hygiene,
graph engineering — is in [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md).

## The idea

A normal AdamW fine-tune pays 3–4× the weight memory for optimizer state: an fp32 master
copy plus two moment tensors. Concord stores **everything in one int32 per weight** — the
weight *is* the optimizer state:

```
bits 31..16  s_fast  (int16)  fast accumulator — catches each step's gradient
bits 15..8   s_slow  (int8)   consolidated position
bits  7..0   v_slow  (int8)   long-time anchor

weight = (s_slow·128 + s_fast + v_slow·128) · 2^(row_exp + col_exp − 15)
```

The shared per-row/per-column exponents give the ~17-bit signed mantissa its dynamic range
(finer than bf16's 8-bit mantissa, at the same 32 bits/param as the raw weights alone).

Three mechanisms make this an optimizer rather than just a quantization format:

1. **Self-stepping backward.** Every swapped `nn.Linear`/`nn.Conv2d` is an autograd
   `Function`; the moment its gradient exists, a Triton kernel folds it into the packed
   accumulators with stochastic rounding (unbiased low-bit accumulation). There is no
   `optimizer.step()` for these layers — only the non-swapped leftovers (norms, biases,
   embeddings) see a plain SGD.
2. **Timescale gaps replace Adam's moments.** The fast→slow "chase" (α = 0.1) and
   slow→anchor "leak" (α_v = 0.001) are mass-preserving redistributions; the gaps between
   the fields carry the gradient's variance and drift — the same signal Adam reads from its
   second moment — without storing one. A coherence gate scales each step by how consistent
   the gradient direction has been. Net behavior: a factored, variance-adaptive AdamW
   (rank-1 v̂, sqrt preconditioner) read from and written back into that one int32.
3. **Fused dequant-matmul.** The forward reconstructs the bf16 weight *inside* the Triton
   matmul, so no persistent bf16 weight copy exists (~5 GB saved on SDXL: ~15 GB footprint
   fused vs ~20 GB cached).

## In familiar terms: EMA Lookahead AdaFactor behind a Kalman gain, in register

Each ingredient of the recipe is a known optimizer idea; Concord's contribution is fusing
them into one packed word and one kernel. Line refs are `concord/packed_b.py` (the same
kernel ships in the fork as `modules/util/optimizer/concord/prototype_packed_b.py`).

- **EMA cascade.** The three fields are exponential moving averages of each other at three
  timescales: `s_slow` tracks the live weight at α = 0.1, `v_slow` tracks `s_slow` at
  α_v = 0.001, and the factored second moment is a β₂ = 0.999 EMA of g².
- **Lookahead, continuous and nested.** The "chase" (L604–611) is exactly Lookahead's
  slow-weight update `slow ← slow + α·(fast − slow)`, run every step instead of every k:
  the slow position absorbs α of the fast displacement and the fast field keeps the rest,
  with the live weight invariant across the transfer (it redistributes mass, it doesn't
  step). The leak (L616–629) is a second Lookahead level at α_v. Deploying from
  `consolidated_weight()` — drop `s_fast` — is Lookahead's "evaluate at the slow weights."
- **AdaFactor preconditioner.** The second moment is rank-1 factored exactly as in
  AdaFactor: row/col marginals of g² kept as EMAs (L1269–1284), reconstructed in-kernel as
  `v̂ = v_row ⊗ v_col / Σv_row` (L498–503), and the step is `g / (v̂ + ε)^0.5` (L539–541;
  at winner settings v_scale = 0, δ² = 1, so v̂ *is* the denominator). The only optimizer
  state outside the packed word is these two O(N+K) fp32 vectors per layer — and the
  storage format's shared row/col exponents echo the same factorization.
- **Kalman-style gating.** The coherence gate (L512–524, the kernel calls it a
  "Wiener/Kalman SNR gate") decomposes the fast field into signal + noise using the two
  time-lagged slow states: signal = C\*·(s_slow − v_slow) — the steady drift, with C\*
  derived analytically so E[noise] = 0 under pure drift (L52–84) — and noise = the
  residual. The gain `coh = S²/(S² + N²)` is the Wiener / steady-state Kalman gain, and
  each consolidation stage accepts its innovation in proportion to estimated SNR: the
  chase and leak gates (L596–597, L619–620) blend toward the floor schedule, and the
  `gf_consol` evaporation (L568) drains the `(1 − coh)` fraction out of the velocity
  before it can consolidate. This is the "measurement update" view: slow state = estimate,
  fast state = observation, per-weight gain set by measured SNR.
- **Everything in register.** All of the above is one Triton kernel launch per layer per
  step (`_apply_packed_adamw_kernel`, L404–685): load the int32 word and the gradient,
  unpack, gate, precondition, stochastically-round all three fields, repack, store — and
  emit the next forward's bf16 weight (L663–671) plus the rebalance watermarks via
  `atomic_max` (L675–685) on the way out. The optimizer state never exists unpacked in
  global memory; momentum, lookahead state, anchor, and the gate's inputs live only in
  registers between load and store.

## The winning configuration

Validated on `main` by same-seed A/B (nanoGPT-char; metric is deployed-sv, lower is
better). The winner is a **fluctuation–dissipation pair on top of the bare recipe**:

| arm | deployed-sv | Δ |
|---|---|---|
| bare recipe (rank-1 v̂ AdamW + fixed coherence gate) | 1.5404 | — |
| + split dissipation (`ratio_coh`, chase/leak floors → 0.1, `gf_consol 50`) | 1.5180 | −0.022 |
| **+ isotropic noise σ = 0.6 (`sf_060`, the winner)** | **1.4967** | −0.021 |
| AdamW baseline (32 b/param) | ~1.534 | |
| native Muon | ~1.578 | |

- **Dissipation** — coherence-gated friction. `gf_consol 50` evaporates low-coherence
  (noisy) mass from the fast field back toward the anchor each step; confident params are
  untouched. The live coherence-ratio gate (`ratio_coh`) replaces the EMA side-buffer, keeping
  the state at a true 32 b/param. Chase/leak gate floors cosine-decay 0.9/0.999 → 0.1 over
  ~1 epoch. This Δ (−0.022) is deterministic same-seed: real.
- **Fluctuation** — isotropic white noise, peak σ = 0.6, rising-late schedule, injected in
  the fused backward off the deploy weight. Single-seed Δ (−0.021) with ~0.01 trajectory
  jitter — mechanism solid, magnitude wants multi-seed confirmation. (The ablation refuted
  structured Σ_g noise: isotropic ≥ Σ_g.)
- **Deploy** — exported weights are `consolidated_weight() = (s_slow + v_slow)·128·2^exp`,
  i.e. **drop `s_fast`**. Shedding the fast velocity at save time is part of the win; saved
  checkpoints are ordinary SDXL safetensors.

`WINNING_CONFIG.md` on `main` is the single source of truth for every number above.

## The optimizer, in its simplest form

Per weight, Concord is a **noisy, damped, driven particle** — a fluctuation–dissipation
pair around a preconditioned gradient flow. The state is a slow position `P` (the weight
you ship) and a velocity `u` (the live weight the network runs is `W = P + u`), plus one
rank-1 second moment `v̂` per layer. Everything else in the kernel is the integer
realization of this rule:

```text
g̃   = g + σ·‖g‖·ξ ,   ξ ~ N(0, I)                   # fluctuation
v̂   ← rank-1 EMA of g̃²   (AdaFactor, β₂ = 0.999)    # preconditioner
coh = μ² / (μ² + (u − μ)²) ,   μ = C*·d              # Kalman gain: SNR of the velocity

u ← u − lr·clip( g̃ / √(v̂ + ε), ±c )                 # drive: preconditioned gradient
      − lr·κ·(1 − coh)·u                             # dissipation: friction on noise
      + β1·coh·u                                     # optional momentum (default β1 = 0)

P ← P + α·gc·u ,   u ← (1 − α·gc)·u                  # consolidation: continuous Lookahead
      gc = φc + (1 − φc)·coh
```

Line by line:

- **Fluctuation.** Isotropic gradient noise at magnitude σ·‖g‖ per layer, σ ramping
  0 → 0.6 as the learning rate decays (rising-late). It is injected before v̂, so it passes
  through the preconditioner. Its job is the classic one: keep shaking coordinates whose
  gradient signal is incoherent so they can't settle into noise-fit minima.
- **Preconditioner.** AdaFactor's factored second moment; the step is RMS-normalized and
  clipped at c = 10. No first moment is stored anywhere.
- **The gain.** The velocity is split into drift + noise. The drift prediction μ = C*·d
  reads a telescope `d` maintained *inside* `P`: P's two halves relax toward each other at
  α_v = 0.001 (gated like the chase, floor 0.999 → 0.1), so their gap is a long-window
  record of consolidated motion — and `P` itself never moves from this. C* ≈ 0.00908 is
  derived analytically so that u − μ is zero-mean under a pure-drift gradient stream,
  which makes `coh = μ²/(μ² + (u−μ)²)` a true Wiener/Kalman gain on the velocity's SNR —
  computed entirely from state already in the packed word.
- **Dissipation.** Friction proportional to the estimated noise fraction: incoherent
  velocity evaporates at rate lr·κ (κ = 50; lr-proportional, so the cosine schedule
  self-fades the friction in the tail and late small signal isn't over-skimmed). Coherent
  velocity feels no friction.
- **Momentum (optional).** A coherence-gated heavy-ball term: only the coherent fraction
  of the velocity is reinforced, so momentum can't run away on noise (ungated, it
  diverges — the velocity is part of the live weight and feeds back through the
  preconditioner). **Off (β1 = 0) in the validated winner.**
- **Consolidation.** Lookahead's slow-weight update run every step instead of every k:
  the position absorbs the fraction α·gc of the velocity (α = 0.1), gated by coherence
  with a floor φc annealing 0.9 → 0.1 over the run. Note the emergent momentum: at full
  coherence `u` recurses with factor (1 − α) = 0.9, so the chase alone gives coherent
  weights a classic momentum memory even at β1 = 0 — and that memory *is* `W − P`.

**The fluctuation–dissipation pair is the validated win.** Noise pumps incoherent
coordinates; friction drains them; only statistically coherent gradient signal survives to
accumulate into `P` — and `P` is what you save (`consolidated_weight()` = drop `u`). On the
same seed, each half contributed ≈ −0.02 deployed-sv (see the winning configuration above).

**Making it exact.** The kernel runs this rule in integers, one int32 per weight: `u` is
`s_fast`, `P = (s_slow + v_slow)·128`, the telescope `d = (s_slow − v_slow)·128`, all
scaled by `2^(row_exp + col_exp − 15)`. Every write is stochastically rounded, so the
expectation equals the rule above and quantization error is unbiased; consolidation
transfers mass without moving `W`, so `W` changes only through the drive and the friction.
Host schedules per step: cosine factor f = 0.2 + 0.4·(1 + cos πt/T); lr = lr_peak · f ·
min(1, (t+1)/100); σ = 0.6·(1 − f); both gate floors anneal on the same cosine. Shared
exponents rebalance (+1, with a stochastically-rounded right shift) only when a layer's
mantissa exceeds 24000. The only optimizer state outside the packed word is v̂'s two
O(N+K) vectors per layer.

### The variational-inference reading

The same dynamics admit a Bayesian interpretation — and it is the one the codebase itself
uses. `src/concord_polyak.py` (on `main`) states it outright: the packed format carries an
implicit variational posterior `q(W) = N(μ, τ²·Σ)`, with the mean in the slow fields and
the spread in the fast/slow gaps.

- **The live weight is a particle, not an estimate.** Injected noise (fluctuation) plus
  coherence-gated friction (dissipation) make `W`'s dynamics Langevin-like: it explores an
  approximate posterior whose temperature is set by σ (the code names its knobs
  accordingly — `v_scale` is documented as "temperature on the preconditioner", and the
  research harness has a `concord_polyak_temperature` Metropolis accept gate).
- **`P` is the posterior mean, computed online.** The continuous-Lookahead chase makes `P`
  an EMA of the sampling trajectory — Polyak averaging, the posterior-mean estimator of
  constant-lr SGD viewed as approximate inference. Deploying `consolidated_weight()` means
  shipping the posterior *mean* rather than a posterior *sample* — which is why dropping
  `s_fast` at save time wins (deployed-sv 1.518 vs live 1.556 on the split arm).
- **The gate is posterior shrinkage.** Under the Gaussian signal+noise split, `coh·u` is
  `E[drift | observed velocity]` — the MMSE estimate (the Wiener gain *is* the posterior
  mean operator). The chase commits exactly this conditional mean into `P`; the
  dissipation evaporates the posterior-noise remainder instead of letting it consolidate.
- **The anchor is the prior.** `v_slow` is the long-window mean — in the code's words,
  "the part of the weight supported by the training distribution" — and the
  Bayesian-anchored decay terms (`wd_sv`/`wd_sf`; `wd_anchor` in the fork's frozen-anchor
  TE) shrink the less-confirmed transients toward it: per-element, confidence-weighted
  regularization, "less decay where the data has spoken, more decay where it hasn't"
  (`CONCORD_README.md`). For fine-tuning, `load_weights_finetune` /
  `load_weights_anchor` place the *pretrained* weight in `v_slow`, centering the prior on
  the pretrained model — the L2-SP/EWC move, which is exactly the frozen-anchor CLIP-L
  mode in the fork.
- **The variance map comes free.** The drift-cancelled residual `u − μ` estimates each
  weight's posterior variance — the "which weights have converged?" diagnostic that the
  C\* fix restored.

One line: the packed word stores a per-weight Gaussian posterior (mean = slow fields,
fluctuation = fast field); the kernel explores it at temperature σ, Wiener-shrinks each
observation to its conditional mean, consolidates that mean, regularizes toward the
prior, and ships the posterior mean — all as integer ticks on the same 32 bits.

## As shipped for SDXL (`concord-integration`)

The fork is stock OneTrainer unless you pick the **CONCORD** optimizer. The preset
`training_presets/#SDXL Concord Fused 24GB.json` is the winner applied to SDXL:

| setting | value | why |
|---|---|---|
| optimizer | CONCORD, winner knobs baked in (`gf_consol 50`, `ratio_coh`, `sigmag_peak 0.6`, `warmup 100`, `lr_min_frac 0.2`) | the sf_060 arm |
| learning rate | 7.5e-5, cosine | fine-tune scale of the validated schedule shape |
| layer filter | `attn-mlp` preset | swap only attention + MLP Linears (794 layers); the rest stay standard and frozen |
| precision / batch | bf16, batch 1, accumulation 1, gradient checkpointing, latent caching | the CUDA-graph-capturable configuration |
| `concord_fused_matmul` | on | dequant inside the matmul; ~15 GB instead of ~20 GB |
| `concord_cuda_graph` | on | manual capture of predict → loss → backward (+ the fused self-step) for batch-size-1 throughput; fresh noise injected per replay |
| diffusion recipe | offset noise 0.0357, input perturbation 0.01, inverted-parabola timesteps | wired through both the eager and captured-graph paths |

Supporting machinery added by the integration:

- **Controller** (`concord_winner.py`) — per-step warmup/cosine LR, rising-late σ, decaying
  coherence floors, all held in device tensors so CUDA-graph replays see live values
  (a scalar would freeze at its capture-time value — a bug that was actually hit).
- **Gated rebalance** — the shared exponents re-center only when a layer's mantissa actually
  approaches overflow (`MAX_M = 24000`); one reduction replaces 794 per-layer kernel
  launches per step in the common no-op case (~1.8× faster iteration).
- **Token control plane** — declaratively train / freeze / zero individual text-encoder
  tokens (`concord_sanitize_tokens`), branch-free and graph-safe; new tokens train through
  the same packed core.
- **Frozen-anchor TE** (newest) — CLIP-L trains as an elastic delta around its pretrained
  weights pinned in `v_slow`, with `wd_anchor` pulling the delta home.
- **Save/resume bridges** — final saves consolidate packed layers back into standard
  `nn.Linear`/`nn.Conv2d` (checkpoints load as ordinary SDXL anywhere); backups carry the
  full packed state and resumes restore it bit-exactly, resyncing the weight cache.
- **Contiguous aspect bucketing** — keeps each resolution bucket a contiguous run so the
  CUDA graph isn't recaptured on every shape change.

Where the code lives: `modules/util/optimizer/concord/prototype_packed_b.py` (the packed
layers + Triton kernels, the core), `concord_winner.py` + `control_plane.py` (controller and
token plane), `modules/util/optimizer/concord_graph.py` (graph capture),
`modules/modelSetup/StableDiffusionXLFineTuneSetup.py` (the SDXL wiring).

## Status and caveats

- SDXL full fine-tuning is functional with validated samples; everything outside that path
  is stock OneTrainer behavior, rough edges expected.
- The dissipation gain is confirmed deterministic same-seed; the noise gain (σ = 0.6) is
  single-seed on nanoGPT — multi-seed it on the target task before treating the magnitude
  as load-bearing.
- The codebase carries deliberate scar tissue: env-gated probes (`CONCORD_MEMLOG`,
  `CONCORD_GRAPHMEM`), ablated mechanisms left as off-by-default knobs, comments documenting
  past failures (int8 `s_fast` saturation, the tick-down oscillation, graph-replay scalar
  freezing), and a restart-on-sample wrapper that works around CUDA-graph memory-pool
  fragmentation. Fallbacks are layered: fused → cached matmul, graph → eager, and every
  save path restores training state in `finally`.
