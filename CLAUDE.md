# CLAUDE.md

This repository is a strange object. Read this before changing anything.

## What this is

Concord: a research optimizer that stores the **entire optimizer state inside the
weights** — one int32 per parameter (int16 `s_fast` velocity + int8 `s_slow` position +
int8 `v_slow` anchor, on shared per-row/col exponents) — with the optimizer step fused
into the autograd backward via Triton kernels. Mechanically: EMA Lookahead AdaFactor
behind a Wiener/Kalman gate, in register. Purpose: Adam-class full fine-tuning at
weight-only memory (full SDXL UNet on a 24 GB card).

**Two unrelated branch histories share this repo:**

- `main` — the research repo. `concord/packed_b.py` is the **canonical package** (the
  2,200-line single file IS the product). `docs/` = the current reports. `dist/
  concord_winner/` = a **frozen snapshot** of the validated winner: sync bug fixes into
  it, never add features. `experiments/cpu_dynamics/` = the active CPU dynamics
  campaign (see below). `notebook/` = the **research notebook**: prototypes
  (`notebook/src/` — `prototype_packed_b.py` is the winning lineage, older than the
  package), theory sims, the `run_*.sh` A/B records (machine-specific, `cd /c/concord`
  — read for flags, don't execute), and session notes; indexed in
  `notebook/README.md`, historical, unmaintained except forward-ported bug fixes.
- `concord-integration` — a full fork of OneTrainer with Concord wired into SDXL
  fine-tuning. The production kernel there is
  `modules/util/optimizer/concord/prototype_packed_b.py` (yes, "prototype" in the name;
  it is the production kernel). Wiring: `concord_winner.py` (controller),
  `concord_ot.py` (ConcordController), `concord_graph.py` (manual CUDA-graph capture),
  `modules/modelSetup/StableDiffusionXLFineTuneSetup.py` (the SDXL surface).

**Read in this order:** `docs/SDXL_WINNER_REPORT.md` (what it is) →
`docs/HOW_IT_WORKS.md` (how the kernel realizes it) → `docs/RESULTS.md` (measured
dynamics, the C\* fix, autotuning) → `WINNING_CONFIG.md` (the exact validated
configuration — single source of truth for numbers) → `docs/INSTALL_SDXL.md` (porting
changes into the fork).

## Invariants that will bite you

1. **The weight IS the optimizer state.** Live weight `W = (s_fast + 128·s_slow +
   128·v_slow)·2^exp`; deploy weight drops `s_fast`. The three fields at different
   timescales are not redundancy to be cleaned up — the gaps between them carry the
   variance/drift information Adam stores in its moments.
2. **Mass preservation is load-bearing.** The chase and leak *transfer* integer mass
   between fields (live weight invariant across the chase, deploy weight invariant
   across the leak). Clamps must run BEFORE mass-preserve subtractions or mass is
   silently destroyed (this bug happened; the comment at the clamp records it).
3. **Constants are derived, not arbitrary.** `drift_cancel_C` (C\*) is the analytic
   function of (α, α_v, mass_preserve) — it has been 11× too large and 2× too small in
   this repo's history; if you touch the chase/leak rates or semantics, **re-derive
   C\*** (see `compute_drift_cancel_C`'s docstring and `RESULTS.md` §2).
   `MANTISSA_BIAS=15`, `S_SLOW_FACTOR=128`, `MAX_M=24000`, `EXP_MIN/MAX` are coupled to
   the bit layout and the rebalance math.
4. **Stability ceilings:** the dissipation diverges past `lr·gf_consol ≈ 2` (the int16
   clamp saturates on GPU instead of NaN-ing — quieter, equally dead). **Warmup is
   load-bearing**: init puts the whole weight in `s_fast`, and warmup holds the
   lr-proportional friction off while the chase consolidates it.
5. **CUDA-graph discipline** (the most-burned bug class in this codebase): anything
   that changes per step MUST cross the graph boundary as a device tensor
   (`.fill_()` outside, `tl.load` inside) — lr, σ, gate floors, consf, step salt all
   do. A Python scalar or bool is baked at capture time. `gf_consol`/`beta1` are
   currently launch-time scalars — fine eager, baked under capture (see
   `INSTALL_SDXL.md` §3d).
6. **Stochastic rounding keeps the integers honest.** Every fractional write is
   floor + Bernoulli(frac) with an xorshift hash keyed by (value, position, step-salt).
   A new update term needs its own SR stream (distinct salt constant) — reusing a
   stream correlates errors and biases the accumulation.
7. **The gate's structural blind spot:** coherence ≠ generalization. Memorization
   drift is temporally coherent, so any new consumer of `coh` (momentum especially)
   must be tested under label noise before adoption (`RESULTS.md` §6).
8. **Gradient accumulation** works by gating consolidation (`consf` device flag), not
   by summing `.grad`: the tick always fires into `s_fast`; chase/leak/evap/weight-emit
   fire only on the update step. The bf16 `weight_buf` doubles as the freeze buffer in
   cached mode; `fast_gain` must stay 1.0 under accumulation (guarded in the SDXL
   setup).
9. **The frozen-anchor TE** runs `alpha_v_fast = 0`: its telescope never advances and
   its coherence reads ~0 regardless of data quality. Exclude TE layers from any
   coherence-based measurement (the autotuner uses UNet layers only).
10. **Checkpoint duality:** final saves consolidate packed layers back to standard
    `nn.Linear`/`Conv2d` (checkpoints load as ordinary SDXL anywhere); backups carry
    the full packed state, and resume must resync `weight_buf` after restoring
    `packed_w`.

## The flow audit: boil and waste (integration branch)

Every `[loss]` stdout line on `concord-integration` carries the kernel's flow audit
next to the loss/gap numbers (TensorBoard: `loss/concord_boil`, `loss/concord_waste`).
The two meters answer different trust questions about the dissipation — read them as a
pair, neither alone is sufficient.

**Plumbing.** The step kernel accumulates three energy sums per device into an fp32[3]
buffer via atomics (graph-safe, same device-buffer pattern as memgap;
`prototype_packed_b.py:661`): `[0] += Σ killed_w²·coh`, `[1] += Σ killed_w²`,
`[2] += Σ chase_w²` — killed_w is evaporated mass, chase_w is consolidated
(fast→slow committed) mass, both in W units.
`ConcordController.read_flow_audit()` (`concord_ot.py:314`) reads-and-zeroes per print
window; an empty denominator drops the field from the line.

- **boil = [0]/[1]** — the drift-recognized fraction of killed energy. *Of what
  friction destroyed this window, how much did the telescope's own coherence meter
  recognize as real drift?* Healthy ≈ 0: the evaporator eats unrecognized mass
  (hygiene working). Sustained high boil = friction is burning weight with evidence
  behind it → λ too high for the regime.
- **waste = [1]/([1]+[2])** — the kill share of total moved energy (kills vs
  commits). The lag tax: weight commits to `s_fast` first, so mass can be destroyed
  *before* its evidence accumulates — and those kills carry coh≈0 at kill time, so
  they look legitimate to boil. **Boil is structurally blind to infanticide; waste is
  the meter that sees it.** High waste + low boil = commit-then-kill churn (raise
  `evap_build_min`, or the lr is outrunning the gate).

**Expected transients — don't tune on them:**

- **Init consolidation** (first ~2/α optimizer steps): coh reads init residue, so
  boil climbs steeply while the fill ramp still holds λ at a few % of target — a high
  ratio on negligible flow. The TRUE UNet trace (measured 2026-06-11 with embeddings
  frozen by the divot, λ=0.5, γ-SNR, epoch window): peak ~0.31 at the warmup
  boundary (~log-step 100), then a slow washout (~0.05 by 250, ~0.02 by 670) as the
  init-residue coherence dies under the rising fill ramp. Waste ≤0.001 throughout.
- **The denominators are SHARED across every packed group**, and that has burned us:
  packed-embedding kills carry coh≡0 and kill energy ∝ lr², so a hot embedding group
  (TI-scale lr 1e-3 = 100× UNet) DOMINATES the denominator — an earlier run's boil
  "collapsed to ≤0.005 by step ~115" and was documented here as the healthy
  signature; it was dilution masking the UNet's real ~0.3 peak (the collapse step
  matched the embedding group's warmup completion exactly, and with embeddings
  frozen the collapse vanished). At lr_emb ≈ 3e-5 the pollution is ~1000× smaller
  and boil reads as UNet-only. Before trusting a boil level, ask what else is
  killing into the shared buffer and how hot its lr is.

(`gap` on the same line is unrelated plumbing: the first-order deploy−live loss
estimate from the memgap buffer. Positive spikes at init and after backup/restore that
re-converge within ~10 prints are bridge behavior, not flow.)

## Methodology norms (the repo's culture — follow them)

- **Same-seed A/B is the validation currency.** Concord is bit-deterministic at fixed
  seed; a deterministic Δ is real, a single-seed Δ gets flagged as such. The metric is
  the **deployed** weight's performance (deployed-sv / deploy accuracy), not the live
  weight's.
- **Provenance discipline:** numeric claims carry file:line references and log paths.
  `WINNING_CONFIG.md` is the single source of truth for the validated config — update
  it only with receipts.
- **Scar tissue is documentation.** The kernel comments record failure modes (int8
  saturation, tick-down oscillation, units-broken gates, graph-replay freezes). Do not
  strip them; add to them when you fix something. Ablated mechanisms stay as
  off-by-default flags rather than being deleted.
- **Honest caveats survive editing.** "Single-seed," "one task deep," "needs the
  nanoGPT A/B" qualifiers are part of the result, not hedging to clean up.

## Working in this repo

- **No CUDA on your box?** `concord/packed_b.py` imports triton at module top and will
  not import on CPU. Two sanctioned patterns: (1) the **CPU reference**
  `experiments/cpu_dynamics/concord_ref.py` — real-valued mirror of the kernel rule
  (kernel == reference in expectation via SR); use it for dynamics experiments, never
  as a training path; (2) **exec-from-source** for pure-torch pieces — see
  `experiments/cpu_dynamics/test_autotuner_parity.py`, which regex-extracts and execs
  the shipped functions to test them without importing triton.
- **Tests:** `python experiments/cpu_dynamics/test_autotuner_parity.py` (CPU);
  `python concord/packed_b.py` (CUDA smoke test); on `concord-integration`,
  `tests/regression.py` runs a real short SDXL fine-tune (~2 min, GPU) — run it before
  and after touching the trainer/kernel surface.
- **Experiment data:** MNIST IDX files go in `experiments/cpu_dynamics/data/`
  (ossci-datasets.s3.amazonaws.com/mnist/, gitignored). Each `expN_*.py` is standalone;
  results land as JSON + PNG next to it; findings go in
  `experiments/cpu_dynamics/EXPERIMENTS.md` (the lab log) and get synthesized into
  `RESULTS.md`.
- **Env probes** (integration): `CONCORD_MEMLOG`, `CONCORD_GRAPHMEM` (on by default),
  `CONCORD_GRAPH_DEBUG`, `CONCORD_NO_RESTORE` (A/B resume), `CONCORD_RESTART_ON_SAMPLE`
  (graph-fragmentation workaround), `CONCORD_FUSED_MATMUL`.
- **Editing the kernel:** match the existing comment density — kernel comments here
  state constraints and history, and reviewers expect new mechanisms to arrive with
  (a) the derivation, (b) the SR stream salt, (c) the device-tensor story for graphs,
  and (d) an A/B or CPU-reference validation.
