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
  2,200-line single file IS the product). `src/` = prototypes and training harnesses
  (`prototype_packed_b.py` is the same lineage, older). `sims/` = closed-loop theory
  sims. `dist/concord_winner/` = a **frozen snapshot** of the validated winner: sync
  bug fixes into it, never add features. `experiments/cpu_dynamics/` = CPU dynamics
  campaign (see below).
- `concord-integration` — a full fork of OneTrainer with Concord wired into SDXL
  fine-tuning. The production kernel there is
  `modules/util/optimizer/concord/prototype_packed_b.py` (yes, "prototype" in the name;
  it is the production kernel). Wiring: `concord_winner.py` (controller),
  `concord_ot.py` (ConcordController), `concord_graph.py` (manual CUDA-graph capture),
  `modules/modelSetup/StableDiffusionXLFineTuneSetup.py` (the SDXL surface).

**Read in this order:** `SDXL_WINNER_REPORT.md` (what it is) → `HOW_IT_WORKS.md` (how
the kernel realizes it) → `RESULTS.md` (measured dynamics, the C\* fix, autotuning) →
`WINNING_CONFIG.md` (the exact validated configuration — single source of truth for
numbers) → `INSTALL_SDXL.md` (porting changes into the fork).

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
