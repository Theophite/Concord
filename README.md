# Concord

An optimizer whose state **is** the weight. Each parameter is a single int32 — an int16
fast accumulator plus int8 slow and anchor fields, on shared per-row/per-column
exponents — and the optimizer step is fused into the autograd backward: the moment a
layer's gradient exists, a Triton kernel folds it into the packed word, in registers,
and stores it back. There is no fp32 master copy, no momentum tensor, no second moment,
and no `optimizer.step()`.

Adaptivity comes from the format itself: the three fields are EMAs of the same
trajectory at different timescales, and the **gaps between them** carry the drift and
variance information Adam keeps in its moment buffers. A drift-cancelled residual gives
a per-weight Wiener gain (signal vs noise in the velocity); the gain steers
consolidation, an evaporation term drains what it calls noise, and the weight you ship
is the slow (Polyak-averaged) position — all read from and written back into the same
32 bits. In familiar terms: **EMA Lookahead AdaFactor behind a Kalman-style gate,
computed in register.**

Why bother: optimizer state drops from ~12–16 bytes/param (mixed-precision AdamW) to
4 total, and per-step optimizer HBM traffic by a similar factor. That is the difference
between LoRA-only and a **full SDXL UNet fine-tune in ~15 GB on a 24 GB card** — the
project's reason to exist.

## Measured

- **Language-model bench** (nanoGPT char-LM, same-seed A/B at fixed lr; deployed-weight
  validation loss, lower better): Concord winner **1.4967** vs AdamW 1.534 vs Muon
  1.578 — Concord carrying 32 bits/param of total state against AdamW's ~96 of
  optimizer state alone. The exact configuration, the A/B logs, and the honest caveats
  (which deltas are deterministic, which are single-seed) live in
  [`WINNING_CONFIG.md`](WINNING_CONFIG.md) — the single source of truth.
- **SDXL full-UNet fine-tuning**: functional with validated samples, ~15 GB training
  footprint with the fused dequant-matmul (~20 GB cached) — the `concord-integration`
  branch.
- **CPU dynamics campaign**: a derivation bug in the gate's drift-cancellation found
  and fixed, the dissipation-vs-noise curve mapped (κ\* ≈ min(1000·ρ, 400)), the
  dissipation made self-tuning from the gate's own coherence, and the autotuned
  optimizer beating AdamW at every label-noise level tested (+0.75 to +1.74) —
  [`docs/RESULTS.md`](docs/RESULTS.md), with reproduction scripts in
  [`experiments/cpu_dynamics/`](experiments/cpu_dynamics/).

## Repository map

| where | what |
|---|---|
| [`concord/`](concord/) | **the canonical package** — `packed_b.py` is the optimizer: layers, Triton kernels, coherence meter, dissipation autotuner |
| [`docs/`](docs/) | [`SDXL_WINNER_REPORT.md`](docs/SDXL_WINNER_REPORT.md) (what it is) → [`HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) (mechanism deep-dive) → [`RESULTS.md`](docs/RESULTS.md) (measured dynamics & autotuning) → [`INSTALL_SDXL.md`](docs/INSTALL_SDXL.md) (porting into the fork) → [`BIBLIOGRAPHY.md`](docs/BIBLIOGRAPHY.md) (relation to prior work) |
| [`WINNING_CONFIG.md`](WINNING_CONFIG.md) | the validated configuration, exact, with provenance |
| [`experiments/cpu_dynamics/`](experiments/cpu_dynamics/) | active CPU experiment suite: reference implementation of the update rule, exps 1–8, parity tests, lab log |
| [`dist/concord_winner/`](dist/concord_winner/) | frozen snapshot of the distilled winner (bug fixes synced in; no new features) |
| [`notebook/`](notebook/) | the research notebook — prototypes, experiment series, run records, session notes; historical, indexed in [`notebook/README.md`](notebook/README.md) |
| [`CLAUDE.md`](CLAUDE.md) | invariants, culture, and traps — read before changing anything |

**Branches.** This is `main`, the research repo. `concord-integration` is a full fork
of [OneTrainer](https://github.com/Nerogar/OneTrainer) with Concord wired into SDXL
fine-tuning (production kernel under `modules/util/optimizer/concord/`; preset:
`training_presets/#SDXL Concord Fused 24GB.json`). The two branch histories are
unrelated; [`docs/INSTALL_SDXL.md`](docs/INSTALL_SDXL.md) is the bridge between them.

## Quick start

```bash
# GPU (the real thing): smoke-test the packed kernels
python concord/packed_b.py

# Minimal use: a self-stepping drop-in for nn.Linear
#   from concord.packed_b import ConcordLinearPackedB
#   layer = ConcordLinearPackedB(in_f, out_f); layer.lr = 5e-4
#   loss.backward()            # the optimizer step happens HERE
#   layer.rebalance()          # per-step exponent guard
#   W = layer.consolidated_weight()   # ship THIS (the slow weight), not the live one

# CPU (no GPU): the reference implementation + dynamics experiments
pip install torch --index-url https://download.pytorch.org/whl/cpu
python experiments/cpu_dynamics/test_autotuner_parity.py
```

Norms, biases, and embeddings take a small standard optimizer alongside (the SDXL fork
uses plain SGD); Concord goes on the 2D weights, which is where the parameters are.

## Status

Active research, not a product. Validated where stated above, with the caveats kept
attached — single-seed deltas are labeled, CPU findings carry their adoption gates
(`docs/RESULTS.md` §"Threats to validity"), and anything not on the SDXL fine-tune path
in the fork falls back to stock OneTrainer behavior.
