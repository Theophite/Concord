# Concord

An optimizer whose state **is** the weight. Each parameter is a single **int32** — an
int16 fast accumulator plus int8 slow and anchor fields, on shared per-row/per-column
exponents — and the optimizer step is fused into the autograd backward: when a layer's
gradient is computed, a Triton kernel folds it into the packed word, in registers, and
stores it back. No fp32 master copy, no momentum or second-moment tensors, no
`optimizer.step()`. Adaptivity comes from the format itself: the fields are EMAs of the
same trajectory at different timescales, and the gaps between them carry the
drift/variance information Adam keeps in its moment buffers (a per-weight Wiener gain,
read for free). The weight you ship is the consolidated slow position
(`consolidated_weight()`), not the live one.

Total state: **32 bits per parameter** (vs ~96 bits of optimizer state for
mixed-precision AdamW) — which is what puts a **full SDXL UNet fine-tune (not LoRA) on
a 24 GB card**.

> **Note:** this README replaces an earlier one describing the predecessor design
> (40 bits/param, separate accumulators, a `CONCORD_SGD` patch applied onto OneTrainer).
> That era's documentation is [`CONCORD_README.md`](CONCORD_README.md) (historical) and
> its integration flow is superseded by the `concord-integration` branch below.

## Validated

- **nanoGPT char-LM bench, same-seed A/B** (deployed-weight validation loss, lower
  better): Concord winner **1.4967** vs AdamW 1.534 vs Muon 1.578 — Concord carrying 32
  bits/param of total state against AdamW's ~96 of optimizer state alone. Exact
  configuration, logs, and caveats: [`WINNING_CONFIG.md`](WINNING_CONFIG.md), the
  single source of truth.
- **SDXL full-UNet fine-tuning**: functional with validated samples, ~15 GB training
  footprint with the fused dequant-matmul — on the
  [`concord-integration`](https://github.com/Theophite/Concord/tree/concord-integration)
  branch (a full OneTrainer fork; pick the **CONCORD** optimizer or load
  `training_presets/#SDXL Concord Fused 24GB.json`).

## Repository map (this branch: `main`, the research repo)

| where | what |
|---|---|
| [`concord/`](concord/) | **the canonical package** — `packed_b.py`: the packed layers + Triton kernels; a bare `ConcordLinearPackedB` IS the validated recipe |
| [`WINNING_CONFIG.md`](WINNING_CONFIG.md) | the validated configuration, exact, with provenance |
| [`dist/concord_winner/`](dist/concord_winner/) | frozen snapshot of the distilled winner |
| [`src/`](src/), [`sims/`](sims/), [`tools/`](tools/), [`docs/`](docs/) | research history: prototypes (including the predecessor designs the old README described), theory sims, A/B run records, session notes |
| [`OneTrainer_integration/`](OneTrainer_integration/) | the design/handoff docs that drove the `concord-integration` fork |
| [`CONCORD_README.md`](CONCORD_README.md) | predecessor-design documentation (historical) |

**Branches:** `main` (research) and
[`concord-integration`](https://github.com/Theophite/Concord/tree/concord-integration)
(the SDXL training fork — the supported way to train with Concord). A pending branch,
[`claude/sharp-franklin-75znkc`](https://github.com/Theophite/Concord/tree/claude/sharp-franklin-75znkc),
carries a repository reorganization plus new material: mechanism reports
(`docs/HOW_IT_WORKS.md`, `docs/SDXL_WINNER_REPORT.md`), a CPU dynamics campaign with
results (`docs/RESULTS.md`), a fix to the gate's drift-cancellation coefficient, a
dissipation autotuner, an annotated bibliography, and porting instructions for the fork.

## Quick start

```bash
python concord/packed_b.py      # CUDA smoke test of the packed kernels
```

```python
from concord.packed_b import ConcordLinearPackedB

layer = ConcordLinearPackedB(in_f, out_f)   # drop-in for nn.Linear; bare = the recipe
layer.lr = 5e-4
loss.backward()                  # the optimizer step happens HERE
layer.rebalance()                # per-step exponent guard
W = layer.consolidated_weight()  # ship THIS (the slow weight), not the live one
```

Norms, biases, and embeddings take a small standard optimizer alongside; Concord goes
on the 2D weights, where the parameters are. Requirements: PyTorch ≥ 2.1 + Triton, CUDA.

## Status

Active research, not a product. Validated where stated, with caveats kept attached;
single-seed deltas are labeled as such in `WINNING_CONFIG.md`.
