# Concord

A research optimizer that stores the **entire optimizer state inside the weights**: one
int32 per parameter — int16 fast accumulator + int8 slow position + int8 anchor, on
shared per-row/column exponents — with the optimizer step fused into the autograd
backward via Triton kernels. Mechanically it is **EMA Lookahead AdaFactor behind a
Wiener/Kalman gate, computed in register**; physically, a fluctuation–dissipation pair;
statistically, variational inference with the posterior mean as the shipped weight.

The point: Adam-class adaptive fine-tuning at weight-only memory — a **full SDXL UNet
fine-tune (not LoRA) in ~15 GB**.

## Validated results

- **nanoGPT/enwik8, same-seed A/B** (deployed-sv, lower better): Concord winner
  **1.4967** vs AdamW ~1.534 vs Muon ~1.578, all at 32 bits/param.
  Exact configuration and provenance: [`WINNING_CONFIG.md`](WINNING_CONFIG.md).
- **SDXL full-UNet fine-tuning**: functional with validated samples on a 24 GB card
  (~15 GB fused) — see the `concord-integration` branch.
- **CPU dynamics campaign** (June 2026): a calibration bug found and fixed (the
  drift-cancel coefficient), the dissipation-vs-noise curve mapped, the dissipation
  made self-tuning, and the autotuned optimizer beating AdamW at every noise level
  tested — [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository map

| where | what |
|---|---|
| [`concord/`](concord/) | **the canonical package** — `packed_b.py` is the optimizer (layers, Triton kernels, coherence meter, dissipation autotuner) |
| [`docs/`](docs/) | current reports: [`SDXL_WINNER_REPORT.md`](docs/SDXL_WINNER_REPORT.md) (what it is) → [`HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) (mechanism deep-dive) → [`RESULTS.md`](docs/RESULTS.md) (measured dynamics & autotuning) → [`INSTALL_SDXL.md`](docs/INSTALL_SDXL.md) (porting into the fork) |
| [`WINNING_CONFIG.md`](WINNING_CONFIG.md) | single source of truth for the validated configuration |
| [`experiments/cpu_dynamics/`](experiments/cpu_dynamics/) | the active CPU experiment suite: reference implementation, exps 1–8, parity tests, lab log |
| [`dist/concord_winner/`](dist/concord_winner/) | frozen snapshot of the distilled winner (bug fixes are synced in; no new features) |
| [`notebook/`](notebook/) | **the research notebook** — prototypes, experiment series, run records, session notes; historical, indexed in [`notebook/README.md`](notebook/README.md) |
| [`CLAUDE.md`](CLAUDE.md) | invariants, culture, and traps — read before changing anything |

**Branches:** `main` (this — the research repo) and `concord-integration` (a full fork
of [OneTrainer](https://github.com/Nerogar/OneTrainer) with Concord wired into SDXL
fine-tuning; its production kernel is `modules/util/optimizer/concord/`). The two
histories are unrelated; [`docs/INSTALL_SDXL.md`](docs/INSTALL_SDXL.md) is the bridge.

## Quick start

```bash
# CPU (no GPU needed): the reference implementation + experiment suite
pip install torch --index-url https://download.pytorch.org/whl/cpu
python experiments/cpu_dynamics/test_autotuner_parity.py   # parity tests
python experiments/cpu_dynamics/exp3_mnist.py              # MNIST ablation (data/ setup in EXPERIMENTS.md)

# GPU: the real thing
python concord/packed_b.py        # CUDA smoke test of the packed kernels
```

For SDXL training, use the `concord-integration` branch (install like stock OneTrainer,
pick the CONCORD optimizer, or load `training_presets/#SDXL Concord Fused 24GB.json`).

## Status

Active research. The optimizer is validated on the benches above; everything else
carries explicit caveats where it stands (single-seed flags, one-task-deep findings,
adoption gates) — those caveats are part of the results, not decoration.
