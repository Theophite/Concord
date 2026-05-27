# Concord AdamW — OneTrainer integration

Storage-efficient AdamW for SDXL fine-tuning. **40 bits per parameter**
of optimizer state vs `torch.optim.AdamW`'s 96 bits/param, with
comparable or slightly better accuracy on the configs we've tested. On
a 2.5B-param SDXL UNet this saves 17.5 GB of optimizer state — what
enables a full fine-tune on a 24 GB card.

The mechanics (three int accumulators + drift-cancelled variance +
Bayesian-anchored weight decay + materialised bf16 + cuDNN/cuBLAS
forward) are described in [CONCORD_README.md](./CONCORD_README.md).
This file is the project overview and integration map.

## Quick start

```bash
# Standalone reference example (CIFAR-10, no SDXL needed):
python cifar_concord_adamw.py
# → 90.91% best / 90.66% final, ~16 min on RTX 4090.
```

For OneTrainer SDXL training, apply the upstream patch and set
`optimizer = CONCORD_SGD` in your `TrainConfig`:

```bash
cd <OneTrainer-clone>
git apply upstream_patches/concord_onetrainer.patch
# Then copy all *.py files from this package alongside the OneTrainer root.
```

See **How to apply this to a fresh OneTrainer clone** below for the
full procedure.

## File map

### Core optimizer (the int-storage Triton kernels + layer modules)

| File | What's in it |
|---|---|
| `concord_triton_fused.py` | Triton kernels (forward, grad_x, fused grad_W+update) + autograd Functions. The hot path. |
| `concord_triton.py` | Rebalance kernel + bf16-weight recon kernel + helper SR tick. |
| `concord_linear_fused.py` | `ConcordLinearFused` / `ConcordConv2dFused` modules. The `.weight` property + chase + v_slow leak + Bayesian-anchored wd live here. |
| `concord_polyak.py` | PolyakHypothesis + BoxVelocityMean (off by default; opt-in via `concord_polyak_bias`). |
| `fused_profiler.py` | CUDA-event timing harness used by the kernel launchers. |
| `train_cifar_qtridiag.py` | Q-aware tridiagonal coupling library (used by the optimizer wrapper to find MLP up/down boundaries). |

### OneTrainer integration

| File | What's in it |
|---|---|
| `concord_optimizer.py` | `wrap_concord_modules()` / `create_concord_optimizer()` + `ConcordSGD` wrapper. Surfaces the `torch.optim.Optimizer` API so OneTrainer's LR scheduler / param-group machinery work unchanged. Also owns qtridiag discovery, BMA centroid accumulator, Polyak selector. |
| `onetrainer_concord_patch.py` | `install()` chain of monkey-patches: tuple stride/padding on Conv2d, `.reshape` instead of `.view` on Linear forward (handles SDXL non-contiguous inputs), tensor-backed step counter (HOP-safe under gradient checkpointing), `state_dict` emits standard `weight`/`bias` keys for the SDXL checkpoint converter, forces `use_reentrant=True` in `torch.utils.checkpoint`. |
| `concord_trainer.py` | Standalone SDXL full fine-tune trainer that bypasses OneTrainer's `GenericTrainer` / `BaseStableDiffusionXLSetup` (which has its own HOP/Dynamo headaches). Loads SDXL via diffusers, wraps the UNet's Linear / Conv2d modules in Concord layers, builds a disk-backed latent + text-embed cache, runs the training loop, calls `_generate_samples()` at the sample-after cadence, saves safetensors. Loss path includes noise offset, LOGIT_NORMAL timesteps, v-prediction, and min-SNR gamma. Also supports training the text encoders and the token embeddings (gated on `text_encoder{,_2}.train` / `.train_embedding`). |
| `concord_dataset.py` | Concept-driven dataset planner. Reuses OneTrainer's `ConceptConfig` schema so the existing UI Concepts tab produces a valid dataset for us. Supports `prompt_source` ∈ `{sample, concept, filename}`. |

### Canonical reference (CIFAR-10)

| File | What's in it |
|---|---|
| `cifar_concord_adamw.py` | **The canonical reference** — runs with no flags, reproduces the 90.91 best from CONCORD_README.md's results table. |
| `_tmp_cifar_bigger_vanilla.py` | Plain `torch.optim.AdamW` baseline on the same `WiderConvNet` — used for the apples-to-apples results table. |
| `cifar_in_memory.py` | GPU-side CIFAR-10 loader with vectorised augmentation. (Avoids the Windows `persistent_workers=True` hang.) |
| `train_cifar.py` | Shared `evaluate()` helper + classic `BaselineConvNet` (used by older reference scripts). |

### Smokes / tests

| File | What's in it |
|---|---|
| `smoke_concord_t1.py` | No-model dispatch / sampler routing smoke. ~1s. |
| `smoke_concord_t2.py` | Full SDXL load + 4 training steps + save. ~70s end to end. Validates the safetensors output has the right SDXL keys. |
| `smoke_concord_t3.py` | Per-step timing at realistic SDXL config (bsz=1, 1024²). 16 steps, separates warm-up from steady state. |
| `test_concord_state_dict.py` | Round-trip test for the patched `state_dict`. Verifies concord layers emit `weight`/`bias` keys, no leaked `s_slow`/`s_fast`, round-trip drift within SR LSB budget. |
| `_tmp_sampler_smoke.py` | Diffusers-shaped introspection smoke: verifies `m.weight.dtype` / `.shape` / `.device` are correct, `state_dict()` round-trips, eval-mode inference works after training. |
| `_tmp_loss_helpers_test.py` | Unit test for the four loss-level features added to `concord_trainer._train_step`: `_sample_noise`, `_sample_timesteps`, `_is_v_prediction`, `_snr`, `_loss_weight`. |
| `_tmp_loss_features_bench.py` | Wallclock benchmark: concord vs `torch.optim.AdamW` at SDXL-shaped batches with all four loss features active. |

### Older CIFAR scripts (kept for reproducing the original-headline numbers)

| File | What's in it |
|---|---|
| `concord_cifar_full_train.py` | 80-epoch CIFAR-10 fine-tune. The original headline configuration on the 188k `BaselineConvNet`. |
| `concord_cifar_full_plus.py` | Variant with Polyak + BMA stacked. |

### Docs

| File | What's in it |
|---|---|
| `CONCORD_README.md` | **How the optimizer works.** Three-accumulator structure, drift-cancel variance, Bayesian-anchored wd, materialised bf16 + cuBLAS/cuDNN path, knob reference, results table, "what didn't help" log, SDXL notes. |
| `README.md` | This file. Project overview + file map + integration steps. |

### Upstream patches

`upstream_patches/concord_onetrainer.patch` — the diff
against OneTrainer's tree (base commit
`61f4d9ff628c09a3d2c9a85db4f048f9c3180969`). Adds:

| File touched | Lines | What it does |
|---|---|---|
| `modules/util/enum/Optimizer.py` | +3 | Adds `CONCORD_SGD = 'CONCORD_SGD'` enum value. |
| `modules/util/optimizer_util.py` | +32 | Default `concord_*` field set for the optimizer presets. |
| `modules/util/config/TrainConfig.py` | +49 | Schema additions for all `concord_*` fields on `TrainOptimizerConfig`. |
| `modules/util/create.py` | +25 | Two hooks: (1) `optimizer == CONCORD_SGD` → `create_concord_optimizer`; (2) on SDXL fine-tune with that optimizer, `create_trainer` returns `ConcordTrainer` instead of `GenericTrainer`. |
| `modules/ui/OptimizerParamsWindow.py` | +24 | GUI tooltips for the concord knobs. |
| `modules/modelSetup/BaseStableDiffusionXLSetup.py` | +9 -2 | One-shot guard so the diffusers Fisher-loss-metric init doesn't run when the optimizer isn't `FISHER_SGD` (the metric would otherwise reach into a Concord layer's 1-D placeholder `weight` and crash). |

## How to apply this to a fresh OneTrainer clone

1. Clone OneTrainer at the base commit above (or any newer revision —
   the patch is small enough that conflicts will be surgical).
2. Drop all the `*.py` files from this package into the OneTrainer root:
   ```bash
   cp concord_*.py onetrainer_concord_patch.py cifar_in_memory.py \
      cifar_concord_adamw.py train_cifar.py train_cifar_qtridiag.py \
      fused_profiler.py CONCORD_README.md <OT_ROOT>/
   ```
3. Apply the upstream patch:
   ```bash
   cd <OT_ROOT>
   git apply upstream_patches/concord_onetrainer.patch
   ```
4. At the top of `scripts/train.py` and `scripts/train_ui.py`:
   ```python
   import onetrainer_concord_patch
   onetrainer_concord_patch.install()
   ```
5. In the UI, pick **Optimizer = CONCORD_SGD**, model = SDXL,
   training method = Fine-tune. The dispatch hook in `create.py` routes
   to `ConcordTrainer` automatically.

## Current state

**Validated:**
- CIFAR-10 fine-tune to **90.91% best / 90.66% final** on the
  3.2M-param `WiderConvNet` + BN, beating `torch.optim.AdamW + wd=0.01`
  (90.65 best) at 40 vs 96 bits/param of optimizer state. See
  CONCORD_README.md for the full results table.
- End-to-end SDXL full fine-tune via `ConcordTrainer` (validated by
  `smoke_concord_t2.py`: 4 steps + save produces a 6.9 GB safetensors
  with all 2526 expected SDXL keys).
- SDXL state_dict patch — concord UNet exports standard
  `conv_in.weight` style keys.
- Diffusers introspection — concord layers expose a real bf16
  `.weight` with correct shape/dtype/device/values (validated by
  `_tmp_sampler_smoke.py`).
- Four trainer-level loss features ported from OneTrainer's mixins
  into `concord_trainer._train_step`: `offset_noise_weight`,
  `timestep_distribution=LOGIT_NORMAL`, v-prediction (auto-detected
  or `force_v_prediction`), `loss_weight_fn=MIN_SNR_GAMMA`.
- Disk-backed cache. Pass 1 lookup is O(N) stat calls; pass 2 encodes
  misses at VAE bsz=1.

**Per-step wallclock vs `torch.optim.AdamW`** (TinyUNet 10.4M params,
all four loss features active in both columns):

| Workload | bsz | latent | Vanilla | Concord | Ratio |
|---|---|---|---|---|---|
| CIFAR-class | 4 | 64 | 5.8 ms | 10.8 ms | 1.87× |
| SDXL bsz=1 1024px | 1 | 128 | 7.2 ms | 14.5 ms | 2.02× |
| SDXL bsz=4 1024px | 4 | 128 | 16.8 ms | 21.9 ms | 1.30× |
| SDXL bsz=8 1024px | 8 | 128 | 35.4 ms | 39.2 ms | **1.11×** |

Concord is launch-overhead-bound at small total work, GPU-bound at
larger work. At the target SDXL workloads (bsz≥4 at 1024px) the
wallclock cost is 10–30% over vanilla AdamW in exchange for a 17.5 GB
HBM saving on the optimizer state.

**Known soft spots:**
- The orphan-Parameter ~5 GB VRAM cost we didn't fully chase (there's
  a workaround comment in `concord_optimizer.py`).
- First-step Triton autotune is ~14 s for the concord kernels (one
  time per training run, cached by Triton across runs).
- Sample generation goes through `_generate_samples()`, which builds a
  fresh `StableDiffusionXLPipeline` each time — heavy. Could be
  cached if it ever matters.

**Watch out for:**
- Stale `.pyc` cache when iterating on the concord files. Python's
  import cache + OneTrainer's UI keeping the same process alive can
  mask edits. Fully kill the process when you change a concord file.

## Where the kernels live

Inline in this package: `concord_triton_fused.py` (the big one) +
`concord_triton.py` (rebalance + recon helpers). No external dependency.
