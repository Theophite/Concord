# Concord × OneTrainer: SDXL full-UNet finetune in 24 GB — **AS-BUILT**

**Status: BUILT and in production.** This documents the integration as it actually exists and
trains SDXL today — it is not a plan. (Earlier revisions of this file framed it as future work
for a "next bot"; that premise was wrong and has been corrected.)

## Where it lives (two trees, one remote)

- **Optimizer library** (the kernel + modules): developed in **this repo** (`C:\concord`, `src/`),
  mirrored 1:1 into the OneTrainer fork's `modules/util/optimizer/concord/`.
- **OneTrainer-side integration** (the glue: controller, graph, saver/setup hooks): a **OneTrainer
  fork** at `C:\fisher\OneTrainer-clean` — branch **`concord-integration`** on the same remote
  (`Theophite/Concord.git`; unrelated history to this repo's `main`). Concord is a first-class
  registered optimizer there: `Optimizer.CONCORD` (`modules/util/enum/Optimizer.py:83`).
- The `C:\concord\OneTrainer/` clone in this repo was only ever **recon scratch** — it has **no**
  integration code (gitignored). Ignore it; the real integration is the fork above.

## Why it fits 24 GB (the thesis — validated in practice)

Standard AdamW holds, per UNet weight: fp32 master (32) + bf16 working (16) + m (32) + v (32) ≈
**112 b/param**. Concord holds the entire state (weight + both momenta + variance) in **one int32 =
32 b/param**; the weight *is* the storage (no fp32 master). That 3.5× state reduction turns
"doesn't fit" into "fits." **Validated:** a full SDXL UNet finetune runs end-to-end — 794
nn.Linear/nn.Conv2d swapped to packed layers — and saves a standard sharded UNet (see §4).

## As-built architecture (file:line into `C:\fisher\OneTrainer-clean`)

### 1. Optimizer slot — Concord layers self-step; OneTrainer's optimizer is just the aux
`create_optimizer` (`modules/util/create.py:164`) for `Optimizer.CONCORD` returns a plain
`torch.optim.SGD` over the **non-swapped** params (norms/biases/embeddings). The swapped UNet
layers have **no nn.Parameter weight** — they self-update inside their autograd `Function.backward`,
bypassing `optimizer.step()`. So "the Concord optimizer" is really: self-stepping packed layers +
an aux SGD. (SGD not AdamW for the aux on purpose — AdamW normalizes every step and collapses group
norms; rationale at `concord/concord_winner.py:333`.) `supports_fused_back_pass()` does **not**
special-case CONCORD — it's orthogonal.

### 2. The swap (the VRAM win)
`StableDiffusionXLFineTuneSetup.setup_model()`
(`modules/modelSetup/StableDiffusionXLFineTuneSetup.py:150-174`) builds a **`ConcordController`**
(`modules/util/optimizer/concord_ot.py`). The controller calls `swap_unet_to_winner()`
(`concord/concord_winner.py:114`): every `nn.Linear`/`nn.Conv2d` →
`ConcordLinearPackedB`/`ConcordConv2dPackedB`, subject to OneTrainer's **Layer Filter** preset
(`concord_winner.py:135`); the original fp32/bf16 weight is dropped. `before_step`/`after_step`
hooks (`setup:375/395`) drive the per-step schedule (lr warmup-cosine, rising-late noise σ, ratio
floors via `winner_step`) and `GatedRebalance` (fires only on real mantissa overflow). The
`concord_fused_matmul` flag is set **before** the swap (layers read it at `__init__`).

### 3. Text encoder + new-token embeddings
- **TE anchor** (`swap_text_encoder_to_anchor`, `concord_winner.py:185`): CLIP-L Linears → packed
  layers in **frozen-v_slow anchor mode** (`alpha_v_fast=0` pins the pretrained anchor; `wd_anchor`
  elastic-pulls the trainable delta back). Default on when TE training is enabled (`concord_te_anchor`).
- **Control plane** (`control_plane.py : ControlPlaneEmbedding`): replaces the TE `token_embedding`
  and routes trainable new tokens to a `ConcordPackedEmbedding` (packed, self-stepping,
  norm-preserving). `forward` is branch-free (`torch.where`) → CUDA-graph-safe. Installed via
  `setup_packed_embeddings` (`concord_ot.py`), conditional on `concord_packed_embeddings`.

### 4. Save / deploy — consolidate back to standard layers
`StableDiffusionXLModelSaver.save`
(`modules/modelSaver/stableDiffusionXL/StableDiffusionXLModelSaver.py:88-90`) →
`ConcordController.consolidate_into_unet` (`concord_ot.py:144-173`): each packed layer →
`consolidated_weight()` (drops s_fast — the deploy-slow weight) → a standard `nn.Linear`/`nn.Conv2d`,
yielding a normal SDXL `.safetensors`. Prints `[concord] consolidated <n> layers -> standard
nn.Linear/nn.Conv2d for deploy` (`concord_ot.py:172`). **Final-save only / destructive**; internal
**backups keep packed state** for resume. TE Linears + trained new-token vectors are reversibly
materialized to standard tensors for the save and restored afterward so training continues.

### 5. CUDA graph (opt-in, experimental)
`modules/util/optimizer/concord_graph.py`. The validated path is **`ManualUNetGraph`** (captures
fwd+loss+bwd with a real `loss.backward()` inside the graph). Gated by `should_graph(config)`:
`concord_cuda_graph` flag **AND** CONCORD **AND** bf16 **AND** latent caching **AND** gradient
checkpointing **AND** single-GPU. **Default OFF; eager is the default path.** The older
`make_graphed_callables` v1 is left in but **NaNs on the first step** (static-buffer backward ×
self-stepping × checkpointing) — explicitly marked experimental.

### 6. Gradient accumulation (~0 extra memory)
`set_consolidate(device, is_update_step)` (`modules/trainer/GenericTrainer.py:826`): only the
cycle's **last** micro-step consolidates (full apply); earlier micro-steps tick the gradient into
s_fast with the live weight frozen. No separate grad-accumulation buffer.

### 7. Config knobs → kernel (`make_concord_config`, `concord_ot.py:66`)
Optimizer panel: `gf_consol`, `noise`, `sigmag_peak`, `ratio_coh`, `lazy_gate`, `lazy_active_thresh`.
General: `concord_cuda_graph` (off), `concord_fused_matmul` (on), `concord_packed_embeddings` (on),
`concord_bucket_contiguous` (on), `concord_te_anchor` (on), `concord_te_wd_anchor` (0.5),
`concord_sanitize_tokens`. These flow into the `ConcordConfig` → `swap_unet_to_winner` / `winner_step`
and the kernel `set_*` setters.

## The recipe (the validated winner)
Baked in `concord/concord_winner.py : WINNER` / `ConcordConfig`; full spec + numbers in
[`WINNING_CONFIG.md`](../WINNING_CONFIG.md). In short: rank-1 v̂ AdamW (32 b/param) + **dissipation**
(`ratio_coh` + chase/leak floors + `gf_consol`) + **fluctuation** (isotropic noise), deploy off
`consolidated_weight()`.

## De-risk status
- [x] Module-swap (794 layers), self-stepping in backward, aux SGD — works.
- [x] bf16 finetune, **no** grad scaler — no double-scale conflict.
- [x] Non-reentrant gradient checkpointing fires the fused step **once** (forward 2× / backward 1×).
      CAVEAT: the `use_reentrant=True` path (layer-offload only) is **not** verified for the fused
      step — target non-reentrant + no offload (the 32-b state is what makes offload unnecessary).
- [x] Save → consolidate to standard SDXL safetensors — works.
- [x] Lazy-update gate no-op-safe on dense (ot_noop A/B, OFF==ON at τ=1e-4, 2026-06-09).
- [x] Fused packed matmul kernels match cuBLAS (fwd exact, bwd within bf16).
- [ ] CUDA-graph **v1** NaNs first step — use `ManualUNetGraph` (v2) / eager. (open)
- [ ] Noise σ magnitude is single-seed (nanoGPT); multi-seed on SDXL before trusting it. (open)
- [ ] EMA over param-less swapped layers — confirm OneTrainer EMA (if enabled) reads the deploy
      weight or is disabled for swapped layers. (verify if EMA is turned on)
