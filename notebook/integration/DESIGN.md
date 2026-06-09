# Concord × OneTrainer: SDXL full-UNet finetune in 24 GB

Design spec. Goal: full SDXL UNet finetune on a 24 GB card, using Concord's packed-int
optimizer (32 b/param weight-as-storage) wired into OneTrainer with **minimal monkeypatching**,
including the fused Triton kernels and CUDA-graph capture.

## Why this fits 24 GB (the thesis)

Standard AdamW finetune holds, per UNet weight: fp32 master (32) + bf16 working (16) + m (32)
+ v (32) ≈ **112 bits/param**. SDXL UNet ≈ 2.6 B params → the optimizer state alone is the
VRAM wall. Concord holds the **entire** state (weight + both momenta + variance) in **one int32
= 32 b/param**, the weight *is* the storage (no fp32 master). That's the 3.5× state reduction
that turns "doesn't fit" into "fits".

## The seam (verified in OneTrainer 2026-06 HEAD)

OneTrainer ALREADY swaps UNet modules in-place: `quantization_util.__replace_linear_layers`
walks `named_modules()` and does `setattr(parent, attr, replacement)` to swap nn.Linear/Conv2d
for quantized variants (called from `BaseStableDiffusionXLSetup` ~L85 `quantize_layers`). We
mirror this idiom -> **low monkeypatch, OneTrainer's own pattern**.

Key facts established by recon:
- `create_optimizer` (modules/util/create.py:124) returns a `torch.optim.Optimizer`; it has a
  `supports_fused_back_pass()` path (Optimizer.py:101) + `step_parameter(tensor, group, i)` via
  `register_post_accumulate_grad_hook` (GenericTrainer.__apply_fused_back_pass L556).
- **Concord modules have NO nn.Parameter weights** (weight = packed_w buffer; step fused in the
  module's autograd Function.backward). So Concord layers BYPASS OneTrainer's optimizer for the
  swapped weights -- they self-update. OneTrainer's optimizer only handles the NON-swapped
  trainable params (norms/embeddings/bias) = the "aux AdamW" split, exactly like nanoGPT.
- **bf16 finetune uses NO grad scaler** (dtype_util.enable_grad_scaling: scaler only for FLOAT_16
  + fp32 params). So no double-scaling conflict with Concord's in-backward step. bf16 is the
  target (SDXL trains bf16; Concord internals are bf16).
- Concord already has the pieces: `ConcordConv2dPackedB` (SDXL UNet is conv-heavy), a bf16
  `.weight` property shim (packed_b.py:1481) so the module looks normal to sampling/EMA/saving,
  `load_weights()` to ingest a pretrained weight into packed_w, `consolidated_weight()` to
  materialize bf16 for checkpoint export, `rebalance()` per step, and `--cuda_graph`-style
  capture (device-tensor scalars).

## The 3 integration points

### 1. Module swap (the VRAM win)
`concord_onetrainer/inject.py : replace_unet_with_concord(unet, config)`
- Mirror `__replace_linear_layers`: recurse, swap nn.Linear -> ConcordLinearPackedB and
  nn.Conv2d -> ConcordConv2dPackedB via setattr, load the pretrained weight into packed_w
  (`load_weights`), drop the original fp32/bf16 weight (the win).
- Filter: only the big UNet hidden weights (skip tiny norm/bias; those go to aux AdamW). Use
  OneTrainer's ModuleFilter pattern (same as Muon's hidden-layer filter).
- Conv2d: pass stride/padding/dilation/groups (ConcordConv2dPackedB already takes them).
- Frozen-base correctness: load_weights at swap time = bit-identical forward at step 0
  (Concord's live weight == init until the first backward).

### 2. Optimizer slot (marker + aux + rebalance driver)
- Add `Optimizer.CONCORD` to the enum; in `create_optimizer`, return a thin wrapper:
  - the swapped Concord modules self-step in backward (no optimizer involvement);
  - a real AdamW (or Concord-via-nanoGPT-aux) over the NON-swapped params;
  - `.step()` ALSO drives `rebalance()` on every Concord module + advances lr/sigma device
    tensors (the per-step schedule).
- `supports_fused_back_pass()` -> the Concord layers are already fused; the wrapper reports
  compatibility so OneTrainer's loop doesn't fight it.
- lr: Concord reads `_lr_buf` (device tensor); the wrapper's lr schedule .fill_()s it.

### 3. CUDA graph
- OneTrainer's train step (GenericTrainer) host side -> capture the UNet fwd+loss+bwd into one
  graph (our proven recipe: capture iter0, side-stream warmup, no eager pre-roll, device-tensor
  scalars). The aux AdamW step + rebalance stay eager after replay.
- This is the trickiest piece (OneTrainer's loop has grad accum, EMA, sampling interleaved) ->
  build LAST behind a flag, fall back to eager if capture fails. Correctness-gate vs eager
  (within SR-noise floor, the nanoGPT method).

## Build order: all three, code-first (no SDXL download needed to write/import-check)
inject.py (swap) -> optimizer wrapper + enum/create.py hooks -> graph wrapper. Validate each
imports against OneTrainer's modules; then download SDXL + bf16 finetune run once GPU is free.

## Open risks to verify during build
- EMA over param-less modules (OneTrainer EMA likely iterates parameters() -> Concord modules
  contribute none; the bf16 .weight shim is read-only -> EMA of the deployed weight may need a
  hook, or disable EMA for Concord layers).
- Checkpoint save: SDXL .safetensors expects bf16 weight tensors -> materialize via
  consolidated_weight() at save time (a save hook).
- Sampling mid-train reads .weight (the bf16 shim) -> should just work (shim is the live weight).
- Gradient checkpointing interaction with the fused-in-backward step (recompute -> the Function
  backward runs on recompute; need the step to fire once, not per-recompute).
  **RESOLVED (probe, 2026-06-02):** under `use_reentrant=False` (OneTrainer's DEFAULT,
  checkpointing_util.py:97), the recomputed FORWARD fires 2x but the custom Function.backward
  (= Concord's fused step) fires exactly **1x** -> SAFE. Probe: forward=2 backward=1.
  **CAVEAT:** OneTrainer ALSO has a `use_reentrant=True` path (L140), used ONLY with LAYER
  OFFLOADING (LayerOffloadConductor). Reentrant checkpointing has different nested-autograd
  semantics -> NOT verified safe for the fused step. MITIGATION: target non-reentrant
  checkpointing + NO layer-offload. The whole thesis is that Concord's 32-b state makes SDXL
  fit in 24 GB WITHOUT offloading, so this should be moot. If offloading is ever needed,
  re-test the reentrant path before enabling.

## De-risk status (before writing integration code)
- [x] Module-swap seam exists (OneTrainer's own __replace_linear_layers idiom).
- [x] Concord has Conv2d + bf16 .weight shim + load_weights + consolidated_weight + rebalance.
- [x] bf16 finetune uses NO grad scaler -> no double-scale conflict with the in-backward step.
- [x] Gradient checkpointing (non-reentrant) fires the fused step ONCE -> safe under the
      checkpointing 24GB-SDXL requires.
- [ ] EMA / checkpoint-save / sampling over param-less modules (build-time verify).
- [ ] CUDA-graph capture over OneTrainer's loop (build LAST, behind flag, eager fallback).
