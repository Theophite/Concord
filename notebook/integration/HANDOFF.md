# HANDOFF — Concord × OneTrainer (SDXL full-UNet finetune, 24 GB)

For the next session. Read `DESIGN.md` first (the architecture); this is the build plan +
exact attachment points. Everything below was verified against OneTrainer HEAD (cloned
2026-06-02 to `C:\concord\OneTrainer`) and torch 2.5.1+cu121 — **trust the file:line refs but
re-grep before editing** (OneTrainer may have moved on if re-cloned).

---

## 0. Constraints (do not violate)
- **Filesystem rule:** only touch the session subtree, `C:\concord\`, and `C:\fisher\`. OneTrainer
  lives at `C:\concord\OneTrainer`; integration code goes in `C:\concord\OneTrainer_integration`
  and/or `C:\concord\concord_onetrainer/` (a small package). Do NOT edit OneTrainer's files
  in-place beyond the 2-3 minimal hooks listed in step 4 — keep the swap/optimizer logic in OUR
  package and import it.
- **GPU contention:** the user runs ComfyUI (SDXL app) on the same 4090; it grabs ~17 GB. Check
  `nvidia-smi --query-gpu=memory.free` before any GPU run; ASK the user to free it rather than
  killing ComfyUI.
- **Don't over-ask / prefer doing.** The architecture is SETTLED (see "Decisions locked"). Build,
  don't re-survey.

## Decisions locked (do not re-litigate)
1. **Module-shim, not optimizer-API.** Swap UNet nn.Linear/Conv2d → ConcordLinear/Conv2dPackedB
   so the weight IS packed_w (32 b/param = the 24 GB win). Concord layers self-step in their
   own autograd Function.backward; they BYPASS OneTrainer's optimizer.
2. **OneTrainer's optimizer handles only the NON-swapped params** (norms/biases/embeddings) =
   the "aux AdamW" split, exactly like the nanoGPT harness.
3. **Target: bf16 full-UNet finetune + non-reentrant gradient checkpointing + NO layer-offload.**
4. **All three pieces (swap + optimizer + CUDA graph) get built** (user: "all three together"),
   but code-first / download-later, and graph is built LAST behind a flag with eager fallback.

## What's DONE + VERIFIED this session (evidence — don't re-check)
- OneTrainer cloned (shallow) → `C:\concord\OneTrainer`. NOT yet: its python env, the SDXL model.
- **Seam exists:** OneTrainer swaps UNet modules in-place via
  `modules/util/quantization_util.py : __replace_linear_layers` (L104) — recurses
  `named_modules()`, `setattr(parent, attr, replacement)`, handles Linear AND Conv2d
  (stride/padding/dilation/groups at L48-50). Called from `quantize_layers` (L260) ←
  `modules/modelSetup/BaseStableDiffusionXLSetup.py` ~L85. **Mirror this.**
- **Fused-back-pass precedent:** `modules/util/enum/Optimizer.py:101 supports_fused_back_pass()`;
  `GenericTrainer.__apply_fused_back_pass` (L556) uses `register_post_accumulate_grad_hook` +
  `optimizer.step_parameter(tensor, group, i)`. (Concord doesn't USE this — its weights are
  param-less buffers — but it's why OneTrainer tolerates non-standard optimizers.)
- **bf16 → NO grad scaler:** `dtype_util.enable_grad_scaling` (L18-20) enables a scaler ONLY for
  FLOAT_16 + fp32 params. bf16 finetune ⇒ no scaler ⇒ no double-scale conflict with Concord's
  in-backward step. ✓
- **Gradient checkpointing SAFE (the big one):** probe proved under `use_reentrant=False`
  (OneTrainer default, `checkpointing_util.py:97`) the recomputed FORWARD fires 2× but a custom
  autograd Function.backward (= Concord's fused step, `concord/packed_b.py:1237`) fires exactly
  **1×**. Re-run the probe if doubtful:
  ```python
  # in C:\concord: counts a custom Function's bwd under checkpoint(use_reentrant=False)
  # expect forward=2 backward=1  (see git log / this session for the snippet)
  ```
  CAVEAT: `use_reentrant=True` path exists (L140) but ONLY with layer-offload — NOT verified,
  out of scope (we target no-offload).
- **Concord already has** (in `C:\concord\concord/packed_b.py`): `ConcordLinearPackedB` (L1337),
  `ConcordConv2dPackedB` (L2004), bf16 `.weight` property shim (L1481), `load_weights()`,
  `consolidated_weight()` (materialize bf16 for save), `rebalance()`, `get_weight()`, and the
  device-tensor scalars (`_lr_buf`, sigma, ratio-floors) needed for CUDA-graph capture.

## BUILD PLAN (in order)

### Step 1 — `concord_onetrainer/inject.py : replace_unet_with_concord(unet, filters=None)`
Mirror `__replace_linear_layers`. Recurse the UNet; for each target `nn.Linear`→
`ConcordLinearPackedB(in,out,bias=...)`, each `nn.Conv2d`→ `ConcordConv2dPackedB(in,out,k,
stride,padding,dilation,groups,bias=...)`; `setattr(parent, attr, new)`; `new.load_weights(
old.weight)` (+ copy bias); `del old.weight` (drop the fp32/bf16 copy = the win).
- FILTER: only big UNet hidden weights. Reuse OneTrainer's `ModuleFilter` + the Muon default
  patterns (`modules/util/optimizer/muon_util.py`: SDXL = `['block', 'text_model.encoder.layers']`).
  Skip norms/bias/tiny → those stay nn.Parameter for the aux optimizer.
- Return the list of created Concord modules (the optimizer wrapper needs them for rebalance).
- VALIDATE: build a tiny toy `nn.Sequential(Conv2d, Linear)`, swap it, assert forward ==
  pre-swap forward to ~bf16 tol at step 0 (load_weights ⇒ live weight == init). No SDXL needed.

### Step 2 — optimizer wrapper + OneTrainer hooks
- `concord_onetrainer/optimizer.py : ConcordAuxOptimizer(torch.optim.Optimizer-compatible)`:
  wraps a real AdamW over the NON-swapped params; holds the Concord module list; its `.step()`
  (a) advances the lr device tensor (`m.lr = lr`) + any schedule on each Concord module,
  (b) calls `m.rebalance()` on each, (c) steps the aux AdamW. `.zero_grad()` zeroes aux grads.
- OneTrainer hooks (MINIMAL — 3 edits, keep logic in our package):
  1. `modules/util/enum/Optimizer.py`: add `CONCORD = 'CONCORD'`; `supports_fused_back_pass`
     returns appropriately (Concord layers are self-fused; the aux is standard — simplest:
     report False, let OneTrainer call .step() normally, since the heavy weights self-update).
  2. `modules/util/create.py create_optimizer` (~L124): `case Optimizer.CONCORD:` → build
     ConcordAuxOptimizer from our package.
  3. `BaseStableDiffusionXLSetup` (~L85, near `quantize_layers`): when optimizer==CONCORD, call
     `replace_unet_with_concord(model.unet, ...)` BEFORE params are collected for the optimizer.
- VALIDATE: import-check against OneTrainer modules; a CPU/meta toy of the create_optimizer path.

### Step 3 — CUDA graph (LAST, behind a flag, eager fallback)
- Capture OneTrainer's per-step UNet fwd+loss+bwd into one graph (our PROVEN recipe: capture at
  the first step, side-stream warmup ~3-5× full fwd_bwd, NO eager pre-roll, device-tensor
  scalars so lr/sigma/floors survive replay; aux step + rebalance EAGER after replay). The host
  seam is `GenericTrainer`'s train loop (find the `loss.backward()` site near L590-625).
- HAZARD: OneTrainer's loop has grad accumulation, EMA, sampling interleaved + a non-static data
  pipeline → capturing the WHOLE step is harder than nanoGPT. Likely need static input buffers
  the dataloader copies into. If full-step capture fights the loop, fall back to eager (the VRAM
  win is from the module-swap, not the graph — graph is speed only).
- CORRECTNESS GATE (non-negotiable, we burned time this session by NOT gating): eager vs graph,
  same seed, a few steps; loss must track within the SR-noise floor and CONVERGE (a real bug
  GROWS). Do NOT claim graph-correct without this.

## OPEN RISKS to resolve DURING build (verify, don't assume)
- **EMA:** OneTrainer EMA likely iterates `parameters()` → Concord modules contribute none (their
  weight is a buffer). Either hook EMA to read `consolidated_weight()`, or disable EMA for the
  swapped layers. Grep `modules/` for the EMA impl; decide at build time.
- **Checkpoint save:** SDXL `.safetensors` wants bf16 weight tensors. Add a save hook that
  materializes each Concord module's weight via `consolidated_weight()` (the DEPLOY weight —
  drop s_fast; it's the BEST weight, confirmed sv 1.518/1.497). Find OneTrainer's UNet-save path.
- **Sampling mid-train:** reads `.weight` (the bf16 shim) → should just work; verify the shim is
  the LIVE weight (it is — `_bf16_weight_buf`, kept fresh by the apply kernel).
- **VRAM reality:** the thesis (32 b/param fits SDXL UNet finetune in 24 GB w/o offload) is
  UNTESTED. First real run: measure peak_mem; if it doesn't fit, the fallbacks are (a) the
  confirmed split config is already minimal, (b) Concord on UNet only (TEs stay frozen/LoRA),
  (c) only-then consider offload (re-test the reentrant-checkpoint hazard).

## The recipe to ship (the validated/confirmed config)
Use the **split** config (confirmed best, same-seed A/B this session, deploy-sv 1.518 vs bare
1.540, and noise-sweep -0.021 at sigma~0.6): the bare ConcordLinear/Conv2dPackedB defaults
(rank-1 v̂ AdamW + fixed coherence gate) PLUS `--ratio_coh --ratio_chase_floor_min 0.1
--ratio_leak_floor_min 0.1 --gf_consol 50`. Deploy/save off `consolidated_weight()`. Noise
injection (`--sigmag_iso ~0.4-0.6`, off the deploy weight) is a PROVEN +~0.02 but single-seed/
nanoGPT-only — treat as optional, validate on SDXL before enabling. Concord's winner package is
at `C:\concord\concord/` (committed main, `f8423fd`/`4ea7f15`); import THAT.

## First moves for the next session
1. `cd C:\concord; git log --oneline -5` (confirm main = the noise/graph commit `4ea7f15`).
2. Re-grep the 5 file:line refs above (OneTrainer may have changed if re-cloned).
3. Write `concord_onetrainer/inject.py` (step 1) + its toy-swap validation. NO download needed.
4. Then step 2; import-check against OneTrainer.
5. Ask the user to free the GPU + confirm the SDXL base model path before the first real run.
   Download: SDXL base (~7 GB, e.g. stabilityai/stable-diffusion-xl-base-1.0) + set up
   OneTrainer's env (`requirements.txt`/`install.bat`) — do this AFTER steps 1-2 compile.

## Anti-patterns from this session (don't repeat)
- Don't claim a CUDA graph is correct without the eager-vs-graph SR-floor gate (cost us a wrong
  "compute-bound" + a fabricated-number log entry earlier).
- Don't fabricate measurements — run the probe, read the real output, THEN write the number.
- The user's momentum WIP is entangled in `src/prototype_packed_b.py` + `src/train_nanogpt.py`
  (off by default). Don't try to disentangle it; the WINNER package `concord/` is clean — import
  from `concord/`, not `src/`.
