# Concord

A fork of **[OneTrainer](https://github.com/Nerogar/OneTrainer)** that adds **Concord** — a self-stepping, packed-weight optimizer for SDXL fine-tuning. Concord folds the optimizer state *into the weights themselves*, so a **full UNet fine-tune** (not LoRA) fits comfortably on a 24 GB card.

Everything OneTrainer does still works exactly as before. Concord is an extra optimizer choice plus the kernels and CUDA-graph machinery that make it fast and memory-light. If you don't select the Concord optimizer, this is just OneTrainer.

## What Concord is

A normal optimizer stores the weights **plus** a separate optimizer state — an fp32 master copy, Adam moments, and so on — several extra bytes per parameter on top of the model. Concord throws that out. It **swaps the UNet's `nn.Linear` / `nn.Conv2d` for packed self-stepping layers** (`ConcordLinearPackedB` / `ConcordConv2dPackedB`):

- Each weight is a single **int32** (~32 bits/param): a fast / slow / value mantissa packed into one word, sharing a per-row and per-column exponent.
- The **optimizer step runs inside the autograd backward** — every layer updates its own packed state directly from its own gradient. There is no separate moment or master tensor: the optimizer state *is* the weight.
- The OneTrainer-visible "optimizer" is just a plain SGD over the **non-swapped** parameters (norms, biases, embeddings). A controller drives the Concord half — its learning-rate schedule, consolidation, and rebalancing.

Because the optimizer state is the weight, a full SDXL UNet fine-tune avoids the 3–4× memory overhead a standard Adam fine-tune pays for its master copy and moments — which is what lets a full fine-tune fit where normally only LoRA would.

## How it works

**One int32 per weight.** Each weight is a single 32-bit word holding three integer accumulators at different timescales, with a shared exponent per row and per column carrying the dynamic range:

```text
bits 31..16   s_fast  (int16)  — fast tick: catches each step's gradient
bits 15..8    s_slow  (int8)   — coarse consolidated position
bits  7..0    v_slow  (int8)   — long-time anchor

weight = (s_slow*128 + s_fast + v_slow*128) * 2^(row_exp + col_exp - bias)
```

The weight the forward pass uses is reconstructed from those on demand. There is no fp32 master copy and no moment buffers — the entire persistent state is 32 bits/param, all integer.

**The accumulators *are* the optimizer.** The fast accumulator catches each gradient; the slower ones consolidate it. The trick is that the **gaps between the timescales** (fast-vs-slow, slow-vs-anchor) carry the same information Adam reads from its second moment — the gradient's variance and drift — without ever storing a second moment. A drift-cancellation term subtracts the steady part so what remains estimates just the noise, and a coherence gate then scales each step by how consistent the gradient direction has been. The net behaviour is a factored, variance-adaptive AdamW (sqrt preconditioner; consolidation that keeps coherent signal and evaporates incoherent jitter) — every piece of it read from and written back into that one int32.

**The step is folded into the backward.** Each swapped layer is an autograd `Function`; the instant its gradient is computed, the *same* Triton kernel folds it into the packed accumulators. There is no `optimizer.step()` over a separate state — the layer steps itself, which is also why it captures cleanly into a CUDA graph.

**Keeping the integers honest.** Stochastic rounding makes the low-bit accumulation unbiased, and the shared row/column exponents are periodically rebalanced so the mantissas stay in range as the weights drift. On the forward, the fused dequant-matmul reconstructs the bf16 weight *inside* the matmul, so a full bf16 copy is never materialized.

The net result is the headline: Adam-class adaptive training at roughly weight-only memory — which is what puts a full SDXL UNet fine-tune on a 24 GB card.

## Key additions over OneTrainer

| Feature | Config field | What it does |
| --- | --- | --- |
| **Fused dequant-matmul** | `concord_fused_matmul` *(on by default)* | Dequantizes the packed weight **inside** a Triton matmul, eliminating the persistent bf16 weight cache (~5 GB on SDXL). Requires `gradient_accumulation_steps == 1`; if accumulation is on it transparently falls back to the cached path. |
| **CUDA-graph step** | `concord_cuda_graph` | Captures UNet *predict → loss → backward* in a CUDA graph for a batch-size-1 speedup, injecting fresh noise on every replay so the capture doesn't pin the RNG. |
| **Diffusion recipe** | OneTrainer noise fields | Offset noise, input perturbation, min-SNR-γ and the timestep distributions are wired through **both** the eager and the captured-graph training paths. |
| **Token control plane** | `concord_sanitize_tokens`, layer filter | Train / freeze / zero-sanitize individual embedding tokens, and restrict the Concord swap to selected layers (the rest stay standard and frozen). |

Measured on a full-UNet SDXL fine-tune (24 GB card): **~15 GB** training footprint with fused on, versus **~20 GB** with the cached path — a ~5 GB saving that is the difference between fitting and spilling.

## Using it

1. **Install** exactly like OneTrainer — clone this repo and run `install.bat` (Windows) or `install.sh` (Linux):
   ```sh
   git clone https://github.com/tok/Concord.git
   ```
   See the [upstream wiki](https://github.com/Nerogar/OneTrainer/wiki) for full setup and troubleshooting; nothing about installation changes in this fork.
2. **Pick the optimizer.** In the GUI, choose **CONCORD** as the optimizer for an SDXL fine-tune, or load a preset:
   - `training_presets/#SDXL Concord Fused 24GB.json` — a blank full-UNet template; set your own base model + dataset.
3. **Keep `gradient_accumulation_steps = 1`** to use the fused path (it's on by default). With accumulation > 1, fused steps aside automatically and you get the cached path.

> Concord currently targets **SDXL full fine-tuning**. Other model types and training methods fall back to stock OneTrainer behavior.

## Diagnostics

A few env-gated probes are available on the Concord paths:

- `CONCORD_MEMLOG=1` — per-epoch VRAM (allocated / reserved / peak) at the top of training.
- `CONCORD_GRAPHMEM` — per-sample memory around the CUDA-graph release/recapture. **On by default**; set `CONCORD_GRAPHMEM=0` to silence the `[graphmem]` lines.
- `CONCORD_FUSED_MATMUL=1` — force the fused path on from the environment (the `concord_fused_matmul` config field is the normal way).

## Where the code lives

- `modules/util/optimizer/concord/` — the packed self-stepping layers, the Triton kernels, and the fused dequant-matmul.
- `modules/util/optimizer/concord_ot.py` — the controller and the layer swap.
- `modules/util/optimizer/concord_graph.py` — the manual CUDA-graph capture of the UNet step.
- `modules/modelSetup/StableDiffusionXLFineTuneSetup.py` — the SDXL wiring (swap, fused flag, graph gate, resume).

## Status

Concord SDXL full fine-tuning is functional and has produced validated samples. This is an active research integration on top of OneTrainer, not a separate product — expect rough edges outside the SDXL fine-tune path.

## Attribution & license

Built on **[Nerogar/OneTrainer](https://github.com/Nerogar/OneTrainer)**; all of its functionality, documentation, and license are retained (see `LICENSE.txt`). Concord is an additive layer on top — for the base trainer, supported models, the wiki, and the full feature set, refer to the upstream project.
