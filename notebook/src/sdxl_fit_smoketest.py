"""SDXL UNet VRAM-fit smoketest for Concord (old / fused lineage).

Answers ONE question: does a Concord-swapped SDXL UNet (32 bits/param,
weight-as-storage) physically fit a 24 GB card under bf16 + gradient
checkpointing?

We build the UNet from its CONFIG (random init -- identical shapes and param
count to real SDXL-base, so the memory footprint is identical and no gated
~7 GB download is needed), swap its nn.Linear / nn.Conv2d to the fused-int
Concord modules, run a few fwd/bwd steps, and report peak CUDA memory.

This is a MEMORY test, not a quality test. The old fused lineage does NOT carry
the validated "winner" (ratio_coh / gf_consol) config -- that lives only in
concord/packed_b.py and is not yet wired to an SDXL swap. The VRAM thesis is
lineage-independent: both lineages store the weight + momenta as 32 b/param.

NOTE vs real OneTrainer: wrap_model in OneTrainer deliberately keeps the
original fp32 weights pinned (~5 GB on SDXL) because OneTrainer's parameter
collection still references them. In THIS standalone harness those originals
are unreferenced after the swap and get GC'd, so we measure the *intrinsic*
32 b/param footprint -- the true thesis number. The OneTrainer +~5 GB is a
known, separately-fixable overhead.

Sizes:
  --size tiny  : small UNet (<1 GB) -- validates the whole pipeline end-to-end
                 (install patches, wrap_model, Triton compile, stride-2
                 downsampler conv, attention Linears, grad checkpointing, the
                 fused-in-backward step, aux AdamW) on the GPU even while
                 something else holds most of the card.
  --size sdxl  : the real ~2.6 B SDXL-base UNet config -- the fit measurement.

Run:
  python src/sdxl_fit_smoketest.py --size tiny
  python src/sdxl_fit_smoketest.py --size sdxl --res 1024 --batch 1
"""
import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.resolve()))  # src/ on path

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

import onetrainer_concord_patch
from concord_optimizer import wrap_model


# Real SDXL-base UNet config -> ~2.567 B params. Inputs (below) use the SDXL
# conditioning dims: encoder_hidden_states 2048, text_embeds 1280, time_ids 6.
SDXL_UNET_CONFIG = dict(
    sample_size=128, in_channels=4, out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(320, 640, 1280),
    layers_per_block=2,
    cross_attention_dim=2048,
    transformer_layers_per_block=(1, 2, 10),
    attention_head_dim=(5, 10, 20),
    use_linear_projection=True,
    addition_embed_type="text_time",
    addition_time_embed_dim=256,
    projection_class_embeddings_input_dim=2816,
    norm_num_groups=32,
)

# Tiny UNet: same conditioning dims (so the SAME fabricated inputs work) but
# minimal channels/depth. Still has a stride-2 downsampler and an upsampler
# conv, plus cross-attention Linears -> exercises every kernel path.
TINY_UNET_CONFIG = dict(
    sample_size=16, in_channels=4, out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(32, 64),
    layers_per_block=1,
    cross_attention_dim=2048,
    transformer_layers_per_block=1,
    attention_head_dim=(2, 4),
    use_linear_projection=True,
    addition_embed_type="text_time",
    addition_time_embed_dim=256,
    projection_class_embeddings_input_dim=2816,
    norm_num_groups=32,
)


def _gb(nbytes):
    return nbytes / (1024 ** 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["tiny", "sdxl"], default="tiny")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--res", type=int, default=1024, help="image px; latent = res/8")
    ap.add_argument("--no_ckpt", action="store_true", help="disable grad checkpointing")
    ap.add_argument("--pin_originals", action="store_true",
                    help="hold references to the pre-swap bf16 weights, "
                         "simulating OneTrainer's NamedParameterGroupCollection "
                         "pinning (~5 GB on SDXL) -- the real-integration footprint")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[fatal] CUDA not available -- Concord Triton kernels need a GPU.")
        sys.exit(1)
    dev = torch.device("cuda")
    free0, total0 = torch.cuda.mem_get_info()
    print(f"[gpu] {torch.cuda.get_device_name(0)} | free {_gb(free0):.2f} / "
          f"{_gb(total0):.2f} GB at start")
    torch.cuda.reset_peak_memory_stats()

    # Install the SDXL-specific monkeypatches (tuple conv args, reshape-not-view,
    # state_dict weight keys, force-reentrant checkpoint, dynamo permissive).
    onetrainer_concord_patch.install()

    cfg = SDXL_UNET_CONFIG if args.size == "sdxl" else TINY_UNET_CONFIG
    print(f"[build] UNet2DConditionModel size={args.size} (random init)")
    unet = UNet2DConditionModel(**cfg)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"[build] UNet params: {n_params/1e9:.4f} B ({n_params:,})")
    if args.size == "sdxl":
        assert 2.4e9 < n_params < 2.7e9, f"SDXL param count off: {n_params:,}"

    unet = unet.to(dev, dtype=torch.bfloat16)
    unet.train()
    if not args.no_ckpt:
        unet.enable_gradient_checkpointing()
    torch.cuda.synchronize()
    print(f"[mem] after UNet -> cuda bf16: "
          f"{_gb(torch.cuda.memory_allocated()):.2f} GB allocated")

    # Optionally pin the pre-swap weights alive to model OneTrainer, which keeps
    # the originals referenced in its NamedParameterGroupCollection (the swap
    # can't GC them). This `pinned` list stays referenced until main() returns.
    pinned = None
    if args.pin_originals:
        import torch.nn as nn
        pinned = [m.weight for m in unet.modules()
                  if isinstance(m, (nn.Linear, nn.Conv2d)) and m.weight is not None]
        pb = sum(w.numel() * w.element_size() for w in pinned)
        print(f"[pin] holding {len(pinned)} original weights = {_gb(pb):.2f} GB "
              f"(simulates OneTrainer pinned originals)")

    # --- swap to Concord (old fused lineage) ---
    cfg_ns = SimpleNamespace(
        concord_alpha=0.1, concord_beta1=0.0, concord_tickdown="row",
        concord_refit_target=16384, concord_target_modules=".*",
        concord_wrap_embeddings=False, concord_finetune_init=False,
    )
    t0 = time.time()
    concord_layers = wrap_model(unet, cfg_ns, device=dev)
    torch.cuda.synchronize()
    t_swap = time.time() - t0
    gc.collect(); torch.cuda.empty_cache()
    print(f"[swap] wrapped {len(concord_layers)} modules in {t_swap:.1f}s")
    print(f"[mem] after Concord swap (originals GC'd): "
          f"{_gb(torch.cuda.memory_allocated()):.2f} GB allocated")

    # --- bits/param accounting ---
    state_bytes = 0
    weight_numel = 0
    for m in concord_layers:
        weight_numel += m.s_slow.numel()
        for _, b in m.named_buffers():
            state_bytes += b.numel() * b.element_size()
    if weight_numel:
        bpp = state_bytes * 8 / weight_numel
        adamw_bytes = weight_numel * (112 / 8)  # fp32 master+bf16+m+v
        print(f"[state] concord-managed weights: {weight_numel/1e9:.3f} B params")
        print(f"[state] concord int state: {_gb(state_bytes):.2f} GB "
              f"({bpp:.1f} bits/param)")
        print(f"[state] same params under AdamW (112 b/p) would be "
              f"{_gb(adamw_bytes):.2f} GB -> Concord saves "
              f"{_gb(adamw_bytes - state_bytes):.2f} GB of state")

    for m in concord_layers:
        m.set_lr(args.lr)

    # aux params = everything still trainable (norms, biases, anything unswapped)
    aux_params = [p for p in unet.parameters() if p.requires_grad]
    aux_n = sum(p.numel() for p in aux_params)
    print(f"[aux] {len(aux_params)} aux tensors, {aux_n/1e6:.2f} M elements "
          f"-> AdamW")
    aux_opt = torch.optim.AdamW(aux_params, lr=args.lr) if aux_params else None

    # --- fabricate SDXL inputs ---
    B = args.batch
    lat = args.res // 8

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=torch.bfloat16)

    add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}

    print(f"[run] {args.steps} steps | bsz={B} latent={lat}x{lat} "
          f"(res {args.res}) | grad_ckpt={not args.no_ckpt}")
    for step in range(args.steps):
        if aux_opt:
            aux_opt.zero_grad(set_to_none=True)
        # requires_grad on the input so the graph has a grad path even though
        # concord weights are non-grad buffers (else: "element 0 ... does not
        # require grad" -- the documented wrap_model failure mode).
        sample = rnd(B, 4, lat, lat).requires_grad_(True)
        ts = torch.randint(0, 1000, (B,), device=dev)
        ehs = rnd(B, 77, 2048)
        target = rnd(B, 4, lat, lat)
        out = unet(sample, ts, encoder_hidden_states=ehs,
                   added_cond_kwargs=add_cond).sample
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        if aux_opt:
            aux_opt.step()
        torch.cuda.synchronize()
        print(f"[step {step}] loss={loss.item():.4f} | "
              f"peak_alloc={_gb(torch.cuda.max_memory_allocated()):.2f}GB | "
              f"peak_reserved={_gb(torch.cuda.max_memory_reserved()):.2f}GB")

    peak_a = _gb(torch.cuda.max_memory_allocated())
    peak_r = _gb(torch.cuda.max_memory_reserved())
    verdict = "FITS 24GB" if peak_r < 24.0 else "OVER 24GB"
    print()
    print("=" * 64)
    print(f"[RESULT] size={args.size} res={args.res} bsz={B} "
          f"ckpt={not args.no_ckpt}")
    print(f"[RESULT] peak allocated : {peak_a:.2f} GB")
    print(f"[RESULT] peak reserved  : {peak_r:.2f} GB   <- {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
