"""Faithfulness check: does the Concord swap preserve a REAL SDXL finetune?

Loads the UNet from a single-file SDXL checkpoint (e.g. albedobaseXL_v21),
runs a reference forward, swaps every Linear/Conv2d to Concord (load_weights
ingests the real pretrained weights), runs the forward again on the SAME input,
and reports the delta.

The DESIGN claims the swap is non-destructive ("bit-identical forward at step 0,
Concord's live weight == init until the first backward"). This measures that on
real finetuned weights, and isolates Concord's 32 b/param int-representation
error by comparing bf16-vs-bf16 (so generic bf16 rounding cancels out -- the
residual is what Concord's int16 split + shared-exponent reconstruction adds).

Run:
  python src/sdxl_real_checkpoint.py --checkpoint C:\\Concord\\albedobaseXL_v21.safetensors
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from diffusers import UNet2DConditionModel

import onetrainer_concord_patch
from concord_optimizer import wrap_model
from sdxl_fit_smoketest import SDXL_UNET_CONFIG, _gb


def load_unet_single_file(path, dtype):
    """Load just the UNet from an original-format SDXL .safetensors.
    Try diffusers' from_single_file; fall back to manual key conversion
    against our known SDXL config (no network needed)."""
    try:
        unet = UNet2DConditionModel.from_single_file(path, torch_dtype=dtype)
        print("[load] via UNet2DConditionModel.from_single_file")
        return unet
    except Exception as e:
        print(f"[load] from_single_file failed ({type(e).__name__}: {e}); "
              f"falling back to manual conversion")
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
    sd = load_file(path)
    unet = UNet2DConditionModel(**SDXL_UNET_CONFIG)
    converted = convert_ldm_unet_checkpoint(sd, config=unet.config)
    missing, unexpected = unet.load_state_dict(converted, strict=False)
    print(f"[load] manual convert: {len(missing)} missing, "
          f"{len(unexpected)} unexpected keys")
    return unet.to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    dt = torch.bfloat16
    free0, total0 = torch.cuda.mem_get_info()
    print(f"[gpu] free {_gb(free0):.2f} / {_gb(total0):.2f} GB at start")
    torch.cuda.reset_peak_memory_stats()
    onetrainer_concord_patch.install()

    print(f"[load] {args.checkpoint}")
    t0 = time.time()
    unet = load_unet_single_file(args.checkpoint, dt).to(dev).eval()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"[load] UNet {n_params/1e9:.4f} B params, bf16, in {time.time()-t0:.1f}s")

    # fixed input
    g = torch.Generator(device=dev).manual_seed(args.seed)
    B, lat = 1, args.res // 8

    def rnd(*shape):
        return torch.randn(*shape, device=dev, dtype=dt, generator=g)

    sample = rnd(B, 4, lat, lat)
    ts = torch.tensor([500], device=dev)
    ehs = rnd(B, 77, 2048)
    add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}

    # reference forward (real bf16 weights)
    with torch.no_grad():
        ref = unet(sample, ts, encoder_hidden_states=ehs,
                   added_cond_kwargs=add_cond).sample.float()
    print(f"[ref] forward done | out norm {ref.norm().item():.3f} "
          f"mean {ref.mean().item():.5f} std {ref.std().item():.4f}")

    # swap to Concord (ingest real weights)
    cfg_ns = SimpleNamespace(
        concord_alpha=0.1, concord_beta1=0.0, concord_tickdown="row",
        concord_refit_target=16384, concord_target_modules=".*",
        concord_wrap_embeddings=False, concord_finetune_init=False,
    )
    t0 = time.time()
    layers = wrap_model(unet, cfg_ns, device=dev)
    torch.cuda.synchronize()
    print(f"[swap] {len(layers)} modules in {time.time()-t0:.1f}s")

    # Concord forward (int16-reconstructed weights), same input
    with torch.no_grad():
        out = unet(sample, ts, encoder_hidden_states=ehs,
                   added_cond_kwargs=add_cond).sample.float()

    # compare
    diff = out - ref
    rel_l2 = (diff.norm() / ref.norm()).item()
    max_abs = diff.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0).item()
    ref_scale = ref.abs().mean().item()
    print()
    print("=" * 64)
    print("[FAITHFULNESS] Concord-swapped vs original (bf16-vs-bf16, same input)")
    print(f"  relative L2 error : {rel_l2:.4%}")
    print(f"  max abs error     : {max_abs:.4f}  (ref |mean| {ref_scale:.4f})")
    print(f"  cosine similarity : {cos:.6f}")
    print(f"  peak reserved     : {_gb(torch.cuda.max_memory_reserved()):.2f} GB")
    print("=" * 64)
    if cos > 0.999 and rel_l2 < 0.05:
        print("[verdict] swap is faithful -- finetune starts from the real model.")
    else:
        print("[verdict] WARNING: swap perturbs the model more than bf16 noise; "
              "investigate load_weights / reconstruction.")


if __name__ == "__main__":
    main()
