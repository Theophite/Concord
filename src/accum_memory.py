"""Measure the grad_W_accum footprint for graphed gradient accumulation, on the
REAL SDXL UNet structure + the user's `attentions` layer-filter. Meta device =
structure only, no weights, no GPU. Sum out*in over the swapped Linears (+Conv2d)
= the persistent accumulator memory paid ONLY when accum>1.
"""
import torch
import torch.nn as nn

# Canonical SDXL base UNet config (structure is fixed regardless of checkpoint).
SDXL = dict(
    sample_size=128, in_channels=4, out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(320, 640, 1280),
    layers_per_block=2, cross_attention_dim=2048,
    transformer_layers_per_block=(1, 2, 10),
    attention_head_dim=(5, 10, 20),
    use_linear_projection=True, addition_embed_type="text_time",
    addition_time_embed_dim=256, projection_class_embeddings_input_dim=2816,
)


def matches(name, filt):
    return filt in name      # OneTrainer ModuleFilter "attentions" = substring whitelist


def measure(layer_filter="attentions"):
    from diffusers import UNet2DConditionModel
    with torch.device("meta"):
        unet = UNet2DConditionModel(**SDXL)

    cats = {"attn(self/cross q/k/v/out)": 0, "ff (geglu)": 0, "proj_in/out": 0, "other": 0}
    n_lin = n_conv = 0
    elems = 0
    for name, m in unet.named_modules():
        if not isinstance(m, (nn.Linear, nn.Conv2d)):
            continue
        if layer_filter and not matches(name, layer_filter):
            continue
        if isinstance(m, nn.Linear):
            sz = m.out_features * m.in_features; n_lin += 1
        else:
            sz = m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1]; n_conv += 1
        elems += sz
        if ".ff." in name:
            cats["ff (geglu)"] += sz
        elif name.endswith(("to_q", "to_k", "to_v")) or ".to_out" in name:
            cats["attn(self/cross q/k/v/out)"] += sz
        elif "proj_in" in name or "proj_out" in name:
            cats["proj_in/out"] += sz
        else:
            cats["other"] += sz

    print(f"=== filter='{layer_filter}': {n_lin} Linear + {n_conv} Conv2d swapped ===")
    print(f"total grad_W elements: {elems/1e6:.1f} M")
    print(f"\ngrad_W_accum memory (the accum>1-only cost):")
    print(f"   bf16 (2B/elem): {elems*2/1e9:.2f} GB")
    print(f"   fp32 (4B/elem): {elems*4/1e9:.2f} GB")
    print(f"\nexisting Concord state for these layers (packed int32 + weight_buf bf16 = 6B):")
    print(f"   {elems*6/1e9:.2f} GB  -> accum adds +{2/6*100:.0f}% (bf16) / +{4/6*100:.0f}% (fp32)")
    print(f"\nbreakdown (bf16):")
    for k, v in cats.items():
        if v:
            print(f"   {k:<28} {v*2/1e9:.3f} GB  ({100*v/elems:.0f}%)")


if __name__ == "__main__":
    measure("attentions")
