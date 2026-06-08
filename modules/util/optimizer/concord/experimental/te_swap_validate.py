"""EXPERIMENTAL: validate swap_text_encoder_to_anchor on a real CLIP text-encoder
architecture, under the FULL production globals (ratio_coh + fixed_coh + sigma_g), EAGERLY.

Covers exactly the parts te_frozen_vslow.py didn't:
 - the swap actually walks a CLIP TE and packs its attention/MLP Linears;
 - the fused forward+backward self-steps those Linears inside a real transformer;
 - sigma_g fluctuation noise is ON (the UNet swap turns it on module-wide; [F] omitted it);
 - the v_slow anchor stays frozen and get_weight stays finite under all of it.

The CUDA-graph CAPTURE of the TE is the same path the UNet already uses (ConcordLinearPackedB
self-stepping in the captured backward), so it isn't re-tested here -- the full-run is the
final confirmation of that + the save round-trip.

Run: python experimental/te_swap_validate.py   (needs CUDA + transformers + the concord pkg).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import prototype_packed_b as ppb
import concord_winner as cw


def vslow(L):
    return ((L.packed_w << 24) >> 24).clone()


def main():
    if not torch.cuda.is_available():
        print("NEED CUDA. Aborting."); return
    from transformers import CLIPTextModel, CLIPTextConfig
    dev = "cuda"
    torch.manual_seed(0)
    cfg = CLIPTextConfig(hidden_size=256, intermediate_size=1024, num_hidden_layers=4,
                         num_attention_heads=4, vocab_size=1000, max_position_embeddings=77)
    te = CLIPTextModel(cfg).to(dev)

    # Production globals: exactly what swap_unet_to_winner flips on module-wide -- INCLUDING
    # sigma_g, which te_frozen_vslow [F] did not exercise.
    ppb.set_fixed_coh(True)
    ppb.set_ratio_coh(True)
    if hasattr(ppb, "set_sigmag_noise"):
        ppb.set_sigmag_noise(True, isotropic=True)

    n_lin = sum(1 for m in te.modules() if isinstance(m, torch.nn.Linear))
    layers = cw.swap_text_encoder_to_anchor(te, dev, lr=1e-3, wd_anchor=0.5, verbose=True)
    print(f"swapped {len(layers)} of {n_lin} Linear layers")

    anchors = [vslow(L) for L in layers]
    ppb.set_consolidate(dev, True)
    ids = torch.randint(0, 1000, (2, 77), device=dev)
    target = torch.randn(2, 77, 256, device=dev)

    def step():
        out = te(input_ids=ids).last_hidden_state.float()
        loss = torch.nn.functional.mse_loss(out, target)
        te.zero_grad(set_to_none=False)
        loss.backward()
        return loss.item()

    packed0 = [L.packed_w.clone() for L in layers]
    l0 = step()
    for _ in range(40):
        lN = step()

    stepped = sum(1 for L, b in zip(layers, packed0) if not torch.equal(L.packed_w, b))
    anchor_held = all(torch.equal(vslow(L), a) for L, a in zip(layers, anchors))
    finite = all(torch.isfinite(L.get_weight()).all().item() for L in layers)

    print(f"loss {l0:.4f} -> {lN:.4f}")
    print(f"self-stepped (packed_w changed): {stepped}/{len(layers)}")
    print(f"v_slow anchor frozen across all layers: {anchor_held}")
    print(f"get_weight finite under sigma_g (no NaN/Inf): {finite}")
    ok = stepped == len(layers) and anchor_held and finite and lN < l0
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
