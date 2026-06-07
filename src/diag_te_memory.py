"""How much more VRAM to also train the SDXL text encoders (CLIP-L + OpenCLIP-G)
under the Concord winner? Build both from config (random init -> footprint is
shape-only), Concord-swap their Linear layers, run fwd/bwd, report state + peak.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from sdxl_fit_smoketest import _gb
from concord_winner import swap_unet_to_winner, winner_step, make_aux_optimizer
import prototype_packed_b as ppb

dev, dt = torch.device("cuda"), torch.bfloat16

# SDXL text encoders.
cfg_l = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072,
                       num_hidden_layers=12, num_attention_heads=12,
                       max_position_embeddings=77, hidden_act="quick_gelu")
cfg_g = CLIPTextConfig(vocab_size=49408, hidden_size=1280, intermediate_size=5120,
                       num_hidden_layers=32, num_attention_heads=20,
                       max_position_embeddings=77, projection_dim=1280, hidden_act="gelu")
torch.manual_seed(0)
te1 = CLIPTextModel(cfg_l).to(dev, dt).train()             # CLIP-L
te2 = CLIPTextModelWithProjection(cfg_g).to(dev, dt).train()  # OpenCLIP-G

n1 = sum(p.numel() for p in te1.parameters())
n2 = sum(p.numel() for p in te2.parameters())
print(f"[build] CLIP-L {n1/1e6:.0f}M + OpenCLIP-G {n2/1e6:.0f}M = {(n1+n2)/1e9:.3f} B params")
print(f"[mem] after TEs -> cuda bf16: {_gb(torch.cuda.memory_allocated()):.2f} GB")

ppb.set_sigmag_noise(False)
layers = swap_unet_to_winner(te1, dev, 1e-5, verbose=False) \
    + swap_unet_to_winner(te2, dev, 1e-5, verbose=False)
import gc; gc.collect(); torch.cuda.empty_cache()

int_state = bf16buf = wrapped = 0
for m in layers:
    int_state += m.packed_w.numel() * 4 + m.row_exp.numel() * m.row_exp.element_size() \
        + m.col_exp.numel() * m.col_exp.element_size()
    b = getattr(m, "_bf16_weight_buf", None)
    bf16buf += b.numel() * b.element_size() if b is not None else 0
    wrapped += m.packed_w.numel()
print(f"[swap] {len(layers)} Linear -> Concord ({wrapped/1e6:.0f}M wrapped params)")
print(f"  int state    : {_gb(int_state):.2f} GB")
print(f"  bf16 buffer  : {_gb(bf16buf):.2f} GB")
resident = torch.cuda.memory_allocated()
print(f"[resident after swap]: {_gb(resident):.2f} GB")

aux = [p for p in te1.parameters() if p.requires_grad] \
    + [p for p in te2.parameters() if p.requires_grad]
aux_opt = make_aux_optimizer(aux, 1e-5)
print(f"[aux] {sum(p.numel() for p in aux)/1e6:.0f}M (embeddings+norms) -> SGD")

# training fwd/bwd: encode a batch of prompts (bsz x 77 tokens) through both TEs.
B = 1
ids = torch.randint(0, 49408, (B, 77), device=dev)
torch.cuda.reset_peak_memory_stats()
for it in range(3):
    winner_step(it, 3, layers, 1e-5, warmup=1, noise=False)
    aux_opt.zero_grad(set_to_none=True)
    out1 = te1(ids).last_hidden_state
    out2 = te2(ids).last_hidden_state
    loss = out1.float().pow(2).mean() + out2.float().pow(2).mean()
    loss.backward()
    aux_opt.step()
    for m in layers:
        m.rebalance()
    torch.cuda.synchronize()

peak = _gb(torch.cuda.max_memory_reserved())
print("=" * 60)
print(f"[RESULT] TE training peak reserved: {peak:.2f} GB")
print(f"[RESULT] incremental over UNet-only (21.7 GB): ~{peak:.1f} GB more")
print(f"[RESULT] UNet + TEs together  ≈ 21.7 + state(~{_gb(int_state+bf16buf):.1f}) "
      f"+ TE acts  -> {21.7 + _gb(int_state+bf16buf):.1f}+ GB")
print("=" * 60)
