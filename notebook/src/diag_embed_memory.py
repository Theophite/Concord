"""Embedding-only training footprint: TE transformer FROZEN, train just the
token/position embeddings (backprop through the frozen TEs to reach them).
No Concord state on the TE Linears. Measure peak with and without checkpointing
the (frozen) TE forward.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from sdxl_fit_smoketest import _gb

dev, dt = torch.device("cuda"), torch.bfloat16
cfg_l = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072,
                       num_hidden_layers=12, num_attention_heads=12,
                       max_position_embeddings=77, hidden_act="quick_gelu")
cfg_g = CLIPTextConfig(vocab_size=49408, hidden_size=1280, intermediate_size=5120,
                       num_hidden_layers=32, num_attention_heads=20,
                       max_position_embeddings=77, projection_dim=1280, hidden_act="gelu")


def run(use_ckpt):
    torch.manual_seed(0)
    te1 = CLIPTextModel(cfg_l).to(dev, dt).train()
    te2 = CLIPTextModelWithProjection(cfg_g).to(dev, dt).train()
    if use_ckpt:
        te1.gradient_checkpointing_enable()
        te2.gradient_checkpointing_enable()
    # freeze everything, then unfreeze ONLY the embeddings.
    trainable = []
    for te in (te1, te2):
        for p in te.parameters():
            p.requires_grad_(False)
        emb = te.text_model.embeddings
        for p in emb.parameters():           # token_embedding + position_embedding
            p.requires_grad_(True)
            trainable.append(p)
    n_train = sum(p.numel() for p in trainable)
    frozen_bytes = torch.cuda.memory_allocated()
    opt = torch.optim.AdamW(trainable, lr=1e-3)   # heaviest case (m+v); SGD/Concord lighter

    ids = torch.randint(0, 49408, (1, 77), device=dev)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out1 = te1(ids).last_hidden_state
        out2 = te2(ids).last_hidden_state
        loss = out1.float().pow(2).mean() + out2.float().pow(2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
    peak = _gb(torch.cuda.max_memory_reserved())
    del te1, te2, opt
    import gc; gc.collect(); torch.cuda.empty_cache()
    return n_train, _gb(frozen_bytes), peak


for use_ckpt in (False, True):
    n_train, frozen, peak = run(use_ckpt)
    tag = "TE grad-ckpt" if use_ckpt else "no ckpt"
    print(f"[{tag:12}] trainable embeddings {n_train/1e6:.0f}M | "
          f"frozen TE resident {frozen:.2f} GB | peak {peak:.2f} GB")

print("=" * 60)
print("Embedding training keeps the TE transformer frozen -> no Concord state,")
print("no optimizer state on the 716M Linears. Only the ~101M embeddings + acts.")
