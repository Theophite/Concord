"""Wire the norm-preserving Concord new-token embedding INTO a real CLIP text
encoder by swapping its token_embedding for the hybrid. Confirms: new tokens flow
through the unmodified CLIP forward, train via the in-backward Concord step, and
their deploy embedding stays pinned to the vocab MEDIAN norm. (Random CLIP weights
+ synthetic target -> validates the WIRING/training mechanism, not semantics.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from transformers import CLIPTextConfig, CLIPTextModel

from concord_embedding import HybridCLIPEmbedding
from concord_embedding_packed import ConcordPackedEmbedding

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Real CLIP-L text encoder (SDXL text_encoder), frozen.
cfg = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072,
                     num_hidden_layers=12, num_attention_heads=12,
                     max_position_embeddings=77, hidden_act="quick_gelu")
te = CLIPTextModel(cfg).to(dev, torch.bfloat16).eval()
for p in te.parameters():
    p.requires_grad_(False)

tok_emb = te.text_model.embeddings.token_embedding          # nn.Embedding [49408, 768]
vocab_size, dim = tok_emb.weight.shape
median = ConcordPackedEmbedding.vocab_median_norm(tok_emb.weight)
print(f"[clip] vocab {vocab_size} dim {dim} | frozen vocab median norm {median:.2f}")

# Two new concept tokens, pinned to the vocab median -- PACKED (32 b/param).
K = 2
new_mod = ConcordPackedEmbedding(K, dim, device=dev, lr=5e-2, target_norm=median)
new_mod.init_tokens()

# INSERT: swap the token_embedding for the hybrid. Nothing else in CLIP changes.
te.text_model.embeddings.token_embedding = HybridCLIPEmbedding(tok_emb, new_mod, vocab_size)
print(f"[insert] swapped token_embedding -> HybridCLIPEmbedding "
      f"(new ids {vocab_size}..{vocab_size+K-1})")

# A prompt that USES the new tokens (ids >= vocab_size), mid-sequence.
ids = torch.tensor([[0, 5, vocab_size + 0, 42, vocab_size + 1, 7, 9, 2]], device=dev)
torch.manual_seed(1)
target = torch.randn(1, ids.shape[1], dim, device=dev) * 0.3      # synthetic target hidden state

before = new_mod.deploy_weight().norm(dim=1).mean().item()
losses = []
for it in range(150):
    out = te(ids).last_hidden_state                              # full CLIP forward
    loss = F.mse_loss(out.float(), target)
    loss.backward()                                              # Concord new-token step fires here
    losses.append(loss.item())
moved = (new_mod.deploy_weight() - 0).norm(dim=1)                # current norms
print(f"[train] loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({'DESCENDS' if losses[-1] < losses[0] else 'no'})")
print(f"[norm]  new-token deploy norm {moved.mean().item():.2f}  (pinned to median {median:.2f})")
ok = abs(moved.mean().item() - median) < 0.05 * median and losses[-1] < losses[0]
print(f"[verdict] {'OK -- new tokens train through real CLIP, norm preserved' if ok else 'CHECK'}")
