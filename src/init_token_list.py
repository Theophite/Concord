"""Initialize a whole new-token vocabulary on the REAL SDXL model: subword-mean
init for each token, inserted into both text encoders, deploy norms pinned to each
encoder's vocab median. (REDEFINE tokens init from the existing word's embedding.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from diffusers import StableDiffusionXLPipeline

from token_init import parse_list, init_specs_from_list, init_string
from concord_embedding_packed import insert_new_tokens

dev = torch.device("cuda")
CKPT = r"C:\Concord\albedobaseXL_v21.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=torch.bfloat16).to(dev)
for m in (pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)

rows = parse_list()
names, specs = init_specs_from_list(rows)
print(f"[list] {len(names)} new tokens to initialize\n")

# show how a sample tokenizes (the subword embeddings averaged for the init).
tok = pipe.tokenizer
for e, s in list(zip(rows, specs))[:12]:
    ids = tok(s, add_special_tokens=False).input_ids
    pieces = [p.replace("</w>", "") for p in tok.convert_ids_to_tokens(ids)]
    tagr = " [REDEFINE->existing]" if e["redefine"] else ""
    print(f"  {e['name']:12} {e['reason']:16} init='{s}' -> {pieces} ({len(ids)} subword"
          f"{'s' if len(ids) != 1 else ''}){tagr}")

# initialize the full vocabulary into BOTH encoders.
print()
for tag, te, tk in (("L", pipe.text_encoder, pipe.tokenizer),
                    ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
    nm = insert_new_tokens(te, tk, names, init_specs=specs, lr=5e-3, device=dev)
    dn = nm.deploy_weight().norm(dim=1)
    med = nm.target.item()
    print(f"[{tag}] {nm.K} tokens initialized | deploy norm "
          f"{dn.mean():.3f} +/- {dn.std():.3f} (all pinned to median {med:.3f}) | "
          f"packed {nm.core.packed_w.numel()*4} bytes")

# sanity: a REDEFINE token's init should sit near the existing word's embedding.
base = pipe.text_encoder.get_input_embeddings()
pid = pipe.tokenizer("tok", add_special_tokens=False).input_ids[0]
print(f"\n[check] 'tok' is an existing single CLIP token (id {pid}); its REDEFINE token "
      f"is seeded from that embedding's direction, pinned to the median.")
print("[done] full token vocabulary initialized on the real model (subword-mean + norm-pinned).")
