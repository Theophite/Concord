"""Isolate every CUDA-graph sync in the encode path (both CLIP encoders + control
plane + EOS pooling + token self-step) WITHOUT the UNet/diffusion -- so I can debug
the TE capture in ~30s, not a 5-min full-trainer run. Captures encode_from_ids +
a dummy loss + backward (so the token self-steps through the frozen TE)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from diffusers import StableDiffusionXLPipeline

from control_plane import TokenSpec, apply_token_spec
from sdxl_train import tokenize_prompt, encode_from_ids, dev, dt

pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)

specs = [TokenSpec("<cncd>", "train", init="dog"), TokenSpec("tok", "sanitize")]
cps = {}
for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                     ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
    apply_token_spec(te, tok, specs, lr=5e-2)
    cps[tag] = (te, tok)
static_ids = tokenize_prompt("a photo of <cncd> by the sea", cps)


def step():
    ehs, pooled = encode_from_ids(static_ids, cps)
    loss = ehs.float().pow(2).mean() + pooled.float().pow(2).mean()
    loss.backward()
    return loss


s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        step()
torch.cuda.current_stream().wait_stream(s)

cp = cps["L"][0].get_input_embeddings()
try:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap = step()
    print("[capture] encode path (both CLIP encoders + token) captured OK")
    b = cp.trainable.deploy_weight().float().clone()
    g.replay(); g.replay()
    a = cp.trainable.deploy_weight().float().clone()
    print(f"[replay] token moved {(a-b).norm().item():.3e} | deploy norm {a.norm().item():.3f}")
    print("[RESULT] TE+token CAPTURABLE -> full graph can include the encoders")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n[RESULT] sync remains: {type(e).__name__}: {e}")
