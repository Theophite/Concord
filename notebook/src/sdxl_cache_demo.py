"""Train briefly, then SAVE EVERYTHING to a cache and verify the round-trip:
the Concord UNet deploy weights, the trained token embeddings, and the control
plane (sanitize/fix/train) -- all reloadable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from safetensors.torch import load_file

from concord_winner import ConcordConfig
from control_plane import TokenSpec
from sdxl_train import train
from run_cache import save_cache

dev, dt = torch.device("cuda"), torch.bfloat16
CACHE = r"C:\Concord\run_cache_demo"

pipe = StableDiffusionXLPipeline.from_single_file(
    r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)
sched = DDPMScheduler.from_config(pipe.scheduler.config)
ref = pipe("a corgi puppy", num_inference_steps=20, height=1024, width=1024,
           generator=torch.Generator(dev).manual_seed(1)).images[0]

config = ConcordConfig(lr=5e-5, aux="sgd")
specs = [TokenSpec("<cncd>", "train", init="dog"), TokenSpec("tok", "sanitize")]
RES = 512
arr = np.array(ref.resize((RES, RES))).astype("float32") / 255.0
img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev, dt) * 2 - 1
with torch.no_grad():
    lat = pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor
tids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)

layers, cps_te = train(pipe, config, specs, lat, "a photo of <cncd>", tids, 30, sched)
cps = {tag: te.get_input_embeddings() for tag, (te, tok) in cps_te.items()}

# SAVE EVERYTHING.
import gc; gc.collect(); torch.cuda.empty_cache()        # free training acts before materializing deploy
cache = save_cache(CACHE, pipe, cps, config, specs, layers=layers)
print(f"\n[cache] saved to {cache}")
for f in sorted(Path(cache).iterdir()):
    print(f"  {f.name:26} {f.stat().st_size/1e6:8.1f} MB")

# Verify the round-trip on the small parts (no full reload needed to prove it saved).
emb_saved = torch.load(Path(cache) / "emb_L.pt")
emb_live = cps["L"].trainable.deploy_weight().cpu()
match = torch.allclose(emb_saved.float(), emb_live.float(), atol=1e-3)
unet_keys = len(load_file(str(Path(cache) / "unet_deploy.safetensors")).keys())
ctrl = torch.load(Path(cache) / "control_L.pt")
print(f"\n[verify] token embeddings round-trip: {match} | UNet deploy: {unet_keys} layers saved | "
      f"control plane state keys: {sorted(k for k in ctrl)[:4]}...")
print("[done] everything (UNet deploy + token embeddings + control plane + config/specs) "
      "is cached and reloadable via run_cache.load_cache.")
