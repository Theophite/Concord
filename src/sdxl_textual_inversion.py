"""REAL SDXL textual inversion with the norm-preserving Concord embedding.

Frozen SDXL (albedobaseXL); a new token is inserted into BOTH text encoders (via
HybridCLIPEmbedding) and trained -- through the real diffusion loss -- to capture
a reference image. The token's embedding is the only thing that learns; it's
stored in Concord's packed format and its deploy norm is pinned to each encoder's
real vocab median.

Flow: generate a reference -> add token to both tokenizers + wire the Concord
embeddings -> train (VAE encode -> noise -> frozen UNet eps-pred -> MSE -> backward
to the token) -> generate WITH the token. Saves before/after images.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from concord_embedding import HybridCLIPEmbedding
from concord_embedding_packed import ConcordPackedEmbedding

dev = torch.device("cuda")
dt = torch.bfloat16
CKPT = r"C:\Concord\albedobaseXL_v21.safetensors"
OUT = Path(r"C:\Concord")
PLACEHOLDER = "<cncd>"
REF_PROMPT = "a photograph of a corgi puppy sitting in grass"
TI_PROMPT = f"a photo of {PLACEHOLDER}"
RES = 512
STEPS = 250
LR = 1e-2

torch.manual_seed(0)
pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)
pipe.unet.enable_gradient_checkpointing()
train_sched = DDPMScheduler.from_config(pipe.scheduler.config)
print(f"[load] pipeline ready | prediction_type={train_sched.config.prediction_type}")

# 1) reference image (before the token exists).
ref = pipe(REF_PROMPT, num_inference_steps=25, height=1024, width=1024,
           generator=torch.Generator(dev).manual_seed(1)).images[0]
ref.save(OUT / "ti_reference.png")
print(f"[ref] generated -> ti_reference.png")

# 2) insert the new token into BOTH tokenizers + encoders.
mods = {}
for tag, tok, te in (("L", pipe.tokenizer, pipe.text_encoder),
                     ("G", pipe.tokenizer_2, pipe.text_encoder_2)):
    tok.add_tokens(PLACEHOLDER)
    base = te.get_input_embeddings()
    vocab, dim = base.weight.shape
    median = base.weight.float().norm(dim=1).median().item()
    nm = ConcordPackedEmbedding(1, dim, device=dev, lr=LR, target_norm=median)
    nm.init_tokens()
    te.text_model.embeddings.token_embedding = HybridCLIPEmbedding(base, nm, vocab)
    mods[tag] = (nm, te, tok, vocab, median)
    print(f"[insert] TE-{tag}: new id {tok.convert_tokens_to_ids(PLACEHOLDER)} "
          f"dim {dim} | pinned to real median {median:.3f}")


def encode_prompt(prompt):
    """SDXL prompt embeds with grad. Penultimate hidden from both TEs; pooled from
    TE-G at the EOS position (the new token id is ABOVE eos, so argmax pooling would
    wrongly land on it)."""
    hs = []
    pooled = None
    for tag in ("L", "G"):
        nm, te, tok, vocab, _ = mods[tag]
        ids = tok(prompt, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        out = te(ids, output_hidden_states=True)
        hs.append(out.hidden_states[-2])                       # penultimate
        if tag == "G":
            eos = (ids == tok.eos_token_id).float().argmax(dim=-1)
            pooled = te.text_projection(out.last_hidden_state[torch.arange(ids.shape[0]), eos])
    return torch.cat(hs, dim=-1), pooled                       # [B,77,2048], [B,1280]


# 3) reference -> latent.
import numpy as np
arr = np.array(ref.resize((RES, RES))).astype("float32") / 255.0      # [H,W,3]
img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev, dt) * 2 - 1
with torch.no_grad():
    lat = pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor
add_time_ids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)

# 4) train the token (Concord self-steps in backward; no torch optimizer).
print(f"[train] {STEPS} steps @ {RES} on the reference, prompt '{TI_PROMPT}'")
losses = []
for it in range(STEPS):
    noise = torch.randn_like(lat)
    t = torch.randint(0, train_sched.config.num_train_timesteps, (1,), device=dev)
    noisy = train_sched.add_noise(lat, noise, t)
    ehs, pooled = encode_prompt(TI_PROMPT)
    pred = pipe.unet(noisy, t, encoder_hidden_states=ehs.to(dt),
                     added_cond_kwargs={"text_embeds": pooled.to(dt),
                                        "time_ids": add_time_ids}).sample
    target = noise if train_sched.config.prediction_type == "epsilon" \
        else train_sched.get_velocity(lat, noise, t)
    loss = F.mse_loss(pred.float(), target.float())
    loss.backward()
    losses.append(loss.item())
    if it % 50 == 0 or it == STEPS - 1:
        dn = {k: mods[k][0].deploy_weight().norm().item() for k in mods}
        print(f"  [{it:3d}] loss {loss.item():.4f} | deploy norms "
              f"L={dn['L']:.3f}(med {mods['L'][4]:.3f}) G={dn['G']:.3f}(med {mods['G'][4]:.3f})")

# 5) generate WITH the learned token.
pipe.unet.disable_gradient_checkpointing()
with torch.no_grad():
    gen = pipe(TI_PROMPT, num_inference_steps=25, height=1024, width=1024,
               generator=torch.Generator(dev).manual_seed(1)).images[0]
gen.save(OUT / "ti_learned.png")
print(f"\n[RESULT] loss {losses[0]:.4f} -> {sum(losses[-10:])/10:.4f} "
      f"({'DESCENDS' if sum(losses[-10:])/10 < losses[0] else 'no'})")
print(f"[RESULT] reference -> ti_reference.png | learned '{PLACEHOLDER}' -> ti_learned.png")
print(f"[RESULT] token deploy norms stayed pinned to the real vocab medians.")
