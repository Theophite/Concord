"""Joint SDXL training: the Concord UNet WINNER + norm-preserving new-token
embeddings, together, through one diffusion loss (dreambooth + textual-inversion
style). The UNet is swapped to the winner (32 b/param, self-steps in backward);
the new token is inserted into both TEs (Concord packed, deploy norm pinned to the
vocab median); VAE + TE transformers stay frozen. Both halves learn the concept.

Memory: at 512 the UNet winner (~14 GB resident) + TEs/VAE + acts ~18 GB -- the
+embeddings combo that fits 24 GB. NOTE the winner's noise flag is global, so it
applies to the embedding step too (norm preservation pins after it).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from concord_embedding import HybridCLIPEmbedding
from concord_embedding_packed import ConcordPackedEmbedding
from concord_winner import swap_unet_to_winner, winner_step, make_aux_optimizer

dev, dt = torch.device("cuda"), torch.bfloat16
CKPT = r"C:\Concord\albedobaseXL_v21.safetensors"
OUT = Path(r"C:\Concord")
PH, REF_PROMPT = "<cncd>", "a photograph of a corgi puppy sitting in grass"
TI_PROMPT, RES, STEPS = "a photo of <cncd>", 512, 150
UNET_LR, EMB_LR = 5e-5, 5e-3

torch.manual_seed(0)
pipe = StableDiffusionXLPipeline.from_single_file(CKPT, torch_dtype=dt).to(dev)
pipe.set_progress_bar_config(disable=True)
for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
    m.requires_grad_(False)
sched = DDPMScheduler.from_config(pipe.scheduler.config)

# reference (original UNet, before the swap).
ref = pipe(REF_PROMPT, num_inference_steps=25, height=1024, width=1024,
           generator=torch.Generator(dev).manual_seed(1)).images[0]
ref.save(OUT / "joint_reference.png")
print("[ref] -> joint_reference.png")

# swap the UNet to the Concord winner; collect its aux (norm/bias) params for SGD.
unet_layers = swap_unet_to_winner(pipe.unet, dev, UNET_LR)
import gc; gc.collect(); torch.cuda.empty_cache()   # release the reference-gen pool + the
                                                    # GC'd original UNet before training (else +~10GB -> spill)
aux = [p for p in pipe.unet.parameters() if p.requires_grad]
aux_opt = make_aux_optimizer(aux, UNET_LR)
print(f"[unet] winner swap: {len(unet_layers)} Concord layers + "
      f"{sum(p.numel() for p in aux)/1e6:.1f}M aux (SGD)")

# insert the new token into both TEs.
mods = {}
for tag, tok, te in (("L", pipe.tokenizer, pipe.text_encoder),
                     ("G", pipe.tokenizer_2, pipe.text_encoder_2)):
    tok.add_tokens(PH)
    base = te.get_input_embeddings()
    vocab, dim = base.weight.shape
    median = base.weight.float().norm(dim=1).median().item()
    nm = ConcordPackedEmbedding(1, dim, device=dev, lr=EMB_LR, target_norm=median)
    nm.init_tokens()
    te.text_model.embeddings.token_embedding = HybridCLIPEmbedding(base, nm, vocab)
    mods[tag] = (nm, te, tok, median)
print(f"[token] inserted into both TEs, pinned to medians "
      f"L={mods['L'][3]:.3f} G={mods['G'][3]:.3f}")


def encode_prompt(prompt):
    hs, pooled = [], None
    for tag in ("L", "G"):
        nm, te, tok, _ = mods[tag]
        ids = tok(prompt, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        out = te(ids, output_hidden_states=True)
        hs.append(out.hidden_states[-2])
        if tag == "G":
            eos = (ids == tok.eos_token_id).float().argmax(dim=-1)
            pooled = te.text_projection(out.last_hidden_state[torch.arange(ids.shape[0]), eos])
    return torch.cat(hs, dim=-1), pooled


arr = np.array(ref.resize((RES, RES))).astype("float32") / 255.0
img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev, dt) * 2 - 1
with torch.no_grad():
    lat = pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor
tids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)

w0 = unet_layers[0].weight.detach().float().clone()           # watch a UNet weight move
torch.cuda.reset_peak_memory_stats()
print(f"[train] {STEPS} steps @ {RES}: UNet winner + token, joint")
for it in range(STEPS):
    winner_step(it, STEPS, unet_layers, peak_lr=UNET_LR, warmup=10)   # UNet lr/sigma/floors
    aux_opt.zero_grad(set_to_none=True)
    noise = torch.randn_like(lat)
    t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
    ehs, pooled = encode_prompt(TI_PROMPT)
    pred = pipe.unet(sched.add_noise(lat, noise, t), t, encoder_hidden_states=ehs.to(dt),
                     added_cond_kwargs={"text_embeds": pooled.to(dt), "time_ids": tids}).sample
    loss = F.mse_loss(pred.float(), noise.float())
    loss.backward()
    aux_opt.step()
    for m in unet_layers:
        m.rebalance()
    if it % 30 == 0 or it == STEPS - 1:
        dw = ((unet_layers[0].weight.detach().float() - w0).norm() / w0.norm()).item()
        dn = {k: mods[k][0].deploy_weight().norm().item() for k in mods}
        print(f"  [{it:3d}] loss {loss.item():.4f} | UNet w moved {dw:.2e} | "
              f"token norms L={dn['L']:.3f} G={dn['G']:.3f}")

peak = torch.cuda.max_memory_reserved() / 1024**3
gc.collect(); torch.cuda.empty_cache()              # free training acts before the 1024 sample
with torch.no_grad():
    gen = pipe(TI_PROMPT, num_inference_steps=25, height=1024, width=1024,
               generator=torch.Generator(dev).manual_seed(1)).images[0]
gen.save(OUT / "joint_learned.png")
print(f"\n[RESULT] peak {peak:.1f} GB (fits 24) | UNet winner weights moved + token norms "
      f"pinned to medians | joint_reference.png vs joint_learned.png")
