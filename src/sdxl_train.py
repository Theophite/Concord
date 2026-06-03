"""SDXL trainer wired from a declarative config + token spec -- no monkeypatching.

  - OPTIMIZER PICKER: a ConcordConfig selects + configures the UNet optimizer
    (the winner, or a baseline). configure_optimizer() does the swap + aux.
  - CONTROL PLANE: a list of TokenSpec drives per-token sanitize / fix / train.

The loop just runs winner_step(config) -> diffusion loss -> backward -> aux.step ->
rebalance. The Concord UNet layers AND the trainable tokens self-step in backward;
sanitized/fixed tokens are buffers and don't move. The whole behaviour is specified
by the two declarative objects, not patched into the loop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from concord_winner import ConcordConfig, configure_optimizer, winner_step, GatedRebalance
from control_plane import TokenSpec, apply_token_spec

dev, dt = torch.device("cuda"), torch.bfloat16


def encode_prompt(prompt, cps):
    """SDXL embeds with grad through the control planes (so trainable tokens learn);
    penultimate hidden from both, pooled from G at the EOS position."""
    hs, pooled = [], None
    for tag in ("L", "G"):
        te, tok = cps[tag]
        ids = tok(prompt, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        out = te(ids, output_hidden_states=True)
        hs.append(out.hidden_states[-2])
        if tag == "G":
            eos = (ids == tok.eos_token_id).float().argmax(dim=-1)
            pooled = te.text_projection(out.last_hidden_state[torch.arange(ids.shape[0]), eos])
    return torch.cat(hs, dim=-1), pooled


def train(pipe, opt_config, token_specs, lat, prompt, time_ids, steps, sched):
    # PICKER: configure the UNet optimizer from the config.
    layers, aux, cfg = configure_optimizer(pipe.unet, dev, opt_config)
    # CONTROL PLANE: apply the token spec to both encoders.
    cps = {}
    for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                         ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
        apply_token_spec(te, tok, token_specs, lr=5e-2)
        cps[tag] = (te, tok)
    import gc; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()                    # measure the TRAINING peak

    rebalance = GatedRebalance(layers)                      # fire only on actual overflow
    w0 = layers[0].weight.detach().float().clone() if layers else None
    for it in range(steps):
        if layers:
            winner_step(it, steps, layers, config=cfg)          # config-driven schedule
            if aux:
                aux.zero_grad(set_to_none=True)
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
        ehs, pooled = encode_prompt(prompt, cps)
        pred = pipe.unet(sched.add_noise(lat, noise, t), t, encoder_hidden_states=ehs.to(dt),
                         added_cond_kwargs={"text_embeds": pooled.to(dt), "time_ids": time_ids}).sample
        F.mse_loss(pred.float(), noise.float()).backward()      # UNet + trainable tokens self-step
        if aux:
            aux.step()
        rebalance()                                             # gated: skips the 794 no-op launches
        if it % 30 == 0 or it == steps - 1:
            dw = ((layers[0].weight.detach().float() - w0).norm() / w0.norm()).item() if layers else 0
            print(f"  [{it:3d}] UNet w moved {dw:.2e}")
    if layers:
        print(f"  [rebalance] fired {rebalance.fires}/{steps} steps "
              f"(gated: the rest skipped 794 no-op launches each)")
    return layers, cps


if __name__ == "__main__":
    pipe = StableDiffusionXLPipeline.from_single_file(
        r"C:\Concord\albedobaseXL_v21.safetensors", torch_dtype=dt).to(dev)
    pipe.set_progress_bar_config(disable=True)
    for m in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        m.requires_grad_(False)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    ref = pipe("a photograph of a corgi puppy in grass", num_inference_steps=25,
               height=1024, width=1024, generator=torch.Generator(dev).manual_seed(1)).images[0]

    # the TWO declarative objects that specify the whole run:
    opt_config = ConcordConfig(lr=5e-5, gf_consol=50.0, noise=True, aux="sgd")   # the picker
    token_specs = [TokenSpec("<cncd>", "train", init="dog"),                     # learn, init 'dog'
                   TokenSpec("tok",  "sanitize")]                              # suppress

    RES = 512
    arr = np.array(ref.resize((RES, RES))).astype("float32") / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev, dt) * 2 - 1
    with torch.no_grad():
        lat = pipe.vae.encode(img).latent_dist.sample() * pipe.vae.config.scaling_factor
    tids = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)

    print(f"[train] config={opt_config.kind} lr={opt_config.lr} | specs: "
          f"{[(s.token, s.mode) for s in token_specs]}")
    torch.cuda.reset_peak_memory_stats()
    layers, cps = train(pipe, opt_config, token_specs, lat, "a photo of <cncd>", tids, 150, sched)

    teL, tokL = cps["L"]
    cp = teL.get_input_embeddings()
    z = lambda w: cp(torch.tensor([[tokL.convert_tokens_to_ids(w)]], device=dev)).norm().item()
    print(f"\n[RESULT] peak {torch.cuda.max_memory_reserved()/1024**3:.1f} GB | "
          f"UNet winner trained ({len(layers)} layers) | "
          f"'<cncd>' (trained) norm {z('<cncd>'):.3f}, 'tok' (sanitized) norm {z('tok'):.3f} | "
          f"one config + one token list drove it all, no monkeypatching")
