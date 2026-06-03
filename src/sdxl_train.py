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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from concord_winner import ConcordConfig, configure_optimizer, winner_step, GatedRebalance
from control_plane import TokenSpec, apply_token_spec

dev, dt = torch.device("cuda"), torch.bfloat16


@dataclass
class DiffusionConfig:
    """The diffusion training recipe -- separate from the optimizer (ConcordConfig) and
    the tokens (TokenSpec). All zero == bare eps-MSE on plain Gaussian noise (so the
    default changes nothing). Capture-safe: the on/off are python floats compared to 0
    (constants baked at capture time), the math inside is pure tensor ops."""
    noise_offset: float = 0.0          # per-channel constant added to the target noise
                                       # (lets the model reach very dark / very bright)
    input_perturbation: float = 0.0    # extra noise on the INPUT latent only, target stays
                                       # the original noise (anti-overfit; Ning et al. 2023)
    min_snr_gamma: float = 0.0         # 0=off; ~5 -> Min-SNR-gamma loss weighting, which
                                       # down-weights the easy low-noise timesteps


def make_noise(lat, dcfg):
    """Target noise, with optional per-channel offset. torch.randn rides the default
    generator -> fresh per CUDA-graph replay."""
    noise = torch.randn_like(lat)
    if dcfg.noise_offset:
        noise = noise + dcfg.noise_offset * torch.randn(
            lat.shape[0], lat.shape[1], 1, 1, device=lat.device, dtype=lat.dtype)
    return noise


def noisy_latent(lat, noise, t, sched, dcfg):
    """The noised input. With input_perturbation the INPUT is noised by a perturbed
    draw, but the loss target stays the original `noise`."""
    inp = noise
    if dcfg.input_perturbation:
        inp = noise + dcfg.input_perturbation * torch.randn_like(noise)
    return sched.add_noise(lat, inp, t)


def diffusion_loss(pred, noise, t, sched, dcfg):
    """eps-MSE, optionally Min-SNR-gamma weighted. eps-pred weight = min(SNR,gamma)/SNR,
    SNR(t) = alpha_bar_t / (1 - alpha_bar_t). Pre-move sched.alphas_cumprod to the latent
    device (the graph trainer does) so the gather is sync-free under capture."""
    if dcfg.min_snr_gamma > 0:
        ac = sched.alphas_cumprod.to(t.device)[t]
        snr = ac / (1.0 - ac)
        w = torch.clamp(snr, max=dcfg.min_snr_gamma) / snr
        per = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(
            dim=list(range(1, pred.ndim)))
        return (w * per).mean()
    return F.mse_loss(pred.float(), noise.float())


def tokenize_prompt(prompt, cps):
    """Host-side (NOT capturable): fixed prompt -> static input_ids per encoder."""
    return {tag: tok(prompt, padding="max_length", max_length=77, truncation=True,
                     return_tensors="pt").input_ids.to(dev)
            for tag, (te, tok) in cps.items()}


def encode_from_ids(ids, cps):
    """GPU-side (capturable): TE forward through the control planes (so trainable
    tokens learn); penultimate hidden from both, pooled from G at the EOS position."""
    hs, pooled = [], None
    for tag in ("L", "G"):
        te, tok = cps[tag]
        out = te(ids[tag], output_hidden_states=True)
        hs.append(out.hidden_states[-2])
        if tag == "G":
            eos = (ids[tag] == tok.eos_token_id).int().argmax(dim=-1)
            ar = torch.arange(ids[tag].shape[0], device=ids[tag].device)   # on-device: no
            pooled = te.text_projection(out.last_hidden_state[ar, eos])    # CPU->CUDA sync
    return torch.cat(hs, dim=-1), pooled


def encode_prompt(prompt, cps):
    """Eager convenience: tokenize (host) then encode (GPU) in one call."""
    return encode_from_ids(tokenize_prompt(prompt, cps), cps)


def train(pipe, opt_config, token_specs, lat, prompt, time_ids, steps, sched, dcfg=None):
    dcfg = dcfg or DiffusionConfig()                        # default: bare eps-MSE
    sched.alphas_cumprod = sched.alphas_cumprod.to(dev)     # for min-SNR (and graph parity)
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
        noise = make_noise(lat, dcfg)                           # + optional noise offset
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev)
        ehs, pooled = encode_prompt(prompt, cps)
        pred = pipe.unet(noisy_latent(lat, noise, t, sched, dcfg), t,
                         encoder_hidden_states=ehs.to(dt),
                         added_cond_kwargs={"text_embeds": pooled.to(dt), "time_ids": time_ids}).sample
        diffusion_loss(pred, noise, t, sched, dcfg).backward()  # min-SNR-weighted; UNet+token self-step
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
