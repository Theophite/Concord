"""Save/load a whole training run to a cache directory -- everything needed to
deploy or resume, in one place:

  cache/
    manifest.json              config (ConcordConfig) + token specs + file list
    unet_deploy.safetensors    the Concord UNet's CONSOLIDATED (deployable) weights
    control_L.pt / control_G.pt each control plane's deltas (sanitize/fix routes +
                               the trainable Concord packed state) -- NOT the frozen base
    emb_L.pt / emb_G.pt        the trained tokens' deploy embeddings (TI-style)

load_cache rebuilds it on a fresh pipeline (re-adds the tokens, restores the control
planes, loads the UNet deploy). Round-trips deterministically.
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from concord_winner import ConcordConfig, consolidated_state_dict
from control_plane import TokenSpec, apply_token_spec
from prototype_packed_b import ConcordLinearPackedB, ConcordConv2dPackedB


def save_cache(cache_dir, pipe, cps, config, token_specs, layers=None):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {"config": asdict(config),
                "specs": [{"token": s.token, "mode": s.mode} for s in token_specs],
                "files": {}}
    if layers:
        named = [(n, m) for n, m in pipe.unet.named_modules()
                 if isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB))]
        sd = consolidated_state_dict([m for _, m in named], [n for n, _ in named])
        save_file({k: v.contiguous() for k, v in sd.items()}, str(cache / "unet_deploy.safetensors"))
        manifest["files"]["unet"] = "unet_deploy.safetensors"
    for tag, cp in cps.items():
        state = {k: v for k, v in cp.state_dict().items() if not k.startswith("base.")}  # not the base
        torch.save(state, cache / f"control_{tag}.pt")
        manifest["files"][f"control_{tag}"] = f"control_{tag}.pt"
        if cp.trainable is not None:
            torch.save(cp.trainable.deploy_weight().cpu(), cache / f"emb_{tag}.pt")
            manifest["files"][f"emb_{tag}"] = f"emb_{tag}.pt"
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache


def load_cache(cache_dir, pipe):
    cache = Path(cache_dir)
    man = json.loads((cache / "manifest.json").read_text())
    config = ConcordConfig(**man["config"])
    specs = [TokenSpec(s["token"], s["mode"], init="zero") for s in man["specs"]]  # init overwritten below
    if "unet" in man["files"]:
        pipe.unet.load_state_dict(load_file(str(cache / man["files"]["unet"])), strict=False)
    cps = {}
    for tag, te, tok in (("L", pipe.text_encoder, pipe.tokenizer),
                         ("G", pipe.text_encoder_2, pipe.tokenizer_2)):
        apply_token_spec(te, tok, specs, lr=config.lr)         # rebuild structure (tokens/modes)
        cp = te.get_input_embeddings()
        cp.load_state_dict(torch.load(cache / man["files"][f"control_{tag}"]), strict=False)
        cps[tag] = cp
    return config, specs, cps
