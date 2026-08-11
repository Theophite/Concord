"""Offline gap-guarded hard-negative audit: replay a fixed audit batch against a BACKUP's
packed Concord state under three weight views (deploy / arm-L / arm-H), and log the mining
telemetry -- per-example deploy loss, branch disagreement, the would-be-mined set.

Why offline: the live meter needs a per-layer bf16 weight cache to swap views into, which
fused matmul does not keep -- but the BACKUP carries the full packed state, and the branch
views are pure integer arithmetic on it. Running here (e.g. in the exit-42 relaunch gap, or
on demand) has zero interaction with training, no CUDA-graph or OOM constraints, and a FIXED
audit set measured against every backup yields longitudinal per-example curves.

The audit cache (workspace/concord_hardneg_audit.pt) is captured once by the trainer when
concord_hardneg_meter is on: the exact replayable UNet inputs + target (deterministic noise),
so this script needs no VAE, text encoder, scheduler, or dataset -- just the base UNet
architecture and the backup's unet/*.safetensors shards.

Views (the same algebra as the live meter, ConcordController.set_branch_view):
  deploy = (s_slow + v_slow) * 128 * 2^(row_exp + col_exp - 15)
  view_A = deploy + 2 * e_A * 2^(arm_row_exp + arm_col_exp) * 2^(row_exp + col_exp - 15)
(the factor 2 restores full magnitude: under the held-out router each arm integrates ~half
the data, so the two views are the two half-data estimates of the weight). MANTISSA_BIAS=15
is the packed format constant (concord invariants doc).

Mining rule metered (CPU receipts: exp48 [epic-williamson], exps 33/34 [mechanics lineage]):
trusted = branch disagreement |loss_L - loss_H| <= its p75; mined = trusted AND deploy loss
>= the trusted p67 -- high loss the two data halves AGREE about. Telemetry only.

Usage:
  python scripts/concord_hardneg_audit.py --backup <backup dir> --base <sdxl model dir/id>
      [--cache <audit .pt>] [--out <jsonl>] [--device cuda]
Defaults: cache = <backup>/../../concord_hardneg_audit.pt (the workspace layout);
out = <cache dir>/concord_hardneg_audit.jsonl (appends one row per invocation).
"""
import argparse
import glob
import json
import os
import sys

import torch

MANTISSA_BIAS = 15


def load_backup_unet_sd(backup):
    files = sorted(glob.glob(os.path.join(backup, "unet", "*.safetensors")))
    if not files:
        raise SystemExit(f"no unet/*.safetensors under {backup} -- is this an internal backup?")
    from safetensors.torch import load_file
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd


def decode_views(sd, prefix, weight_shape):
    """Materialize (deploy, view_L, view_H) fp32 weights for one packed layer."""
    p = sd[prefix + "packed_w"].to(torch.int32)
    e_L = (p >> 24).float()
    e_H = ((p << 8) >> 24).float()
    s8 = ((p << 16) >> 24).float()
    v8 = ((p << 24) >> 24).float()
    scale_fwd = torch.pow(2.0, sd[prefix + "row_exp"].float()[:, None]
                          + sd[prefix + "col_exp"].float()[None, :] - MANTISSA_BIAS)
    ascale = torch.pow(2.0, sd[prefix + "arm_row_exp"].float()[:, None]
                       + sd[prefix + "arm_col_exp"].float()[None, :])
    deploy = (s8 + v8) * 128.0 * scale_fwd
    views = {
        "deploy": deploy,
        "L": deploy + 2.0 * e_L * ascale * scale_fwd,
        "H": deploy + 2.0 * e_H * ascale * scale_fwd,
    }
    return {k: v.reshape(weight_shape) for k, v in views.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", required=True, help="internal backup dir (contains unet/)")
    ap.add_argument("--base", required=True, help="base SDXL model dir or hub id (UNet architecture)")
    ap.add_argument("--cache", default=None, help="audit cache .pt (default: <backup>/../../concord_hardneg_audit.pt)")
    ap.add_argument("--out", default=None, help="output jsonl (default: next to the cache)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache_path = args.cache or os.path.normpath(
        os.path.join(args.backup, "..", "..", "concord_hardneg_audit.pt"))
    if not os.path.exists(cache_path):
        raise SystemExit(f"audit cache not found: {cache_path} -- enable concord_hardneg_meter "
                         f"for one segment to capture it (works under fused matmul too)")
    out_path = args.out or os.path.join(os.path.dirname(cache_path), "concord_hardneg_audit.jsonl")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    dev = torch.device(args.device)

    print(f"[hardneg-audit] backup={os.path.basename(os.path.normpath(args.backup))} "
          f"cache step={cache.get('captured_step')} B={cache['target'].shape[0]}")

    sd = load_backup_unet_sd(args.backup)
    packed_prefixes = sorted(k[:-len("packed_w")] for k in sd if k.endswith("packed_w"))
    if not packed_prefixes:
        raise SystemExit("backup carries no packed layers (standard backup?) -- nothing to view")

    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet",
                                                torch_dtype=torch.bfloat16)
    # non-swapped weights (norms, biases, everything unpacked) come from the backup verbatim
    unet.load_state_dict({k: v for k, v in sd.items() if k in unet.state_dict()
                          and v.shape == unet.state_dict()[k].shape}, strict=False)
    unet.to(dev).eval()
    params = dict(unet.named_parameters())

    inputs = {
        "sample": cache["latent_input"].to(dev, torch.bfloat16),
        "timestep": cache["timestep"].to(dev),
        "encoder_hidden_states": cache["encoder_hidden_states"].to(dev, torch.bfloat16),
        "added_cond_kwargs": {k: (v.to(dev, torch.bfloat16) if torch.is_tensor(v) else v)
                              for k, v in (cache.get("added_cond_kwargs") or {}).items()},
    }
    target = cache["target"].to(dev, torch.float32)
    B = target.shape[0]

    per_view = {}
    for which in ("deploy", "L", "H"):
        with torch.no_grad():
            n_set = 0
            for pref in packed_prefixes:
                wkey = pref + "weight"
                if wkey not in params:
                    continue
                w = decode_views(sd, pref, params[wkey].shape)[which]
                params[wkey].data.copy_(w.to(dev, params[wkey].dtype))
                n_set += 1
            pred = unet(**inputs).sample.float()
            d = (pred - target).pow(2)
            per_view[which] = d.reshape(B, -1).mean(dim=1).cpu()
        print(f"[hardneg-audit] view={which}: {n_set} packed layers materialized, "
              f"loss p50={float(per_view[which].median()):.5f}")

    ld = per_view["deploy"]
    dis = (per_view["L"] - per_view["H"]).abs()
    q75 = dis.quantile(0.75)
    trusted = dis <= q75
    thr = ld[trusted].quantile(2.0 / 3.0) if int(trusted.sum()) > 0 else ld.quantile(2.0 / 3.0)
    mined = trusted & (ld >= thr)
    emb_ids = cache.get("emb_ids")     # per-example trainable-embedding token ids (numeric identity)
    rel = float((dis / ld.clamp_min(1e-12)).median())
    row = {
        "backup": os.path.basename(os.path.normpath(args.backup)),
        "captured_step": cache.get("captured_step"),
        "loss_deploy": [round(float(x), 6) for x in ld],
        "loss_L": [round(float(x), 6) for x in per_view["L"]],
        "loss_H": [round(float(x), 6) for x in per_view["H"]],
        "trusted": [bool(x) for x in trusted],
        "mined": [bool(x) for x in mined],
        "emb_ids": emb_ids,
        "rel_disagree_p50": round(rel, 6),
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    mtag = ""
    if emb_ids:
        picked = [str(emb_ids[i]) for i in torch.nonzero(mined).flatten().tolist()[:6]]
        mtag = " mined_emb_ids=" + ";".join(picked)
    print(f"[hardneg-audit] B={B} loss(dep) p50={float(ld.median()):.5f} | "
          f"branch-disagree p50={float(dis.median()):.2e} rel(p50)={rel:.3f} | "
          f"trusted={int(trusted.sum())}/{B} mined={int(mined.sum())}{mtag}\n"
          f"[hardneg-audit] appended -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
