"""Decompose the winner's NO-checkpointing footprint at 1024, to answer: is there
room to fit it in 24 GB (option B)? Reports per-component bytes (int state, the
materialized bf16 weight buffer, Adafactor v, rebalance scratch) + the no-ckpt
peak (activations = peak - resident). Random init (footprint is value-independent).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

from sdxl_fit_smoketest import SDXL_UNET_CONFIG, _gb
from concord_winner import swap_unet_to_winner, winner_step, make_aux_optimizer
import prototype_packed_b as ppb

dev, dt = torch.device("cuda"), torch.bfloat16
torch.manual_seed(0)
unet = UNet2DConditionModel(**SDXL_UNET_CONFIG).to(dev, dt).train()
layers = swap_unet_to_winner(unet, dev, 1e-4, verbose=False)
ppb.set_sigmag_noise(False)
import gc; gc.collect(); torch.cuda.empty_cache()


def nbytes(t):
    return t.numel() * t.element_size() if t is not None else 0


int_state = bf16buf = vbuf = rebscratch = cohpre = 0
for m in layers:
    int_state += nbytes(m.packed_w) + nbytes(m.row_exp) + nbytes(m.col_exp)
    bf16buf += nbytes(getattr(m, "_bf16_weight_buf", None))
    vbuf += nbytes(getattr(m, "v_row", None)) + nbytes(getattr(m, "v_col", None)) \
        + nbytes(getattr(m, "_sum_v_inv", None))
    for b in ("_row_max_buf", "_col_max_buf", "_row_max_hwm", "_col_max_hwm", "_reb_seed"):
        rebscratch += nbytes(getattr(m, b, None))
    cohpre += nbytes(getattr(m, "_coh_pre", None))

print(f"[components] {len(layers)} concord layers")
print(f"  int state (packed_w + exps) : {_gb(int_state):.2f} GB")
print(f"  bf16 weight buffer          : {_gb(bf16buf):.2f} GB   <- materialized fwd weight")
print(f"  Adafactor v (row/col)       : {_gb(vbuf):.3f} GB")
print(f"  rebalance scratch           : {_gb(rebscratch):.3f} GB")
print(f"  coh_pre (should be 0)        : {_gb(cohpre):.3f} GB")
resident = torch.cuda.memory_allocated()
print(f"[resident after swap]         : {_gb(resident):.2f} GB allocated")

# no-checkpointing fwd/bwd to hit peak activations (will spill >24GB -> slow but
# max_memory_reserved still reports the true demand).
B, lat = 1, 1024 // 8
g = torch.Generator(device=dev).manual_seed(1)
rnd = lambda *s: torch.randn(*s, device=dev, dtype=dt, generator=g)
sample0 = rnd(B, 4, lat, lat)
ts = torch.tensor([500], device=dev)
ehs = rnd(B, 77, 2048)
add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}
target = sample0.clone()
aux = [p for p in unet.parameters() if p.requires_grad]
aux_opt = make_aux_optimizer(aux, 1e-4)

torch.cuda.reset_peak_memory_stats()
print("[run] 2 NO-CKPT steps at 1024 (may spill -> slow)...", flush=True)
for it in range(2):
    winner_step(it, 2, layers, 1e-4, warmup=1, noise=False)
    aux_opt.zero_grad(set_to_none=True)
    s = sample0.clone().requires_grad_(True)
    out = unet(s, ts, encoder_hidden_states=ehs, added_cond_kwargs=add_cond).sample
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    aux_opt.step()
    for m in layers:
        m.rebalance()
    torch.cuda.synchronize()

peak = _gb(torch.cuda.max_memory_reserved())
print(f"[no-ckpt peak reserved]       : {peak:.2f} GB")
print(f"[activations (peak - resident)]: ~{peak - _gb(resident):.2f} GB")
print(f"[verdict] over 24 GB by {peak - 24:.2f} GB; "
      f"dropping the bf16 weight buffer (fused-recon fwd) frees {_gb(bf16buf):.2f} GB")
