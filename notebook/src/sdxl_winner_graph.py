"""CUDA-graph capture for the winner step on SDXL -- eliminate the bsz=1 Triton
launch overhead. Mirrors the validated train_nanogpt recipe:

  - per-step scalars (lr, sigma, ratio floors) are DEVICE TENSORS updated OUTSIDE
    the graph by winner_step() -> they ride replays (no recapture);
  - capture ONE fwd+loss+bwd (Concord step + isotropic noise fused in backward);
    side-stream warmup x3, NO eager pre-roll; aux SGD step + rebalance() stay EAGER
    after replay (rebalance reads the row/col-max buffers the captured bwd wrote).

Correctness gate (non-negotiable, per the project log): run eager and graph from
the SAME seed/init and confirm the graph CONVERGES like eager (a real capture bug
diverges/grows). Use --target input --no_noise for a clean deterministic compare.

Run:
  python src/sdxl_winner_graph.py --mode both --size tiny --target input --no_noise
  python src/sdxl_winner_graph.py --mode graph --checkpoint <ckpt> --ckpt
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

import prototype_packed_b as ppb
from sdxl_fit_smoketest import SDXL_UNET_CONFIG, TINY_UNET_CONFIG, _gb
from concord_winner import swap_unet_to_winner, winner_step, make_aux_optimizer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
ap.add_argument("--size", choices=["tiny", "sdxl"], default="tiny")
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target", choices=["random", "input"], default="input")
ap.add_argument("--no_noise", action="store_true")
ap.add_argument("--ckpt", action="store_true")
ap.add_argument("--keep_rng", action="store_true",
                help="preserve_rng_state=True in checkpoint (correct but blocks capture)")
args = ap.parse_args()

dev, dt = torch.device("cuda"), torch.bfloat16

if args.ckpt:
    # Non-reentrant checkpointing (diffusers default) is NOT CUDA-graph capturable
    # (it does RNG-state save/restore = CPU syncs). Force the reentrant backend with
    # preserve_rng_state=False (SDXL UNet is dropout-free -> RNG preservation moot).
    import torch.utils.checkpoint as _ckpt
    _orig_ckpt = _ckpt.checkpoint

    def _capturable_checkpoint(function, *a, use_reentrant=None, preserve_rng_state=True, **kw):
        # NON-reentrant (modern; verified to fire the fused step ONCE) + drop RNG-state
        # save/restore (SDXL UNet is dropout-free, so it's a no-op) -> removes the CPU
        # syncs that block CUDA-graph capture, WITHOUT the reentrant step-breakage.
        return _orig_ckpt(function, *a, use_reentrant=False,
                          preserve_rng_state=args.keep_rng, **kw)

    _ckpt.checkpoint = _capturable_checkpoint
    print(f"[graph] patched checkpoint -> non-reentrant, "
          f"preserve_rng_state={args.keep_rng}")


def build_swap():
    torch.manual_seed(args.seed)
    if args.checkpoint:
        from sdxl_real_checkpoint import load_unet_single_file
        unet = load_unet_single_file(args.checkpoint, dt).to(dev)
    else:
        cfg = SDXL_UNET_CONFIG if args.size == "sdxl" else TINY_UNET_CONFIG
        unet = UNet2DConditionModel(**cfg).to(dev, dt)
    unet.train()
    if args.ckpt:
        unet.enable_gradient_checkpointing()
    layers = swap_unet_to_winner(unet, dev, args.lr, verbose=False)
    return unet, layers


def make_batch():
    g = torch.Generator(device=dev).manual_seed(args.seed + 1)
    B, lat = 1, args.res // 8
    rnd = lambda *s: torch.randn(*s, device=dev, dtype=dt, generator=g)
    sample = rnd(B, 4, lat, lat).requires_grad_(True)   # static, needs grad
    ts = torch.tensor([500], device=dev)
    ehs = rnd(B, 77, 2048)
    add_cond = {"text_embeds": rnd(B, 1280), "time_ids": rnd(B, 6)}
    target = sample.detach().clone() if args.target == "input" else rnd(B, 4, lat, lat)
    return sample, ts, ehs, add_cond, target


def run(mode):
    unet, layers = build_swap()
    if args.no_noise:
        ppb.set_sigmag_noise(False)   # actually turn the branch OFF (no wasted torch.randn draw)
    aux = [p for p in unet.parameters() if p.requires_grad]
    aux_opt = make_aux_optimizer(aux, args.lr)
    sample, ts, ehs, add_cond, target = make_batch()
    warmup = max(1, args.steps // 10)

    def step_fn():
        for p in aux:
            p.grad = None
        if sample.grad is not None:
            sample.grad = None
        out = unet(sample, ts, encoder_hidden_states=ehs, added_cond_kwargs=add_cond).sample
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        for m in layers:
            m.rebalance()        # captured too: 794 per-layer launches -> one graph
        return loss

    losses, times = [], []
    cg, cap_loss = None, None
    for it in range(args.steps):
        winner_step(it, args.steps, layers, args.lr, warmup=warmup, noise=not args.no_noise)
        torch.cuda.synchronize(); t0 = time.time()
        if mode == "eager":
            l = step_fn().item()
        else:
            if cg is None:                       # capture on first step
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        step_fn()
                torch.cuda.current_stream().wait_stream(s)
                cg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cg):
                    cap_loss = step_fn()
                l = cap_loss.item()
            else:
                cg.replay()                      # pure fwd+loss+bwd replay
                l = cap_loss.item()
        aux_opt.step()                           # eager (rebalance now inside step_fn/graph)
        torch.cuda.synchronize()
        times.append(time.time() - t0); losses.append(l)
    return losses, times, _gb(torch.cuda.max_memory_reserved())


def summarize(name, losses, times):
    steady = statistics.median(times[1:]) if len(times) > 1 else times[0]
    print(f"[{name:5}] loss {losses[0]:.5f} -> {losses[-1]:.5f} | "
          f"steady {steady*1e3:7.1f} ms/step (step0 {times[0]*1e3:.0f}ms)")
    return steady, losses[-1]


print(f"=== CUDA-graph winner: size={args.size} res={args.res} ckpt={args.ckpt} "
      f"noise={not args.no_noise} target={args.target} steps={args.steps} ===")

results = {}
try:
    if args.mode in ("eager", "both"):
        lo, ti, pk = run("eager")
        results["eager"] = summarize("eager", lo, ti)
        print(f"        peak {pk:.2f} GB")
        import gc
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if args.mode in ("graph", "both"):
        lo, ti, pk = run("graph")
        results["graph"] = summarize("graph", lo, ti)
        print(f"        peak {pk:.2f} GB | capture OK")
except Exception as e:
    import traceback
    print(f"[graph] CAPTURE FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("[graph] -> eager fallback is the mitigation (graph is speed-only)")

if "eager" in results and "graph" in results:
    es, ef = results["eager"]; gs, gf = results["graph"]
    conv = abs(gf - ef) < max(0.05 * abs(ef), 1e-3) or (gf < ef * 1.5)
    print("=" * 60)
    print(f"[SPEEDUP] {es/gs:.1f}x  ({es*1e3:.0f} -> {gs*1e3:.0f} ms/step)")
    print(f"[GATE] eager final {ef:.5f} vs graph final {gf:.5f} -> "
          f"{'CONVERGES (ok)' if conv else 'DIVERGES (BUG)'}")
    print("=" * 60)
