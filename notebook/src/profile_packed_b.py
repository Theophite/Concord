"""Profile the packed-B training step at bsz=16.

Runs ~50 microbatches with torch.profiler active, dumps the top ops by
CUDA time and CPU time. Separates training-loop sections (forward / loss /
backward / aux_opt.step) so we can see where wallclock goes.

Usage:
    python profile_packed_b.py [--batch_size 16] [--warmup 10] [--profile 50]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_in_memory import get_loaders_in_memory
from prototype_packed_b import (ConcordLinearPackedB, ConcordConv2dPackedB,
                                  compute_drift_cancel_C)
from cifar_concord_packed_b import WiderConvNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=20,
                     help="microbatches to run before profiler starts")
    ap.add_argument("--profile", type=int, default=50,
                     help="microbatches to profile")
    ap.add_argument("--data_dir", type=str,
                     default=os.environ.get(
                         "CIFAR_DATA_DIR", "./cifar_data"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", flush=True)
        return
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

    tl, _ = get_loaders_in_memory(args.batch_size, device,
                                    data_dir=args.data_dir)

    model = WiderConvNet(device=device)
    concord_layers = [m for m in model.modules()
                       if isinstance(m, (ConcordLinearPackedB,
                                          ConcordConv2dPackedB))]
    for m in concord_layers:
        if not isinstance(m, ConcordConv2dPackedB):
            m.set_optimizer_kind('adamw', weight_decay=0.01,
                                  eps=1.0, step_cap=10.0)
        m.alpha = 0.1
        m.alpha_v_fast = 0.001
        m.drift_cancel_C = compute_drift_cancel_C(m.alpha, m.alpha_v_fast)
        m.wd_sv = 1e-5
        m.wd_sf = 1e-5
        m.lr = 0.1 * (0.2 if not isinstance(m, ConcordConv2dPackedB) else 1.0)

    bn_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and 'bn' in n.lower()]
    bias_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and 'bn' not in n.lower()]
    aux_opt = torch.optim.SGD(
        [{'params': bn_params, 'lr': 0.01},
         {'params': bias_params, 'lr': 0.02}],
        momentum=0.0)

    # Warmup: run a few steps to JIT-compile Triton kernels and let
    # cudnn benchmark settle.
    print(f"[profile] warmup {args.warmup} microbatches at bsz={args.batch_size}",
          flush=True)
    model.train()
    it = iter(tl)
    for _ in range(args.warmup):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tl)
            x, y = next(it)
        aux_opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        aux_opt.step()
    torch.cuda.synchronize()

    # Wall-clock baseline: time N microbatches with cuda.Event timers,
    # split into forward / loss+backward / aux_opt.step / Python.
    print(f"[profile] wall-clock baseline ({args.profile} microbatches)...",
          flush=True)
    sections = ['lr_update', 'fwd', 'bwd', 'aux', 'host']
    cuda_times = {k: 0.0 for k in sections}
    cpu_t0 = time.perf_counter()
    starts = {k: torch.cuda.Event(enable_timing=True) for k in sections}
    ends = {k: torch.cuda.Event(enable_timing=True) for k in sections}
    for i in range(args.profile):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tl)
            x, y = next(it)

        starts['lr_update'].record()
        # Mimic the cifar driver: per-step LR update (cheap, all on host).
        for m in concord_layers:
            pass  # in real training, m.lr is set; here we leave it
        aux_opt.zero_grad(set_to_none=True)
        ends['lr_update'].record()

        starts['fwd'].record()
        logits = model(x).float()
        ends['fwd'].record()

        starts['bwd'].record()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        ends['bwd'].record()

        starts['aux'].record()
        aux_opt.step()
        ends['aux'].record()
    torch.cuda.synchronize()
    cpu_dt = time.perf_counter() - cpu_t0
    for k in sections[:-1]:
        cuda_times[k] += starts[k].elapsed_time(ends[k])  # ms — last step only
    print(f"[profile] {args.profile} microbatches: {cpu_dt*1000:.1f}ms total "
          f"({cpu_dt*1000/args.profile:.2f}ms/mb on host)", flush=True)
    print(f"[profile] last microbatch GPU breakdown (ms):", flush=True)
    for k in sections[:-1]:
        print(f"  {k:>12s}: {cuda_times[k]:6.3f} ms", flush=True)

    # Detailed torch.profiler trace.
    print(f"\n[profile] running torch.profiler over {args.profile} microbatches",
          flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(args.profile):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(tl)
                x, y = next(it)
            aux_opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            aux_opt.step()
    torch.cuda.synchronize()

    # Top kernels by CUDA self-time.
    print(f"\n[profile] top 25 ops by CUDA self-time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25,
        max_name_column_width=70))

    print(f"\n[profile] top 15 ops by CPU self-time (host overhead):")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=15,
        max_name_column_width=70))


if __name__ == "__main__":
    main()
