"""Diagnostic: after fine-tuning T5 with Concord 2acc at lr=0.08,
inspect the s_fast and s_slow distributions across all 97 layers.

Goal: understand why Concord doesn't blow up at lr=0.08 when SGD does.
Hypothesis: (a) saturation acts as a step-cap, OR (b) the SR-rounding
deadband suppresses small-grad updates.

Per layer, report:
  s_fast/s_slow:  median(|x|), p99(|x|), max(|x|), % saturated (|x|>=32700)
  Leading zeros (in the int16 representation) = 15 - ceil(log2(max(|x|,1)))
"""
import argparse
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset

from prototype_packed_2acc import ConcordLinear2acc
from test_t5_sst2 import wrap_with_concord, encode_sst2


def stats(name, x_int):
    """Report distribution stats on a flat int32 tensor."""
    abs_x = x_int.abs().flatten()
    n = abs_x.numel()
    nz = (abs_x > 0).sum().item()
    sat = (abs_x >= 32700).sum().item()
    near_cap_90 = (abs_x >= 29490).sum().item()   # 90% of int16 cap
    near_cap_75 = (abs_x >= 24576).sum().item()   # 75%
    abs_f = abs_x.float()
    median = abs_f.median().item()
    k = min(max(int(n * 0.99), 1), n)
    p99 = abs_f.kthvalue(k).values.item()
    maxv = abs_x.max().item()
    # Leading zeros in int16: count from MSB (bit 14, since sign bit is 15).
    # leading_zeros = 15 - floor(log2(maxv)) for maxv > 0.
    if maxv > 0:
        bits = int(math.floor(math.log2(maxv))) + 1
        leading = max(0, 15 - bits)
    else:
        leading = 16
    print(f"    {name:>8}:  med={median:>6.1f}  p99={p99:>7.1f}  "
          f"max={maxv:>6}  >75%cap={near_cap_75/n*100:>5.2f}%  "
          f"  >90%cap={near_cap_90/n*100:>5.2f}%  "
          f"  >sat={sat/n*100:>5.3f}%  "
          f"leading={leading}")
    return {'median': median, 'p99': p99, 'max': maxv,
            'sat_pct': sat/n*100, 'nz_pct': nz/n*100,
            'leading_zeros': leading}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()

    device = 'cuda'
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print(f"Loading T5-small + SST-2 + wrapping with 2acc...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    concord_layers, n_wrapped, n_params = wrap_with_concord(
        model, device=device, lr=args.lr, weight_decay=args.wd,
        init_mode='finetune', kind='two_acc', alpha=args.alpha)
    aux_opt = torch.optim.SGD([p for p in model.parameters()
                                  if p.requires_grad],
                                lr=1e-4, momentum=0.0)
    print(f"  {n_wrapped} 2acc layers, alpha={args.alpha}  "
          f"lr={args.lr}  wd={args.wd}", flush=True)

    train_data = load_dataset('glue', 'sst2')['train']
    enc = encode_sst2(tokenizer, train_data.select(range(args.bsz * 4)),
                        max_len=args.max_len)
    pad_id = tokenizer.pad_token_id
    inp = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    lbl = enc['labels'].clone().to(device)
    lbl[lbl == pad_id] = -100

    print(f"  Training {args.steps} steps at bsz={args.bsz}...", flush=True)
    model.train()
    for step in range(args.steps):
        idx = torch.randperm(inp.size(0))[:args.bsz]
        aux_opt.zero_grad(set_to_none=True)
        out = model(input_ids=inp[idx], attention_mask=mask[idx],
                     labels=lbl[idx])
        out.loss.backward()
        aux_opt.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}: loss={out.loss.item():.4f}",
                  flush=True)
    torch.cuda.synchronize()

    print(f"\n=== Per-layer int16 distributions after {args.steps} steps ===\n",
          flush=True)
    # Layer info: name, in_features, out_features
    for name, m in model.named_modules():
        if not isinstance(m, ConcordLinear2acc):
            continue
        s_fast, s_slow = m.get_state()
        print(f"  {name:<50s} ({m.out_features}x{m.in_features})")
        stats('s_fast', s_fast)
        stats('s_slow', s_slow)

    # Aggregate.
    print(f"\n=== Aggregate stats ===\n", flush=True)
    all_sf = []
    all_ss = []
    for m in model.modules():
        if isinstance(m, ConcordLinear2acc):
            s_fast, s_slow = m.get_state()
            all_sf.append(s_fast.flatten())
            all_ss.append(s_slow.flatten())
    all_sf = torch.cat(all_sf)
    all_ss = torch.cat(all_ss)
    print("  All 2acc params combined:")
    stats('s_fast', all_sf)
    stats('s_slow', all_ss)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)
    main()
