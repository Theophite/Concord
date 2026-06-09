"""Does fp32 SGD survive lr=0.08 if we round the forward weight to bf16?

Hypothesis: Concord 2acc stayed stable at lr=0.08 while fp32 SGD blew up.
Earlier analysis suggested the difference is the bf16 forward weight —
a deadband / low-pass filter that prevents the tight feedback loop fp32
SGD has at high lr.

This test isolates that hypothesis: store weights as fp32 (full
precision, standard SGD), but materialize the forward weight as bf16.
No int16 storage, no SR-rounding, no chase. If this is stable at
lr=0.08, the bf16 forward IS the stability mechanism. If it blows up,
something else is at play.

Comparison:
  - sgd_fp32 (control): standard nn.Linear, fp32 SGD, lr=0.08
  - sgd_bf16fwd:        fp32 storage + bf16 forward weight, fp32 SGD, lr=0.08
  - concord_two_acc:    full Concord 2acc, lr=0.08 (reference)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset

from test_t5_sst2 import wrap_with_concord, encode_sst2, eval_sst2
from test_t5_transfer import encode_cola, eval_cola


class BF16ForwardLinear(nn.Module):
    """fp32 weight storage; forward materializes weight as bf16.

    Backward: autograd computes grad w.r.t. bf16 forward weight via cuBLAS,
    then back-propagates through the cast — gradient lands in fp32 weight
    unchanged. Standard SGD on self.weight then steps in fp32, but the
    next forward pass only sees the bf16-rounded version of that weight.
    Sub-bf16-LSB accumulation lives only in fp32 storage; the forward is
    blind to it until enough drift accumulates to cross a bf16 boundary.
    """
    def __init__(self, in_features, out_features, bias=True, device='cuda',
                 init_weight=None, init_bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if init_weight is not None:
            self.weight = nn.Parameter(init_weight.detach().clone().float().to(device))
        else:
            w = torch.empty(out_features, in_features, device=device)
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
            self.weight = nn.Parameter(w)
        if bias:
            if init_bias is not None:
                self.bias = nn.Parameter(init_bias.detach().clone().float().to(device))
            else:
                self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        in_dtype = x.dtype
        w_bf16 = self.weight.to(torch.bfloat16)
        b_bf16 = self.bias.to(torch.bfloat16) if self.bias is not None else None
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        y = F.linear(x_bf16, w_bf16, b_bf16)
        return y.to(in_dtype)


def wrap_with_bf16fwd(model, device='cuda'):
    n_wrapped = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                new_layer = BF16ForwardLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None, device=device,
                    init_weight=child.weight.data,
                    init_bias=child.bias.data if child.bias is not None else None)
                setattr(parent, child_name, new_layer)
                n_wrapped += 1
    return n_wrapped


def run_finetune(args, mode):
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    if mode == 'sgd_fp32':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.0, weight_decay=args.wd)
        print(f"[{mode}] standard nn.Linear, fp32 SGD, lr={args.lr} wd={args.wd}",
              flush=True)
    elif mode == 'sgd_bf16fwd':
        n_wrapped = wrap_with_bf16fwd(model, device=device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.0, weight_decay=args.wd)
        print(f"[{mode}] {n_wrapped} bf16-forward layers, fp32 SGD, "
              f"lr={args.lr} wd={args.wd}", flush=True)
    elif mode == 'concord_two_acc':
        concord_layers, n_wrapped, _ = wrap_with_concord(
            model, device=device, lr=args.lr, weight_decay=args.wd,
            init_mode='finetune', kind='two_acc', alpha=1.0, beta1=0.0)
        opt = torch.optim.SGD([p for p in model.parameters()
                                  if p.requires_grad],
                                lr=1e-4, momentum=0.0)
        print(f"[{mode}] {n_wrapped} Concord(2acc) layers, lr={args.lr} "
              f"wd={args.wd} alpha=1.0", flush=True)
    else:
        raise ValueError(f"Unknown mode {mode}")

    sst2 = load_dataset('glue', 'sst2')
    cola_val = load_dataset('glue', 'cola')['validation']
    train_data = sst2['train']
    sst2_val = sst2['validation']
    if args.max_examples > 0:
        train_data = train_data.select(
            range(min(args.max_examples, len(train_data))))

    sst2_train_enc = encode_sst2(tokenizer, train_data, max_len=args.max_len)
    sst2_val_enc = encode_sst2(tokenizer, sst2_val, max_len=args.max_len)
    cola_val_enc = encode_cola(tokenizer, cola_val, max_len=args.max_len)
    for k in ('input_ids', 'attention_mask', 'labels'):
        sst2_train_enc[k] = sst2_train_enc[k].to(device)
        sst2_val_enc[k] = sst2_val_enc[k].to(device)
        cola_val_enc[k] = cola_val_enc[k].to(device)

    sst2_pre = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_pre = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{mode}] PRE-FT:  SST2={sst2_pre*100:.2f}%  "
          f"CoLA={cola_pre*100:.2f}%", flush=True)

    pad_id = tokenizer.pad_token_id
    n_train = sst2_train_enc['input_ids'].size(0)
    model.train()
    t0 = time.time()
    nan_blowup = False
    for ep in range(args.epochs):
        perm = torch.randperm(n_train, device=device)
        running_loss = 0.0
        seen = 0
        ep_t = time.time()
        for i in range(0, n_train - args.bsz + 1, args.bsz):
            idx = perm[i:i + args.bsz]
            inp = sst2_train_enc['input_ids'][idx]
            mask = sst2_train_enc['attention_mask'][idx]
            lbl = sst2_train_enc['labels'][idx].clone()
            lbl[lbl == pad_id] = -100
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=inp, attention_mask=mask, labels=lbl)
            loss = out.loss
            if not torch.isfinite(loss):
                nan_blowup = True
                print(f"[{mode}] NaN/Inf loss at ep {ep+1} step "
                      f"{i//args.bsz} — aborting", flush=True)
                break
            loss.backward()
            opt.step()
            running_loss += loss.item() * inp.size(0)
            seen += inp.size(0)
        ep_dt = time.time() - ep_t
        if nan_blowup:
            break
        sst2_acc = eval_sst2(model, tokenizer, sst2_val_enc, device)
        print(f"[{mode}] ep {ep+1}/{args.epochs}  "
              f"tr_loss={running_loss/max(seen,1):.4f}  "
              f"SST2={sst2_acc*100:.2f}%  ({ep_dt:.1f}s)", flush=True)
    tot_min = (time.time() - t0) / 60

    if nan_blowup:
        print(f"[{mode}] BLOWUP after {tot_min:.1f} min", flush=True)
        return sst2_pre, cola_pre, float('nan'), float('nan'), tot_min, True

    sst2_post = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_post = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{mode}] POST-FT: SST2={sst2_post*100:.2f}%  "
          f"CoLA={cola_post*100:.2f}%", flush=True)
    print(f"[{mode}] d SST2 = {(sst2_post - sst2_pre)*100:+.2f}%  "
          f"d CoLA = {(cola_post - cola_pre)*100:+.2f}%", flush=True)
    print(f"[{mode}] time={tot_min:.1f} min", flush=True)
    return sst2_pre, cola_pre, sst2_post, cola_post, tot_min, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", type=str,
                     default="sgd_fp32,sgd_bf16fwd,concord_two_acc")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    results = {}
    for mode in args.modes.split(','):
        mode = mode.strip()
        results[mode] = run_finetune(args, mode)
        print(flush=True)

    print("=" * 70)
    print("BF16-FORWARD HYPOTHESIS SUMMARY")
    print("=" * 70)
    print(f"{'mode':<22} | {'SST2 pre':>9} | {'SST2 post':>10} | "
          f"{'CoLA pre':>9} | {'CoLA post':>10} | {'blew up?':>9}")
    print("-" * 85)
    for tag, (sst2_pre, cola_pre, sst2_post, cola_post, tm, blew) \
            in results.items():
        sp = f"{sst2_post*100:>9.2f}%" if not blew else "       NaN"
        cp = f"{cola_post*100:>9.2f}%" if not blew else "       NaN"
        print(f"{tag:<22} | {sst2_pre*100:>8.2f}% | {sp} "
              f"| {cola_pre*100:>8.2f}% | {cp} | "
              f"{'YES' if blew else 'no':>9}")


if __name__ == "__main__":
    main()
