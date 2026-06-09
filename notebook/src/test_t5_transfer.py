"""Transfer / catastrophic-forgetting test.

Pretrained T5-small was multi-task trained on GLUE (and others), so it
knows BOTH SST-2 and CoLA out of the box. We fine-tune on SST-2 with
either AdamW (sharp-minima) or Concord-2acc (flat-minima via chase +
int8/16 quantization noise), then measure how much of the original
CoLA capability survives.

Hypothesis: Concord's stay-near-pretrained chase dynamics should
preserve CoLA accuracy better than AdamW, which specializes to SST-2.

Output (per optimizer):
  Pretrained:   SST-2 = X.XX%   CoLA = X.XX%
  Post-FT:      SST-2 = X.XX%   CoLA = X.XX%
  dSST-2 = +X.XX%   dCoLA = -X.XX%
"""
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset

from test_t5_sst2 import wrap_with_concord, encode_sst2, eval_sst2


def encode_cola(tokenizer, ds, max_len=64):
    sentences = list(ds['sentence'])
    labels = list(ds['label'])
    inputs = [f"cola sentence: {s}" for s in sentences]
    targets = ['acceptable' if l == 1 else 'unacceptable' for l in labels]
    inp = tokenizer(inputs, max_length=max_len, truncation=True,
                     padding='max_length', return_tensors='pt')
    tgt = tokenizer(targets, max_length=4, truncation=True,
                     padding='max_length', return_tensors='pt')
    return {
        'input_ids': inp['input_ids'],
        'attention_mask': inp['attention_mask'],
        'labels': tgt['input_ids'],
    }


@torch.no_grad()
def eval_cola(model, tokenizer, eval_enc, device, bsz=32):
    model.eval()
    pos_id = tokenizer('acceptable', return_tensors='pt')['input_ids'][0, 0].item()
    neg_id = tokenizer('unacceptable', return_tensors='pt')['input_ids'][0, 0].item()
    n = eval_enc['input_ids'].size(0)
    correct = 0
    total = 0
    for i in range(0, n, bsz):
        inp = eval_enc['input_ids'][i:i+bsz].to(device)
        mask = eval_enc['attention_mask'][i:i+bsz].to(device)
        lbl_first = eval_enc['labels'][i:i+bsz, 0].to(device)
        decoder_inp = torch.full((inp.size(0), 1),
                                   model.config.decoder_start_token_id,
                                   device=device, dtype=torch.long)
        out = model(input_ids=inp, attention_mask=mask,
                     decoder_input_ids=decoder_inp)
        logits_first = out.logits[:, 0]
        pred_pos = logits_first[:, pos_id] > logits_first[:, neg_id]
        true_pos = (lbl_first == pos_id)
        correct += (pred_pos == true_pos).sum().item()
        total += inp.size(0)
    return correct / total


def run_finetune(args, mode, kind=None):
    """Returns (sst2_pre, cola_pre, sst2_post, cola_post, time_min, peak_mem)."""
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    tag = f"{mode}{('_' + kind) if kind else ''}"

    if mode == 'concord':
        concord_layers, n_wrapped, n_params = wrap_with_concord(
            model, device=device, lr=args.concord_lr,
            weight_decay=args.concord_wd, init_mode='finetune',
            kind=kind, alpha=args.alpha, beta1=args.beta1)
        # Post-wrap: enable mag_weighted forward emission on each 2acc
        # layer if requested.
        if kind == 'two_acc' and args.mag_weighted_forward:
            for m in concord_layers:
                m.mag_weighted = True
                m._resync_weight_buf()
            print(f"[{tag}] mag-weighted forward weight ENABLED",
                  flush=True)
        aux_opt = torch.optim.SGD([p for p in model.parameters()
                                      if p.requires_grad],
                                    lr=args.aux_lr, momentum=0.0)
        print(f"[{tag}] {n_wrapped} Concord({kind}) layers  "
              f"lr={args.concord_lr}  wd={args.concord_wd}",
              flush=True)
    elif mode == 'sgd':
        aux_opt = torch.optim.SGD(model.parameters(), lr=args.aux_lr,
                                    momentum=0.0,
                                    weight_decay=args.concord_wd)
        print(f"[{tag}] SGD (fp32)  lr={args.aux_lr}  "
              f"wd={args.concord_wd}", flush=True)
    else:
        aux_opt = torch.optim.AdamW(model.parameters(), lr=args.aux_lr,
                                      weight_decay=0.01)
        print(f"[{tag}] AdamW  lr={args.aux_lr}", flush=True)

    # Load datasets.
    sst2 = load_dataset('glue', 'sst2')
    cola_val = load_dataset('glue', 'cola')['validation']
    train_data = sst2['train']
    sst2_val = sst2['validation']
    if args.max_examples > 0:
        train_data = train_data.select(
            range(min(args.max_examples, len(train_data))))

    sst2_train_enc = encode_sst2(tokenizer, train_data,
                                    max_len=args.max_len)
    sst2_val_enc = encode_sst2(tokenizer, sst2_val, max_len=args.max_len)
    cola_val_enc = encode_cola(tokenizer, cola_val, max_len=args.max_len)
    for k in ('input_ids', 'attention_mask', 'labels'):
        sst2_train_enc[k] = sst2_train_enc[k].to(device)
        sst2_val_enc[k] = sst2_val_enc[k].to(device)
        cola_val_enc[k] = cola_val_enc[k].to(device)

    # Pre-fine-tune eval.
    sst2_pre = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_pre = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{tag}] PRE-FT:  SST2={sst2_pre*100:.2f}%  "
          f"CoLA={cola_pre*100:.2f}%", flush=True)

    # Set up graph capture (concord only — adamw eager is fine for fairness).
    pad_id = tokenizer.pad_token_id
    static_inp = torch.zeros(args.bsz, args.max_len,
                              dtype=torch.long, device=device,
                              requires_grad=False)
    static_mask = torch.zeros(args.bsz, args.max_len,
                               dtype=torch.long, device=device)
    static_lbl = torch.full((args.bsz, 4), -100,
                              dtype=torch.long, device=device)
    static_loss = None
    g = None

    if mode == 'concord' and args.use_graph:
        # Warm up on capture stream.
        idx = torch.arange(args.bsz, device=device)
        static_inp.copy_(sst2_train_enc['input_ids'][idx])
        static_mask.copy_(sst2_train_enc['attention_mask'][idx])
        lbl_seed = sst2_train_enc['labels'][idx].clone()
        lbl_seed[lbl_seed == pad_id] = -100
        static_lbl.copy_(lbl_seed)
        s_capture = torch.cuda.Stream()
        s_capture.wait_stream(torch.cuda.current_stream())
        model.train()
        with torch.cuda.stream(s_capture):
            for _ in range(3):
                aux_opt.zero_grad(set_to_none=True)
                out = model(input_ids=static_inp,
                              attention_mask=static_mask,
                              labels=static_lbl)
                out.loss.backward()
                aux_opt.step()
        torch.cuda.current_stream().wait_stream(s_capture)
        torch.cuda.synchronize()
        aux_opt.zero_grad(set_to_none=True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s_capture):
            aux_opt.zero_grad(set_to_none=False)
            out_g = model(input_ids=static_inp,
                            attention_mask=static_mask,
                            labels=static_lbl)
            static_loss = out_g.loss
            static_loss.backward()
            aux_opt.step()
        print(f"[{tag}] CUDA graph captured.", flush=True)

    # Train.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n_train = sst2_train_enc['input_ids'].size(0)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_t = time.time()
        running_loss = 0.0
        seen = 0
        for i in range(0, n_train - args.bsz + 1, args.bsz):
            idx = perm[i:i+args.bsz]
            inp = sst2_train_enc['input_ids'][idx]
            mask = sst2_train_enc['attention_mask'][idx]
            lbl = sst2_train_enc['labels'][idx].clone()
            lbl[lbl == pad_id] = -100
            if g is not None:
                static_inp.copy_(inp, non_blocking=True)
                static_mask.copy_(mask, non_blocking=True)
                static_lbl.copy_(lbl, non_blocking=True)
                g.replay()
                loss_value = static_loss.detach()
            else:
                aux_opt.zero_grad(set_to_none=True)
                out = model(input_ids=inp, attention_mask=mask,
                              labels=lbl)
                out.loss.backward()
                aux_opt.step()
                loss_value = out.loss.detach()
            running_loss += loss_value.item() * inp.size(0)
            seen += inp.size(0)
        ep_dt = time.time() - ep_t
        sst2_acc = eval_sst2(model, tokenizer, sst2_val_enc, device)
        print(f"[{tag}] ep {ep+1}/{args.epochs}  "
              f"tr_loss={running_loss/max(seen,1):.4f}  "
              f"SST2={sst2_acc*100:.2f}%  ({ep_dt:.1f}s)", flush=True)
    tot_min = (time.time() - t0) / 60
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    # Post-fine-tune eval.
    sst2_post = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_post = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{tag}] POST-FT: SST2={sst2_post*100:.2f}%  "
          f"CoLA={cola_post*100:.2f}%", flush=True)
    print(f"[{tag}] d SST2 = {(sst2_post - sst2_pre)*100:+.2f}%  "
          f"d CoLA = {(cola_post - cola_pre)*100:+.2f}%", flush=True)
    print(f"[{tag}] time={tot_min:.1f} min  peak_mem={peak_mem:.0f} MB",
          flush=True)
    return sst2_pre, cola_pre, sst2_post, cola_post, tot_min, peak_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--aux_lr", type=float, default=1e-4)
    ap.add_argument("--concord_lr", type=float, default=0.01)
    ap.add_argument("--concord_wd", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.1,
                     help="Chase rate (s_fast -> s_slow migration per "
                          "step). Higher = faster cumulative learning.")
    ap.add_argument("--beta1", type=float, default=0.0,
                     help="Momentum dampening on s_fast: "
                          "delta_t -= beta1 * s_fast.")
    ap.add_argument("--mag_weighted_forward", action='store_true',
                     default=False,
                     help="2acc only: emit bf16 forward weight as a "
                          "magnitude-weighted blend of s_fast and "
                          "s_slow (heavier on whichever is larger "
                          "in absolute value).")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_examples", type=int, default=0,
                     help="Cap training examples (0 = full SST-2).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_graph", action='store_true', default=True)
    ap.add_argument("--no_graph", dest='use_graph', action='store_false')
    ap.add_argument("--modes", type=str, default='adamw,concord_two_acc',
                     help="Comma-separated: adamw, concord_packed_b, "
                          "concord_two_acc")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    results = {}
    for spec in args.modes.split(','):
        spec = spec.strip()
        if spec == 'adamw':
            results[spec] = run_finetune(args, mode='adamw')
        elif spec == 'sgd':
            results[spec] = run_finetune(args, mode='sgd')
        elif spec.startswith('concord_'):
            kind = spec.split('_', 1)[1]
            results[spec] = run_finetune(args, mode='concord', kind=kind)
        else:
            print(f"Unknown mode: {spec}")
            sys.exit(1)
        print()

    # Summary.
    print("=" * 60)
    print("TRANSFER EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'optimizer':<22} | {'SST2 pre':>9} | {'SST2 post':>10} | "
          f"{'CoLA pre':>9} | {'CoLA post':>10} | {'dCoLA':>7}")
    print("-" * 80)
    for tag, (sst2_pre, cola_pre, sst2_post, cola_post, tm, mem) \
            in results.items():
        print(f"{tag:<22} | {sst2_pre*100:>8.2f}% | {sst2_post*100:>9.2f}% "
              f"| {cola_pre*100:>8.2f}% | {cola_post*100:>9.2f}% | "
              f"{(cola_post - cola_pre)*100:>+6.2f}%")


if __name__ == "__main__":
    main()
