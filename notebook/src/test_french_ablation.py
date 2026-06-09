"""Targeted unlearning via mag-weighted chain-rule + chase Concord-2acc.

The chain-rule projection through the mag-weighted blend f(s_fast,s_slow)
= (sf|sf| + ss|ss|)/(|sf|+|ss|) is negative near the cusp at sf=0
(f has a local *maximum* along sf axis there). The chase then carries
that negative-sign step into s_slow as a persistent displacement.
Net dynamic: weights move AWAY from the gradient direction. Fed
targets the model already produces, it ACTIVELY UNLEARNS them.

Goal: ablate T5-small's English→French translation capability while
leaving English→German, SST-2, and CoLA intact.
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


def encode_translation(tokenizer, pairs, src_lang, tgt_lang,
                        prefix_lang_name, max_len_in=64, max_len_out=64):
    """pairs: list of {src_lang: ..., tgt_lang: ...} dicts."""
    inputs = [f"translate English to {prefix_lang_name}: {p[src_lang]}"
              for p in pairs]
    targets = [p[tgt_lang] for p in pairs]
    inp = tokenizer(inputs, max_length=max_len_in, truncation=True,
                     padding='max_length', return_tensors='pt')
    tgt = tokenizer(targets, max_length=max_len_out, truncation=True,
                     padding='max_length', return_tensors='pt')
    return {
        'input_ids': inp['input_ids'],
        'attention_mask': inp['attention_mask'],
        'labels': tgt['input_ids'],
    }


@torch.no_grad()
def eval_translation_loss(model, enc, device, bsz=16, pad_id=0):
    model.eval()
    n = enc['input_ids'].size(0)
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, n, bsz):
        inp = enc['input_ids'][i:i+bsz].to(device)
        mask = enc['attention_mask'][i:i+bsz].to(device)
        lbl = enc['labels'][i:i+bsz].clone().to(device)
        lbl[lbl == pad_id] = -100
        out = model(input_ids=inp, attention_mask=mask, labels=lbl)
        n_tokens = (lbl != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def sample_translations(model, tokenizer, prompts, device, max_length=40):
    model.eval()
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        g = model.generate(ids, max_length=max_length, num_beams=1)
        out.append(tokenizer.decode(g[0], skip_special_tokens=True))
    return out


def filter_pairs(ds, src='en', tgt='fr', max_chars=200, min_chars=15,
                  cap=20000):
    out = []
    for r in ds:
        t = r['translation']
        if not (min_chars < len(t[src]) < max_chars and
                 min_chars < len(t[tgt]) < max_chars):
            continue
        out.append(t)
        if len(out) >= cap:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=150)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Loading T5-small...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    pad_id = tokenizer.pad_token_id

    print("Wrapping with Concord-2acc + mag-weighted "
          "(chain-rule + chase UNLEARNING kernel)...", flush=True)
    concord_layers, n_wrapped, n_params = wrap_with_concord(
        model, device=device, lr=args.lr, weight_decay=args.wd,
        init_mode='finetune', kind='two_acc',
        alpha=args.alpha, beta1=args.beta1)
    for m in concord_layers:
        m.mag_weighted = True
        m._resync_weight_buf()
    print(f"  {n_wrapped} layers, lr={args.lr}, wd={args.wd}, "
          f"alpha={args.alpha}, beta1={args.beta1}", flush=True)
    aux_opt = torch.optim.SGD([p for p in model.parameters()
                                  if p.requires_grad],
                                lr=1e-4, momentum=0.0)

    print("Loading opus100 (en-fr + de-en) and GLUE (SST-2 + CoLA)...",
          flush=True)
    fr_train_ds = load_dataset('opus100', 'en-fr', split='train')
    fr_val_ds = load_dataset('opus100', 'en-fr', split='validation')
    de_val_ds = load_dataset('opus100', 'de-en', split='validation')
    sst2_val = load_dataset('glue', 'sst2')['validation']
    cola_val = load_dataset('glue', 'cola')['validation']

    fr_train_pairs = filter_pairs(fr_train_ds, 'en', 'fr',
                                    cap=max(args.n_train * 2, 10000))
    fr_val_pairs = filter_pairs(fr_val_ds, 'en', 'fr',
                                  cap=args.n_eval * 2)[:args.n_eval]
    de_val_pairs = filter_pairs(de_val_ds, 'en', 'de',
                                  cap=args.n_eval * 2)[:args.n_eval]
    fr_train_pairs = fr_train_pairs[:args.n_train]
    print(f"  fr_train: {len(fr_train_pairs)}  "
          f"fr_val: {len(fr_val_pairs)}  "
          f"de_val: {len(de_val_pairs)}", flush=True)

    fr_train_enc = encode_translation(tokenizer, fr_train_pairs,
                                        'en', 'fr', 'French',
                                        max_len_in=args.max_len,
                                        max_len_out=args.max_len)
    fr_val_enc = encode_translation(tokenizer, fr_val_pairs,
                                      'en', 'fr', 'French',
                                      max_len_in=args.max_len,
                                      max_len_out=args.max_len)
    de_val_enc = encode_translation(tokenizer, de_val_pairs,
                                      'en', 'de', 'German',
                                      max_len_in=args.max_len,
                                      max_len_out=args.max_len)
    sst2_val_enc = encode_sst2(tokenizer, sst2_val, max_len=args.max_len)
    cola_val_enc = encode_cola(tokenizer, cola_val, max_len=args.max_len)
    for k in ('input_ids', 'attention_mask', 'labels'):
        fr_train_enc[k] = fr_train_enc[k].to(device)
        fr_val_enc[k] = fr_val_enc[k].to(device)
        de_val_enc[k] = de_val_enc[k].to(device)
        sst2_val_enc[k] = sst2_val_enc[k].to(device)
        cola_val_enc[k] = cola_val_enc[k].to(device)

    fr_prompts = [
        "translate English to French: The cat sat on the mat.",
        "translate English to French: I love programming in Python.",
        "translate English to French: The weather is beautiful today.",
    ]
    de_prompts = [
        "translate English to German: The cat sat on the mat.",
        "translate English to German: I love programming in Python.",
        "translate English to German: The weather is beautiful today.",
    ]

    def run_eval(tag):
        fr_loss = eval_translation_loss(model, fr_val_enc, device,
                                          pad_id=pad_id)
        de_loss = eval_translation_loss(model, de_val_enc, device,
                                          pad_id=pad_id)
        sst2_acc = eval_sst2(model, tokenizer, sst2_val_enc, device)
        cola_acc = eval_cola(model, tokenizer, cola_val_enc, device)
        print(f"  [{tag}]  FR_loss={fr_loss:.3f}  "
              f"DE_loss={de_loss:.3f}  "
              f"SST2={sst2_acc*100:.2f}%  "
              f"CoLA={cola_acc*100:.2f}%", flush=True)
        return fr_loss, de_loss, sst2_acc, cola_acc

    print("\n=== PRE-ablation eval ===", flush=True)
    pre = run_eval("PRE")
    print("  FR samples (PRE):", flush=True)
    for s in sample_translations(model, tokenizer, fr_prompts, device):
        print(f"    {s!r}", flush=True)
    print("  DE samples (PRE):", flush=True)
    for s in sample_translations(model, tokenizer, de_prompts, device):
        print(f"    {s!r}", flush=True)

    print(f"\n=== Training {args.steps} steps on EN->FR pairs ===",
          flush=True)
    n_train = fr_train_enc['input_ids'].size(0)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randperm(n_train, device=device)[:args.bsz]
        inp = fr_train_enc['input_ids'][idx]
        mask = fr_train_enc['attention_mask'][idx]
        lbl = fr_train_enc['labels'][idx].clone()
        lbl[lbl == pad_id] = -100
        aux_opt.zero_grad(set_to_none=True)
        out = model(input_ids=inp, attention_mask=mask, labels=lbl)
        out.loss.backward()
        aux_opt.step()
        if (step + 1) % args.eval_every == 0:
            print(f"  step {step+1}/{args.steps}  "
                  f"tr_loss={out.loss.item():.3f}", flush=True)
    train_min = (time.time() - t0) / 60
    print(f"  Training: {train_min:.1f} min", flush=True)

    print("\n=== POST-ablation eval ===", flush=True)
    post = run_eval("POST")
    print("  FR samples (POST):", flush=True)
    for s in sample_translations(model, tokenizer, fr_prompts, device):
        print(f"    {s!r}", flush=True)
    print("  DE samples (POST):", flush=True)
    for s in sample_translations(model, tokenizer, de_prompts, device):
        print(f"    {s!r}", flush=True)

    print(f"\n=== Deltas ===", flush=True)
    print(f"  FR_loss: {pre[0]:.3f} -> {post[0]:.3f}  "
          f"(d={post[0]-pre[0]:+.3f})  [target: LARGE positive]",
          flush=True)
    print(f"  DE_loss: {pre[1]:.3f} -> {post[1]:.3f}  "
          f"(d={post[1]-pre[1]:+.3f})  [target: ~0]", flush=True)
    print(f"  SST2:    {pre[2]*100:.2f}% -> {post[2]*100:.2f}%  "
          f"(d={(post[2]-pre[2])*100:+.2f}%)  [target: ~0]", flush=True)
    print(f"  CoLA:    {pre[3]*100:.2f}% -> {post[3]*100:.2f}%  "
          f"(d={(post[3]-pre[3])*100:+.2f}%)  [target: ~0]", flush=True)


if __name__ == "__main__":
    main()
