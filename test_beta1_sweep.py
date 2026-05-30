"""β1 sweep with shift-on-saturation enabled.

Question: does Concord with β1<0 (velocity amplification) blow up at
lr=0.08 the way SGD-momentum=0.9 does — now that proper exponent
ticking removes my hypothesized 'int16-saturation = velocity clip'?

If yes → the chase/quantization gives no special W-space stability;
β1 is just a span from damping (>0) to standard SGD-momentum (<0).

If no → some other Concord-specific mechanism (SR noise? bf16 forward
deadband?) is preventing the velocity blow-up.

Plus SGD-mom reference for the known blow-up shape.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset

from test_t5_sst2 import wrap_with_concord, encode_sst2, eval_sst2
from test_t5_transfer import encode_cola, eval_cola


def run(args, tag, optimizer_kind, opt_kwargs, concord_kwargs=None):
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    if optimizer_kind == 'concord':
        wrap_with_concord(model, device=device,
                            lr=concord_kwargs['lr'],
                            weight_decay=concord_kwargs['wd'],
                            init_mode='finetune', kind='two_acc',
                            alpha=concord_kwargs['alpha'],
                            beta1=concord_kwargs['beta1'])
        opt = torch.optim.SGD([p for p in model.parameters()
                                  if p.requires_grad],
                                lr=1e-4, momentum=0.0)
        print(f"[{tag}] Concord-2acc  "
              f"lr={concord_kwargs['lr']} "
              f"alpha={concord_kwargs['alpha']} "
              f"beta1={concord_kwargs['beta1']} "
              f"wd={concord_kwargs['wd']}  (rebalance ON)", flush=True)
    elif optimizer_kind == 'sgd':
        opt = torch.optim.SGD(model.parameters(), **opt_kwargs)
        print(f"[{tag}] SGD {opt_kwargs}", flush=True)
    else:
        raise ValueError(optimizer_kind)

    sst2 = load_dataset('glue', 'sst2')
    cola_val = load_dataset('glue', 'cola')['validation']
    train_data = sst2['train']
    sst2_val = sst2['validation']
    sst2_train_enc = encode_sst2(tokenizer, train_data, max_len=args.max_len)
    sst2_val_enc = encode_sst2(tokenizer, sst2_val, max_len=args.max_len)
    cola_val_enc = encode_cola(tokenizer, cola_val, max_len=args.max_len)
    for k in ('input_ids', 'attention_mask', 'labels'):
        sst2_train_enc[k] = sst2_train_enc[k].to(device)
        sst2_val_enc[k] = sst2_val_enc[k].to(device)
        cola_val_enc[k] = cola_val_enc[k].to(device)

    pad_id = tokenizer.pad_token_id
    sst2_pre = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_pre = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{tag}] PRE  SST2={sst2_pre*100:.2f}%  "
          f"CoLA={cola_pre*100:.2f}%", flush=True)

    n_train = sst2_train_enc['input_ids'].size(0)
    model.train()
    t0 = time.time()
    blew = False
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
                blew = True
                print(f"[{tag}] BLOWUP at ep{ep+1} step {i//args.bsz}",
                      flush=True)
                break
            loss.backward()
            opt.step()
            running_loss += loss.item() * inp.size(0)
            seen += inp.size(0)
        if blew:
            break
        ep_dt = time.time() - ep_t
        sst2_acc = eval_sst2(model, tokenizer, sst2_val_enc, device)
        print(f"[{tag}] ep{ep+1}  tr_loss={running_loss/max(seen,1):.4f}  "
              f"SST2={sst2_acc*100:.2f}%  ({ep_dt:.0f}s)", flush=True)
    tot_min = (time.time() - t0) / 60
    if blew:
        return sst2_pre, cola_pre, float('nan'), float('nan'), tot_min, True
    sst2_post = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_post = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{tag}] POST SST2={sst2_post*100:.2f}%  "
          f"CoLA={cola_post*100:.2f}%  "
          f"dSST2={(sst2_post-sst2_pre)*100:+.2f}%  "
          f"dCoLA={(cola_post-cola_pre)*100:+.2f}%  "
          f"({tot_min:.1f} min)", flush=True)
    print(flush=True)
    return sst2_pre, cola_pre, sst2_post, cola_post, tot_min, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit(0)

    runs = []
    # Reference: known SGD-momentum blowup
    runs.append(('R_sgd_mom0.9_lr0.08', 'sgd',
                  {'lr': 0.08, 'momentum': 0.9, 'weight_decay': args.wd},
                  None))
    # β1 sweep on Concord α=0.1 at lr=0.08
    for b1 in [-0.20, -0.10, -0.05, 0.0, 0.10]:
        tag = f'C_a0.1_b1_{b1:+.2f}_lr0.08'
        runs.append((tag, 'concord', None,
                      {'lr': 0.08, 'alpha': 0.1, 'beta1': b1,
                        'wd': args.wd}))

    results = {}
    for tag, kind, opt_kwargs, concord_kwargs in runs:
        results[tag] = run(args, tag, kind, opt_kwargs, concord_kwargs)

    print("=" * 80)
    print(f"BETA1 SWEEP (lr=0.08, alpha=0.1, rebalance ON)")
    print("=" * 80)
    print(f"{'tag':<28} | {'dSST2':>8} | {'dCoLA':>8} | {'blew?':>5}")
    print("-" * 80)
    for tag, (sst2_pre, cola_pre, sst2_post, cola_post, tm, blew) \
            in results.items():
        if blew:
            print(f"{tag:<28} |   BLEW UP")
        else:
            print(f"{tag:<28} | {(sst2_post-sst2_pre)*100:+7.2f}% | "
                  f"{(cola_post-cola_pre)*100:+7.2f}% | "
                  f"{'no':>5}")


if __name__ == "__main__":
    main()
