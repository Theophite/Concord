"""Momentum comparison: AdamW-style β1 vs Concord-style α/β1.

Question: in earlier runs Concord with β1>0 hurt SST-2 accuracy. AdamW
with β1=0.9 doesn't hurt. Why?

The mechanism difference:
  SGD-momentum:    v = μ·v + grad;  W ← W - lr·v
  AdamW (no β2):   m = β1·m + (1-β1)·grad;  m_hat = m/(1-β1^t); W ← W - lr·m_hat
  Concord:         s_fast = (1-α)·s_fast - lr·grad·scale_inv (after chase)
                   plus β1 damp:  s_fast ← (1-β1)·s_fast each tick
                   m_eff = s_fast + s_slow (forward weight = m_eff · scale_fwd)

The chase preserves m_eff = s_fast + s_slow per step, so the chase itself
doesn't add momentum at the W level. The Concord β1 term subtracts β1·s_fast
from m_eff each step — that's the actual "momentum effect" in Concord.

Configs (all bsz=128, 3 epochs, wd=0.001):
  A.  sgd_mom0          lr=0.08    SGD no momentum (baseline)
  B.  sgd_mom09         lr=0.08    SGD momentum=0.9 at high lr
  C.  sgd_mom09_lo      lr=0.008   SGD momentum=0.9 at low lr (effective ~SGD)
  D.  adamw_no_b2_b1_0  lr=0.08    AdamW(β1=0, β2=∞) (~SGD-with-bias-corr)
  E.  adamw_no_b2_b1_09 lr=0.008   AdamW(β1=0.9, β2=∞) — first-moment EMA
  F.  concord_a1_b1_0   lr=0.08    Concord α=1.0 β1=0 (no momentum)
  G.  concord_a01_b1_0  lr=0.08    Concord α=0.1 β1=0 (chase residual only)
  H.  concord_a01_b1_01 lr=0.08    Concord α=0.1 β1=0.1 (chase + damp)
  I.  adamw_full        lr=1e-4    standard AdamW (β1=0.9, β2=0.999)
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


class AdamWNoBeta2(torch.optim.Optimizer):
    """AdamW with the variance term disabled.

    Update:
        m ← β1·m + (1-β1)·grad
        m_hat = m / (1 - β1^t)
        W ← W·(1 - lr·wd) - lr·m_hat
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=beta1, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            b1 = group['beta1']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 't' not in state:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                state['t'] += 1
                t = state['t']
                m = state['m']
                m.mul_(b1).add_(p.grad, alpha=1.0 - b1)
                bias_corr = 1.0 - b1 ** t
                if wd:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(m, alpha=-lr / bias_corr)
        return loss


def make_optimizer(model, spec):
    """spec: dict with 'name' and configs."""
    name = spec['name']
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(),
                                 lr=spec['lr'],
                                 momentum=spec.get('momentum', 0.0),
                                 weight_decay=spec.get('wd', 0.0))
    if name == 'adamw_no_b2':
        return AdamWNoBeta2(model.parameters(),
                              lr=spec['lr'],
                              beta1=spec.get('beta1', 0.9),
                              weight_decay=spec.get('wd', 0.0))
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(),
                                   lr=spec['lr'],
                                   betas=(spec.get('beta1', 0.9),
                                           spec.get('beta2', 0.999)),
                                   weight_decay=spec.get('wd', 0.0))
    raise ValueError(name)


def run_one(args, tag, optimizer_spec, concord_spec=None):
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    if concord_spec is not None:
        wrap_with_concord(model, device=device,
                            lr=concord_spec['lr'],
                            weight_decay=concord_spec['wd'],
                            init_mode='finetune', kind='two_acc',
                            alpha=concord_spec['alpha'],
                            beta1=concord_spec['beta1'])
        aux_opt = torch.optim.SGD([p for p in model.parameters()
                                      if p.requires_grad],
                                    lr=1e-4, momentum=0.0)
        opt = aux_opt
        print(f"[{tag}] Concord-2acc  lr={concord_spec['lr']} "
              f"alpha={concord_spec['alpha']} beta1={concord_spec['beta1']} "
              f"wd={concord_spec['wd']}", flush=True)
    else:
        opt = make_optimizer(model, optimizer_spec)
        spec_str = ' '.join(f"{k}={v}" for k, v in optimizer_spec.items()
                              if k != 'name')
        print(f"[{tag}] {optimizer_spec['name']}  {spec_str}", flush=True)

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
    print(f"[{tag}] PRE-FT  SST2={sst2_pre*100:.2f}%  "
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
                print(f"[{tag}] BLOWUP at ep {ep+1} step {i//args.bsz}",
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
        print(f"[{tag}] ep {ep+1}  tr_loss={running_loss/max(seen,1):.4f}  "
              f"SST2={sst2_acc*100:.2f}%  ({ep_dt:.1f}s)", flush=True)
    tot_min = (time.time() - t0) / 60
    if blew:
        print(f"[{tag}] BLEW UP", flush=True)
        return sst2_pre, cola_pre, float('nan'), float('nan'), tot_min, True
    sst2_post = eval_sst2(model, tokenizer, sst2_val_enc, device)
    cola_post = eval_cola(model, tokenizer, cola_val_enc, device)
    print(f"[{tag}] POST-FT  SST2={sst2_post*100:.2f}%  "
          f"CoLA={cola_post*100:.2f}%  "
          f"d SST2={(sst2_post-sst2_pre)*100:+.2f}%  "
          f"d CoLA={(cola_post-cola_pre)*100:+.2f}%  "
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
        print("CUDA not available")
        sys.exit(0)

    runs = [
        # A: SGD no momentum at high lr (matches +1.03%)
        ('A_sgd_mom0_lr0.08',
            {'name': 'sgd', 'lr': 0.08, 'momentum': 0.0, 'wd': args.wd},
            None),
        # B: SGD momentum 0.9 at high lr (likely too aggressive)
        ('B_sgd_mom0.9_lr0.08',
            {'name': 'sgd', 'lr': 0.08, 'momentum': 0.9, 'wd': args.wd},
            None),
        # C: SGD momentum 0.9 at 10x lower lr (matched effective lr)
        ('C_sgd_mom0.9_lr0.008',
            {'name': 'sgd', 'lr': 0.008, 'momentum': 0.9, 'wd': args.wd},
            None),
        # D: AdamW with β2 disabled, β1=0 — first-moment EMA at high lr
        ('D_adamwno2_b1_0_lr0.08',
            {'name': 'adamw_no_b2', 'lr': 0.08, 'beta1': 0.0, 'wd': args.wd},
            None),
        # E: AdamW with β2 disabled, β1=0.9 at 10x lower lr
        ('E_adamwno2_b1_0.9_lr0.008',
            {'name': 'adamw_no_b2', 'lr': 0.008, 'beta1': 0.9, 'wd': args.wd},
            None),
        # F: Concord 2acc α=1.0 β1=0 (no momentum, ~SGD)
        ('F_concord_a1_b1_0_lr0.08',
            None,
            {'lr': 0.08, 'alpha': 1.0, 'beta1': 0.0, 'wd': args.wd}),
        # G: Concord 2acc α=0.1 β1=0 (chase residual only)
        ('G_concord_a0.1_b1_0_lr0.08',
            None,
            {'lr': 0.08, 'alpha': 0.1, 'beta1': 0.0, 'wd': args.wd}),
        # H: Concord 2acc α=0.1 β1=0.1 (chase + damping)
        ('H_concord_a0.1_b1_0.1_lr0.08',
            None,
            {'lr': 0.08, 'alpha': 0.1, 'beta1': 0.1, 'wd': args.wd}),
        # I: Standard AdamW for reference
        ('I_adamw_lr1e-4',
            {'name': 'adamw', 'lr': 1e-4, 'beta1': 0.9, 'beta2': 0.999,
              'wd': 0.01},
            None),
    ]

    results = {}
    for tag, opt_spec, concord_spec in runs:
        results[tag] = run_one(args, tag, opt_spec, concord_spec)

    print("=" * 90)
    print("MOMENTUM COMPARISON SUMMARY (bsz=128, 3 epochs)")
    print("=" * 90)
    print(f"{'tag':<32} | {'SST2 pre':>9} | {'SST2 post':>10} | "
          f"{'dSST2':>7} | {'dCoLA':>7} | {'blew?':>6}")
    print("-" * 90)
    for tag, (sst2_pre, cola_pre, sst2_post, cola_post, tm, blew) \
            in results.items():
        if blew:
            sp = '   NaN'
            dsst = '   NaN'
            dcola = '   NaN'
        else:
            sp = f'{sst2_post*100:>9.2f}%'
            dsst = f'{(sst2_post-sst2_pre)*100:+6.2f}%'
            dcola = f'{(cola_post-cola_pre)*100:+6.2f}%'
        print(f"{tag:<32} | {sst2_pre*100:>8.2f}% | {sp} | "
              f"{dsst:>7} | {dcola:>7} | "
              f"{'YES' if blew else 'no':>6}")


if __name__ == "__main__":
    main()
