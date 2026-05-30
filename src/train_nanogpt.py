"""Raw Concord (packed-B) vs AdamW on nanoGPT, char-level.

Concord wraps every nn.Linear (attn + mlp projections + lm_head) with
ConcordLinearPackedB (eps=1 shipped recipe: SGD-chase + v_slow leak), the step
fused into backward. Aux AdamW handles the tiny non-Linear params (token/pos
embeddings + LayerNorm). Per-step rebalance (from-scratch -> weights move far).

This is the regime that matters: LM is non-realizable (next token is genuinely
stochastic) and data >> capacity is reachable, so v-hat is finally load-bearing
-- unlike clean CIFAR where memorization dominates and SGD>=Adam.

Run from repo root:
    python src/train_nanogpt.py --mode concord
    python src/train_nanogpt.py --mode adamw
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn

from nanogpt import GPT, GPTConfig, load_char_data, get_batch
from prototype_packed_b import ConcordLinearPackedB


def wrap_with_concord(model, device, lr, alpha=0.1, beta1=0.0,
                      weight_decay=0.0, eps=1.0, step_cap=10.0):
    """Replace every nn.Linear with ConcordLinearPackedB, loading the
    from-scratch random init into s_fast (load_weights -> live weight = init
    at step 0; the chase redistributes mantissa over the first steps)."""
    layers = []
    n_params = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                c = ConcordLinearPackedB(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=device, alpha=alpha, beta1=beta1, lr=lr)
                c.set_optimizer_kind('adamw', weight_decay=weight_decay,
                                     eps=eps, step_cap=step_cap)
                with torch.no_grad():
                    c.load_weights(child.weight.data.float())
                    if child.bias is not None:
                        c.bias.data.copy_(child.bias.data.to(torch.bfloat16))
                setattr(parent, name, c)
                layers.append(c)
                n_params += child.in_features * child.out_features
    return layers, n_params


@torch.no_grad()
def estimate_loss(model, train, val, bsz, block_size, device, eval_iters):
    model.eval()
    out = {}
    for split, data in (("train", train), ("val", val)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, bsz, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["concord", "adamw"], default="concord")
    ap.add_argument("--data", default="nanogpt_data/input.txt")
    ap.add_argument("--max_iters", type=int, default=3000)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--concord_lr", type=float, default=0.05)
    ap.add_argument("--concord_wd", type=float, default=0.0)
    ap.add_argument("--step_cap", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--aux_lr", type=float, default=1e-3)
    ap.add_argument("--adamw_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--lr_min_frac", type=float, default=0.1)
    ap.add_argument("--rebalance_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.mode
    device = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train, val, vocab, _ = load_char_data(args.data, device)
    cfg = GPTConfig(vocab_size=vocab, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"[{tag}] GPT {model.num_params()/1e6:.2f}M params  vocab={vocab}  "
          f"block={args.block_size}  bsz={args.bsz}", flush=True)

    if args.mode == "concord":
        layers, npacked = wrap_with_concord(
            model, device, lr=args.concord_lr, alpha=args.alpha,
            weight_decay=args.concord_wd, step_cap=args.step_cap)
        aux = [p for p in model.parameters() if p.requires_grad]
        aux_opt = torch.optim.AdamW(aux, lr=args.aux_lr, weight_decay=0.0)
        print(f"[{tag}] Concord on {len(layers)} Linears "
              f"({npacked/1e6:.2f}M packed)  aux AdamW {sum(p.numel() for p in aux)/1e6:.2f}M "
              f"(embed+LN)  concord_lr={args.concord_lr} wd={args.concord_wd} "
              f"step_cap={args.step_cap}  aux_lr={args.aux_lr}", flush=True)
        peak_lr = args.concord_lr
    else:
        layers = []
        aux_opt = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr,
                                    weight_decay=args.weight_decay,
                                    betas=(0.9, 0.95))
        print(f"[{tag}] AdamW over {model.num_params()/1e6:.2f}M  "
              f"lr={args.adamw_lr} wd={args.weight_decay}", flush=True)
        peak_lr = args.adamw_lr

    def lr_at(it):
        if it < args.warmup_iters:
            f = (it + 1) / args.warmup_iters
        else:
            p = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
            f = args.lr_min_frac + 0.5 * (1 - args.lr_min_frac) * (1 + math.cos(math.pi * p))
        return peak_lr * f

    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    best_val = 1e9
    for it in range(args.max_iters):
        lr = lr_at(it)
        if args.mode == "concord":
            for m in layers:
                m.lr = lr
            # aux follows the same cosine shape, scaled to aux_lr
            for g in aux_opt.param_groups:
                g['lr'] = args.aux_lr * (lr / peak_lr)
        else:
            for g in aux_opt.param_groups:
                g['lr'] = lr

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            L = estimate_loss(model, train, val, args.bsz, args.block_size,
                              device, args.eval_iters)
            best_val = min(best_val, L['val'])
            print(f"[{tag}] iter {it:>5}/{args.max_iters}  lr={lr:.4f}  "
                  f"train {L['train']:.4f}  val {L['val']:.4f}  "
                  f"best_val {best_val:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        x, y = get_batch(train, args.bsz, args.block_size, device)
        aux_opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()              # Concord layers step in backward
        aux_opt.step()
        if args.mode == "concord" and (it + 1) % args.rebalance_every == 0:
            for m in layers:
                m.rebalance()

    L = estimate_loss(model, train, val, args.bsz, args.block_size, device,
                      args.eval_iters)
    print(f"\n[{tag}] DONE {(time.time()-t0)/60:.1f} min  "
          f"final val {L['val']:.4f}  best val {best_val:.4f}  "
          f"peak_mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
