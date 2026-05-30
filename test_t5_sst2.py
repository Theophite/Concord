"""T5-small SST-2 fine-tune: Concord packed-B vs vanilla AdamW.

Measures:
  - Correctness: final SST-2 validation accuracy (target: within ~1% of AdamW)
  - Speed: wall-clock per epoch
  - Memory: peak GPU memory during training

Concord setup wraps every nn.Linear in T5 with ConcordLinearPackedB.
Pretrained weights are loaded into the three-accumulator state via
load_weights_finetune (steady-state init: v_slow ≈ s_slow ≈ live/2,
s_fast ≈ quantization residual), so the model behaves like the
pretrained T5 at step 0 — fine-tuning just nudges it from there.

LayerNorm + embeddings + bias stay fp32 and are handled by
torch.optim.SGD (we don't compare optimizer behavior on these tiny
params, just on the Linears).

Run:
    # Concord:
    python test_t5_sst2.py --mode concord
    # AdamW baseline:
    python test_t5_sst2.py --mode adamw
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

from prototype_packed_b import ConcordLinearPackedB
from prototype_packed_2acc import ConcordLinear2acc


def wrap_with_concord(model, device='cuda', alpha=0.1, beta1=0.0, lr=0.001,
                       weight_decay=0.01, init_mode='finetune',
                       kind='packed_b'):
    """Walk the model and replace every nn.Linear with ConcordLinearPackedB.

    init_mode controls how pretrained weights are loaded into the
    three-accumulator state:
      'finetune': v_slow_i8 ≈ live/2, s_slow_i8 ≈ live/2, s_fast = residual
                  (steady-state — model behaves like pretrained at step 0,
                   d_fs ≈ 0 and d_sv = 0 so no spurious updates)
      'fast':     v_slow_i8 = 0, s_slow_i8 = 0, s_fast = full mantissa
                  (the chase will redistribute over the first ~10 steps —
                   creates a transient where the optimizer first "discovers"
                   what the weight is supposed to be)

    Returns (list_of_concord_layers, n_wrapped, n_params_wrapped).
    """
    concord_layers = []
    n_wrapped = 0
    n_params = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                if kind == 'two_acc':
                    # 2-accumulator: int16 s_fast + int16 s_slow. Init
                    # always puts all mantissa in s_slow (load_weights).
                    concord = ConcordLinear2acc(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        device=device, alpha=alpha, beta1=beta1, lr=lr,
                        weight_decay=weight_decay, step_cap=10.0,
                    )
                    with torch.no_grad():
                        concord.load_weights(child.weight.data.float())
                        if child.bias is not None:
                            concord.bias.data.copy_(
                                child.bias.data.to(torch.bfloat16))
                else:
                    # packed_b: 3-accumulator (s_fast i16 + s_slow_i8 +
                    # v_slow_i8). AdamW-mode with eps=1.0 → SGD-chase +
                    # v_slow leak + Bayesian wd behavior (CIFAR recipe).
                    concord = ConcordLinearPackedB(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        device=device, alpha=alpha, beta1=beta1, lr=lr,
                    )
                    concord.set_optimizer_kind('adamw',
                                                weight_decay=weight_decay,
                                                eps=1.0,
                                                step_cap=10.0)
                    with torch.no_grad():
                        if init_mode == 'finetune':
                            concord.load_weights_finetune(
                                child.weight.data.float())
                        else:
                            concord.load_weights(child.weight.data.float())
                        if child.bias is not None:
                            concord.bias.data.copy_(
                                child.bias.data.to(torch.bfloat16))
                setattr(parent, child_name, concord)
                concord_layers.append(concord)
                n_wrapped += 1
                n_params += child.in_features * child.out_features
    return concord_layers, n_wrapped, n_params


def encode_sst2(tokenizer, examples, max_len=128):
    """SST-2 → T5 format: input 'sst2 sentence: <text>',
    target 'positive'/'negative'."""
    sentences = examples['sentence'] if isinstance(examples['sentence'], list) \
                else list(examples['sentence'])
    labels = examples['label'] if isinstance(examples['label'], list) \
              else list(examples['label'])
    inputs = [f"sst2 sentence: {s}" for s in sentences]
    targets = ['positive' if l == 1 else 'negative' for l in labels]
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
def eval_sst2(model, tokenizer, eval_enc, device, bsz=32):
    """Eval: greedy-decode the first target token, compare to ground truth."""
    model.eval()
    pos_id = tokenizer('positive', return_tensors='pt')['input_ids'][0, 0].item()
    neg_id = tokenizer('negative', return_tensors='pt')['input_ids'][0, 0].item()

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
        logits_first = out.logits[:, 0]   # [B, vocab]
        # Compare logit at pos_id vs neg_id to determine prediction.
        pred_pos = logits_first[:, pos_id] > logits_first[:, neg_id]
        true_pos = (lbl_first == pos_id)
        correct += (pred_pos == true_pos).sum().item()
        total += inp.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=['concord', 'adamw'], default='concord')
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                     help="LR. For AdamW: standard 1e-4. For Concord, "
                          "this seeds the chase lr — typical fine-tune "
                          "range is 0.01-0.1.")
    ap.add_argument("--concord_lr", type=float, default=0.05,
                     help="Concord chase lr (overrides --lr in concord mode).")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--max_examples", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--init_mode", choices=['finetune', 'fast'],
                     default='finetune',
                     help="Concord init mode (packed_b only): 'finetune' "
                          "puts mantissa into slow+very_slow (steady "
                          "state), 'fast' puts it all in s_fast.")
    ap.add_argument("--kind", choices=['packed_b', 'two_acc'],
                     default='packed_b',
                     help="Concord layer kind: packed_b (3-accumulator) "
                          "or two_acc (2-accumulator, both int16, "
                          "everything inited to s_slow).")
    ap.add_argument("--use_graph", action='store_true', default=False,
                     help="Capture the training step into a CUDA graph "
                          "and replay each microbatch. Static input "
                          "tensors are pre-allocated; eval uses eager "
                          "path. Disabled by default — enable with "
                          "--use_graph to opt in.")
    ap.add_argument("--cosine", action='store_true', default=False,
                     help="Cosine LR schedule on the Concord layers' lr "
                          "(from --concord_lr down to --concord_lr * "
                          "--cosine_min_frac). LR is updated per outer "
                          "step by writing to each layer's _lr_buf "
                          "device tensor; the captured graph reads "
                          "the fresh value on every replay.")
    ap.add_argument("--cosine_min_frac", type=float, default=0.01,
                     help="Cosine schedule floor as a fraction of "
                          "concord_lr. Default 0.01 (lr ends at 1%% of "
                          "starting lr).")
    ap.add_argument("--concord_wd", type=float, default=0.0,
                     help="weight_decay for Concord layers. Concord's "
                          "wd is NOT decoupled (gradient-additive): "
                          "per-step decay = concord_lr * wd * W. For "
                          "fine-tune at concord_lr=0.05, wd=0.01 gives "
                          "~40% decay/epoch, destroying pretrained "
                          "weights. Default 0 (no wd) for fine-tune.")
    args = ap.parse_args()

    tag = args.tag or args.mode
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[{tag}] Loading T5-small + tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    if args.mode == 'concord':
        concord_layers, n_wrapped, n_params = wrap_with_concord(
            model, device=device, lr=args.concord_lr,
            weight_decay=args.concord_wd,
            init_mode=args.init_mode,
            kind=args.kind)
        print(f"[{tag}] kind={args.kind}  init={args.init_mode}  "
              f"concord_wd={args.concord_wd}", flush=True)
        # Set per-layer LR (we update it externally via the property
        # setter, which writes through to the kernel's lr device tensor).
        for m in concord_layers:
            m.lr = args.concord_lr
        # Non-Concord params (embeddings, LayerNorm, biases): handled by
        # a small SGD optimizer.
        aux_params = [p for p in model.parameters() if p.requires_grad]
        aux_opt = torch.optim.SGD(aux_params, lr=args.lr, momentum=0.0)
        print(f"[{tag}] Wrapped {n_wrapped} Linear layers "
              f"({n_params/1e6:.2f}M packed params).  Aux params: "
              f"{sum(p.numel() for p in aux_params)/1e6:.2f}M (fp32).",
              flush=True)
        print(f"[{tag}] concord_lr={args.concord_lr}  aux_lr={args.lr}",
              flush=True)
    else:
        aux_opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=0.01)
        total_params = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
        print(f"[{tag}] AdamW over {total_params/1e6:.2f}M params  "
              f"lr={args.lr}", flush=True)

    print(f"[{tag}] Loading SST-2...", flush=True)
    ds = load_dataset('glue', 'sst2')
    train_data = ds['train']
    val_data = ds['validation']
    if args.max_examples > 0:
        train_data = train_data.select(
            range(min(args.max_examples, len(train_data))))

    print(f"[{tag}] Encoding...  train={len(train_data)}  val={len(val_data)}",
          flush=True)
    train_enc = encode_sst2(tokenizer, train_data, max_len=args.max_len)
    val_enc = encode_sst2(tokenizer, val_data, max_len=args.max_len)

    # Push to device.
    for k in ('input_ids', 'attention_mask', 'labels'):
        train_enc[k] = train_enc[k].to(device)
        val_enc[k] = val_enc[k].to(device)

    n_train = train_enc['input_ids'].size(0)
    n_steps = (n_train // args.bsz) * args.epochs
    print(f"[{tag}] Training: {args.epochs} epochs × "
          f"{n_train // args.bsz} steps  bsz={args.bsz}  "
          f"max_len={args.max_len}", flush=True)

    pad_id = tokenizer.pad_token_id

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Initial accuracy (sanity check that Concord wrapping preserves
    # pretrained behavior).
    val0 = eval_sst2(model, tokenizer, val_enc, device)
    print(f"[{tag}] Pretrained val_acc (no fine-tune): {val0*100:.2f}%",
          flush=True)

    # ──────────────────────────────────────────────────────────────
    # CUDA graph capture of the training step. Static input buffers;
    # per-microbatch we .copy_() the next batch into them and replay.
    # Eval uses the eager path (different shapes).
    # ──────────────────────────────────────────────────────────────
    use_graph = args.use_graph and torch.cuda.is_available()
    static_inp = torch.zeros(args.bsz, args.max_len,
                              dtype=torch.long, device=device)
    static_mask = torch.zeros(args.bsz, args.max_len,
                               dtype=torch.long, device=device)
    static_lbl = torch.full((args.bsz, 4), -100,
                              dtype=torch.long, device=device)
    static_loss = None
    g = None
    if use_graph:
        # Warmup on capture stream so all autograd nodes / .grad
        # buffers are bound to that stream — otherwise the captured
        # backward tries to sync with the legacy stream → capture fails.
        print(f"[{tag}] Warmup on capture stream (3 steps)...", flush=True)
        s_capture = torch.cuda.Stream()
        s_capture.wait_stream(torch.cuda.current_stream())
        model.train()
        # Seed static buffers with a real example so warmup doesn't see
        # all-zero labels (which would mask out the whole loss).
        idx = torch.arange(args.bsz, device=device)
        static_inp.copy_(train_enc['input_ids'][idx])
        static_mask.copy_(train_enc['attention_mask'][idx])
        lbl_seed = train_enc['labels'][idx].clone()
        lbl_seed[lbl_seed == pad_id] = -100
        static_lbl.copy_(lbl_seed)
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
        # Reset grads to fresh state, then capture.
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

    # Cosine LR schedule (Concord layers only): updates m.lr (and
    # therefore the device-side _lr_buf the kernel reads) per outer
    # step. The .fill_() lives outside the captured graph so each
    # replay picks up the fresh value.
    total_outer_steps = (n_train // args.bsz) * args.epochs
    def cosine_lr(step):
        import math as _m
        frac = step / max(total_outer_steps, 1)
        return (args.concord_lr * args.cosine_min_frac
                 + 0.5 * args.concord_lr * (1.0 - args.cosine_min_frac)
                 * (1.0 + _m.cos(_m.pi * frac)))

    if args.cosine and args.mode == 'concord':
        print(f"[{tag}] cosine LR: {args.concord_lr:.5f} -> "
              f"{args.concord_lr * args.cosine_min_frac:.5f} "
              f"over {total_outer_steps} outer steps", flush=True)

    step_counter = 0
    t_run = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_t = time.time()
        running_loss = 0.0
        seen = 0
        for i in range(0, n_train - args.bsz + 1, args.bsz):
            idx = perm[i:i+args.bsz]
            inp = train_enc['input_ids'][idx]
            mask = train_enc['attention_mask'][idx]
            lbl = train_enc['labels'][idx].clone()
            # Mask pad tokens out of loss.
            lbl[lbl == pad_id] = -100

            # Cosine LR update (Concord layers only — aux_opt LR is
            # fixed at args.lr to avoid messing up the embedding/LN
            # path that's already well-tuned at 1e-4).
            if args.cosine and args.mode == 'concord':
                cur_lr = cosine_lr(step_counter)
                for m in concord_layers:
                    m.lr = cur_lr   # property setter fills _lr_buf
            step_counter += 1

            if g is not None:
                # Graph fast path: copy into static buffers, replay.
                static_inp.copy_(inp, non_blocking=True)
                static_mask.copy_(mask, non_blocking=True)
                static_lbl.copy_(lbl, non_blocking=True)
                g.replay()
                loss_value = static_loss.detach()
                running_loss += loss_value.item() * inp.size(0)
                seen += inp.size(0)
                continue

            aux_opt.zero_grad(set_to_none=True)
            out = model(input_ids=inp, attention_mask=mask, labels=lbl)
            out.loss.backward()
            aux_opt.step()
            running_loss += out.loss.item() * inp.size(0)
            seen += inp.size(0)
        ep_dt = time.time() - ep_t
        val_acc = eval_sst2(model, tokenizer, val_enc, device)
        print(f"[{tag}] ep {ep+1}/{args.epochs}  "
              f"tr_loss={running_loss/max(seen,1):.4f}  "
              f"val_acc={val_acc*100:.2f}%  ({ep_dt:.1f}s)", flush=True)

    tot = time.time() - t_run
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    print()
    print(f"[{tag}] DONE  total {tot/60:.1f} min  avg {tot/args.epochs:.1f}s/ep")
    print(f"[{tag}] Peak GPU memory: {peak_mem:.1f} MB")


if __name__ == "__main__":
    main()
