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
from prototype_packed_b import (ConcordLinearPackedB, reset_reb_stats,
                                get_reb_stats)
from optim_factored import FactoredAdam


@torch.no_grad()
def _vproxy_vhat(m):
    """Per-element (v_proxy, v_hat) from a layer's LIVE state -- watch only,
    nothing is fed back into the step. v_proxy = drift-cancelled velocity-noise^2
    (what eps<1 would whiten by); v_hat = Adafactor rank-1 row x col E[g^2]
    (what actually drives the step here). Both from stored accumulators / EMAs."""
    pw = m.packed_w
    s_fast = (pw >> 16).to(torch.float32)
    s_slow = ((pw << 16) >> 24).to(torch.float32)
    v_slow = ((pw << 24) >> 24).to(torch.float32)
    exp = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - m.MANTISSA_BIAS)
    scale_fwd = torch.exp2(exp)
    C = float(m.drift_cancel_C)
    noise_w = (s_fast - C * (s_slow - v_slow) * 128.0) * scale_fwd
    vproxy = noise_w * noise_w
    vr = m.v_row.float(); vc = m.v_col.float()
    vhat = (vr[:, None] * vc[None, :]) / (vr.sum() + 1e-30)
    return vproxy, vhat


@torch.no_grad()
def _vproxy_only(m):
    """Just the per-element v_proxy (velocity-noise^2) -- cheap, for per-step
    EMA accumulation (skips the v_hat outer product)."""
    pw = m.packed_w
    s_fast = (pw >> 16).to(torch.float32)
    s_slow = ((pw << 16) >> 24).to(torch.float32)
    v_slow = ((pw << 24) >> 24).to(torch.float32)
    exp = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - m.MANTISSA_BIAS)
    scale_fwd = torch.exp2(exp)
    C = float(m.drift_cancel_C)
    noise_w = (s_fast - C * (s_slow - v_slow) * 128.0) * scale_fwd
    return noise_w * noise_w


@torch.no_grad()
def _coh_diag(m):
    """Full per-element diagnostic from live state: v_proxy (incoherent-velocity^2),
    v_hat (Adafactor E[g^2]), coh (the gate's coherence = signal^2/E[g^2] in [0,1]),
    and |W|. Lets us identify which axis v_proxy tracks (coherence vs magnitude)
    and characterize the coherence gate on the known-working v-hat run."""
    pw = m.packed_w
    s_fast = (pw >> 16).to(torch.float32)
    s_slow = ((pw << 16) >> 24).to(torch.float32)
    v_slow = ((pw << 24) >> 24).to(torch.float32)
    exp = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - m.MANTISSA_BIAS)
    scale_fwd = torch.exp2(exp)
    C = float(m.drift_cancel_C); av = float(m.alpha_v_fast)
    d_sv = (s_slow - v_slow) * 128.0
    noise_w = (s_fast - C * d_sv) * scale_fwd
    vproxy = noise_w * noise_w
    vr = m.v_row.float(); vc = m.v_col.float()
    vhat = (vr[:, None] * vc[None, :]) / (vr.sum() + 1e-30)
    mean_grad_w = av * d_sv * scale_fwd                      # the gate's signal est
    coh = ((mean_grad_w * mean_grad_w) / (vhat + 1e-12)).clamp(0, 1)
    # dimensionally-correct SNR gate: signal=C*d_sv (the drift in the SAME
    # velocity decomposition as the noise), denom = signal^2 + v_proxy (both
    # weight-velocity^2 -> lr cancels -> true gradient-SNR, Kalman/Wiener form).
    sig_w = C * d_sv * scale_fwd
    coh_fixed = (sig_w * sig_w) / (sig_w * sig_w + vproxy + 1e-30)
    absW = (s_slow * 128.0 + s_fast + v_slow * 128.0).mul(scale_fwd).abs()
    return vproxy, vhat, coh, coh_fixed, absW


@torch.no_grad()
def _spearman(a, b, cap=20000):
    a = a.flatten().float(); b = b.flatten().float()
    n = a.numel()
    if n > cap:
        s = max(1, n // cap)
        a = a[::s]; b = b[::s]
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-30)
    rb = (rb - rb.mean()) / (rb.std() + 1e-30)
    return (ra * rb).mean().item()


def wrap_with_concord(model, device, lr, alpha=0.1, beta1=0.0,
                      weight_decay=0.0, eps=1.0, step_cap=10.0,
                      precond_p=0.5, gf_consol=0.0, v_scale=1.0,
                      gf_trust_delta_sq=0.0):
    """Replace every nn.Linear with ConcordLinearPackedB, loading the
    from-scratch random init into s_fast (load_weights -> live weight = init
    at step 0; the chase redistributes mantissa over the first steps).

    Rank-1 v-hat (factored-Adam) preconditioning: v_scale=0 (kill the
    velocity-noise v_proxy) + gf_trust_delta_sq=1 -> denom=(v_hat+eps)^0.5
    where v_hat = v_row(x)v_col Adafactor rank-1 E[g^2] (fp32, out+in floats
    per layer -- cheap). The packed-int analog of FactoredAdam."""
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
                c.precond_p = precond_p     # eps<1 + precond_p>0 engages the
                c.gf_consol = gf_consol     # drift-cancel v_proxy whitening
                c.v_scale = v_scale         # 0 = kill velocity-noise v_proxy
                c.gf_trust_delta_sq = gf_trust_delta_sq  # 1 = v_hat is the denom
                with torch.no_grad():
                    c.load_weights(child.weight.data.float())
                    if child.bias is not None:
                        c.bias.data.copy_(child.bias.data.to(torch.bfloat16))
                setattr(parent, name, c)
                layers.append(c)
                n_params += child.in_features * child.out_features
    return layers, n_params


@torch.no_grad()
def estimate_loss(model, train, val, bsz, block_size, device, eval_iters,
                  gen=None):
    model.eval()
    out = {}
    for split, data in (("train", train), ("val", val)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, bsz, block_size, device, generator=gen)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["concord", "adamw", "factored"],
                    default="concord")
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
    ap.add_argument("--eps", type=float, default=1.0,
                    help="Concord precond eps. 1.0 (shipped) -> v_proxy inert "
                         "(uniform chase). <1 engages the drift-cancel v_proxy "
                         "whitening (the noise filter).")
    ap.add_argument("--precond_p", type=float, default=0.5,
                    help="Padam precond power on v_proxy (0=SGD, 0.5=Adam-sqrt).")
    ap.add_argument("--gf_consol", type=float, default=0.0,
                    help="coherence-gated consolidation rate (garbage filter).")
    ap.add_argument("--v_scale", type=float, default=1.0,
                    help="scale on the velocity-noise v_proxy term. 0 kills it "
                         "(use with gf_trust_delta_sq=1 for pure rank-1 v_hat).")
    ap.add_argument("--gf_trust_delta_sq", type=float, default=0.0,
                    help="weight on rank-1 v_hat in the denom. 1 + v_scale=0 + "
                         "small eps => denom=(v_hat+eps)^0.5 = factored-Adam.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--aux_lr", type=float, default=1e-3)
    ap.add_argument("--adamw_lr", type=float, default=1e-3)
    ap.add_argument("--factored_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--lr_min_frac", type=float, default=0.1)
    ap.add_argument("--rebalance_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=1234,
                    help="dedicated RNG seed for batch sampling, DECOUPLED from "
                         "model/optimizer RNG -> identical batch order across "
                         "optimizers (the comparability fix).")
    ap.add_argument("--tick_down", action="store_true",
                    help="enable bidirectional rebalance (median-gated tick-down "
                         "to reclaim exponent precision). Default off (1.17 recipe).")
    ap.add_argument("--watch_coh", action="store_true",
                    help="characterize v_proxy's axis (corr vs coherence/v_hat/|W|) "
                         "+ measure the coherence gate (coh distribution, coh vs "
                         "v_hat) on the run. Watch only.")
    ap.add_argument("--watch_vproxy_ema", action="store_true",
                    help="accumulate a per-step EMA of v_proxy (beta=0.999, v-hat's "
                         "horizon) and log rho(EMA, v_hat) vs rho(instant, v_hat) -- "
                         "tests if the noise is zero-mean (EMA sharpens) or structural.")
    ap.add_argument("--watch_vproxy", action="store_true",
                    help="WATCH ONLY (not used in the step): log the rank "
                         "correlation between v_proxy (velocity-noise^2) and the "
                         "Adafactor row/col v_hat per eval, + a final binned view.")
    ap.add_argument("--reb_stats", action="store_true",
                    help="instrument rebalance: log mean exponent per eval + "
                         "cumulative tick-up/down + clip counts (the 'why').")
    ap.add_argument("--save_prefix", default=None,
                    help="if set, save {prefix}_{mode}.pt with per-Linear init "
                         "+ final effective weights for the where-the-gap-lives "
                         "analysis (AdamW .weight vs Concord materialized).")
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
            weight_decay=args.concord_wd, step_cap=args.step_cap,
            eps=args.eps, precond_p=args.precond_p, gf_consol=args.gf_consol,
            v_scale=args.v_scale, gf_trust_delta_sq=args.gf_trust_delta_sq)
        if args.tick_down:
            for m in layers:
                m.allow_tickdown = True
        aux = [p for p in model.parameters() if p.requires_grad]
        aux_opt = torch.optim.AdamW(aux, lr=args.aux_lr, weight_decay=0.0)
        print(f"[{tag}] Concord on {len(layers)} Linears "
              f"({npacked/1e6:.2f}M packed)  aux AdamW {sum(p.numel() for p in aux)/1e6:.2f}M "
              f"(embed+LN)  concord_lr={args.concord_lr} wd={args.concord_wd} "
              f"eps={args.eps} precond_p={args.precond_p} gf_consol={args.gf_consol} "
              f"step_cap={args.step_cap}  aux_lr={args.aux_lr}", flush=True)
        peak_lr = args.concord_lr
    elif args.mode == "factored":
        layers = []
        aux_opt = FactoredAdam(model.parameters(), lr=args.factored_lr,
                               weight_decay=args.weight_decay, betas=(0.9, 0.95))
        print(f"[{tag}] FactoredAdam (rank-1 v-hat) over "
              f"{model.num_params()/1e6:.2f}M  lr={args.factored_lr} "
              f"wd={args.weight_decay}", flush=True)
        peak_lr = args.factored_lr
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

    # --- comparability: dedicated batch-sampling RNG, decoupled from the
    # model/optimizer RNG (Concord's stochastic rounding draws RNG that AdamW
    # does not -> the default-generator batch sequences would drift apart after
    # step 1). A separate, identically-seeded generator guarantees AdamW and
    # Concord see the SAME batch order. eval uses its own gen so periodic evals
    # don't perturb the train sequence.
    train_gen = torch.Generator(device=device); train_gen.manual_seed(args.data_seed)
    eval_gen = torch.Generator(device=device); eval_gen.manual_seed(args.data_seed + 1)

    # capture per-Linear init (identical across modes given --seed): for Concord
    # the wrapped layer already holds the loaded init in its packed state.
    wlayers = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (nn.Linear, ConcordLinearPackedB))]
    init_w = {n: m.weight.detach().float().cpu().clone() for n, m in wlayers} \
        if args.save_prefix else None

    if args.reb_stats:
        reset_reb_stats()
    vp_emas = {}
    VP_BETA = 0.999
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
                              device, args.eval_iters, gen=eval_gen)
            best_val = min(best_val, L['val'])
            expstr = ""
            if args.reb_stats and args.mode == "concord" and layers:
                em = sum((m.row_exp.float().mean() + m.col_exp.float().mean()).item()
                         for m in layers) / len(layers)
                expstr = f"  exp~{em:+.2f}"
            if (args.watch_vproxy or args.watch_vproxy_ema) and \
                    args.mode == "concord" and layers and it > 0:
                ri, re_ = [], []
                for m in layers:
                    vp, vh = _vproxy_vhat(m)
                    ri.append(_spearman(vp, vh))
                    if args.watch_vproxy_ema and id(m) in vp_emas:
                        re_.append(_spearman(vp_emas[id(m)], vh))
                expstr += f"  rho_inst={sum(ri)/len(ri):+.3f}"
                if re_:
                    expstr += f"  rho_ema={sum(re_)/len(re_):+.3f}"
            print(f"[{tag}] iter {it:>5}/{args.max_iters}  lr={lr:.4f}  "
                  f"train {L['train']:.4f}  val {L['val']:.4f}  "
                  f"best_val {best_val:.4f}{expstr}  ({time.time()-t0:.0f}s)", flush=True)

        x, y = get_batch(train, args.bsz, args.block_size, device, generator=train_gen)
        aux_opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()              # Concord layers step in backward
        aux_opt.step()
        if args.mode == "concord" and (it + 1) % args.rebalance_every == 0:
            for m in layers:
                m.rebalance()
        if args.watch_vproxy_ema and args.mode == "concord":
            for m in layers:
                vp = _vproxy_only(m)
                k = id(m)
                if k not in vp_emas:
                    vp_emas[k] = vp
                else:
                    vp_emas[k].mul_(VP_BETA).add_(vp, alpha=1 - VP_BETA)

    L = estimate_loss(model, train, val, args.bsz, args.block_size, device,
                      args.eval_iters, gen=eval_gen)
    print(f"\n[{tag}] DONE {(time.time()-t0)/60:.1f} min  "
          f"final val {L['val']:.4f}  best val {best_val:.4f}  "
          f"peak_mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB", flush=True)

    if args.watch_vproxy and args.mode == "concord" and layers:
        mid = len(layers) // 2
        m = layers[mid]
        vp, vh = _vproxy_vhat(m)
        vpf, vhf = vp.flatten(), vh.flatten()
        rho = _spearman(vpf, vhf)
        q = torch.quantile(vhf.float(), torch.linspace(0, 1, 11, device=vhf.device))
        print(f"\n[{tag}] WATCH v_proxy(noise^2) vs v_hat(Adafactor E[g^2]) "
              f"-- layer {mid}, rank corr rho={rho:+.3f}")
        print(f"  {'v_hat decile':<13}{'med v_hat':>12}{'med v_proxy':>13}")
        for i in range(10):
            lo, hi = q[i], q[i + 1]
            sel = (vhf >= lo) & (vhf <= hi if i == 9 else vhf < hi)
            if sel.any():
                print(f"  {i+1:<13}{vhf[sel].median().item():>12.2e}"
                      f"{vpf[sel].median().item():>13.2e}", flush=True)
        rho_all = sum(_spearman(*_vproxy_vhat(m)) for m in layers) / len(layers)
        print(f"  mean rho across {len(layers)} layers = {rho_all:+.3f}  "
              f"(med v_hat={vhf.median():.2e}, med v_proxy={vpf.median():.2e})", flush=True)

    if args.watch_coh and args.mode == "concord" and layers:
        mean = lambda x: sum(x) / len(x)
        cm, ch, cl = [], [], []        # broken coh: mean, frac>0.5, frac<0.1
        fm, fh, fl = [], [], []        # fixed coh
        r_f_vh = []
        for m in layers:
            vp, vh, coh, cohf, aw = _coh_diag(m)
            cm.append(coh.mean().item()); ch.append((coh > 0.5).float().mean().item())
            cl.append((coh < 0.1).float().mean().item())
            fm.append(cohf.mean().item()); fh.append((cohf > 0.5).float().mean().item())
            fl.append((cohf < 0.1).float().mean().item())
            r_f_vh.append(_spearman(cohf, vh))
        print(f"\n[{tag}] COHERENCE GATE: broken (signal/v_hat) vs fixed (signal^2/(signal^2+v_proxy))")
        print(f"  BROKEN: mean={mean(cm):.4f}  frac>0.5={mean(ch)*100:.1f}%  frac<0.1={mean(cl)*100:.1f}%")
        print(f"  FIXED : mean={mean(fm):.4f}  frac>0.5={mean(fh)*100:.1f}%  frac<0.1={mean(fl)*100:.1f}%"
              f"  rho(fixed,v_hat)={mean(r_f_vh):+.3f}", flush=True)

    if args.reb_stats:
        s = get_reb_stats()
        if s and s['dim_total']:
            print(f"[{tag}] REB STATS: tick-up {s['tickup_dim']/s['dim_total']*100:.1f}%"
                  f" of row/col-dims, tick-down {s['tickdown_dim']/s['dim_total']*100:.1f}%, "
                  f"clip {s['clip_elems']/max(s['elem_total'],1)*100:.2f}% of elems "
                  f"(over {s['calls']} rebalance calls)", flush=True)

    if args.save_prefix:
        final_w = {n: m.weight.detach().float().cpu().clone() for n, m in wlayers}
        # aux params (embeddings + LayerNorm) = everything that's a live
        # Parameter (Concord Linears are packed buffers, not Parameters), so
        # the full fp32 model can be rebuilt at the solution for gradient probes.
        aux_final = {n: p.detach().float().cpu().clone()
                     for n, p in model.named_parameters()}
        path = f"{args.save_prefix}_{args.mode}.pt"
        torch.save({"mode": args.mode, "tag": tag, "seed": args.seed,
                    "data_seed": args.data_seed, "init": init_w,
                    "final": final_w, "aux_final": aux_final,
                    "final_val": L['val'], "best_val": best_val,
                    "peak_lr": peak_lr, "max_iters": args.max_iters}, path)
        print(f"[{tag}] saved init+final weights for {len(final_w)} Linears "
              f"+ {len(aux_final)} aux params -> {path}", flush=True)


if __name__ == "__main__":
    main()
