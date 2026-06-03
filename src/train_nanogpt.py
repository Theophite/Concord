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
                                get_reb_stats, set_fixed_coh, set_gate_gain,
                                set_coh_weighted_v, set_ratio_coh,
                                set_ratio_coh_floors,
                                set_v_bias_correction, bias_correction_factor,
                                set_gap_feedback,
                                set_sigmag_noise, set_sigmag_sigma)
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
def _consolidated_buf(m, which="s"):
    """The CONSOLIDATED (trusted) weight -- drop the transient s_fast and deploy only
    the slow path. m_eff = s_slow*128 + s_fast + v_slow*128, so cutting s_fast leaves
    the denoised position. If a gate strands noise in s_fast, deploying this should
    beat the live m_eff. Variants:
      'sv'  -> (s_slow + v_slow)*128   : EXACTLY m_eff minus s_fast (the true slow sum).
      's2v' -> (s_slow + 2*v_slow)*128 : weight the long ANCHOR double (v_slow = leak
               rate ~0.001 => ~1000-step EMA, the most denoised term).
      's'   -> 2*s_slow*128            : 2x the medium position (legacy; v_slow~=s_slow).
      'v'   -> 2*v_slow*128            : 2x the long anchor    (legacy).
    Returns a bf16 buffer to swap into the forward's _bf16_weight_buf."""
    pw = m.packed_w
    s_slow = ((pw << 16) >> 24).to(torch.float32)     # bits 15:8 s_slow_i8
    v_slow = ((pw << 24) >> 24).to(torch.float32)     # bits 7:0  v_slow_i8
    if   which == "sv":  slow = s_slow + v_slow
    elif which == "s2v": slow = s_slow + 2.0 * v_slow
    elif which == "v":   slow = 2.0 * v_slow
    else:                slow = 2.0 * s_slow          # "s"
    exp = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - m.MANTISSA_BIAS)
    scale = torch.exp2(exp)
    return (slow * 128.0 * scale).to(m._bf16_weight_buf.dtype)


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
    ap.add_argument("--beta1", type=float, default=0.0,
                    help="FAST-accumulator momentum: reinforces s_fast by beta1*velocity "
                         "(d_fs) each step (heavy-ball over the v-hat-RMS step). Net s_fast "
                         "decay = alpha-beta1, so keep 0<=beta1<alpha (=%s); beta1>=alpha "
                         "diverges. 0 = today's recipe." % "alpha")
    ap.add_argument("--gap_feedback", action="store_true",
                    help="Conserved pass<->evaporate split gated by the gap MAGNITUDE: "
                         "pass fraction c=min(1, coh+exp(-|d_sv|/gap_scale)); the (1-c) "
                         "fraction evaporates. Strips the constant chase floor (floor "
                         "survives only in the gap->0 ignition limit). Replaces the "
                         "ratio-coh floor + gf_consol with one rate alpha.")
    ap.add_argument("--gap_scale", type=float, default=500.0,
                    help="gap-feedback scale in MANTISSA units (|d_sv|); g=exp(-|d_sv|/scale).")
    ap.add_argument("--aux_lr", type=float, default=1e-3)
    ap.add_argument("--adamw_lr", type=float, default=1e-3)
    ap.add_argument("--factored_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--lr_min_frac", type=float, default=0.1)
    ap.add_argument("--sigmag", type=float, default=0.0,
                    help="EXPERIMENTAL: peak centered-Sigma_g gradient-noise magnitude "
                         "(units of ||grad_W||). Scheduled rising-late: sigma = sigmag*(1-lr/"
                         "lr_peak). 0=off. Pair with a raised --lr_min_frac + deploy off S+V.")
    ap.add_argument("--sigmag_iso", action="store_true",
                    help="ABLATION: isotropic white noise instead of Sigma_g-shaped.")
    ap.add_argument("--sigmag_const", action="store_true",
                    help="ABLATION: constant sigma (=sigmag) instead of rising-late.")
    ap.add_argument("--cuda_graph", action="store_true",
                    help="capture per-step fwd+loss+bwd into ONE CUDA graph + replay (concord "
                         "only) to cut launch overhead. lr + sigmag sigma are device tensors "
                         "(rising-late noise survives capture). aux step + rebalance eager.")
    ap.add_argument("--rebalance_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=1234,
                    help="dedicated RNG seed for batch sampling, DECOUPLED from "
                         "model/optimizer RNG -> identical batch order across "
                         "optimizers (the comparability fix).")
    ap.add_argument("--eval_consolidated", action="store_true",
                    help="also eval the CONSOLIDATED weight (2*s_slow, drop s_fast = "
                         "Lookahead slow weights) on the same batches -> consV. Tests "
                         "'deploy the trusted weight, not the s_fast-polluted live one'.")
    ap.add_argument("--const_lr", action="store_true",
                    help="hold lr at peak (warmup then constant; no cosine decay).")
    ap.add_argument("--gate_cosine", action="store_true",
                    help="route the cosine schedule onto the commitment GATE (gate_gain) "
                         "instead of the lr. Use with --const_lr + --coh_gate.")
    ap.add_argument("--watch_dw", action="store_true",
                    help="log median per-step |dW| (the adjusted-gradient magnitude) "
                         "of the deployed weight, per eval.")
    ap.add_argument("--watch_accum", action="store_true",
                    help="end-of-run readout of where the live weight mass sits across "
                         "the cascade: mean|s_fast| / |s_slow|*128 / |v_slow|*128 and the "
                         "per-element s_fast share. Tests whether a gate that refuses to "
                         "chase noise into s_slow merely STRANDS it in s_fast (still in "
                         "m_eff = s_slow*128 + s_fast + v_slow*128, so still in the weight).")
    ap.add_argument("--coh_gate", action="store_true",
                    help="engage the FIXED coherence gate (Wiener coh=S/(S+noise^2) "
                         "via enable_cohpre + set_fixed_coh): SNR-gated commitment, "
                         "freeze the incoherent/stuck coords. Use with gf_trust=1.")
    ap.add_argument("--coh_weighted_v", action="store_true",
                    help="EXPERIMENTAL: weight the rank-1 variance accumulation "
                         "(v_row/v_col) by coh_pre so v-hat fits COHERENT gradient "
                         "power only. Requires --coh_gate (needs coh_pre).")
    ap.add_argument("--ratio_coh", action="store_true",
                    help="EXPERIMENTAL: gate BOTH chase and v_slow leak by live coh "
                         "and DROP coh_pre -- the s_fast:s_slow:v_slow ratio carries "
                         "established coherence. 32 bits/param (no fp32 coh_pre buffer).")
    ap.add_argument("--ratio_chase_floor", type=float, default=0.9,
                    help="ratio-coh fast->slow gate floor START (1=normal chase, "
                         "0=coh-gated); cosine-decays to 0 over ratio_coh_floor_epochs.")
    ap.add_argument("--ratio_leak_floor", type=float, default=0.999,
                    help="ratio-coh slow->v_slow leak floor START (1=normal leak rate, "
                         "0=coh-gated). Start at the normal leak, move to noise-gating it.")
    ap.add_argument("--ratio_coh_floor_epochs", type=float, default=1.0,
                    help="cosine-decay both ratio-coh floors to 0 over this many epochs.")
    ap.add_argument("--ratio_chase_floor_min", type=float, default=0.0,
                    help="PERMANENT floor the chase decays TO (>0 keeps minimum "
                         "consolidation -> bounds s_fast stranding).")
    ap.add_argument("--ratio_leak_floor_min", type=float, default=0.0,
                    help="PERMANENT floor the leak decays TO. 1.0 = drift-leak (leak "
                         "coh-INDEPENDENT, pure alpha_v_fast*d_sv -> v_slow tracks).")
    ap.add_argument("--bias_correct", action="store_true",
                    help="(1) Adam bias-correction of v_hat: multiply by 1/(1-beta2^t) "
                         "(beta2=0.999) each step. Tames the cold-start; ported from "
                         "concord_ratio_coh. Off = legacy (uncorrected) behaviour.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (Concord layers stay opaque custom "
                         "autograd Functions; inductor fuses + cudagraphs the rest). "
                         "Requires torch/Triton aligned (torch 2.5.1 <-> Triton 3.1).")
    ap.add_argument("--compile_mode", default="reduce-overhead",
                    help="'reduce-overhead' (cudagraphs; best for this launch-bound model) "
                         "or 'default' (fusion only).")
    ap.add_argument("--fast_gain_anneal", action="store_true",
                    help="anneal s_fast share of the FORWARD weight gamma:1->0 over "
                         "training; the loss drives signal into the slow path and the "
                         "deployed weight becomes s_slow*128 + v_slow*128 (drop s_fast).")
    ap.add_argument("--fast_gain_frac", type=float, default=1.0,
                    help="fraction of max_iters over which gamma goes 1->fast_gain_final.")
    ap.add_argument("--fast_gain_final", type=float, default=0.0,
                    help="target gamma at end of anneal (0.0 = fully drop s_fast).")
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
        if args.beta1 != 0.0:
            for m in layers:
                m.beta1 = args.beta1   # fast-accumulator momentum (reinforce s_fast)
            print(f"[{tag}] FAST-accum momentum beta1={args.beta1} (COHERENCE-GATED: "
                  f"reinforces beta1*coh*velocity; per-coord net s_fast decay = "
                  f"alpha - beta1*coh, so beta1>=alpha={args.alpha} diverges on coh~1 "
                  f"coords)", flush=True)
        if args.tick_down:
            for m in layers:
                m.allow_tickdown = True
        if args.coh_gate:
            set_fixed_coh(True)             # Wiener coh = S/(S+noise^2)
            for m in layers:
                m.enable_cohpre()           # engage coh_pre-gated commitment
            print(f"[{tag}] FIXED coherence gate ENGAGED (Wiener SNR-gated "
                  f"commitment) on {len(layers)} layers", flush=True)
        else:
            for m in layers:
                m.disable_cohpre()          # gate is default-ON now; ablate it off
        set_coh_weighted_v(args.coh_weighted_v)
        if args.coh_weighted_v:
            print(f"[{tag}] COH-WEIGHTED v-hat (normalized): rank-1 variance "
                  f"fits coherent power", flush=True)
        set_ratio_coh(args.ratio_coh)
        if args.ratio_coh:
            set_fixed_coh(True)
            for m in layers:
                m.disable_cohpre()      # ratio-coh: no coh_pre buffer (32 bits/param)
            print(f"[{tag}] RATIO-COH: chase {args.ratio_chase_floor}->0 / leak "
                  f"{args.ratio_leak_floor}->0 over ~{args.ratio_coh_floor_epochs} epoch, "
                  f"then coh-gated; coh_pre dropped (32 bits/param) on {len(layers)} "
                  f"layers", flush=True)
        set_gap_feedback(args.gap_feedback, args.gap_scale)
        if args.gap_feedback:
            print(f"[{tag}] GAP-FEEDBACK ON: conserved pass/evap, c=min(1,coh+exp(-|d_sv|/"
                  f"{args.gap_scale})); floor stripped except gap->0", flush=True)
        set_sigmag_noise(args.sigmag > 0.0, isotropic=args.sigmag_iso)
        if args.sigmag > 0.0:
            _shape = "ISOTROPIC" if args.sigmag_iso else "Sigma_g-shaped"
            _sched = "CONSTANT" if args.sigmag_const else "rising-late*(1-lr/lr_peak)"
            print(f"[{tag}] NOISE ON: {_shape}, {_sched}, sigma_peak={args.sigmag}, "
                  f"lr_min_frac={args.lr_min_frac} -- deploy off S+V (sv)", flush=True)
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

    # ratio-coh bootstrap floors: cosine-decay chase (0.9) / leak (0.999) -> 0 over
    # ~one epoch, i.e. start at the normal (ungated) chase+leak and move to pure
    # coherence-gating once the s_fast:s_slow:v_slow ratio carries the memory.
    floor_horizon = max(1, int(args.ratio_coh_floor_epochs
                               * (len(train) // (args.bsz * args.block_size))))

    def cos_floor(start, it, end=0.0):
        if it >= floor_horizon:
            return end
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * it / floor_horizon))

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
    _CONS_VARIANTS = ("sv", "s2v", "v", "s")   # deployed-weight candidates (drop s_fast)
    best_cons = {k: 1e9 for k in _CONS_VARIANTS}
    prev_W = None; prev_it = 0
    # ---- CUDA graph capture (concord only): ONE graph over fwd+loss+bwd(+rebalance),
    # replayed. Validated bit-exact (research_notebook probe_loss.py). Captured on the FIRST
    # iter (NO eager pre-roll on the default stream -- that re-trips the legacy-stream error;
    # mirror probe_loss). Per-step scalars the captured backward reads are DEVICE TENSORS
    # updated outside the graph: lr (_lr_buf) + sigmag sigma (_get_sigmag_sigma), so the
    # rising-late noise schedule rides replays with no recapture. The Sigma_g noise matmul
    # lives in the backward -> it is captured automatically. aux_opt.step() stays eager.
    _gst = {"on": bool(getattr(args, "cuda_graph", False) and args.mode == "concord" and layers),
            "cap": False, "g": None, "loss": None, "sx": None, "sy": None}
    _aux_params = [p for p in model.parameters() if p.requires_grad]
    if _gst["on"]:
        _gst["sx"] = torch.zeros(args.bsz, args.block_size, dtype=torch.long, device=device)
        _gst["sy"] = torch.zeros(args.bsz, args.block_size, dtype=torch.long, device=device)

    def _graph_step():
        # captured region = fwd + loss + bwd ONLY (Concord step fused in bwd; the Sigma_g
        # noise matmul rides along). rebalance() stays EAGER after replay -- matches the
        # bit-exact probe_loss structure (rebalance reads _row_max_buf the captured bwd wrote).
        for p in _aux_params:
            if p.grad is not None:
                p.grad = None
        _, gl = model(_gst["sx"], _gst["sy"])
        gl.backward()
        return gl

    if args.compile:
        # Windows torch.compile cache bug: inductor's atomic write uses os.rename, which
        # ERRORS on Windows if the destination exists (POSIX rename overwrites) -> swap in
        # os.replace (cross-platform atomic-overwrite).
        import os as _os
        _os.rename = _os.replace
        # Compile AFTER layers/aux are collected (they reference the real Concord modules,
        # so lr/rebalance still work). Concord's custom autograd Function stays opaque;
        # inductor fuses + (reduce-overhead) cudagraphs the rest (attention/LN/embeddings).
        model = torch.compile(model, mode=args.compile_mode)
        print(f"[{tag}] torch.compile mode={args.compile_mode} (first iters pay compile cost)",
              flush=True)
    for it in range(args.max_iters):
        if args.fast_gain_anneal and args.mode == "concord" and layers:
            _fh = max(1, int(args.fast_gain_frac * args.max_iters))
            _fg = args.fast_gain_final if it >= _fh else (
                args.fast_gain_final + (1.0 - args.fast_gain_final)
                * 0.5 * (1.0 + math.cos(math.pi * it / _fh)))
            for _m in layers:
                _m.fast_gain = _fg
        f = lr_at(it) / peak_lr            # schedule factor in [min_frac, 1] (= lr/lr_peak)
        warm = min(1.0, (it + 1) / args.warmup_iters) if args.warmup_iters > 0 else 1.0
        lr = peak_lr * (warm if args.const_lr else f)
        if args.mode == "concord" and args.sigmag > 0.0:
            # rising-late sigma = sigmag_peak * (1 - lr/lr_peak): ~0 early, max late
            # (the doc's only-helpful schedule). --sigmag_const ablates to constant.
            set_sigmag_sigma(args.sigmag if args.sigmag_const else args.sigmag * (1.0 - f))
        if args.mode == "concord" and args.gate_cosine:
            set_gate_gain(f)               # route the cosine onto the commitment gate
        if args.mode == "concord" and args.ratio_coh:
            set_ratio_coh_floors(cos_floor(args.ratio_chase_floor, it, args.ratio_chase_floor_min),
                                 cos_floor(args.ratio_leak_floor, it, args.ratio_leak_floor_min))
        if args.mode == "concord" and args.bias_correct:
            set_v_bias_correction(bias_correction_factor(it))   # (1) 1/(1-beta2^t)
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
            gstate = eval_gen.get_state() if args.eval_consolidated else None
            L = estimate_loss(model, train, val, args.bsz, args.block_size,
                              device, args.eval_iters, gen=eval_gen)
            best_val = min(best_val, L['val'])
            expstr = ""
            if args.eval_consolidated and args.mode == "concord" and layers:
                saved = [m._bf16_weight_buf for m in layers]
                Lcons = {}
                for which in _CONS_VARIANTS:        # (s+v), (s+2v), 2v, 2s -- all drop s_fast
                    eval_gen.set_state(gstate)      # identical batches every variant
                    for m in layers:
                        m._bf16_weight_buf = _consolidated_buf(m, which)
                    Lcons[which] = estimate_loss(model, train, val, args.bsz,
                                   args.block_size, device, args.eval_iters, gen=eval_gen)
                    best_cons[which] = min(best_cons[which], Lcons[which]['val'])
                for m, s in zip(layers, saved):
                    m._bf16_weight_buf = s
                expstr += "  " + " ".join(f"{w}:V={Lcons[w]['val']:.4f}"
                                          for w in _CONS_VARIANTS)
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
            if args.watch_dw and args.mode == "concord" and layers:
                curW = torch.cat([m.weight.detach().flatten().float() for m in layers])
                if prev_W is not None and it > prev_it:
                    dwm = ((curW - prev_W).abs().median() / (it - prev_it)).item()
                    expstr += f"  dW/step={dwm:.2e}"
                prev_W = curW; prev_it = it
            print(f"[{tag}] iter {it:>5}/{args.max_iters}  lr={lr:.4f}  "
                  f"train {L['train']:.4f}  val {L['val']:.4f}  "
                  f"best_val {best_val:.4f}{expstr}  ({time.time()-t0:.0f}s)", flush=True)

        x, y = get_batch(train, args.bsz, args.block_size, device, generator=train_gen)
        if _gst["on"] and not _gst["cap"]:
            # FIRST iter: side-stream warmup of the FULL fwd+bwd (3x; settles autograd +
            # Triton autotune so capture has nothing on the legacy stream), then record ONE
            # graph. NO eager pre-roll on the default stream. Warmup+capture = ~4 real steps
            # on batch 0 (one-time blip).
            _gst["sx"].copy_(x); _gst["sy"].copy_(y)
            _s = torch.cuda.Stream()
            _s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_s):
                for _ in range(3):
                    _graph_step()
            torch.cuda.current_stream().wait_stream(_s)
            _cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_cg):
                _gst["loss"] = _graph_step()
            _gst["g"] = _cg; _gst["cap"] = True
            aux_opt.step()
            if (it + 1) % args.rebalance_every == 0:
                for m in layers:
                    m.rebalance()
            loss = _gst["loss"]
            print(f"[{tag}] CUDA graph captured at iter {it}", flush=True)
        elif _gst["on"]:
            _gst["sx"].copy_(x); _gst["sy"].copy_(y)
            _gst["g"].replay()       # fwd+loss+bwd (Concord step + Sigma_g noise, fused)
            aux_opt.step()           # eager, outside the graph
            if (it + 1) % args.rebalance_every == 0:
                for m in layers:
                    m.rebalance()
            loss = _gst["loss"]
        else:
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
    if args.eval_consolidated and args.mode == "concord" and layers:
        print(f"[{tag}] BEST val by deployed weight:  live(m_eff)={best_val:.4f}  "
              + "  ".join(f"{w}={best_cons[w]:.4f}" for w in _CONS_VARIANTS), flush=True)

    if args.watch_accum and args.mode == "concord" and layers:
        # Where does the mass live at convergence? s_fast is part of m_eff, so noise the
        # gate keeps OUT of s_slow but leaves IN s_fast is still in the deployed weight.
        sf = ss = vs = sf_sh = 0.0
        for m in layers:
            pw = m.packed_w
            a_f = (pw >> 16).to(torch.float32).abs()
            a_s = ((pw << 16) >> 24).to(torch.float32).abs() * 128.0
            a_v = ((pw << 24) >> 24).to(torch.float32).abs() * 128.0
            sf += a_f.mean().item(); ss += a_s.mean().item(); vs += a_v.mean().item()
            sf_sh += (a_f / (a_f + a_s + a_v + 1e-9)).mean().item()   # scale-invariant
        n = len(layers); sf /= n; ss /= n; vs /= n; sf_sh /= n
        tot = sf + ss + vs + 1e-9
        print(f"\n[{tag}] WATCH accum ({n} layers, mean|.| in m_eff mantissa units): "
              f"s_fast={sf:.2f} ({100*sf/tot:.1f}%)  s_slow*128={ss:.2f} ({100*ss/tot:.1f}%)  "
              f"v_slow*128={vs:.2f} ({100*vs/tot:.1f}%)  |  per-elem s_fast share="
              f"{100*sf_sh:.1f}%", flush=True)

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
