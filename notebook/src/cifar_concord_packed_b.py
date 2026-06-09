"""CIFAR-10 training with packed-B layers + CUDA graph capture.

Same model as before (WiderConvNet, 4 conv + 3 fc + BN, ~3.2M params)
with packed-B int32 storage. The per-microbatch training step (zero_grad
+ forward + loss + backward + aux_opt step) is captured into a CUDA
graph and replayed each microbatch. Eliminates the autograd / dispatch
overhead that dominated the wall-clock at bsz=16.

LR is read by the kernels from device-side tensors (`m._lr_buf`) — we
update those between replays via the property setter `m.lr = X`, which
.fill_()'s the buffer outside the captured graph.

Run:
    python cifar_concord_packed_b.py --epochs 80 --batch_size 16
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_in_memory import get_loaders_in_memory
from prototype_packed_b import (ConcordLinearPackedB, ConcordConv2dPackedB)


class WiderConvNet(nn.Module):
    """~3.2M-param CIFAR conv net with BatchNorm, using packed-B layers
    for conv + fc weights. BN params stay as standard nn.Parameter
    (fp32, small)."""

    def __init__(self, device='cuda'):
        super().__init__()
        self.conv1 = ConcordConv2dPackedB(3, 64, 3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(64).to(device)
        self.conv2 = ConcordConv2dPackedB(64, 128, 3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(128).to(device)
        self.conv3 = ConcordConv2dPackedB(128, 256, 3, padding=1, device=device)
        self.bn3 = nn.BatchNorm2d(256).to(device)
        self.conv4 = ConcordConv2dPackedB(256, 256, 3, padding=1, device=device)
        self.bn4 = nn.BatchNorm2d(256).to(device)
        self.fc1 = ConcordLinearPackedB(256 * 4 * 4, 512, device=device)
        self.bn_fc1 = nn.BatchNorm1d(512).to(device)
        self.fc2 = ConcordLinearPackedB(512, 256, device=device)
        self.bn_fc2 = nn.BatchNorm1d(256).to(device)
        self.fc3 = ConcordLinearPackedB(256, 10, device=device)

    def forward(self, x):
        x = x.to(torch.bfloat16)
        # Channels-last on the input lets cuDNN pick faster NHWC conv
        # kernels on Ampere/Ada — bf16 TensorCore paths typically do
        # 10-20% better in NHWC than NCHW at small bsz. Our packed
        # weight stays NCHW (cuDNN handles the layout mismatch
        # internally); the cost is one tensor-format copy on the input.
        x = x.contiguous(memory_format=torch.channels_last)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        # .reshape (not .view) handles either memory format — won't error
        # if x is still in channels_last layout.
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.1,
                     help="Single global lr — applied uniformly to all "
                          "Concord layers (Linear and Conv2d). Biases "
                          "also use this lr.")
    ap.add_argument("--v_lr_scale", type=float, default=1.0,
                     help="DEPRECATED — legacy per-Linear lr multiplier. "
                          "Default 1.0 means one lr across all layers. "
                          "Setting !=1.0 reproduces the old split-lr "
                          "behaviour where Linears ran at lr*v_lr_scale.")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--wd_sv", type=float, default=1e-5)
    ap.add_argument("--wd_sf", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.0,
                     help="Damping on s_fast: per-step δs_fast -= β1·s_fast. "
                          "β1>0 shortens the velocity's effective memory.")
    ap.add_argument("--alpha_v_fast", type=float, default=0.001)
    ap.add_argument("--drift_cancel_C", type=float, default=None,
                     help="None = auto (compute_drift_cancel_C from rates).")
    ap.add_argument("--gf_trust_radius", type=float, default=0.0,
                     help="Garbage-fraction trust region: step is bounded "
                          "by ~radius in normalized SNR units, smoothly. "
                          "0 disables (legacy step_cap hard clamp). "
                          "Setting to step_cap gives the same asymptotic "
                          "max step as the legacy clamp.")
    ap.add_argument("--step_cap", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=1.0,
                     help="Denominator floor in step=grad/sqrt(v_proxy+eps). "
                          "At v_scale=1 this constant dominates the W²-scale "
                          "v_proxy, so it effectively normalizes the step "
                          "(eps=1 => step~=grad, i.e. SGD-on-mantissa).")
    ap.add_argument("--v_scale", type=float, default=1.0,
                     help="Multiplier on the drift-cancel noise² term in "
                          "v_proxy = noise_in_w²·v_scale. Raise it (e.g. 1e6) "
                          "to make the drift-cancel preconditioner actually "
                          "compete with eps in the denominator.")
    ap.add_argument("--precond_p", type=float, default=0.5,
                     help="Padam-style preconditioner power: step = grad / "
                          "(v_proxy+eps)^p. p=0.5 is the usual sqrt; p=0 gives "
                          "step=grad (pure SGD); p in (0,0.5) interpolates "
                          "between the linear (SGD) and smoothed (trust-region) "
                          "regimes — the knob to dial back over-conservative "
                          "drift normalization.")
    ap.add_argument("--gf_consol", type=float, default=0.0,
                     help="gf-gated consolidation evaporation rate kappa. 0 = "
                          "off (uniform cautious wd). >0 enables drift-aware "
                          "routing: gradient flows freely into s_fast, the "
                          "chase consolidates unconditionally, and only the "
                          "incoherent part of s_fast evaporates at kappa*(1-coh). "
                          "Keep kappa < alpha (0.1) so the chase bootstraps "
                          "coherence. Replaces weight_decay (set wd=0).")
    ap.add_argument("--cohpre", action="store_true",
                     help="Enable coh_pre-gated acceptance: gate the chase by "
                          "coh + coh_pre*(1-coh), coh_pre = per-coord EMA of "
                          "coh (init 1, rate alpha_v_fast). Bounds noise "
                          "diffusion into s_slow at the source while holding "
                          "converged coords. Per-element fp32 buffer.")
    ap.add_argument("--eps_warm_steps", type=int, default=0,
                     help="If >0: anneal eps (log-cosine) from --eps down to "
                          "--eps_final over this many steps, then hold. Starts "
                          "as SGD (eps=1) while accumulators warm, hands off to "
                          "the drift/gf preconditioner. Forces eager (eps is "
                          "baked into the CUDA graph at capture).")
    ap.add_argument("--eps_final", type=float, default=1e-6,
                     help="Target eps after the warmup ramp.")
    ap.add_argument("--lr_min_frac", type=float, default=0.01)
    ap.add_argument("--bn_lr", type=float, default=0.01,
                     help="lr for BN params (handled by torch.optim.SGD).")
    ap.add_argument("--polyak_window", type=int, default=10,
                     help="Last N epochs to maintain a CPU-fp32 Polyak "
                          "average of (s_slow+v_slow). 0 = disabled.")
    ap.add_argument("--polyak_beta", type=float, default=0.9,
                     help="Polyak EMA coefficient, applied per epoch.")
    ap.add_argument("--diag_steps", type=int, default=0,
                     help="If >0: enable the denominator-term diagnostic, "
                          "force eager (no graph), run this many steps, "
                          "print per-layer term magnitudes, then exit "
                          "before full training.")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--data_dir", type=str,
                     default=os.environ.get(
                         "CIFAR_DATA_DIR", "./cifar_data"))
    ap.add_argument("--tag", type=str, default="packed-b")
    ap.add_argument("--no_graph", action="store_true",
                     help="Disable CUDA graph capture (fall back to "
                          "eager per-step). Useful for debugging.")
    ap.add_argument("--warmup_steps", type=int, default=8,
                     help="Microbatches to run eagerly before graph "
                          "capture (lets Triton JIT-compile + cuDNN "
                          "benchmark settle).")
    args = ap.parse_args()

    # Diagnostic forces eager so the launcher (and its diag hook) runs every
    # step. Must set this BEFORE use_graph = not args.no_graph is computed.
    if args.diag_steps > 0:
        args.no_graph = True
    # eps warmup updates m.eps each step; eps is now a device tensor (_eps_buf)
    # that the kernel reads at runtime, so the schedule propagates across CUDA
    # graph replays (same mechanism as lr). No need to force eager.

    if not torch.cuda.is_available():
        print(f"[{args.tag}] CUDA not available, SKIP.", flush=True)
        return
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    tl, vl = get_loaders_in_memory(args.batch_size, device,
                                     data_dir=args.data_dir)
    total_steps = args.epochs * len(tl)

    model = WiderConvNet(device=device)

    # Configure all packed-B layers for AdamW three_accum.
    concord_layers = [m for m in model.modules()
                      if isinstance(m, (ConcordLinearPackedB,
                                         ConcordConv2dPackedB))]
    for m in concord_layers:
        # One update rule: AdamW (drift-cancel preconditioner + optional
        # gf trust region) on every Concord layer — Linear and Conv2d
        # alike — with the same weight_decay. The legacy recipe used
        # wd=0 on convs ("crushes conv features"); that's now a single
        # hyperparameter that you set explicitly via --weight_decay if
        # you want the old behaviour. The drift-cancel variance +
        # Adafactor v̂ floor work identically on both layer types.
        m.set_optimizer_kind('adamw',
                              weight_decay=args.weight_decay,
                              eps=args.eps,
                              step_cap=args.step_cap)
        m.v_scale = args.v_scale
        m.precond_p = args.precond_p
        m.gf_consol = args.gf_consol
        if args.cohpre:
            m.enable_cohpre()   # coh_pre-gated acceptance (λ = alpha_v_fast)
        else:
            m.disable_cohpre()  # gate default-ON now; CIFAR recipe keeps it off unless --cohpre
        m.alpha = args.alpha
        m.beta1 = args.beta1
        m.alpha_v_fast = args.alpha_v_fast
        if args.drift_cancel_C is None:
            from prototype_packed_b import compute_drift_cancel_C
            m.drift_cancel_C = compute_drift_cancel_C(m.alpha,
                                                       m.alpha_v_fast)
        else:
            m.drift_cancel_C = args.drift_cancel_C
        # gf trust region applies uniformly. The Adafactor v_row/v_col
        # EMA always tracks (cheap) and feeds the diagnostic.
        if args.gf_trust_radius > 0:
            m.gf_trust_delta_sq = 1.0 / (args.gf_trust_radius ** 2)
        m.wd_sv = args.wd_sv
        m.wd_sf = args.wd_sf
        # Track row/col max and rebalance per step. Costs the per-apply
        # atomic-max reductions (~5-15% of kernel time) plus a separate
        # rebalance kernel launch per layer per step. At aggressive lr
        # this is what prevents int16 saturation from acting as a
        # silent implicit step cap — saturating accumulators get their
        # exponent ticked instead of being clamped indefinitely.
        m.track_rebalance = True

    # BN params + biases go through torch.optim.SGD (tiny, fp32 native).
    bn_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and 'bn' in n.lower()]
    bias_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and 'bn' not in n.lower()]
    # Biases go through plain SGD at args.lr — no v_lr_scale, no
    # Concord packed dynamics. BN params get their own bn_lr (typically
    # lower) and ride the cosine schedule proportionally.
    aux_opt = torch.optim.SGD(
        [{'params': bn_params, 'lr': args.bn_lr},
         {'params': bias_params, 'lr': args.lr}],
        momentum=0.0)

    n_conv = sum(1 for m in concord_layers
                  if isinstance(m, ConcordConv2dPackedB))
    n_lin = sum(1 for m in concord_layers
                 if isinstance(m, ConcordLinearPackedB)
                 and not isinstance(m, ConcordConv2dPackedB))
    n_concord_params = sum(m.packed_w.numel() for m in concord_layers)
    use_graph = not args.no_graph
    print(f"[{args.tag}] WiderConvNet ({n_concord_params/1e6:.2f}M concord params)  "
          f"{n_conv} Conv2d + {n_lin} Linear  bsz={args.batch_size}  "
          f"graph={use_graph}", flush=True)
    print(f"[{args.tag}] AdamW(three_accum, packed, drift-cancel + gf trust) "
          f"on ALL {n_conv} Conv2d + {n_lin} Linear  "
          f"lr={args.lr}  (v_lr_scale={args.v_lr_scale})  "
          f"wd={args.weight_decay}  alpha={args.alpha}  "
          f"gf_trust_radius={args.gf_trust_radius}  "
          f"eps={args.eps}  v_scale={args.v_scale}  precond_p={args.precond_p}"
          f"  gf_consol={args.gf_consol}"
          + (f"  eps_warm: {args.eps}->{args.eps_final} over "
             f"{args.eps_warm_steps} steps" if args.eps_warm_steps > 0
             else ""),
          flush=True)

    # Denominator diagnostic: enable the per-layer term breakdown and
    # force eager execution (graph capture would hide the prints / sync).
    if args.diag_steps > 0:
        from prototype_packed_b import _DENOM_DIAG
        _DENOM_DIAG["enabled"] = True
        args.no_graph = True
        print(f"[{args.tag}] DENOM DIAG enabled: eager, {args.diag_steps} "
              f"steps then exit. Columns: drift=noise^2*v_scale, "
              f"gf=delta^2*vhat, eps, denom, |g|, |step_raw|=|g|/sqrt(denom), "
              f"cap%=frac clamped, |delta|=mantissa tick.", flush=True)

    # Helper: set per-layer + aux_opt LRs from a cosine-scheduled scalar.
    # The packed-B `m.lr = X` setter writes through to `m._lr_buf`
    # (1-elem device tensor) which the apply kernel reads — that's how
    # the captured graph picks up the new LR without re-capture.
    # aux_opt has two groups: [0]=bn_params, [1]=bias_params (set at
    # construction). Use index to identify them — pg['params'] is the
    # list PyTorch made internally, never the same object as bn_params.
    BN_GROUP = 0
    BIAS_GROUP = 1

    def set_lr(cur_lr):
        # One lr across all Concord layers — Linear and Conv2d alike —
        # honouring the (deprecated) v_lr_scale knob for backward
        # compatibility. With v_lr_scale=1.0 every Concord layer uses
        # cur_lr exactly. Biases use cur_lr (plain SGD). BN params
        # scale their bn_lr cosine'ly via (cur_lr / args.lr).
        linear_lr = cur_lr * args.v_lr_scale
        for m in concord_layers:
            if isinstance(m, ConcordConv2dPackedB):
                m.lr = cur_lr
            else:
                m.lr = linear_lr
        aux_opt.param_groups[BN_GROUP]['lr'] = (
            args.bn_lr * (cur_lr / args.lr))
        aux_opt.param_groups[BIAS_GROUP]['lr'] = cur_lr

    def cosine_lr(step):
        return (args.lr * args.lr_min_frac
                + 0.5 * args.lr * (1.0 - args.lr_min_frac)
                * (1.0 + math.cos(math.pi * step / max(total_steps, 1))))

    # ------------------------------------------------------------------
    # CUDA graph capture setup.
    # ------------------------------------------------------------------
    # Static input/label buffers. Per microbatch we .copy_() fresh data
    # in, then g.replay() runs the captured forward+backward+aux_step
    # over the new data.
    static_x = torch.zeros(args.batch_size, 3, 32, 32,
                            dtype=torch.float32, device=device,
                            requires_grad=True)
    static_y = torch.zeros(args.batch_size,
                            dtype=torch.long, device=device)
    # Static scalar loss (lives in graph allocator pool).
    static_loss = None  # filled by capture

    set_lr(args.lr)
    model.train()

    # ------------------------------------------------------------------
    # Capture (or eager fallback).
    # ------------------------------------------------------------------
    if use_graph:
        # All warmup happens on the dedicated capture stream — this
        # ensures every grad tensor and autograd node is bound to that
        # stream, otherwise the captured backward will try to sync with
        # the legacy default stream and capture fails.
        print(f"[{args.tag}] warmup on capture stream "
              f"({args.warmup_steps} steps)...", flush=True)
        s_capture = torch.cuda.Stream()
        s_capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_capture):
            for _ in range(args.warmup_steps):
                aux_opt.zero_grad(set_to_none=True)
                logits = model(static_x).float()
                loss = F.cross_entropy(logits, static_y)
                loss.backward()
                aux_opt.step()
        torch.cuda.current_stream().wait_stream(s_capture)
        torch.cuda.synchronize()

        # Capture.
        aux_opt.zero_grad(set_to_none=True)
        g = torch.cuda.CUDAGraph()
        # Helper for post-backward rebalance — runs on every layer
        # after the apply kernel has populated row_max/col_max. The
        # rebalance kernel reads those, ticks exponents where needed,
        # and SR-shifts mantissas. Graph-captured along with the rest
        # of the step so it's part of the fused inner loop.
        def _rebalance_all():
            for m in concord_layers:
                m.rebalance()
        with torch.cuda.graph(g, stream=s_capture):
            aux_opt.zero_grad(set_to_none=False)
            logits = model(static_x).float()
            static_loss = F.cross_entropy(logits, static_y)
            static_loss.backward()
            aux_opt.step()
            _rebalance_all()
        print(f"[{args.tag}] CUDA graph captured.", flush=True)
    else:
        # Eager warmup so cudnn benchmark picks algorithms.
        print(f"[{args.tag}] eager warmup ({args.warmup_steps} steps)...",
              flush=True)
        it_warmup = iter(tl)
        for _ in range(args.warmup_steps):
            try:
                x, y = next(it_warmup)
            except StopIteration:
                it_warmup = iter(tl)
                x, y = next(it_warmup)
            aux_opt.zero_grad(set_to_none=False)
            logits = model(x).float()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            aux_opt.step()
        torch.cuda.synchronize()
        g = None
        static_loss = None

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    best_acc = 0.0
    best_epoch = -1
    final_acc = 0.0
    step = 0
    t_run = time.time()

    # ── CPU-fp32 Polyak average of (s_slow + v_slow), built over the
    # last `polyak_window` epochs. Per-epoch EMA at coefficient
    # `polyak_beta`. Kept on CPU so it doesn't fight training memory.
    from prototype_packed_b import (S_SLOW_FACTOR as _SF, V_SLOW_FACTOR as _VF,
                                      MANTISSA_BIAS as _MB)
    polyak_cpu = {}
    polyak_first_epoch = max(0, args.epochs - args.polyak_window)

    @torch.no_grad()
    def _layer_fp32_slow_weight(m):
        """Return fp32 weight = (s_slow_full + v_slow_full) * envelope, on GPU."""
        s_slow_i8 = ((m.packed_w << 16) >> 24).to(torch.int32).float()
        v_slow_i8 = ((m.packed_w << 24) >> 24).to(torch.int32).float()
        m_eff = s_slow_i8 * _SF + v_slow_i8 * _VF
        exp = (m.row_exp[:, None].to(torch.int32)
               + m.col_exp[None, :].to(torch.int32) - _MB).to(torch.float32)
        return m_eff * torch.pow(2.0, exp)
    for epoch in range(args.epochs):
        model.train()
        ep_t0 = time.time()
        running_loss, seen = 0.0, 0
        for x, y in tl:
            cur_lr = cosine_lr(step)
            set_lr(cur_lr)
            # eps warmup: log-cosine ramp from args.eps -> args.eps_final over
            # args.eps_warm_steps, then hold. SGD-like while accumulators warm,
            # then hand off to the drift/gf preconditioner.
            if args.eps_warm_steps > 0:
                if step < args.eps_warm_steps:
                    frac = step / args.eps_warm_steps           # 0 -> 1
                    ramp = 0.5 * (1.0 - math.cos(math.pi * frac))  # 0 -> 1
                    log_eps = (math.log(args.eps) * (1.0 - ramp)
                               + math.log(args.eps_final) * ramp)
                    cur_eps = math.exp(log_eps)
                else:
                    cur_eps = args.eps_final
                for m in concord_layers:
                    m.eps = cur_eps
            if g is not None and x.size(0) == args.batch_size:
                # Graph fast path. Use .data.copy_ to bypass the leaf-
                # tensor in-place check (static_x has requires_grad=True
                # so autograd can build the backward graph through it).
                static_x.data.copy_(x, non_blocking=True)
                static_y.data.copy_(y, non_blocking=True)
                g.replay()
                loss_value = static_loss.detach()
            else:
                # Eager path (last partial batch, or graph disabled).
                aux_opt.zero_grad(set_to_none=False)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x).float()
                loss = F.cross_entropy(logits, y)
                loss.backward()
                aux_opt.step()
                loss_value = loss.detach()
            running_loss += loss_value.item() * x.size(0)
            seen += x.size(0)
            step += 1
            if args.diag_steps > 0 and step >= args.diag_steps:
                print(f"[{args.tag}] DENOM DIAG done ({step} steps). Exit.",
                      flush=True)
                return
        val_acc, val_loss = evaluate(model, vl, device)
        final_acc = val_acc
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1

        # Polyak EMA on CPU of (s_slow + v_slow) in fp32, active only in
        # the last `polyak_window` epochs. Initialize at first eligible
        # epoch, then EMA = β*EMA + (1-β)*current per subsequent epoch.
        if args.polyak_window > 0 and epoch >= polyak_first_epoch:
            beta = args.polyak_beta
            for m in concord_layers:
                w_gpu = _layer_fp32_slow_weight(m)
                if id(m) not in polyak_cpu:
                    polyak_cpu[id(m)] = w_gpu.detach().cpu().clone()
                else:
                    polyak_cpu[id(m)].mul_(beta).add_(
                        w_gpu.detach().cpu(), alpha=1.0 - beta)
        ep_dt = time.time() - ep_t0
        if (epoch + 1) % args.log_every == 0 or epoch == 0 \
                or epoch == args.epochs - 1:
            tr_loss = running_loss / max(seen, 1)
            print(f"[{args.tag}] ep {epoch+1:>3}/{args.epochs}  "
                  f"lr={cur_lr:.4f}  tr_loss={tr_loss:.4f}  "
                  f"val_acc={val_acc*100:.2f}%  val_loss={val_loss:.4f}  "
                  f"best={best_acc*100:.2f}% (ep {best_epoch})  "
                  f"({ep_dt:.1f}s)", flush=True)
    tot = time.time() - t_run
    print()
    print(f"[{args.tag}] DONE  total {tot/60:.1f} min  "
          f"avg {tot/args.epochs:.1f}s/ep")
    print(f"[{args.tag}] BEST  val_acc = {best_acc*100:.2f}% (epoch {best_epoch})")
    print(f"[{args.tag}] FINAL val_acc = {final_acc*100:.2f}% (epoch {args.epochs})")

    # ── CPU Polyak average eval (s_slow+v_slow, fp32, last N epochs) ──
    if polyak_cpu:
        for m in concord_layers:
            w_polyak = polyak_cpu[id(m)].to(device).to(torch.bfloat16)
            m._bf16_weight_buf.copy_(w_polyak)
        acc_p, loss_p = evaluate(model, vl, device)
        print(f"[{args.tag}] POLYAK (s_slow+v_slow, fp32 CPU EMA, "
              f"beta={args.polyak_beta}, last {args.polyak_window} eps): "
              f"val_acc={acc_p*100:.2f}%  val_loss={loss_p:.4f}", flush=True)
        # Restore live weight from packed for downstream ablation.
        for m in concord_layers:
            m._resync_weight_buf()

    # ── Per-accumulator ablation (magnitude-aware fp32 recon) ─────
    # Compose live weight = a*s_slow_full + b*s_fast + c*v_slow_full per
    # layer in fp32, write into weight_buf as bf16, evaluate. Restore by
    # re-materializing from packed_w. The (2,0,0) and (0,0,2) probes
    # rescale s_slow / v_slow alone to match the trained magnitude so
    # BN's frozen running stats stay calibrated.

    @torch.no_grad()
    def _eval_with_coeffs(a, b, c, label):
        for m in concord_layers:
            s_fast    = (m.packed_w >> 16).to(torch.int32)
            s_slow_i8 = ((m.packed_w << 16) >> 24).to(torch.int32)
            v_slow_i8 = ((m.packed_w << 24) >> 24).to(torch.int32)
            m_eff = (a * s_slow_i8.float() * _SF
                     + b * s_fast.float()
                     + c * v_slow_i8.float() * _VF)
            exp = (m.row_exp[:, None].to(torch.int32)
                   + m.col_exp[None, :].to(torch.int32)
                   - _MB).to(torch.float32)
            w_fp32 = m_eff * torch.pow(2.0, exp)
            m._bf16_weight_buf.copy_(w_fp32.to(torch.bfloat16))
        acc_a, loss_a = evaluate(model, vl, device)
        print(f"[{args.tag}]   {label}: val_acc={acc_a*100:.2f}%  "
              f"val_loss={loss_a:.4f}", flush=True)

    print(f"[{args.tag}] Per-accumulator ablation (fp32 recon, "
          f"magnitude-aware):")
    _eval_with_coeffs(1, 1, 1, "all (1,1,1)          ")
    _eval_with_coeffs(1, 0, 1, "s_slow+v_slow (1,0,1)")
    _eval_with_coeffs(2, 0, 0, "2×s_slow only        ")
    _eval_with_coeffs(0, 0, 2, "2×v_slow only        ")

    # ── gf-weighted per-element blend ─────────────────────────────
    # Where signal is clean (gf low), trust 2×s_slow (recent).
    # Where signal is noisy (gf high), trust 2×v_slow (Polyak).
    # w_ij = (1 - gf_ij^p) * 2*s_slow_full_ij + gf_ij^p * 2*v_slow_full_ij
    # Test linear blend (p=1) and sharper variants (p=0.5 toward more v_slow,
    # p=2 toward more s_slow).
    @torch.no_grad()
    def _eval_gf_blend(p, label):
        for m in concord_layers:
            s_fast    = (m.packed_w >> 16).to(torch.int32).float()
            s_slow_i8 = ((m.packed_w << 16) >> 24).to(torch.int32).float()
            v_slow_i8 = ((m.packed_w << 24) >> 24).to(torch.int32).float()
            s_slow_full = s_slow_i8 * _SF
            v_slow_full = v_slow_i8 * _VF
            # Per-element gf (same formula as get_garbage_fraction_stats)
            d_fs = s_fast
            d_sv = s_slow_full - v_slow_full
            noise_m = d_fs - float(m.drift_cancel_C) * d_sv
            exp = (m.row_exp[:, None].to(torch.float32)
                   + m.col_exp[None, :].to(torch.float32) - _MB)
            scale_fwd = torch.pow(2.0, exp)
            var_W = (noise_m * scale_fwd) ** 2
            v_row = m.v_row.float()
            v_col = m.v_col.float()
            sum_v = v_row.sum().clamp(min=1e-30)
            v_hat = v_row[:, None] * v_col[None, :] / sum_v
            gf = (var_W / (v_hat + 1e-30)).clamp(0.0, 1.0)
            if p != 1.0:
                gf = gf.pow(p)
            # Blend in mantissa space; both terms doubled so magnitude matches.
            m_eff = (1.0 - gf) * (2 * s_slow_full) + gf * (2 * v_slow_full)
            w_fp32 = m_eff * scale_fwd
            m._bf16_weight_buf.copy_(w_fp32.to(torch.bfloat16))
        acc_a, loss_a = evaluate(model, vl, device)
        print(f"[{args.tag}]   {label}: val_acc={acc_a*100:.2f}%  "
              f"val_loss={loss_a:.4f}", flush=True)

    _eval_gf_blend(1.0, "gf-blend p=1.0       ")
    _eval_gf_blend(0.5, "gf-blend p=0.5 (->v) ")
    _eval_gf_blend(2.0, "gf-blend p=2.0 (->s) ")

    # ── Hard-threshold variants ───────────────────────────────────
    @torch.no_grad()
    def _eval_gf_threshold(thr, label):
        for m in concord_layers:
            s_fast    = (m.packed_w >> 16).to(torch.int32).float()
            s_slow_i8 = ((m.packed_w << 16) >> 24).to(torch.int32).float()
            v_slow_i8 = ((m.packed_w << 24) >> 24).to(torch.int32).float()
            s_slow_full = s_slow_i8 * _SF
            v_slow_full = v_slow_i8 * _VF
            d_fs = s_fast
            d_sv = s_slow_full - v_slow_full
            noise_m = d_fs - float(m.drift_cancel_C) * d_sv
            exp = (m.row_exp[:, None].to(torch.float32)
                   + m.col_exp[None, :].to(torch.float32) - _MB)
            scale_fwd = torch.pow(2.0, exp)
            var_W = (noise_m * scale_fwd) ** 2
            v_row = m.v_row.float()
            v_col = m.v_col.float()
            sum_v = v_row.sum().clamp(min=1e-30)
            v_hat = v_row[:, None] * v_col[None, :] / sum_v
            gf = (var_W / (v_hat + 1e-30)).clamp(0.0, 1.0)
            use_v = (gf > thr).float()
            m_eff = (1.0 - use_v) * (2 * s_slow_full) + use_v * (2 * v_slow_full)
            w_fp32 = m_eff * scale_fwd
            m._bf16_weight_buf.copy_(w_fp32.to(torch.bfloat16))
        acc_a, loss_a = evaluate(model, vl, device)
        print(f"[{args.tag}]   {label}: val_acc={acc_a*100:.2f}%  "
              f"val_loss={loss_a:.4f}", flush=True)

    _eval_gf_threshold(0.5, "gf-thresh > 0.5 -> v ")
    _eval_gf_threshold(0.3, "gf-thresh > 0.3 -> v ")
    _eval_gf_threshold(0.1, "gf-thresh > 0.1 -> v ")

    # Restore baseline weight_buf from packed state.
    for m in concord_layers:
        m._resync_weight_buf()

    # Diagnostic: print packed state magnitudes per layer.
    print(f"[{args.tag}] Final packed state stats:")
    for name, m in model.named_modules():
        if not isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB)):
            continue
        s_fast, s_slow_i8, v_slow_i8 = m.get_state()
        s_slow_full = s_slow_i8.float() * 128
        v_slow_full = v_slow_i8.float() * 128
        print(f"[{args.tag}]   {name:>10}: "
              f"|s_fast|={s_fast.abs().float().mean().item():5.1f}  "
              f"|s_slow_full|={s_slow_full.abs().mean().item():7.1f}  "
              f"|v_slow_full|={v_slow_full.abs().mean().item():7.1f}  "
              f"row_exp.max={m.row_exp.max().item()}  "
              f"col_exp.max={m.col_exp.max().item()}")

    # Rebalance high-watermark: max |s_slow*128 + s_fast + v_slow*128|
    # seen across training, per row and per col. Compare to MAX_M=24000
    # (the rebalance trigger threshold) to see how close we ever came
    # to firing the exponent-tick mechanism.
    print(f"[{args.tag}] Rebalance high-watermark per layer "
          f"(trigger MAX_M={ConcordLinearPackedB.MAX_M}):")
    for name, m in model.named_modules():
        if not isinstance(m, ConcordLinearPackedB):
            continue
        row_hwm, col_hwm = m.get_rebalance_watermark_stats()
        if row_hwm is None:
            print(f"[{args.tag}]   {name:>10}: (track_rebalance=False)")
            continue
        kind = 'conv' if isinstance(m, ConcordConv2dPackedB) else 'fc'
        rmax = row_hwm.max().item()
        rmed = row_hwm.float().median().item()
        cmax = col_hwm.max().item()
        cmed = col_hwm.float().median().item()
        rfrac = (row_hwm > m.MAX_M).float().mean().item() * 100
        cfrac = (col_hwm > m.MAX_M).float().mean().item() * 100
        print(f"[{args.tag}]   {name:>10} ({kind}): "
              f"row max={rmax:>6}  median={rmed:>7.0f}  "
              f"col max={cmax:>6}  median={cmed:>7.0f}  "
              f"rows>thr={rfrac:5.1f}%  cols>thr={cfrac:5.1f}%")

    # Garbage fraction per layer. Both Linear and Conv2d EMA the
    # Adafactor row/col second moment now, so the convergence map is
    # visible on the convs too — but note convs precondition by SGD
    # (no v_proxy = noise²·v_scale), so for them gf is purely
    # diagnostic, not a step gate.
    print(f"[{args.tag}] Garbage fraction Var/E[g²] per layer:")
    for name, m in model.named_modules():
        if not isinstance(m, ConcordLinearPackedB):
            continue
        stats = m.get_garbage_fraction_stats()
        if stats is None:
            print(f"[{args.tag}]   {name:>10}: (v_row not populated)")
            continue
        kind = 'conv' if isinstance(m, ConcordConv2dPackedB) else 'fc'
        print(f"[{args.tag}]   {name:>10} ({kind}): "
              f"median={stats['median']:.3f}  "
              f"p25={stats['p25']:.3f}  "
              f"p75={stats['p75']:.3f}  "
              f"mean={stats['mean']:.3f}  "
              f"signal<0.1={stats['frac_signal_dominated']*100:.1f}%  "
              f"noise>0.9={stats['frac_noise_dominated']*100:.1f}%")


if __name__ == "__main__":
    main()
