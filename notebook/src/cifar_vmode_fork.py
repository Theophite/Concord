"""Mechanism-localization study: where does the second moment (v-hat) change
the weights?

Three optimizers, IDENTICAL except for the v-hat treatment:
  none   : update = m_hat                       (momentum SGD; v-hat ablated)
  rank1  : update = m_hat / (sqrt(v_row⊗v_col)+eps)   (Adafactor, factored v)
  full   : update = m_hat / (sqrt(v_elem)+eps)        (per-element b2 = Adam)
Same Adam-style m, same bias correction, same decoupled wd, same eps slot.
Only the denominator differs -> any weight difference is attributable to v-hat.

Protocol: same init, FIRST 10 epochs run in `none` mode (shared trajectory) ->
common W10 checkpoint -> fork each mode (fresh opt state, own peak LR) to N ep.
Save per-weight final W and coh = EMA[g]^2 / EMA[g^2] (concord garbage factor:
coh = signal^2/(signal^2+noise^2) = SNR^2/(1+SNR^2)). coh measured during the
SHARED warmup is a mode-independent landscape property for the analysis.
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


class WiderConvNet(nn.Module):
    """Identical architecture/forward to the packed-B / vanilla-AdamW net."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.float()
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)


# weight params we analyze (2D-flattenable conv/linear weights), in fwd order
ANALYZE = ["conv1.weight", "conv2.weight", "conv3.weight", "conv4.weight",
           "fc1.weight", "fc2.weight", "fc3.weight"]


def _single_denom(g, p, eps=0.01):
    """Per-step (no-EMA) rank-1 factored g^2 whitener denom = the SS.2.8
    single-sample whitener: aggressively washes out incoherent (high per-step-
    variance) gradient directions. A memorization filter, not an EMA v-hat."""
    if p.ndim >= 2:
        R = p.shape[0]; C = p.numel() // R
        g2 = g.reshape(R, C) ** 2
        r = g2.mean(1); c = g2.mean(0)
        est = (r[:, None] * c[None, :]) / (r.mean() + 1e-30)
        est = est / (est.mean() + 1e-30)
        return (est + eps).sqrt().reshape(p.shape)
    g2 = g * g
    return (g2 / (g2.mean() + 1e-30) + eps).sqrt()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0; loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total, loss_sum / total


class VOpt:
    """Adam-family optimizer with switchable second moment. Decoupled wd.
    Also maintains diagnostic coh EMAs (md1, md2) for the analyzed weights."""
    def __init__(self, named_params, mode, lr, b1, b2, eps, wd, coh_beta=0.99):
        self.mode = mode; self.lr = lr; self.b1 = b1; self.b2 = b2
        self.eps = eps; self.wd = wd; self.coh_beta = coh_beta
        self.t = 0; self.graph = False
        self.params = []          # list of (name, tensor, is2d)
        self.m = {}; self.v = {}; self.vr = {}; self.vc = {}
        self.md1 = {}; self.md2 = {}
        for name, p in named_params:
            self.params.append((name, p))
            self.m[name] = torch.zeros_like(p)
            if mode == "full":
                self.v[name] = torch.zeros_like(p)
            elif mode == "rank1" and p.ndim >= 2:
                R, C = p.shape[0], p.numel() // p.shape[0]
                self.vr[name] = torch.zeros(R, device=p.device)
                self.vc[name] = torch.zeros(C, device=p.device)
            elif mode == "rank1":           # 1D params (bias/BN): use full v
                self.v[name] = torch.zeros_like(p)
            if name in ANALYZE:
                self.md1[name] = torch.zeros_like(p)
                self.md2[name] = torch.zeros_like(p)

    def set_lr(self, lr):
        self.lr = lr

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        cb = self.coh_beta
        for name, p in self.params:
            g = p.grad
            if g is None:
                continue
            m = self.m[name]; m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            mhat = m / bc1
            if self.mode == "none":
                denom = None
            elif self.mode == "single":            # per-step g^2 whitener (2.8)
                denom = _single_denom(g, p)
            elif self.mode == "full" or (self.mode == "rank1" and p.ndim < 2):
                v = self.v[name]; v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                denom = (v / bc2).sqrt_().add_(self.eps)
            else:   # rank1 on a 2D+ param
                R = p.shape[0]; C = p.numel() // R
                g2 = (g.reshape(R, C) ** 2)
                vr = self.vr[name]; vc = self.vc[name]
                vr.mul_(self.b2).add_(g2.mean(dim=1), alpha=1 - self.b2)
                vc.mul_(self.b2).add_(g2.mean(dim=0), alpha=1 - self.b2)
                vrh = vr / bc2; vch = vc / bc2
                vest = (vrh[:, None] * vch[None, :]) / (vrh.mean() + 1e-30)
                denom = (vest.sqrt_().add_(self.eps)).reshape(p.shape)
            upd = mhat if denom is None else mhat / denom
            if self.wd != 0.0:
                p.add_(p, alpha=-self.lr * self.wd)        # decoupled
            p.add_(upd, alpha=-self.lr)
            # diagnostic coh EMAs (on the raw gradient)
            if name in self.md1:
                self.md1[name].mul_(cb).add_(g, alpha=1 - cb)
                self.md2[name].mul_(cb).addcmul_(g, g, value=1 - cb)

    def coh_snapshot(self):
        out = {}
        for name in self.md1:
            out[name] = (self.md1[name] ** 2 / (self.md2[name] + 1e-30)).cpu()
        return out

    def state_dict(self):
        cpu = lambda d: {k: v.detach().cpu() for k, v in d.items()}
        return {"t": self.t, "m": cpu(self.m), "v": cpu(self.v),
                "vr": cpu(self.vr), "vc": cpu(self.vc),
                "md1": cpu(self.md1), "md2": cpu(self.md2)}

    def load_state_dict(self, sd, device):
        self.t = sd["t"]
        for attr in ("m", "v", "vr", "vc", "md1", "md2"):
            d = getattr(self, attr)
            for k in d:
                if k in sd[attr]:
                    d[k].copy_(sd[attr][k].to(device))   # in-place: graph-safe
        if self.graph:
            self.t_buf.fill_(float(self.t))

    # ---- CUDA-graph-capturable variant (tensor lr + self-incrementing t) ----
    def enable_graph(self, device):
        self.graph = True
        self.lr_buf = torch.zeros(1, device=device)
        self.t_buf = torch.full((1,), float(self.t), device=device)

    def set_lr_graph(self, lr):
        self.lr_buf.fill_(lr)        # outside graph; replay reads current value

    @torch.no_grad()
    def step_graph(self):
        self.t_buf.add_(1.0)                       # increments each replay
        bc1 = 1.0 - self.b1 ** self.t_buf          # on-device bias correction
        bc2 = 1.0 - self.b2 ** self.t_buf
        cb = self.coh_beta; lr = self.lr_buf
        for name, p in self.params:
            g = p.grad
            m = self.m[name]; m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            mhat = m / bc1
            if self.mode == "none":
                upd = mhat
            elif self.mode == "single":
                upd = mhat / _single_denom(g, p)
            elif self.mode == "full" or (self.mode == "rank1" and p.ndim < 2):
                v = self.v[name]
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                upd = mhat / ((v / bc2).sqrt() + self.eps)
            else:
                R = p.shape[0]; C = p.numel() // R
                g2 = g.reshape(R, C) ** 2
                vr = self.vr[name]; vc = self.vc[name]
                vr.mul_(self.b2).add_(g2.mean(dim=1), alpha=1 - self.b2)
                vc.mul_(self.b2).add_(g2.mean(dim=0), alpha=1 - self.b2)
                vrh = vr / bc2; vch = vc / bc2
                vest = (vrh[:, None] * vch[None, :]) / (vrh.mean() + 1e-30)
                upd = mhat / (vest.sqrt() + self.eps).reshape(p.shape)
            if self.wd != 0.0:
                p.mul_(1.0 - lr * self.wd)         # decoupled wd, tensor lr
            p.sub_(upd * lr)
            if name in self.md1:
                self.md1[name].mul_(cb).add_(g, alpha=1 - cb)
                self.md2[name].mul_(cb).addcmul_(g, g, value=1 - cb)


def cosine_lr(peak, step, total, min_frac):
    return peak * min_frac + 0.5 * peak * (1 - min_frac) * \
        (1 + math.cos(math.pi * step / max(total, 1)))


def train_phase(model, opt, tl, vl, device, peak_lr, min_frac,
                step0, total_steps, ep0, ep1, tag, traj=None, ckpt_every=0,
                best0=0.0, best_ep0=-1, resume_path=None, resume_every=0,
                label_noise=0.0):
    """Train epochs [ep0, ep1); cosine over global [0,total_steps). traj/
    ckpt_every: snapshot ANALYZE weights every ckpt_every ep. resume_path/
    resume_every: atomically checkpoint model+opt+traj every resume_every ep
    for intra-arm crash recovery (flaky box)."""
    best = best0; best_ep = best_ep0; final = 0.0; step = step0
    for epoch in range(ep0, ep1):
        model.train(); ep_t = time.time(); run = seen = 0
        for x, y in tl:
            opt.set_lr(cosine_lr(peak_lr, step, total_steps, min_frac))
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if label_noise > 0:        # inject Bayes error: resample y per batch
                flip = torch.rand(y.shape, device=device) < label_noise
                y = torch.where(flip, torch.randint(0, 10, y.shape,
                                                    device=device), y)
            for _, p in opt.params:
                p.grad = None
            loss = F.cross_entropy(model(x).float(), y)
            loss.backward(); opt.step()
            run += loss.item() * x.size(0); seen += x.size(0); step += 1
        acc, vloss = evaluate(model, vl, device); final = acc
        if acc > best:
            best, best_ep = acc, epoch + 1
        print(f"[{tag}] ep {epoch+1:>3}/{ep1}  lr={opt.lr:.4f}  "
              f"tr_loss={run/max(seen,1):.4f}  val_acc={acc*100:.2f}%  "
              f"best={best*100:.2f}% (ep {best_ep})  ({time.time()-ep_t:.1f}s)",
              flush=True)
        if traj is not None and ckpt_every and (epoch + 1) % ckpt_every == 0:
            traj[epoch + 1] = {nm: p.detach().cpu().clone()
                               for nm, p in model.named_parameters()
                               if nm in ANALYZE}
        if resume_path and resume_every and (epoch + 1) % resume_every == 0 \
                and (epoch + 1) < ep1:
            torch.save({"next_ep": epoch + 1, "step": step, "best": best,
                        "best_ep": best_ep, "traj": traj,
                        "model": {k: v.cpu() for k, v in
                                  model.state_dict().items()},
                        "opt": opt.state_dict()}, resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)   # atomic
    return step, best, best_ep, final


def train_phase_graph(model, opt, tl, vl, device, peak_lr, min_frac, step0,
                      total_steps, ep0, ep1, tag, bsz, traj=None, ckpt_every=0,
                      best0=0.0, best_ep0=-1, resume_path=None, resume_every=0,
                      label_noise=0.0, warmup_steps=5):
    """Same training as train_phase, but the fwd+bwd+opt.step is captured into
    a CUDA graph and replayed (bsz=16 is launch-bound -> big speedup). LR + step
    counter are device tensors so the cosine schedule + bias correction work
    without re-capture. Warmup+capture perturb the model on a repeated batch, so
    we snapshot clean state before and restore after -> the fork still starts at
    the exact W10/resume state. Label noise (host-side, outside graph) + eval
    stay eager. 50000/16=3125 (no partial batch)."""
    opt.enable_graph(device)
    model.train()
    static_x = torch.zeros(bsz, 3, 32, 32, device=device, requires_grad=True)
    static_y = torch.zeros(bsz, dtype=torch.long, device=device)
    xb, yb = next(iter(tl))
    static_x.data.copy_(xb[:bsz].to(device)); static_y.copy_(yb[:bsz].to(device))
    snap_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snap_opt = opt.state_dict(); snap_t = opt.t
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    opt.set_lr_graph(peak_lr)
    with torch.cuda.stream(s):
        for _ in range(warmup_steps):
            for _, p in opt.params:
                if p.grad is not None:
                    p.grad.zero_()
            F.cross_entropy(model(static_x).float(), static_y).backward()
            opt.step_graph()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _, p in opt.params:
            if p.grad is not None:
                p.grad.zero_()
        static_logits = model(static_x).float()
        static_loss = F.cross_entropy(static_logits, static_y)
        static_loss.backward(); opt.step_graph()
    # undo warmup+capture perturbation -> clean W10/resume start
    model.load_state_dict(snap_model)
    opt.load_state_dict(snap_opt, device)
    opt.t = snap_t; opt.t_buf.fill_(float(snap_t))
    print(f"[{tag}] CUDA graph captured (warmup {warmup_steps}); replaying "
          f"from ep {ep0}", flush=True)

    best = best0; best_ep = best_ep0; final = 0.0; step = step0
    for epoch in range(ep0, ep1):
        model.train(); ep_t = time.time(); run = seen = 0
        for x, y in tl:
            if x.size(0) != bsz:
                continue
            opt.set_lr_graph(cosine_lr(peak_lr, step, total_steps, min_frac))
            xd = x.to(device, non_blocking=True)
            yd = y.to(device, non_blocking=True)
            if label_noise > 0:
                flip = torch.rand(yd.shape, device=device) < label_noise
                yd = torch.where(flip, torch.randint(0, 10, yd.shape,
                                                     device=device), yd)
            static_x.data.copy_(xd); static_y.copy_(yd)
            g.replay()
            run += static_loss.item() * bsz; seen += bsz; step += 1
        acc, vloss = evaluate(model, vl, device); final = acc
        if acc > best:
            best, best_ep = acc, epoch + 1
        print(f"[{tag}] ep {epoch+1:>3}/{ep1}  lr={opt.lr_buf.item():.4f}  "
              f"tr_loss={run/max(seen,1):.4f}  val_acc={acc*100:.2f}%  "
              f"best={best*100:.2f}% (ep {best_ep})  ({time.time()-ep_t:.1f}s)",
              flush=True)
        opt.t = int(opt.t_buf.item())            # sync for checkpoint/state
        if traj is not None and ckpt_every and (epoch + 1) % ckpt_every == 0:
            traj[epoch + 1] = {nm: p.detach().cpu().clone()
                               for nm, p in model.named_parameters()
                               if nm in ANALYZE}
        if resume_path and resume_every and (epoch + 1) % resume_every == 0 \
                and (epoch + 1) < ep1:
            torch.save({"next_ep": epoch + 1, "step": step, "best": best,
                        "best_ep": best_ep, "traj": traj,
                        "model": {k: v.cpu() for k, v in
                                  model.state_dict().items()},
                        "opt": opt.state_dict()}, resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)
    return step, best, best_ep, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_none", type=float, default=0.02)
    ap.add_argument("--lr_adam", type=float, default=1e-3)
    ap.add_argument("--wd_none", type=float, default=5e-3)
    ap.add_argument("--wd_adam", type=float, default=0.05)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--lr_min_frac", type=float, default=0.001)
    ap.add_argument("--modes", type=str, default="none,rank1,full")
    ap.add_argument("--data_dir", type=str,
                    default=os.environ.get("CIFAR_DATA_DIR", "./cifar_data"))
    ap.add_argument("--out", type=str, default="vmode")
    ap.add_argument("--force_warm", action="store_true",
                    help="redo warmup even if {out}_warm.pt exists")
    ap.add_argument("--force", action="store_true",
                    help="redo a mode even if {out}_{mode}.pt exists")
    ap.add_argument("--ckpt_every", type=int, default=15,
                    help="snapshot ANALYZE weights every N epochs per fork "
                         "(settling-timeline / freezing analysis)")
    ap.add_argument("--resume_every", type=int, default=10,
                    help="atomically checkpoint model+opt every N epochs for "
                         "intra-arm crash recovery (0 disables)")
    ap.add_argument("--label_noise", type=float, default=0.0,
                    help="per-batch random label flip prob on TRAIN only -> "
                         "injects ~Bayes error (irreducible gradient noise); "
                         "the regime where v-hat should matter. test/val clean.")
    ap.add_argument("--use_graph", action="store_true",
                    help="capture fwd+bwd+opt.step into a CUDA graph + replay "
                         "(bsz=16 is launch-bound -> ~4x). forks only; eval "
                         "eager. default off (eager path unchanged).")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    tl, vl = get_loaders_in_memory(args.batch_size, device, data_dir=args.data_dir)
    total_steps = args.epochs * len(tl)
    warm_steps = args.warmup_epochs * len(tl)

    def lr_wd(mode):
        return (args.lr_none, args.wd_none) if mode == "none" \
            else (args.lr_adam, args.wd_adam)

    # ---- shared warmup in `none` mode (resume from {out}_warm.pt if present) --
    warm_path = f"{args.out}_warm.pt"
    model = WiderConvNet().to(device)
    if os.path.exists(warm_path) and not args.force_warm:
        ck = torch.load(warm_path, map_location=device, weights_only=False)
        W10 = ck["W10"]
        print(f"[resume] loaded W10 from {warm_path}; skipping "
              f"{args.warmup_epochs}-ep warmup", flush=True)
    else:
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model = WiderConvNet().to(device)
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        np_ = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        lr0, wd0 = lr_wd("none")
        opt = VOpt(np_, "none", lr0, args.b1, args.b2, args.eps, wd0)
        n = sum(p.numel() for _, p in np_)
        print(f"[warmup] none-mode {n/1e6:.2f}M params  {args.warmup_epochs} ep  "
              f"lr_none={args.lr_none} lr_adam={args.lr_adam}  "
              f"bsz={args.batch_size}", flush=True)
        train_phase(model, opt, tl, vl, device, lr0, args.lr_min_frac, 0,
                    total_steps, 0, args.warmup_epochs, "warmup",
                    label_noise=args.label_noise)
        W10 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        coh_warm = opt.coh_snapshot()
        torch.save({"coh_warm": coh_warm, "W10": W10,
                    "init": {k: v.cpu() for k, v in init_state.items()}}, warm_path)
        print(f"[warmup] done -> W10 + coh_warm saved", flush=True)

    # ---- fork each mode from W10 (fresh opt state, own peak lr) ----
    results = {}
    for mode in args.modes.split(","):
        mode_path = f"{args.out}_{mode}.pt"
        if os.path.exists(mode_path) and not args.force:
            d = torch.load(mode_path, map_location="cpu", weights_only=False)
            results[mode] = (d["best"], d["best_ep"], d["final"])
            print(f"[{mode}] already done -> {mode_path} "
                  f"(best {d['best']*100:.2f}%); skipping", flush=True)
            continue
        np_ = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]
        lr, wd = lr_wd(mode)
        opt = VOpt(np_, mode, lr, args.b1, args.b2, args.eps, wd)
        traj = {}; start_ep = args.warmup_epochs; step = warm_steps
        best = 0.0; best_ep = -1
        resume_path = f"{args.out}_{mode}_resume.pt"
        if os.path.exists(resume_path) and not args.force:
            try:
                rs = torch.load(resume_path, map_location="cpu",
                                weights_only=False)
                model.load_state_dict(rs["model"])
                opt.load_state_dict(rs["opt"], device)
                start_ep = rs["next_ep"]; step = rs["step"]
                best = rs["best"]; best_ep = rs["best_ep"]; traj = rs["traj"]
                print(f"[{mode}] resuming mid-arm from ep {start_ep} "
                      f"(best {best*100:.2f}%)", flush=True)
            except Exception as e:
                print(f"[{mode}] resume unreadable ({e}); restart arm",
                      flush=True)
                model.load_state_dict(W10)
        else:
            model.load_state_dict(W10)
        phase_fn = train_phase_graph if args.use_graph else train_phase
        extra = {"bsz": args.batch_size} if args.use_graph else {}
        step, best, best_ep, final = phase_fn(
            model, opt, tl, vl, device, lr, args.lr_min_frac, step,
            total_steps, start_ep, args.epochs, mode, traj=traj,
            ckpt_every=args.ckpt_every, best0=best, best_ep0=best_ep,
            resume_path=resume_path, resume_every=args.resume_every,
            label_noise=args.label_noise, **extra)
        Wf = {nm: p.detach().cpu().clone()
              for nm, p in model.named_parameters() if nm in ANALYZE}
        coh = opt.coh_snapshot()
        # full state_dict too (BN/bias) -> enables rank-k reconstruction eval;
        # traj = weight snapshots over training -> settling-timeline analysis
        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"mode": mode, "W": Wf, "coh": coh, "best": best,
                    "best_ep": best_ep, "final": final, "state": state,
                    "traj": traj}, f"{args.out}_{mode}.pt")
        if os.path.exists(resume_path):
            os.remove(resume_path)
        results[mode] = (best, best_ep, final)
        print(f"[{mode}] BEST {best*100:.2f}% (ep {best_ep})  "
              f"FINAL {final*100:.2f}%  -> {args.out}_{mode}.pt", flush=True)

    print("\n==== SUMMARY ====")
    for mode, (b, be, f) in results.items():
        print(f"  {mode:6s}  best {b*100:.2f}% (ep {be})  final {f*100:.2f}%")


if __name__ == "__main__":
    main()
