"""Exp 15: pre-NS v̂ scaling — selectivity at the write that survives NS.

The one candidate left standing after exp 12–14: orthogonalize the
SNR-SCALED gradient. NS5 re-whitens magnitudes, but the dominant subspace it
preserves is set by its INPUT — and h = g/√(v̂+ε) weights that input by
per-element SNR. Selectivity acts at the write (the exp-14 law), in the only
channel orthogonalization cannot erase: which directions dominate.

    h = g / sqrt(v̂ + eps)         # the v̂ drive's whole point
    step = sqrt(max(N,K)) * NS5(h/||h||)

Two instruments, stored controls:
  A. exp-14 rank-4 synthetic (leak + deploy MSE, 3 seeds, λ=0):
     prediction — leak drops from muon's ~44% toward v̂'s ~24%.
  B. exp-12 MNIST protocol (4 noise levels, λ ∈ {0, 0.1}, 3 seeds):
     prediction — competitive with muon at λ=0, and the λ=0.1 penalty
     SHRINKS vs muon's (the velocity regains per-element legibility, so the
     element gate stops being a uniform tax).
"""
import json
import math
import time

import torch

from concord_ref import ConcordRef, evap_term, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, LR, SEEDS
from exp9_muon import ns5
import exp14_rank_deficient as e14

torch.set_num_threads(4)


class PreNSMuon(ConcordRef):
    """Orthogonalize the v̂-scaled gradient (pre-NS SNR weighting)."""

    @torch.no_grad()
    def step(self):
        self._advance_schedules()
        lr = self.lr
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            g = g.float()
            g2 = g * g
            st["v_row"].mul_(self.beta2).add_(g2.sum(1), alpha=1 - self.beta2)
            st["v_col"].mul_(self.beta2).add_(g2.sum(0), alpha=1 - self.beta2)
            vhat = (st["v_row"][:, None] * st["v_col"][None, :]
                    / st["v_row"].sum().clamp_min(1e-30))
            u, S, A = st["u"], st["S"], st["A"]
            mu = self.Cstar * (S - A)
            nse = u - mu
            coh = (mu * mu) / (mu * mu + nse * nse + 1e-30)
            h = g / (vhat + self.eps).sqrt()
            hn = h / h.norm().clamp_min(1e-12)
            step_ = math.sqrt(max(g.shape)) * ns5(hn)
            u += -lr * step_ - evap_term(lr, self.kappa, coh, u, self.min_leak)
            gc = self.phic + (1 - self.phic) * coh
            tr = self.alpha * gc * u
            S += tr
            u -= tr
            gl = self.phil + (1 - self.phil) * coh
            lk = self.alpha_v * gl * (S - A)
            A += lk
            S -= lk
            st["p"].copy_(u + S + A)
            st["last_coh"] = float(coh.mean())
        for st in self.aux:
            if st["p"].grad is None:
                continue
            st["m"].mul_(0.9).add_(st["p"].grad)
            st["p"].add_(st["m"], alpha=-lr)
        self.t += 1


def run_mnist(nf, kappa, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = PreNSMuon(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                    noise=False, generator=torch.Generator().manual_seed(seed + 10))
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            if not math.isfinite(float(loss.detach())):
                return None
            opt.zero_grad()
            loss.backward()
            opt.step()
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        return accuracy(net, xte, yte), fit, opt.mean_coh()


if __name__ == "__main__":
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    t0 = time.time()

    # ---- A. rank-4 synthetic (controls: exp14_results.json) ----
    e14.DRIVES["prens"] = PreNSMuon
    e14ref = json.load(open("exp14_results.json"))
    print("A. rank-4 synthetic (controls: vhat / muon from exp 14):")
    for key in ("vhat|0", "muon|0", "vhat|1", "muon|1"):
        v = e14ref[key]
        print(f"   {key:8s}  MSE={v[0]:.4f}±{v[1]:.4f}  leak={v[2]*100:.1f}%")
    for noisy in (False, True):
        rs = [e14.run("prens", noisy, 0.0, s) for s in (0, 1, 2)]
        m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
        lk = mean([r[1] for r in rs])
        out[f"rank|prens|{int(noisy)}"] = (m, sp, lk)
        print(f"   prens|{int(noisy)}   MSE={m:.4f}±{sp:.4f}  leak={lk*100:.1f}%"
              f"   [{time.time()-t0:.0f}s]", flush=True)

    # ---- B. MNIST (controls: exp12_results.json muon rows) ----
    data = load_mnist()
    e12 = json.load(open("exp12_results.json"))
    print("B. MNIST (controls: muon-NS5 from exp 12):")
    for nf in (0.0, 0.10, 0.30, 0.45):
        for k in (0.0, 100.0):
            v = e12.get(f"muon|{nf}|{k}")
            if v:
                print(f"   ns5   noise={nf:.0%} lam={k*LR:.1f}: {v[0]*100:.2f}±{v[1]*100:.2f}%")
    for nf in (0.0, 0.10, 0.30, 0.45):
        for k in (0.0, 100.0):
            rs = [run_mnist(nf, k, s, data) for s in SEEDS]
            if any(r is None for r in rs):
                out[f"mnist|{nf}|{k}"] = None
                print(f"   prens noise={nf:.0%} lam={k*LR:.1f}: DIVERGED"
                      f"   [{time.time()-t0:.0f}s]", flush=True)
                continue
            m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
            mf = mean([r[1] for r in rs])
            out[f"mnist|{nf}|{k}"] = (m, sp, mf)
            print(f"   prens noise={nf:.0%} lam={k*LR:.1f}: {m*100:.2f}±{sp*100:.2f}%"
                  f"  memorized={mf*100:.1f}%   [{time.time()-t0:.0f}s]", flush=True)
            json.dump(out, open("exp15_results.json", "w"), indent=1)
