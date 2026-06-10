"""Exp 13: the spectral gate INSIDE Newton-Schulz — Wiener-NS.

Exp 12's verdict: the per-element Wiener meter is the wrong basis for a
whitened drive (NS5 writes signal and noise element-wise indistinguishable,
so the dissipation taxes both). The surviving reopen point is a SPECTRAL
gate — and it can live inside the Newton-Schulz itself, no SVD:

    X @ (X^T X) = U diag(sigma^3) V^T        (two matmuls, basis-preserving)

Cube-and-renormalize c times BEFORE the NS5 orthogonalization:

    X = G/||G||_F
    c times:  X <- X (X^T X);  X <- X/||X||_F     # relative gap -> gap^(3^c)
    step = sqrt(max(N,K)) * NS5(X)

The composed sigma-map is a smooth RELATIVE threshold: bulk (noise)
directions are crushed below NS5's rise before it can lift them; spiked
(signal) directions land at 1. Because the threshold is relative to the
renormalized spectrum — not a fixed count or energy as in exp 10's hard/comp
modes — the gate is EMERGENT-ANNEALED: a flat early spectrum passes nearly
unchanged (full NS5, no subspace-finding starvation); a spiked late spectrum
gets tightened. This is the annealed rank restriction of MUON_DRIVE §10,
implemented by 2c matmuls.

Arms (exp-5 protocol; muon-NS5 controls = exp 12 cells, same seeds):
    drive in {wns1, wns2}  (c = 1, 2)
    rho   in {0, 10%, 30%, 45%}
    lam   in {0, 0.1}      — does a small dissipation now HELP (the drive's
                             injection is cleaned at source, so the
                             per-element meter sees legible contrast again)?
"""
import json
import math
import time

import torch

from concord_ref import ConcordRef, evap_term, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, LR, SEEDS
from exp9_muon import ns5

torch.set_num_threads(4)

NOISES = (0.0, 0.10, 0.30, 0.45)
KAPPAS = (0.0, 100.0)            # lam 0, 0.1 at peak lr
CUBES = (1, 2)


def wiener_ns(g, n_cube, eps=1e-12):
    """The spectral gate inside NS: cube-renormalize c times, then NS5."""
    X = g / g.norm().clamp_min(eps)
    for _ in range(n_cube):
        X = X @ (X.T @ X)
        X = X / X.norm().clamp_min(eps)
    return math.sqrt(max(g.shape)) * ns5(X)


class WienerNS(ConcordRef):
    """Muon drive with the in-NS spectral gate (exp-12 MuonLeak shape)."""

    def __init__(self, *a, n_cube=1, **k):
        super().__init__(*a, **k)
        self.n_cube = int(n_cube)

    @torch.no_grad()
    def step(self):
        self._advance_schedules()
        lr = self.lr
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            g = g.float()
            u, S, A = st["u"], st["S"], st["A"]
            mu = self.Cstar * (S - A)
            nse = u - mu
            coh = (mu * mu) / (mu * mu + nse * nse + 1e-30)
            step_ = wiener_ns(g, self.n_cube)
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


def run(n_cube, nf, kappa, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = WienerNS(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                   noise=False, n_cube=n_cube,
                   generator=torch.Generator().manual_seed(seed + 10))
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
        dep = accuracy(net, xte, yte)
    return dep, fit, opt.mean_coh()


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    # muon-NS5 controls at lam {0, 0.1} from exp 12 (same protocol/seeds)
    e12 = json.load(open("exp12_results.json"))
    print("controls (exp 12, NS5):")
    for nf in NOISES:
        for k in KAPPAS:
            v = e12.get(f"muon|{nf}|{k}")
            if v:
                print(f"  ns5  noise={nf:.0%} lam={k*LR:.1f}: "
                      f"deploy={v[0]*100:.2f}±{v[1]*100:.2f}%  "
                      f"memorized={v[2]*100:.1f}%  coh={v[3]:.3f}")
    results = {}
    t0 = time.time()
    for c in CUBES:
        for nf in NOISES:
            for k in KAPPAS:
                rs = [run(c, nf, k, s, data) for s in SEEDS]
                if any(r is None for r in rs):
                    results[(c, nf, k)] = None
                    print(f"wns{c} noise={nf:.0%} lam={k*LR:.1f}: DIVERGED"
                          f"   [{time.time()-t0:.0f}s]", flush=True)
                    continue
                m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
                mf, mc = mean([r[1] for r in rs]), mean([r[2] for r in rs])
                results[(c, nf, k)] = (m, sp, mf, mc)
                print(f"wns{c} noise={nf:.0%} lam={k*LR:.1f}: "
                      f"deploy={m*100:.2f}±{sp*100:.2f}%  "
                      f"memorized={mf*100:.1f}%  coh={mc:.3f}"
                      f"   [{time.time()-t0:.0f}s]", flush=True)
                json.dump({f"{c}|{n}|{kk}": v
                           for (c, n, kk), v in results.items()},
                          open("exp13_results.json", "w"), indent=1)
