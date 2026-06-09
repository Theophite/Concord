"""Exp 7: coherence-gated momentum (beta1) under autotuned dissipation.

The winner ships beta1 = 0; the kernel's comment says ungated momentum
diverges and the gated term was left off. Both decisions predate the C*
rescale — momentum gated by a half-blind coherence is a different animal
from momentum gated by the honest one. This sweeps beta1 with the v2.1
autotuner active (probe epochs 3-8 at kappa=50 -> commit from the exp-6
table), in both regimes:

    clean (rho=0)       4k x 25 epochs
    noisy (rho=30%)     4k x 25 epochs, label noise

Linear stability note: at full coherence the velocity recursion is
u <- (1+beta1)(1-alpha*gc)*u, unstable for beta1 > ~0.11 — but the gate is
self-limiting (velocity outrunning the drift prediction lowers coh, cutting
momentum and raising friction), so the sweep crosses the linear bound on
purpose: {0, 0.05, 0.1, 0.2, 0.4, 0.8}.
"""
import json
import math

import torch

from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, LR, SEEDS

torch.set_num_threads(4)
BETAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
RHOS = (0.0, 0.30)
TABLE = [(0.387, 0.0), (0.314, 100.0), (0.288, 200.0), (0.274, 400.0),
         (0.256, 400.0)]


def kappa_from_coh(c):
    if c >= TABLE[0][0]:
        return TABLE[0][1]
    if c <= TABLE[-1][0]:
        return TABLE[-1][1]
    for (c1, k1), (c2, k2) in zip(TABLE, TABLE[1:]):
        if c2 <= c <= c1:
            return k1 + (c1 - c) / (c1 - c2) * (k2 - k1)
    return TABLE[-1][1]


def run(nf, seed, beta1, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    spe = len(x) // BATCH
    steps = EPOCHS * spe
    opt = ConcordRef(net, lr=LR, total_steps=steps, gate=True, kappa=50.0,
                     noise=False, beta1=beta1,
                     generator=torch.Generator().manual_seed(seed + 10))
    p0, p1 = 3 * spe, 8 * spe
    cohs, committed, t = [], None, 0
    umax_late = 0.0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            if not math.isfinite(float(loss)):
                return None                                 # diverged
            opt.zero_grad()
            loss.backward()
            opt.step()
            if p0 <= t < p1:
                cohs.append(opt.mean_coh())
            elif t == p1:
                committed = kappa_from_coh(sum(cohs) / len(cohs))
                opt.kappa = committed
            if t >= steps - 2 * spe:
                un = max(float(st["u"].abs().max()) for st in opt.swapped)
                umax_late = max(umax_late, un)
            t += 1
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        dep = accuracy(net, xte, yte)
    return dep, fit, committed, sum(cohs) / len(cohs), umax_late


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for nf in RHOS:
        print(f"rho={nf:.0%} (autotuned kappa, 3 seeds):")
        for b1 in BETAS:
            rs = [run(nf, s, b1, data) for s in SEEDS]
            if any(r is None for r in rs):
                print(f"  beta1={b1:.2f}: DIVERGED "
                      f"({sum(r is None for r in rs)}/{len(SEEDS)} seeds)")
                out[(nf, b1)] = None
                continue
            deps = [r[0] for r in rs]
            fits = [r[1] for r in rs]
            kaps = [r[2] for r in rs]
            pcoh = [r[3] for r in rs]
            umax = [r[4] for r in rs]
            out[(nf, b1)] = (mean(deps), spread(deps), mean(kaps), mean(fits))
            print(f"  beta1={b1:.2f}: deploy={mean(deps)*100:.2f}±{spread(deps)*100:.2f}%  "
                  f"kappa->{mean(kaps):.0f}  probe_coh={mean(pcoh):.3f}  "
                  f"memorized={mean(fits)*100:.1f}%  |u|max_late={mean(umax):.3f}",
                  flush=True)
    json.dump({f"{nf}|{b}": v for (nf, b), v in out.items()},
              open("exp7_results.json", "w"), indent=1)
