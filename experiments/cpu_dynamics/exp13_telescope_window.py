"""Exp 13: the telescope window in epoch units — sweeping alpha_v.

alpha_v has been 0.001 in every experiment of this campaign and (per the
records) the repo's history — an ABSOLUTE window of 1/(2*alpha_v) = 500 steps
that means 16 epochs on this protocol and a quarter-epoch on a 2k-image bs1
SDXL run. The matched-filter argument says the natural unit is the data
revisit period: window >= 1 epoch so every example votes before motion counts
as drift; window >> epoch buys averaging at the price of lag. This sweeps the
window in epoch units. C* is recomputed per alpha_v automatically
(compute_drift_cancel_C is rate-coupled in the ref). Watch-items: the
telescope amplitude d* ~ 1/alpha_v (int8-budget analogue), and windows longer
than the run never fill.

Protocol: 4k x 80ep (2480 steps; 31 steps/epoch), NS drive, lr 1e-2,
kappa: clean 0 / 30%-noise 150, 3 seeds. window_epochs -> alpha_v via
alpha_v = 1/(2 * 31 * E).
"""
import json
import math

import torch
import torch.nn.functional as F

from concord_ref import swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, BATCH, SEEDS
from exp9_muon import MuonConcord

torch.set_num_threads(4)
EPOCHS, LR = 80, 1e-2
SPE = 4000 // 128                      # 31 steps/epoch
WINDOWS_EP = (1, 4, 16, 64)            # 16 ~ the historical default (500 steps)
REGIMES = {0.0: 0.0, 0.30: 150.0}


def run(window_ep, nf, kappa, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    alpha_v = 1.0 / (2.0 * SPE * window_ep)
    opt = MuonConcord(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                      noise=False, alpha_v=alpha_v,
                      generator=torch.Generator().manual_seed(seed + 10))
    cohs, dmax = [], 0.0
    t = 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(net(x[idx]), y[idx])
            if not math.isfinite(float(loss.detach())):
                return None
            opt.zero_grad()
            loss.backward()
            opt.step()
            if t >= steps // 2:
                cohs.append(opt.mean_coh())
                if t % 100 == 0:
                    dmax = max(dmax, max(float((st["S"] - st["A"]).abs().max())
                                         for st in opt.swapped))
            t += 1
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        dep = accuracy(net, xte, yte)
    return dep, fit, sum(cohs) / max(1, len(cohs)), dmax


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for nf, kappa in REGIMES.items():
        print(f"rho={nf:.0%} (kappa={kappa:.0f}):")
        for w in WINDOWS_EP:
            rs = [run(w, nf, kappa, s, data) for s in SEEDS]
            if any(r is None for r in rs):
                print(f"  window={w:3d}ep  DIVERGED")
                out[(nf, w)] = None
                continue
            deps = [r[0] for r in rs]
            fits = [r[1] for r in rs]
            cohs = [r[2] for r in rs]
            dmx = [r[3] for r in rs]
            out[(nf, w)] = (mean(deps), spread(deps), mean(fits), mean(cohs), mean(dmx))
            print(f"  window={w:3d}ep (a_v={1/(2*SPE*w):.5f}): "
                  f"deploy={mean(deps)*100:.2f}±{spread(deps)*100:.2f}%  "
                  f"memorized={mean(fits)*100:.1f}%  late_coh={mean(cohs):.3f}  "
                  f"|d|max={mean(dmx):.3f}", flush=True)
    json.dump({f"{n}|{w}": v for (n, w), v in out.items()},
              open("exp13_results.json", "w"), indent=1)
