"""Exp 6: autotuning the dissipation from an online noise estimate.

Exp 5 gave the open-loop curve kappa*(noise) — but it is indexed by ground-truth
label-noise fraction, which a trainer doesn't know. This closes the loop:

  Phase A (the meter): run at fixed kappa and measure an online noise
  statistic eta from optimizer-internal quantities only:
      per layer:  gbar = EMA_b(g)  (bias-corrected),  m2 = EMA_b(||g||^2)
      signal fraction  s = (||gbar||^2/m2 - floor) / (1 - floor),
      floor = (1-b)/(1+b)   (the pure-noise EMA energy floor)
      eta = 1 - s, gradient-energy-weighted across layers.
  This is the same quantity the telescope/v-hat pair estimates per weight,
  measured per layer and kappa-independently (it reads the raw gradient
  stream, not the velocity, so the controller doesn't chase its own tail).

  Phase B (the map): least-squares fit kappa(eta) = clip(a*eta + c, 0, 400)
  through (eta measured at each rho, kappa* from exp 5).

  Phase C (closed loop): kappa_t = map(eta_t) every step (kappa held at 0
  during warmup while the init mass consolidates), across all noise levels,
  vs the exp-5 oracle and the fixed-kappa extremes.

Caveat: the map is calibrated on this task; the transferable object is the
procedure. The statistic conflates label noise with late-training minibatch
noise (near interpolation, clean gradients also decohere) — Phase C shows
what that does in practice.
"""
import json
import time

import torch

from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net

torch.set_num_threads(4)
SUBSET, EPOCHS, BATCH, LR = 4000, 25, 128, 1e-3
SEEDS = (0, 1, 2)
NOISES = (0.0, 0.10, 0.20, 0.30, 0.45)
KAPPA_MAX = 400.0
ORACLE = {0.0: 0.0, 0.10: 100.0, 0.20: 200.0, 0.30: 400.0, 0.45: 400.0}


class AutoConcord(ConcordRef):
    """ConcordRef + online gradient-noise meter + optional kappa controller."""

    def __init__(self, *a, beta_s=0.99, kappa_map=None, **k):
        super().__init__(*a, **k)
        self.beta_s = beta_s
        self.kappa_map = kappa_map
        for st in self.swapped:
            st["gbar"] = torch.zeros_like(st["p"])
            st["m2"] = 0.0
        self.eta_hist, self.kappa_hist = [], []

    def _noise_stat(self):
        b = self.beta_s
        bc = 1.0 - b ** (self.t + 1)
        floor = (1 - b) / (1 + b)
        num = den = 0.0
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            g = g.float()
            st["gbar"].mul_(b).add_(g, alpha=1 - b)
            st["m2"] = b * st["m2"] + (1 - b) * float((g * g).sum())
            m2 = st["m2"] / bc
            if m2 <= 0:
                continue
            gb2 = float((st["gbar"] / bc).pow(2).sum())
            s = (gb2 / m2 - floor) / (1 - floor)
            s = min(max(s, 0.0), 1.0)
            num += (1.0 - s) * m2          # energy-weighted incoherent fraction
            den += m2
        return num / den if den > 0 else None

    @torch.no_grad()
    def step(self):
        eta = self._noise_stat()
        if eta is not None:
            self.eta_hist.append(eta)
            if self.kappa_map is not None:
                self.kappa = self.kappa_map(eta) if self.t >= self.warmup else 0.0
            self.kappa_hist.append(self.kappa)
        super().step()


def make_run(noise_frac, seed, data):
    xtr, ytr, xte, yte = data
    gen = torch.Generator().manual_seed(seed + 100)
    sub = torch.randperm(len(xtr), generator=gen)[:SUBSET]
    x, y = xtr[sub], ytr[sub].clone()
    flip = torch.rand(len(y), generator=gen) < noise_frac
    if flip.any():
        y[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=gen)
    return x, y, flip, gen, xte, yte


def train(opt, net, x, y, gen):
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def run(noise_frac, seed, data, kappa=50.0, kappa_map=None):
    x, y, flip, gen, xte, yte = make_run(noise_frac, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = AutoConcord(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                      noise=False, kappa_map=kappa_map,
                      generator=torch.Generator().manual_seed(seed + 10))
    train(opt, net, x, y, gen)
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        dep = accuracy(net, xte, yte)
    n = len(opt.eta_hist)
    eta_mid = sum(opt.eta_hist[n // 8: n // 2]) / max(1, (n // 2 - n // 8))
    eta_late = sum(opt.eta_hist[-n // 5:]) / max(1, n // 5)
    kap_mean = sum(opt.kappa_hist[n // 2:]) / max(1, n - n // 2)
    return dep, fit, eta_mid, eta_late, kap_mean


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2

    # ── Phase A: does the meter discriminate? ─────────────────────────
    print("Phase A: online noise statistic at fixed kappa=50")
    eta_by_rho = {}
    for nf in NOISES:
        mids, lates = [], []
        for seed in SEEDS:
            _, _, em, el, _ = run(nf, seed, data, kappa=50.0)
            mids.append(em)
            lates.append(el)
        eta_by_rho[nf] = mean(mids)
        print(f"  noise={nf:.0%}: eta_mid={mean(mids):.3f}±{spread(mids):.3f}  "
              f"eta_late={mean(lates):.3f}±{spread(lates):.3f}", flush=True)

    # ── Phase B: fit kappa(eta) through (eta(rho), kappa*(rho)) ───────
    xs = [eta_by_rho[nf] for nf in NOISES]
    ys = [ORACLE[nf] for nf in NOISES]
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    a = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys)) / \
        max(1e-12, sum((xi - mx) ** 2 for xi in xs))
    c = my - a * mx
    print(f"\nPhase B: kappa(eta) = clip({a:.0f}*eta + {c:.0f}, 0, {KAPPA_MAX:.0f})")
    kmap = lambda eta: min(max(a * eta + c, 0.0), KAPPA_MAX)
    for nf in NOISES:
        print(f"  rho={nf:.0%}: eta={eta_by_rho[nf]:.3f} -> kappa={kmap(eta_by_rho[nf]):.0f}"
              f"  (oracle {ORACLE[nf]:.0f})")

    # ── Phase C: closed loop across noise levels ──────────────────────
    print("\nPhase C: closed-loop autotune vs oracle (deploy acc, 3 seeds)")
    results = {}
    for nf in NOISES:
        deps, fits, kaps = [], [], []
        for seed in SEEDS:
            d, f, _, _, km = run(nf, seed, data, kappa=0.0, kappa_map=kmap)
            deps.append(d)
            fits.append(f)
            kaps.append(km)
        results[nf] = (mean(deps), spread(deps), mean(fits), mean(kaps))
        print(f"  noise={nf:.0%}: autotune deploy={mean(deps)*100:.2f}±{spread(deps)*100:.2f}%  "
              f"memorized={mean(fits)*100:.1f}%  settled kappa~{mean(kaps):.0f} "
              f"(oracle kappa*={ORACLE[nf]:.0f})", flush=True)
    with open("exp6_results.json", "w") as f:
        json.dump({"eta_by_rho": {str(k): v for k, v in eta_by_rho.items()},
                   "map": [a, c],
                   "autotune": {str(k): v for k, v in results.items()}}, f, indent=1)
