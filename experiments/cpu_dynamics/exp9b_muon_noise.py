"""Exp 9b: the fluctuation for the Muon drive — noise after NS, not before.

Exp 9 ran the NS drive with sigma off, flagging the interaction: NS maps any
input direction to unit spectral weight, so PRE-NS noise doesn't add a small
fluctuation — it rotates the step and gets re-amplified. POST-NS noise is the
faithful transplant of the winner's fluctuation: a perturbation of the step,
sigma in units of the step norm, rising-late schedule as in the winner.

Arms (MuonConcord c=0, per-regime Muon-arm oracle kappa: clean 0, noisy 100):
    none       reference (exp 9: clean 94.88, noisy 92.02)
    post 0.3   step = D + 0.3_t * ||D|| * xi_hat,  D = gamma*NS5(g_hat)
    post 0.6   same at the winner's sigma_peak
    pre  0.6   g_tilde = g + 0.6_t*||g||*xi_hat BEFORE NS  (the control —
               expected pathological)
"""
import json
import math

import torch

from concord_ref import swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, LR, SEEDS
from exp9_muon import MuonConcord, ns5

torch.set_num_threads(4)
REGIMES = {0.0: 0.0, 0.30: 100.0}     # rho -> Muon-arm kappa* (exp 9)


class NoisyMuonConcord(MuonConcord):
    def __init__(self, *a, noise_mode=None, **k):
        super().__init__(*a, **k)
        self.noise_mode = noise_mode   # None | "pre" | "post"

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
            if self.noise_mode == "pre" and self.sigma > 0:
                xi = torch.randn(g.shape, generator=self.gen)
                g = g + xi * (self.sigma * g.norm() / xi.norm().clamp_min(1e-12))
            gn = g / g.norm().clamp_min(1e-12)
            gamma = math.sqrt(max(g.shape))
            step_ = gamma * ns5(gn)
            if self.noise_mode == "post" and self.sigma > 0:
                xi = torch.randn(g.shape, generator=self.gen)
                step_ = step_ + xi * (self.sigma * step_.norm()
                                      / xi.norm().clamp_min(1e-12))
            evap = lr * self.kappa * (1.0 - coh) * u if self.kappa > 0 else 0.0
            u += -lr * step_ - evap
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


def run(mode, sigma, nf, kappa, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = NoisyMuonConcord(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                           noise=(mode is not None), sigma_peak=sigma,
                           noise_mode=mode,
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
        return accuracy(net, xte, yte), fit


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    REF = {0.0: "no-noise ref 94.88±0.01", 0.30: "no-noise ref 92.02±0.28"}
    out = {}
    for nf, kappa in REGIMES.items():
        print(f"rho={nf:.0%} (kappa={kappa:.0f})  [{REF[nf]}]:")
        for mode, sigma in (("post", 0.3), ("post", 0.6), ("pre", 0.6)):
            rs = [run(mode, sigma, nf, kappa, s, data) for s in SEEDS]
            if any(r is None for r in rs):
                print(f"  {mode}-NS sigma={sigma}: DIVERGED")
                out[(mode, sigma, nf)] = None
                continue
            accs = [r[0] for r in rs]
            fits = [r[1] for r in rs]
            out[(mode, sigma, nf)] = (mean(accs), spread(accs), mean(fits))
            print(f"  {mode}-NS sigma={sigma}: deploy={mean(accs)*100:.2f}"
                  f"±{spread(accs)*100:.2f}%  memorized={mean(fits)*100:.1f}%",
                  flush=True)
    json.dump({f"{m}|{s}|{n}": v for (m, s, n), v in out.items()},
              open("exp9b_results.json", "w"), indent=1)
