"""Exp 12: the muon dissipation curve — does the NS5 drive want a MUCH higher λ?

MUON_DRIVE.md §11: in steady state the drained incoherent power balances the
injected, so λ* = (lr·κ)* scales with the DRIVE's noise-energy injection rate.
NS5 writes every singular direction at equal magnitude — noise amplified to
signal strength instead of suppressed by 1/√v̂ — so the muon λ* should sit
10–100× above the v̂ winner's, plausibly near the Wiener point λ=1 (where the
friction step IS the per-element MMSE filter u ← coh·u). Gate 1's κ sweep was
v̂-calibrated and never visited that regime; neither did exp 9's (grid edge
κ=400 ≡ λ=0.4 at peak lr 1e-3).

This extends the exp-5 oracle protocol (4k MNIST × 25 ep, σ=0, fixed C*, same
SEEDS — old cells stay comparable) along the λ axis with the min-leak servo
floor in place (min_leak=0.1: the valve can't slam shut at λ≥1, no λ>1
ringing; bit-exact no-op on the whole old grid):

    muon arms   κ ∈ {0, 100, 250, 500, 1000, 1500}   (peak-λ 0…1.5)
    v̂ arms     κ ∈ {500, 1000, 1500}                 (peak-λ 0.5…1.5;
                κ ≤ 400 already in exp5_results.json)
    noise ρ ∈ {0, 10%, 30%, 45%}

Sharpest discriminating cell: CLEAN muon. exp 9 found muon-clean κ*=0 on the
old grid, but the injection argument says the whitened-noise fluctuation is
there even with clean labels — muon-clean should want substantial λ where
v̂-clean wants ~0. Prediction table (§11): muon's deploy-acc curve keeps
rising past the old grid edge; λ*(muon) ≫ λ*(v̂) at every ρ; the high-λ muon
arms close (or beat) the gap to the v̂ winner.

λ here is quoted at PEAK lr (κ·1e-3); the cosine schedule scales the realized
λ_t with lr_t, exactly as in the kernels.
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
MUON_KAPPAS = (0.0, 100.0, 250.0, 500.0, 1000.0, 1500.0)
VHAT_KAPPAS = (500.0, 1000.0, 1500.0)      # extension; <=400 lives in exp5


class MuonLeak(ConcordRef):
    """The settled muon drive (exp 9 c-sweep: C_BLEND=0 — per-step NS5(ĝ),
    the chase is the manifold EMA) with the min-leak-floored dissipation.
    No v̂, no cap, σ off — the drive and the friction are the only players."""

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
            gn = g / g.norm().clamp_min(1e-12)
            step_ = math.sqrt(max(g.shape)) * ns5(gn)
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


def run(drive, nf, kappa, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    cls = MuonLeak if drive == "muon" else ConcordRef
    opt = cls(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
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
        dep = accuracy(net, xte, yte)
    return dep, fit, opt.mean_coh()


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    results = {}
    t0 = time.time()
    cells = [("muon", nf, k) for nf in NOISES for k in MUON_KAPPAS] \
          + [("vhat", nf, k) for nf in NOISES for k in VHAT_KAPPAS]
    for drive, nf, k in cells:
        rs = [run(drive, nf, k, s, data) for s in SEEDS]
        if any(r is None for r in rs):
            results[(drive, nf, k)] = None
            print(f"{drive} noise={nf:.0%} kappa={k:5.0f} (lam={k*LR:.2f}): "
                  f"DIVERGED   [{time.time()-t0:.0f}s]", flush=True)
        else:
            m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
            mf, mc = mean([r[1] for r in rs]), mean([r[2] for r in rs])
            results[(drive, nf, k)] = (m, sp, mf, mc)
            print(f"{drive} noise={nf:.0%} kappa={k:5.0f} (lam={k*LR:.2f}): "
                  f"deploy={m*100:.2f}±{sp*100:.2f}%  memorized={mf*100:.1f}%  "
                  f"coh={mc:.3f}   [{time.time()-t0:.0f}s]", flush=True)
        json.dump({f"{d}|{n}|{kk}": v for (d, n, kk), v in results.items()},
                  open("exp12_results.json", "w"), indent=1)

    print("\nlam* (deploy-acc argmax over THIS grid) by drive x noise:")
    for drive, ks in (("muon", MUON_KAPPAS), ("vhat", VHAT_KAPPAS)):
        for nf in NOISES:
            ok = [k for k in ks if results.get((drive, nf, k))]
            if not ok:
                continue
            best = max(ok, key=lambda k: results[(drive, nf, k)][0])
            r = results[(drive, nf, best)]
            print(f"  {drive} noise={nf:.0%}: lam*={best*LR:.2f} "
                  f"(deploy {r[0]*100:.2f}%)")
