"""Exp 9: Muon from the packed state — NS-of-EMA-of-NS vs v-hat.

The hypothesis (manifold-Lookahead): the velocity u is already the momentum
buffer, so orthogonalize it. NS5 is a retraction onto the polar manifold;
EMA-then-retract is the canonical momentum-on-a-manifold pattern, idempotence
makes it degrade to exact Muon when directions are stable, and EMA-of-
directions discards the magnitude spectrum even harder than Muon does.

Arms (per regime: clean rho=0, noisy rho=0.30; 4k x 25ep; sigma OFF in all;
dissipation fixed at the per-regime exp-5 oracle so the preconditioner is the
only variable; lr 1e-3, winner schedule, NOT re-tuned per arm):

  concord_vhat   the standard drive: clip(g/sqrt(vhat+eps), +-10)
  concord_muon   drive = sqrt(max(N,K)) * NS5( c*(-u_hat) + g_hat ), c=3;
                 u-blend engages after warmup (early u is init mass, not
                 momentum); vhat not computed at all
  muon_native    faithful Euclidean Muon baseline (M <- beta*M + g,
                 step = sqrt(max(N,K))*NS5(M), beta=0.95), live weights,
                 aux SGD for 1D params, same schedule

AdamW reference at this exact protocol (exp 8): clean 92.78, noisy 89.15.
"""
import json
import math

import torch

from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, LR, SEEDS

torch.set_num_threads(4)
REGIMES = {0.0: 0.0, 0.30: 400.0}          # rho -> oracle kappa (exp 5)
# C_BLEND: weight of the velocity in the NS input. The c-sweep settled it at
# ZERO: blending u back into the drive is a self-reinforcing direction loop
# (|u| grows ~20x, telescope coherence drops 0.48 -> 0.37, consolidation
# throttles; clean acc 94.88 -> 90.93 as c goes 0 -> 3). The Lookahead idiom
# resolves it: the CHASE is the EMA -- positions integrate orthogonalized
# directions, so no Euclidean momentum blend (and no outer retraction on a
# position) is needed. Per-step NS(g) + chase IS the manifold average.
C_BLEND, NS_STEPS, MUON_BETA = 0.0, 5, 0.95


def ns5(G, steps=NS_STEPS, eps=1e-7):
    """Newton-Schulz quintic orthogonalization (the Muon iteration)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class MuonConcord(ConcordRef):
    """ConcordRef with the drive replaced: NS-of-EMA-of-NS, no v-hat."""

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
            # ── the Muon drive: orthogonalize the blended direction ──
            gn = g / g.norm().clamp_min(1e-12)
            if self.t >= self.warmup:                  # early u = init mass
                un = -u / u.norm().clamp_min(1e-12)    # ascent-pointing
                blend = C_BLEND * un + gn
            else:
                blend = gn
            gamma = math.sqrt(max(g.shape))
            step_ = gamma * ns5(blend)
            evap = lr * self.kappa * (1.0 - coh) * u if self.kappa > 0 else 0.0
            u += self.beta1 * coh * u - lr * step_ - evap
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


class MuonNative:
    """Faithful Euclidean Muon, same lr schedule / aux handling as the ref."""

    def __init__(self, model, lr, total_steps, warmup=100, lr_min_frac=0.2):
        self.peak_lr, self.T, self.warmup = lr, total_steps, warmup
        self.lr_min_frac = lr_min_frac
        self.t = 0
        self.swapped = [{"p": p, "M": torch.zeros_like(p)}
                        for p in model.parameters() if p.dim() == 2]
        self.aux = [{"p": p, "m": torch.zeros_like(p)}
                    for p in model.parameters() if p.dim() != 2 and p.requires_grad]

    @torch.no_grad()
    def step(self):
        p_ = min(1.0, self.t / max(1, self.T))
        f = self.lr_min_frac + 0.5 * (1 - self.lr_min_frac) * (1 + math.cos(math.pi * p_))
        warm = min(1.0, (self.t + 1) / self.warmup)
        lr = self.peak_lr * f * warm
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            st["M"].mul_(MUON_BETA).add_(g.float())
            gamma = math.sqrt(max(g.shape))
            st["p"].add_(gamma * ns5(st["M"]), alpha=-lr)
        for st in self.aux:
            if st["p"].grad is None:
                continue
            st["m"].mul_(0.9).add_(st["p"].grad)
            st["p"].add_(st["m"], alpha=-lr)
        self.t += 1

    def zero_grad(self):
        for st in self.swapped + self.aux:
            st["p"].grad = None


def run(arm, nf, kappa, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    if arm == "concord_vhat":
        opt = ConcordRef(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                         noise=False, generator=torch.Generator().manual_seed(seed + 10))
    elif arm == "concord_muon":
        opt = MuonConcord(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                          noise=False, generator=torch.Generator().manual_seed(seed + 10))
    else:
        opt = MuonNative(net, lr=LR, total_steps=steps)
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
    if arm == "muon_native":
        return accuracy(net, xte, yte), fit
    with swap_to_deploy(opt):
        return accuracy(net, xte, yte), fit


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for nf, kappa in REGIMES.items():
        print(f"rho={nf:.0%} (kappa={kappa:.0f} for Concord arms):")
        for arm in ("concord_vhat", "concord_muon", "muon_native"):
            rs = [run(arm, nf, kappa, s, data) for s in SEEDS]
            if any(r is None for r in rs):
                print(f"  {arm:13s} DIVERGED")
                out[(arm, nf)] = None
                continue
            accs = [r[0] for r in rs]
            fits = [r[1] for r in rs]
            out[(arm, nf)] = (mean(accs), spread(accs), mean(fits))
            print(f"  {arm:13s} acc={mean(accs)*100:.2f}±{spread(accs)*100:.2f}%"
                  f"  memorized={mean(fits)*100:.1f}%", flush=True)
    json.dump({f"{a}|{n}": v for (a, n), v in out.items()},
              open("exp9_results.json", "w"), indent=1)
