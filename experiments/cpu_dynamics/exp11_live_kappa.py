"""Exp 11: computing the dissipation curve LIVE — continuous kappa control vs
the exp-5 oracle, including calibration-free laws and a mid-run noise flip.

Exp 6 v2 validated continuous tracking THROUGH A CALIBRATED TABLE (gate-coh ->
kappa, fit to the exp-5 oracle). This asks the next question: can the curve be
computed live, with no per-domain calibration? Laws, per step (all read the
gate's own mean coherence c_t = mean_coh(), the v2 meter; all start kappa=0
and engage after warmup + the init transient):

  oracle  fixed kappa = kappa*(rho)               [exp-5 ground truth]
  table   kappa_t = interp(table, c_t)            [exp-6 v2: calibrated ref]
  servo   integral law, self-referenced: c_ref = mean c over a reference
          window right after the transient; then
              kappa += G * KBOX * (c_ref - c_t) / c_ref,   clip [0, KBOX]
          KBOX = 0.2 * (2/lr) — 10%% of the lr*kappa<2 stability ceiling
          (note: = 400 at lr 1e-3, the oracle's own plateau — the principled
          box, not a fit). Fixed point: kappa rises until the gate's read of
          the stream stops sagging below the run's own early reference.
          Calibration-free; G only sets approach speed.
  trend   setpoint-free: windowed mean of c every W steps; window dropped vs
          previous -> kappa <- kappa*1.15 + 1; else kappa <- kappa*0.97.
          (Raise friction while coherence degrades, relax when stable.)

Caveat measured, not assumed: the gate meter is kappa-DEPENDENT (friction
drains incoherent d_fs, raising c) — the closed loop interacts with its own
meter. The oracle comparison is exactly the test of whether the equilibria
land anywhere useful.

Flip demo: train clean, corrupt 30%% of labels at epoch 12. The shipped
probe-then-commit (probe epochs 3-8, one commit) is structurally wrong here;
live laws can adapt. Arms: commit / table / servo / trend.
"""
import json
import time

import torch

from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net

torch.set_num_threads(4)
SUBSET, EPOCHS, BATCH, LR = 4000, 25, 128, 1e-3
SEEDS = (0, 1)
NOISES = (0.0, 0.10, 0.30)
ORACLE = {0.0: 0.0, 0.10: 100.0, 0.30: 400.0}
KBOX = 0.2 * (2.0 / LR)            # = 400 at lr 1e-3
TRANSIENT = 60                      # steps after warmup before any law engages
REF_WIN = 60                        # servo reference window length (steps)
TREND_W = 20

TABLE = [(0.3865175302951567, 0.0), (0.3144563574303863, 100.0),
         (0.28822555593265003, 200.0), (0.2738887910080212, 400.0),
         (0.2563569223688495, 400.0)]   # exp6_v2_results.json


def table_kappa(c):
    t = TABLE
    if c >= t[0][0]:
        return t[0][1]
    if c <= t[-1][0]:
        return t[-1][1]
    for (c1, k1), (c2, k2) in zip(t, t[1:]):
        if c2 <= c <= c1:
            return k1 + (c1 - c) / (c1 - c2) * (k2 - k1)
    return t[-1][1]


class LiveConcord(ConcordRef):
    """ConcordRef + a continuous kappa law driven by the gate's mean coherence."""

    def __init__(self, *a, law="oracle", oracle_kappa=0.0, **k):
        super().__init__(*a, **k)
        self.law = law
        self.oracle_kappa = oracle_kappa
        self.kappa = oracle_kappa if law == "oracle" else 0.0
        self._engage = self.warmup + TRANSIENT
        self._ref_sum, self._ref_n, self.c_ref = 0.0, 0, None
        self._win, self._prev_win = [], None
        self.kappa_hist, self.coh_hist = [], []

    @torch.no_grad()
    def step(self):
        super().step()                      # records last_coh per layer
        c = self.mean_coh()
        self.coh_hist.append(c)
        t = self.t
        if self.law == "oracle":
            pass
        elif t < self._engage:
            self.kappa = 0.0
        elif self.law == "table":
            self.kappa = table_kappa(c)
        elif self.law == "servo":
            if self.c_ref is None:
                self._ref_sum += c
                self._ref_n += 1
                if self._ref_n >= REF_WIN:
                    self.c_ref = self._ref_sum / self._ref_n
            else:
                self.kappa = min(max(
                    self.kappa + 1e-3 * KBOX * (self.c_ref - c) / max(self.c_ref, 1e-9),
                    0.0), KBOX)
        elif self.law == "trend":
            self._win.append(c)
            if len(self._win) >= TREND_W:
                m = sum(self._win) / len(self._win)
                if self._prev_win is not None:
                    if m < self._prev_win:
                        self.kappa = min(self.kappa * 1.15 + 1.0, KBOX)
                    else:
                        self.kappa = self.kappa * 0.97
                self._prev_win = m
                self._win = []
        elif self.law == "commit":          # shipped probe-then-commit emulation
            if t == self._engage + REF_WIN:
                self.kappa = table_kappa(
                    sum(self.coh_hist[self._engage:t]) / max(1, t - self._engage))
        self.kappa_hist.append(self.kappa)


def make_run(noise_frac, seed, data):
    xtr, ytr, xte, yte = data
    gen = torch.Generator().manual_seed(seed + 100)
    sub = torch.randperm(len(xtr), generator=gen)[:SUBSET]
    x, y = xtr[sub], ytr[sub].clone()
    flip = torch.rand(len(y), generator=gen) < noise_frac
    if flip.any():
        y[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=gen)
    return x, y, flip, gen, xte, yte


def run(noise_frac, seed, data, law, flip_at_epoch=None):
    x, y, flip, gen, xte, yte = make_run(noise_frac, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = LiveConcord(net, lr=LR, total_steps=steps, gate=True,
                      noise=False, law=law, oracle_kappa=ORACLE.get(noise_frac, 0.0),
                      generator=torch.Generator().manual_seed(seed + 10))
    for ep in range(EPOCHS):
        if flip_at_epoch is not None and ep == flip_at_epoch:
            fl = torch.rand(len(y), generator=gen) < 0.30
            y[fl] = torch.randint(0, 10, (int(fl.sum()),), generator=gen)
            flip = fl
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        dep = accuracy(net, xte, yte)
    n = len(opt.kappa_hist)
    k_late = sum(opt.kappa_hist[-n // 5:]) / max(1, n // 5)
    k_mean = sum(opt.kappa_hist[n // 2:]) / max(1, n - n // 2)
    return dep, fit, k_mean, k_late


if __name__ == "__main__":
    t0 = time.time()
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    out = {"static": {}, "flip": {}}

    print(f"=== Phase A: laws vs oracle (static noise) | KBOX={KBOX:.0f} ===", flush=True)
    for nf in NOISES:
        out["static"][nf] = {}
        for law in ("oracle", "table", "servo", "trend"):
            deps, fits, kms, kls = [], [], [], []
            for seed in SEEDS:
                d, f, km, kl = run(nf, seed, data, law)
                deps.append(d); fits.append(f); kms.append(km); kls.append(kl)
            out["static"][nf][law] = [mean(deps), mean(fits), mean(kms), mean(kls)]
            print(f"  rho={nf:.0%} {law:6s}: deploy={mean(deps)*100:.2f}%  "
                  f"mem={mean(fits)*100:.1f}%  kappa(mean/late)={mean(kms):.0f}/{mean(kls):.0f}"
                  f"  (oracle k*={ORACLE[nf]:.0f})", flush=True)

    print(f"\n=== Phase B: mid-run noise flip (clean -> 30% at epoch 12) ===", flush=True)
    for law in ("commit", "table", "servo", "trend"):
        deps, fits, kms, kls = [], [], [], []
        for seed in SEEDS:
            d, f, km, kl = run(0.0, seed, data, law, flip_at_epoch=12)
            deps.append(d); fits.append(f); kms.append(km); kls.append(kl)
        out["flip"][law] = [mean(deps), mean(fits), mean(kms), mean(kls)]
        print(f"  {law:6s}: deploy={mean(deps)*100:.2f}%  mem={mean(fits)*100:.1f}%  "
              f"kappa(mean/late)={mean(kms):.0f}/{mean(kls):.0f}", flush=True)

    with open("exp11_results.json", "w") as f:
        json.dump({"kbox": KBOX, "static": {str(k): v for k, v in out["static"].items()},
                   "flip": out["flip"]}, f, indent=1)
    print(f"\nDONE {(time.time()-t0)/60:.1f} min -> exp11_results.json", flush=True)
