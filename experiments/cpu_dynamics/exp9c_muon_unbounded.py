"""Exp 9c: does the Muon drive obsolete the trust region and step cap?

The v-hat arm's denominator IS the trust region (winner: v_proxy = delta^2 *
v-hat, delta^2 = 1) and the step cap (+-10) guards its tails. The exp-9 Muon
arm already ran with neither — NS5's output is spectrally bounded by
construction (||gamma*NS(X)||_2 = gamma regardless of gradient magnitude), so
the 94.88 / 92.02 results are capless. This experiment supplies the receipts:

  (a) tail instrumentation — per-element |step| RMS / p99.9 / max over
      training for the NS drive, vs the v-hat drive's PRE-clamp distribution
      and its cap-binding frequency;
  (b) lr stability sweep — capless NS drive vs capped v-hat drive at
      kappa = 0 (friction out of the picture; its lr*kappa < 2 ceiling is a
      separate, drive-independent constraint), clean MNIST.
"""
import json
import math

import torch

from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, SEEDS
from exp9_muon import MuonConcord, ns5

torch.set_num_threads(4)
LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


class InstrumentedMuon(MuonConcord):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.step_rms, self.step_max, self.step_p999 = [], 0.0, []

    @torch.no_grad()
    def step(self):
        for st in self.swapped:
            g = st["p"].grad
            if g is None:
                continue
            gn = g.float() / g.float().norm().clamp_min(1e-12)
            s = math.sqrt(max(g.shape)) * ns5(gn)
            self.step_rms.append(float(s.pow(2).mean().sqrt()))
            self.step_max = max(self.step_max, float(s.abs().max()))
            self.step_p999.append(float(s.abs().flatten().kthvalue(
                int(0.999 * s.numel()))[0]))
        super().step()


class InstrumentedVhat(ConcordRef):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.step_rms, self.step_max, self.step_p999 = [], 0.0, []
        self.clip_frac = []

    @torch.no_grad()
    def step(self):
        # measure the PRE-clamp v-hat step (one-step-lagged v-hat is fine for
        # tail statistics; the kernel uses same-step v-hat)
        for st in self.swapped:
            g = st["p"].grad
            if g is None or float(st["v_row"].sum()) <= 0:
                continue
            vhat = (st["v_row"][:, None] * st["v_col"][None, :]
                    / st["v_row"].sum().clamp_min(1e-30))
            s = g.float() / (vhat + self.eps).sqrt()
            self.step_rms.append(float(s.pow(2).mean().sqrt()))
            self.step_max = max(self.step_max, float(s.abs().max()))
            self.step_p999.append(float(s.abs().flatten().kthvalue(
                int(0.999 * s.numel()))[0]))
            self.clip_frac.append(float((s.abs() > self.cap).float().mean()))
        super().step()


def run(arm, lr, seed, data, instrument=False, nf=0.0, kappa=0.0):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    cls = ((InstrumentedMuon if instrument else MuonConcord)
           if arm == "muon" else
           (InstrumentedVhat if instrument else ConcordRef))
    opt = cls(net, lr=lr, total_steps=steps, gate=True, kappa=kappa,
              noise=False, generator=torch.Generator().manual_seed(seed + 10))
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            if not math.isfinite(float(loss.detach())):
                return None, opt
            opt.zero_grad()
            loss.backward()
            opt.step()
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        return (accuracy(net, xte, yte), fit), opt


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2

    print("(a) per-element |step| tails over training (seed 0; clean and 30% noise):")
    for nf, kap_m, kap_v in ((0.0, 0.0, 0.0), (0.30, 100.0, 400.0)):
        r, om = run("muon", 1e-3, 0, data, instrument=True, nf=nf, kappa=kap_m)
        r2, ov = run("vhat", 1e-3, 0, data, instrument=True, nf=nf, kappa=kap_v)
        print(f"  rho={nf:.0%}  NS drive:    rms={mean(om.step_rms):.2f}  "
              f"p99.9={mean(om.step_p999):.2f}  max={om.step_max:.1f}  (no cap)")
        print(f"  rho={nf:.0%}  v-hat drive: rms={mean(ov.step_rms):.2f}  "
              f"p99.9={mean(ov.step_p999):.2f}  max={ov.step_max:.1f}  "
              f"(pre-clamp; cap=10 binds on {mean(ov.clip_frac)*100:.2f}% of elements)",
              flush=True)

    print("\n(b) lr stability sweep, kappa=0, clean (deploy acc; DIV = diverged):")
    out = {}
    for arm in ("vhat", "muon"):
        row = []
        for lr in LRS:
            rs = [run(arm, lr, s, data)[0] for s in SEEDS]
            if any(r is None for r in rs):
                row.append("DIV")
                out[(arm, lr)] = None
            else:
                accs = [r[0] for r in rs]
                row.append(f"{mean(accs)*100:.2f}±{spread(accs)*100:.2f}")
                out[(arm, lr)] = (mean(accs), spread(accs))
        print(f"  {arm:5s}: " + "  ".join(f"lr={lr:g}: {v}" for lr, v in zip(LRS, row)),
              flush=True)
    json.dump({f"{a}|{lr}": v for (a, lr), v in out.items()},
              open("exp9c_results.json", "w"), indent=1)
