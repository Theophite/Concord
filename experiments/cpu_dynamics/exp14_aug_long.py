"""Exp 14: augmentation + longer horizon — does the gate win back its keep?

Exp 13's findings (long window ~ close the gate; gateless friction optimal)
were earned in the NO-diversity corner. The standing claim from exps 10/12:
augmentation repairs the coherence signal, making the gate an asset. This
tests the interaction at double the horizon (160ep — twice the memorization
opportunity, schedule stretched accordingly): if the claim is right, the
corner's winners (gateless, long window) should LOSE to the gated default
under aug, and stacking mixup's dilution on top of crops should set a new
regime record.

Arms (4k, 30% label noise, pad-2 crop aug, NS drive, lr 1e-2, 160ep, 3 seeds):
  gated        kappa=150, default window      (the 96.31@80ep config, longer)
  long_window  kappa=150, 64-epoch window     (exp-13 corner winner)
  gateless     C*=0, F=1.5                    (exp-13 plateau config)
  full_stack   gated kappa=150 + mixup        (crop + chord dilution + cascade)
plus clean+aug gated kappa=0 at 160ep (ceiling check vs 97.58@80ep).
"""
import json
import math

import torch
import torch.nn.functional as F

from concord_ref import swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, BATCH, SEEDS
from exp9_muon import MuonConcord
from exp10_aug_ablation import augment

torch.set_num_threads(4)
EPOCHS, LR = 160, 1e-2
SPE = 4000 // 128


class GatelessMuon(MuonConcord):
    @torch.no_grad()
    def step(self):
        self.Cstar = 0.0
        super().step()


def run(arm, nf, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    kappa = 0.0 if nf == 0.0 else 150.0
    kw = dict(lr=LR, total_steps=steps, gate=True, kappa=kappa, noise=False,
              generator=torch.Generator().manual_seed(seed + 10))
    if arm == "gateless":
        opt = GatelessMuon(net, **kw)
    elif arm == "long_window":
        opt = MuonConcord(net, alpha_v=1.0 / (2 * SPE * 64), **kw)
    else:
        opt = MuonConcord(net, **kw)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(x[idx], gen)
            yb = y[idx]
            if arm == "full_stack":
                j = torch.randperm(len(xb), generator=gen)
                lam = float(torch.rand((), generator=gen))
                out = net(lam * xb + (1 - lam) * xb[j])
                loss = (lam * F.cross_entropy(out, yb)
                        + (1 - lam) * F.cross_entropy(out, yb[j]))
            else:
                loss = F.cross_entropy(net(xb), yb)
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
    out = {}
    print("clean + aug, 160ep (ref: gated 97.58@80ep):")
    rs = [run("gated", 0.0, s, data) for s in SEEDS]
    accs = [r[0] for r in rs]
    out["clean_gated"] = (mean(accs), spread(accs))
    print(f"  gated       acc={mean(accs)*100:.2f}±{spread(accs)*100:.2f}%", flush=True)
    print("30% noise + aug, 160ep (refs @80ep: gated 96.31, no-aug stack 93.15):")
    for arm in ("gated", "long_window", "gateless", "full_stack"):
        rs = [run(arm, 0.30, s, data) for s in SEEDS]
        if any(r is None for r in rs):
            print(f"  {arm:11s} DIVERGED")
            out[arm] = None
            continue
        accs = [r[0] for r in rs]
        fits = [r[1] for r in rs]
        out[arm] = (mean(accs), spread(accs), mean(fits))
        print(f"  {arm:11s} acc={mean(accs)*100:.2f}±{spread(accs)*100:.2f}%"
              f"  memorized={mean(fits)*100:.1f}%", flush=True)
    json.dump(out, open("exp14_results.json", "w"), indent=1)
