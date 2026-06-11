"""Exp 20: best-of-both synthesis — long telescope windows under the min-leak
floor, at the NS drive's own lr, with augmentation.

The two campaigns each held a piece. Main: NS at its own lr (1e-2) with
crop-aug is the protocol regime (exp 16: 97.51 clean), the telescope window
is the gate's trust timescale (exp 18: 1ep dangerous, long-window plateau ~
the gateless limit), and the gate's exemption at matched F is positive only
clean+high-F (exp 19b: +0.51 clean F=1.5; −2.78 at 10% noise; ~0 with aug).
Muon-drive: the κ=0 arm main never ran (exp 12), and the min-leak servo
floor — WITHOUT which every F=1.5 arm in exps 18/19 ran a NEGATIVE survival
factor on incoherent coordinates (1 − 1.5·(1−coh) < 0 for coh < 1/3:
sign-flip ringing, the regime exp 12's tooltip-fix flagged as meaningless).

So two things are genuinely open: (a) do main's high-F gate-ablation
verdicts survive the floor? (b) does EXTENDING the window past epoch length
(the user's standing trust-timescale intuition: every example should vote
before motion counts as drift) repair the gate's noisy deficit once the
floor stops the ringing?

Grid (NS drive = MuonLeak [floored evap], lr 1e-2, pad-2 crop-aug, 4k x 25ep,
3 seeds; window_ep -> alpha_v = 1/(2*31*E); C* recomputed per alpha_v by
ConcordRef):

    gated   F=1.5 : windows {1, 4, 16, 64, 256} x noise {0, 30%}
    gateless F=1.5: default window               x noise {0, 30%}   (Cstar=0)
    gated   F=0   : default window               x noise {0, 30%}   (exp-12 control)

References to beat (pre-floor, 80-160ep arms, exp 19/19b): clean F=1.5 gated
96.25 / gateless 95.74; 30%+aug F=1.5 gated 96.31 / gateless 96.15; exp 16
25ep aug F=0: 97.51.
"""
import json
import math
import time

import torch
import torch.nn.functional as F

from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, BATCH, SEEDS
from exp10_aug_ablation import augment
from exp12_muon_lambda import MuonLeak
from concord_ref import swap_to_deploy

torch.set_num_threads(4)
EPOCHS, LR = 25, 1e-2
SPE = 4000 // 128
WINDOWS_EP = (1, 4, 16, 64, 256)
F_HI = 1.5


class GatelessMuonLeak(MuonLeak):
    @torch.no_grad()
    def step(self):
        self.Cstar = 0.0
        super().step()


def run(arm, window_ep, nf, friction_F, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    cls = GatelessMuonLeak if arm == "gateless" else MuonLeak
    opt = cls(net, lr=LR, total_steps=steps, gate=True,
              kappa=0.0, friction_F=(friction_F if friction_F > 0 else None),
              alpha_v=1.0 / (2.0 * SPE * window_ep), noise=False,
              generator=torch.Generator().manual_seed(seed + 10))
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(x[idx], gen)
            loss = F.cross_entropy(net(xb), y[idx])
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
    cells = ([("gated", w, nf, F_HI) for nf in (0.0, 0.30) for w in WINDOWS_EP]
             + [("gateless", 16, nf, F_HI) for nf in (0.0, 0.30)]
             + [("gated", 16, nf, 0.0) for nf in (0.0, 0.30)])
    for arm, w, nf, fF in cells:
        rs = [run(arm, w, nf, fF, s, data) for s in SEEDS]
        tag = f"{arm:8s} W={w:3d}ep F={fF:.1f} noise={nf:.0%}"
        if any(r is None for r in rs):
            results[(arm, w, nf, fF)] = None
            print(f"{tag}: DIVERGED   [{time.time()-t0:.0f}s]", flush=True)
            continue
        m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
        mf, mc = mean([r[1] for r in rs]), mean([r[2] for r in rs])
        results[(arm, w, nf, fF)] = (m, sp, mf, mc)
        print(f"{tag}: deploy={m*100:.2f}±{sp*100:.2f}%  memorized={mf*100:.1f}%  "
              f"coh={mc:.3f}   [{time.time()-t0:.0f}s]", flush=True)
        json.dump({f"{a}|{w_}|{n}|{f_}": v for (a, w_, n, f_), v in results.items()},
                  open("exp20_results.json", "w"), indent=1)
