"""Exp 11: fine lr ablation for the Concord NS drive.

Exp 9c's decade grid put lr* at 1e-2 with a flat top (no-aug, 25ep, kappa=0).
This refines it: fine grid around the peak, with and without the pad-2
random-crop augmentation that exp 10 established as the realistic protocol.
4k x 25 epochs (not 80 — per the protocol note in exp 10, 25ep no-aug ends
near the clean sweet spot, and aug curves are about the peak's LOCATION, not
its long-horizon value), kappa=0, sigma off, 3 seeds.

References: 9c no-aug coarse grid — 94.84 / 95.89 / 96.07 / 95.38 / 95.14 at
lr = 1e-3 / 3e-3 / 1e-2 / 3e-2 / 1e-1.
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
EPOCHS = 25
LRS = (3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2)


def run(lr, seed, data, aug):
    x, y, flip, gen, xte, yte = make_run(0.0, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = MuonConcord(net, lr=lr, total_steps=steps, gate=True, kappa=0.0,
                      noise=False,
                      generator=torch.Generator().manual_seed(seed + 10))
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(x[idx], gen) if aug else x[idx]
            loss = F.cross_entropy(net(xb), y[idx])
            if not math.isfinite(float(loss.detach())):
                return None
            opt.zero_grad()
            loss.backward()
            opt.step()
    with swap_to_deploy(opt):
        return accuracy(net, xte, yte)


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for aug in (False, True):
        print(f"aug={'on' if aug else 'off'} (25ep, kappa=0, deploy acc):")
        best = (None, -1.0)
        for lr in LRS:
            rs = [run(lr, s, data, aug) for s in SEEDS]
            if any(r is None for r in rs):
                print(f"  lr={lr:g}: DIVERGED")
                out[(aug, lr)] = None
                continue
            m, sp = mean(rs), spread(rs)
            out[(aug, lr)] = (m, sp)
            if m > best[1]:
                best = (lr, m)
            print(f"  lr={lr:g}: {m*100:.2f}±{sp*100:.2f}%", flush=True)
        print(f"  -> lr* = {best[0]:g} ({best[1]*100:.2f}%)")
    json.dump({f"{int(a)}|{lr}": v for (a, lr), v in out.items()},
              open("exp11_results.json", "w"), indent=1)
