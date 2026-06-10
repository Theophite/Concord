"""Exp 11b: native Muon on the same fine lr grid — apportioning the flatness.

Exp 11 showed the Concord-NS drive is lr-insensitive across a decade. Two
candidate sources: NS normalization (the step magnitude never sees the
gradient scale) and the cascade's regulation (gate/friction/averaging absorb
excess step heat). Native Muon has the first and not the second, so its curve
apportions the credit: equally flat -> mostly Muon's normalization; sharper ->
the cascade's feedback is doing real work.

Same protocol as exp 11: 4k x 25ep, clean, aug off/on, 3 seeds, deploy n/a
(native ships live weights).
"""
import json
import math

import torch
import torch.nn.functional as F

from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, BATCH, SEEDS
from exp9_muon import MuonNative
from exp10_aug_ablation import augment
from exp11_ns_lr import EPOCHS, LRS

torch.set_num_threads(4)


def run(lr, seed, data, aug):
    x, y, flip, gen, xte, yte = make_run(0.0, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = MuonNative(net, lr=lr, total_steps=steps)
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
    return accuracy(net, xte, yte)


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for aug in (False, True):
        print(f"native Muon, aug={'on' if aug else 'off'} (25ep, live weights):")
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
              open("exp11b_results.json", "w"), indent=1)
