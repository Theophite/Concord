"""Exp 4: the regime axis — MNIST with injected label noise.

The repo's session notes claim the fluctuation/dissipation pair only pays off
on tasks with nonzero Bayes error (noisy/heterogeneous gradient streams), and
that on clean tasks v-hat/gating is idle-to-harmful. Exp 3 confirmed the
clean half (bare > dissip > winner on clean MNIST). This injects Bayes error
the way the repo's own cifar_vmode_fork does: a fraction of train labels is
resampled uniformly. Train on the noisy set, evaluate on the CLEAN test set.

10k train subset, 30% label noise, 6 epochs (468 steps), same arms/schedule
as exp 3.
"""
import time

import torch

from concord_ref import (ConcordRef, adamw_with_winner_schedule,
                         compute_drift_cancel_C, swap_to_deploy)
from exp3_mnist import load_mnist, make_net, accuracy

torch.set_num_threads(4)
EPOCHS, BATCH, LR, SEEDS = 6, 128, 1e-3, (0, 1, 2)
SUBSET, NOISE_FRAC = 10_000, 0.30


def run(arm, seed, data):
    xtr, ytr, xte, yte = data
    gen = torch.Generator().manual_seed(seed + 100)
    sub = torch.randperm(len(xtr), generator=gen)[:SUBSET]
    x, y = xtr[sub], ytr[sub].clone()
    flip = torch.rand(len(y), generator=gen) < NOISE_FRAC
    y[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=gen)

    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    kw = dict(lr=LR, total_steps=steps,
              generator=torch.Generator().manual_seed(seed + 10))
    if arm == "adamw":
        opt, set_lr = adamw_with_winner_schedule(net.parameters(), LR, steps)
    elif arm == "bare":
        opt = ConcordRef(net, gate=False, kappa=0.0, noise=False, **kw)
    elif arm == "dissip":
        opt = ConcordRef(net, gate=True, kappa=50.0, noise=False, **kw)
    elif arm == "winner":
        opt = ConcordRef(net, gate=True, kappa=50.0, noise=True, **kw)
    elif arm == "winner_c2":
        opt = ConcordRef(net, gate=True, kappa=50.0, noise=True, **kw)
        opt.Cstar = 2.0 * compute_drift_cancel_C()
    t = 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            if arm == "adamw":
                set_lr(t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t += 1
    fit_noise = accuracy(net, x[flip], y[flip])   # memorization of wrong labels
    acc_live = accuracy(net, xte, yte)
    if arm == "adamw":
        return acc_live, acc_live, fit_noise
    with swap_to_deploy(opt):
        acc_dep = accuracy(net, xte, yte)
    return acc_live, acc_dep, fit_noise


if __name__ == "__main__":
    data = load_mnist()
    print(f"train={SUBSET} noise={NOISE_FRAC:.0%} steps/run="
          f"{EPOCHS * (SUBSET // BATCH)}  (clean-test accuracy)")
    for arm in ("adamw", "bare", "dissip", "winner", "winner_c2"):
        lives, deps, fits = [], [], []
        t0 = time.time()
        for seed in SEEDS:
            al, ad, fn = run(arm, seed, data)
            lives.append(al)
            deps.append(ad)
            fits.append(fn)
        mean = lambda v: sum(v) / len(v)
        spread = lambda v: (max(v) - min(v)) / 2
        print(f"{arm:10s} live={mean(lives)*100:.2f}±{spread(lives)*100:.2f}%  "
              f"deploy={mean(deps)*100:.2f}±{spread(deps)*100:.2f}%  "
              f"noise-memorized={mean(fits)*100:.1f}%  ({time.time()-t0:.0f}s)")
