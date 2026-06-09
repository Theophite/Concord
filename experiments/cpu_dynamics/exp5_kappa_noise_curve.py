"""Exp 5: the dissipation curve — optimal kappa as a function of noise magnitude.

Exp 4 gave two points (clean -> small kappa, 30% label noise -> kappa ~150).
This sweeps the full grid: label-noise fraction x kappa, in the overfitting
regime (4k subset, 25 epochs) where dissipation actually matters, with the
fluctuation OFF (sigma=0) so the curve is for the dissipation in isolation.
Gate uses the fixed (mass-preserve-corrected) C*. Metric: clean-test accuracy
at the deploy weights; secondary: fraction of wrongly-labeled train examples
memorized.
"""
import json
import time

import torch

from concord_ref import ConcordRef, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net

torch.set_num_threads(4)
SUBSET, EPOCHS, BATCH, LR = 4000, 25, 128, 1e-3
SEEDS = (0, 1, 2)
KAPPAS = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0)
NOISES = (0.0, 0.10, 0.20, 0.30, 0.45)


def run(noise_frac, kappa, seed, data):
    xtr, ytr, xte, yte = data
    gen = torch.Generator().manual_seed(seed + 100)
    sub = torch.randperm(len(xtr), generator=gen)[:SUBSET]
    x, y = xtr[sub], ytr[sub].clone()
    flip = torch.rand(len(y), generator=gen) < noise_frac
    if flip.any():
        y[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=gen)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = ConcordRef(net, lr=LR, total_steps=steps, gate=True, kappa=kappa,
                     noise=False,
                     generator=torch.Generator().manual_seed(seed + 10))
    for _ in range(EPOCHS):
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
    return dep, fit


if __name__ == "__main__":
    data = load_mnist()
    results = {}    # (noise, kappa) -> (mean_dep, spread, mean_fit)
    t0 = time.time()
    for nf in NOISES:
        for k in KAPPAS:
            deps, fits = [], []
            for seed in SEEDS:
                d, f = run(nf, k, seed, data)
                deps.append(d)
                fits.append(f)
            m = sum(deps) / len(deps)
            sp = (max(deps) - min(deps)) / 2
            mf = sum(fits) / len(fits)
            results[(nf, k)] = (m, sp, mf)
            print(f"noise={nf:.0%} kappa={k:5.0f}: deploy={m*100:.2f}±{sp*100:.2f}%"
                  f"  memorized={mf*100:.1f}%   [{time.time()-t0:.0f}s]", flush=True)
    with open("exp5_results.json", "w") as f:
        json.dump({f"{nf}|{k}": v for (nf, k), v in results.items()}, f, indent=1)

    # kappa* per noise level
    print("\nkappa* (deploy-acc argmax) by noise level:")
    kstar = {}
    for nf in NOISES:
        best = max(KAPPAS, key=lambda k: results[(nf, k)][0])
        kstar[nf] = best
        print(f"  noise={nf:.0%}: kappa*={best:.0f} "
              f"(acc {results[(nf, best)][0]*100:.2f}%)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.9))
    xpos = range(len(KAPPAS))
    cmap = plt.cm.viridis
    for i, nf in enumerate(NOISES):
        c = cmap(i / (len(NOISES) - 1))
        accs = [results[(nf, k)][0] * 100 for k in KAPPAS]
        sps = [results[(nf, k)][1] * 100 for k in KAPPAS]
        ax1.errorbar(xpos, accs, yerr=sps, color=c, marker="o", ms=4,
                     label=f"label noise {nf:.0%}")
        fits = [results[(nf, k)][2] * 100 for k in KAPPAS]
        if nf > 0:
            ax2.plot(xpos, fits, color=c, marker="o", ms=4)
    for ax in (ax1, ax2):
        ax.set_xticks(list(xpos))
        ax.set_xticklabels([f"{k:g}" for k in KAPPAS])
        ax.set_xlabel("κ (dissipation, gf_consol)")
    ax1.set_ylabel("clean-test acc, deploy P (%)")
    ax1.set_title("accuracy vs dissipation, by noise level")
    ax1.legend(fontsize=7)
    ax2.axhline(10, color="gray", lw=0.8, ls=":", label="chance (no memorization)")
    ax2.set_ylabel("wrong labels memorized (%)")
    ax2.set_title("noise memorization vs dissipation")
    ax2.legend(fontsize=7)
    ax3.plot([nf * 100 for nf in NOISES], [kstar[nf] for nf in NOISES],
             marker="o", color="tab:purple")
    ax3.set_xlabel("label noise (%)")
    ax3.set_ylabel("κ* (best dissipation)")
    ax3.set_title("the dissipation curve: κ*(noise)")
    fig.suptitle("MNIST 4k×25ep, fixed-C* gate, σ=0 — dissipation only", fontsize=10)
    fig.tight_layout()
    fig.savefig("exp5_kappa_noise_curve.png", dpi=130)
    print("saved exp5_kappa_noise_curve.png")
