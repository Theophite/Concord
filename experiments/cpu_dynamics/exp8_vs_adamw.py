"""Exp 8: the head-to-head — fully autotuned Concord vs AdamW, variable noise.

Concord arm = the complete package as now shipped: fixed-C* gate, one probe
(epochs 3-8 at kappa=50, beta1=0) committing BOTH kappa (exp-6 table) and
beta1 (0.1 iff probe coh >= 0.35). AdamW arms: wd=0 (the baseline used
throughout) and wd=0.01 (typical default, fairness arm). Same lr (1e-3
peak), same warmup+cosine schedule, same model/data, 3 seeds.

Regime: 4k train subset x 25 epochs (overfitting room), label noise
rho in {0, 5, 10, 20, 30, 45}%. Metric: clean-test accuracy (Concord:
deploy weights), plus fraction of wrong labels memorized.
"""
import json

import torch

from concord_ref import ConcordRef, adamw_with_winner_schedule, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, EPOCHS, BATCH, LR, SEEDS
from exp7_beta1_sweep import kappa_from_coh

torch.set_num_threads(4)
RHOS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.45)
B1_ON, B1_THRESH = 0.10, 0.35


def run(arm, nf, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    spe = len(x) // BATCH
    steps = EPOCHS * spe
    if arm.startswith("adamw"):
        wd = 0.01 if arm == "adamw_wd" else 0.0
        opt, set_lr = adamw_with_winner_schedule(net.parameters(), LR, steps)
        for grp in opt.param_groups:
            grp["weight_decay"] = wd
    else:
        opt = ConcordRef(net, lr=LR, total_steps=steps, gate=True, kappa=50.0,
                         noise=False, beta1=0.0,
                         generator=torch.Generator().manual_seed(seed + 10))
    p0, p1 = 3 * spe, 8 * spe
    cohs, t, committed = [], 0, (None, None)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
            if arm.startswith("adamw"):
                set_lr(t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if arm == "concord":
                if p0 <= t < p1:
                    cohs.append(opt.mean_coh())
                elif t == p1:
                    c = sum(cohs) / len(cohs)
                    opt.kappa = kappa_from_coh(c)
                    opt.beta1 = B1_ON if c >= B1_THRESH else 0.0
                    committed = (opt.kappa, opt.beta1)
            t += 1
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    if arm == "concord":
        with swap_to_deploy(opt):
            acc = accuracy(net, xte, yte)
    else:
        acc = accuracy(net, xte, yte)
    return acc, fit, committed


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for arm in ("adamw", "adamw_wd", "concord"):
        print(f"{arm}:")
        for nf in RHOS:
            rs = [run(arm, nf, s, data) for s in SEEDS]
            accs = [r[0] for r in rs]
            fits = [r[1] for r in rs]
            out[(arm, nf)] = (mean(accs), spread(accs), mean(fits))
            extra = ""
            if arm == "concord":
                ks = [r[2][0] for r in rs]
                bs = [r[2][1] for r in rs]
                extra = f"  kappa->{mean(ks):.0f} beta1->{mean(bs):.2f}"
            print(f"  noise={nf:.0%}: acc={mean(accs)*100:.2f}±{spread(accs)*100:.2f}%  "
                  f"memorized={mean(fits)*100:.1f}%{extra}", flush=True)
    json.dump({f"{a}|{n}": v for (a, n), v in out.items()},
              open("exp8_results.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    style = {"adamw": ("tab:red", "AdamW (wd=0)"),
             "adamw_wd": ("tab:orange", "AdamW (wd=0.01)"),
             "concord": ("tab:blue", "Concord autotuned (deploy)")}
    xs = [r * 100 for r in RHOS]
    for arm, (c, lbl) in style.items():
        ax1.errorbar(xs, [out[(arm, r)][0] * 100 for r in RHOS],
                     yerr=[out[(arm, r)][1] * 100 for r in RHOS],
                     color=c, marker="o", ms=4, label=lbl)
        ax2.plot(xs, [out[(arm, r)][2] * 100 for r in RHOS],
                 color=c, marker="o", ms=4, label=lbl)
    ax1.set_xlabel("label noise ρ (%)")
    ax1.set_ylabel("clean-test accuracy (%)")
    ax1.set_title("accuracy vs noise")
    ax1.legend(fontsize=8)
    ax2.axhline(10, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel("label noise ρ (%)")
    ax2.set_ylabel("wrong labels memorized (%)")
    ax2.set_title("noise memorization")
    ax2.legend(fontsize=8)
    fig.suptitle("Autotuned Concord (κ+β1 from one probe) vs AdamW — MNIST 4k×25ep",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("exp8_vs_adamw.png", dpi=130)
    print("saved exp8_vs_adamw.png")
