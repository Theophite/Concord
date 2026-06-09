"""Exp 1: how much does Concord move weights, vs AdamW, across LRs over time.

Teacher-student regression with label noise (nonzero Bayes error, so the
gradient stream has a real noise floor — the regime the gate is built for).
Same init for every arm. We track the relative Frobenius displacement of the
2D weights from their init:  ||W_t - W_0|| / ||W_0||  (Concord: live W and
deploy P), plus final train loss.
"""
import math

import torch

from concord_ref import ConcordRef, adamw_with_winner_schedule

torch.set_num_threads(4)
STEPS, BATCH, IN, HID = 800, 64, 32, 64
LRS = [1e-4, 1e-3, 1e-2, 5e-2]
LOG_EVERY = 5


def make_net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(IN, HID), torch.nn.ReLU(),
        torch.nn.Linear(HID, HID), torch.nn.ReLU(),
        torch.nn.Linear(HID, 1))


def disp(params, w0):
    num = sum((p.detach() - w).pow(2).sum() for p, w in zip(params, w0))
    den = sum(w.pow(2).sum() for w in w0)
    return (num / den).sqrt().item()


def run(arm, lr, seed=0):
    net = make_net(seed)
    w2d = [p for p in net.parameters() if p.dim() == 2]
    w0 = [p.detach().clone() for p in w2d]
    torch.manual_seed(seed + 1)
    teacher = torch.nn.Sequential(
        torch.nn.Linear(IN, HID), torch.nn.Tanh(), torch.nn.Linear(HID, 1))
    for p in teacher.parameters():
        p.requires_grad_(False)
    if arm == "adamw":
        opt, set_lr = adamw_with_winner_schedule(net.parameters(), lr, STEPS)
    else:
        opt = ConcordRef(net, lr=lr, total_steps=STEPS,
                         generator=torch.Generator().manual_seed(seed + 2))
    gen = torch.Generator().manual_seed(seed + 3)
    curve_live, curve_dep, losses = [], [], []
    for t in range(STEPS):
        x = torch.randn(BATCH, IN, generator=gen)
        y = teacher(x) + 0.5 * torch.randn(BATCH, 1, generator=gen)
        loss = ((net(x) - y) ** 2).mean()
        if arm == "adamw":
            set_lr(t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(loss.item())
        if t % LOG_EVERY == 0 or t == STEPS - 1:
            curve_live.append(disp(w2d, w0))
            if arm != "adamw":
                dep = opt.deploy_state()
                curve_dep.append(disp([dep[id(p)] for p in w2d], w0))
    tail = sum(losses[-50:]) / 50
    return curve_live, curve_dep, tail


if __name__ == "__main__":
    results = {}
    for lr in LRS:
        for arm in ("adamw", "concord"):
            live, dep, tail = run(arm, lr)
            results[(arm, lr)] = (live, dep, tail)
            print(f"lr={lr:.0e} {arm:8s} final_disp_live={live[-1]:.4f} "
                  + (f"final_disp_deploy={dep[-1]:.4f} " if dep else "")
                  + f"tail_loss={tail:.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [t for t in range(STEPS) if t % LOG_EVERY == 0 or t == STEPS - 1]
        fig, axes = plt.subplots(1, len(LRS), figsize=(4.2 * len(LRS), 3.6), sharey=True)
        for ax, lr in zip(axes, LRS):
            al, _, at = results[("adamw", lr)]
            cl, cd, ct = results[("concord", lr)]
            ax.plot(xs, al, label=f"AdamW (loss {at:.3f})", color="tab:red")
            ax.plot(xs, cl, label=f"Concord live (loss {ct:.3f})", color="tab:blue")
            ax.plot(xs, cd, label="Concord deploy P", color="tab:blue", ls="--")
            ax.set_title(f"lr = {lr:g}")
            ax.set_xlabel("step")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("‖W − W₀‖ / ‖W₀‖")
        fig.suptitle("Weight displacement vs step (noisy regression, Bayes floor 0.25)")
        fig.tight_layout()
        fig.savefig("exp1_lr_sweep.png", dpi=130)
        print("saved exp1_lr_sweep.png")
    except ImportError:
        pass
