"""Exp 2: response to controlled gradient streams — the selectivity claim.

A single 64x64 weight, no task: we hand-feed gradients
    g_t = m·G + s·xi_t ,   G fixed unit-norm direction, xi ~ N(0, I)
for (m, s) = (0,1) pure noise, (1,0) pure drift, (0.3,1) buried signal.

Constant lr (no warmup ramp confound beyond the first 100 steps), Concord's
own noise injection OFF — we are measuring the response to the stream, not
the fluctuation half. Track displacement of the live weight W and deploy
weight P from their post-consolidation reference (step 100), AdamW from the
same step, and the mean coherence gain.
"""
import torch

from concord_ref import ConcordRef

torch.set_num_threads(4)
N, STEPS, LR, REF_T = 64, 2000, 1e-3, 100
STREAMS = {"pure noise (m=0, s=1)": (0.0, 1.0),
           "buried signal (m=0.3, s=1)": (0.3, 1.0),
           "pure drift (m=1, s=0)": (1.0, 0.0)}


def run(arm, m, s, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(N, N, bias=False)
    G = torch.randn(N, N)
    G /= G.norm()
    gen = torch.Generator().manual_seed(seed + 1)
    if arm == "adamw":
        opt = torch.optim.AdamW(lin.parameters(), lr=LR, weight_decay=0.0)
    else:
        opt = ConcordRef(lin, lr=LR, total_steps=STEPS, warmup=100,
                         lr_min_frac=1.0, noise=False)   # constant lr after warmup
    ref_live = ref_dep = None
    xs, dl, dd, cohs = [], [], [], []
    for t in range(STEPS):
        g = m * G + s * torch.randn(N, N, generator=gen)
        lin.weight.grad = g
        opt.step()
        if arm != "adamw":
            opt.zero_grad()
        if t == REF_T:
            ref_live = lin.weight.detach().clone()
            if arm != "adamw":
                dep = opt.deploy_state()
                ref_dep = dep[id(lin.weight)].clone()
        if t > REF_T and (t % 10 == 0 or t == STEPS - 1):
            xs.append(t)
            dl.append((lin.weight.detach() - ref_live).norm().item())
            if arm != "adamw":
                dep = opt.deploy_state()
                dd.append((dep[id(lin.weight)] - ref_dep).norm().item())
                cohs.append(opt.mean_coh())
    return xs, dl, dd, cohs


if __name__ == "__main__":
    results = {}
    for name, (m, s) in STREAMS.items():
        for arm in ("adamw", "concord"):
            results[(arm, name)] = run(arm, m, s)
        xs, al, _, _ = results[("adamw", name)]
        _, cl, cd, coh = results[("concord", name)]
        print(f"{name:30s} disp@end  adamw={al[-1]:8.4f}  concord_live={cl[-1]:8.4f}  "
              f"concord_deploy={cd[-1]:8.4f}  mean_coh={sum(coh)/len(coh):.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, name in zip(axes, STREAMS):
            xs, al, _, _ = results[("adamw", name)]
            _, cl, cd, coh = results[("concord", name)]
            ax.plot(xs, al, color="tab:red", label="AdamW")
            ax.plot(xs, cl, color="tab:blue", label="Concord live W")
            ax.plot(xs, cd, color="tab:blue", ls="--", label="Concord deploy P")
            ax2 = ax.twinx()
            ax2.plot(xs, coh, color="tab:green", alpha=0.5, lw=1, label="coh")
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("mean coh", color="tab:green", fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("step")
            ax.legend(fontsize=7, loc="upper left")
        axes[0].set_ylabel("‖W − W_ref‖ (from step 100)")
        fig.suptitle("Displacement under controlled gradient streams (lr=1e-3)")
        fig.tight_layout()
        fig.savefig("exp2_signal_noise.png", dpi=130)
        print("saved exp2_signal_noise.png")
    except ImportError:
        pass
