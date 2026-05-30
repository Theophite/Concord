"""Temporal miserliness: WHEN does each weight settle, and could you have
freed its variance-tracking memory early?

Uses the per-fork weight trajectory (snapshots every ckpt_every ep). A weight
is "settled by epoch e" when its remaining travel |W_final - W_e| has dropped
below tol * |W_final - W_10| -- i.e. it has essentially arrived and further
variance-tracking is wasted. Reports:
  - f(e) = fraction settled by epoch e  ->  active(e)=1-f(e) = the v-hat budget
    you must still hold at epoch e (peak memory if you free on settle).
  - mean active fraction over training = the time-averaged variance-budget.
  - per mode: does Adam (full) settle weights earlier/later than SGD (none)?
  - CROSS with SVD leverage: do the LATE-settling (always-active) weights live
    in the layer's TOP singular subspace -- i.e. is the persistently-active set
    the same as the where-v-hat-matters set (spatial low-rank)? If yes, the two
    miserly axes are one structure.
"""
import argparse
import numpy as np
import torch

from cifar_vmode_fork import ANALYZE


def ranks(x):
    r = np.empty_like(x, dtype=np.float64); r[np.argsort(x)] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def spearman(a, b):
    a, b = ranks(a), ranks(b); a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def settle_epochs(traj_epochs, Ws, tol):
    """Ws: list of [n] flattened weights at traj_epochs (incl ep10 first,
    final last). Returns per-weight first epoch where remaining<tol*total."""
    W10 = Ws[0]; Wf = Ws[-1]
    total = np.abs(Wf - W10) + 1e-12
    se = np.full(W10.shape, traj_epochs[-1], dtype=np.float64)
    done = np.zeros(W10.shape, dtype=bool)
    for e, W in zip(traj_epochs, Ws):
        rem = np.abs(Wf - W)
        now = (~done) & (rem < tol * total)
        se[now] = e; done |= now
    return se, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="vmode")
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--modes", default="none,full,rank1")
    args = ap.parse_args()
    warm = torch.load(f"{args.prefix}_warm.pt", map_location="cpu", weights_only=False)

    summary = {}
    store = {}
    for mode in args.modes.split(","):
        try:
            d = torch.load(f"{args.prefix}_{mode}.pt", map_location="cpu",
                           weights_only=False)
        except FileNotFoundError:
            continue
        traj = d.get("traj", {})
        if not traj:
            print(f"[{mode}] no trajectory saved; skip"); continue
        eps = sorted(traj.keys())
        all_se, all_lev = [], []
        per_layer_f = {}
        grid = [10] + eps
        for ln in ANALYZE:
            R = d["W"][ln].shape[0]
            W10 = warm["W10"][ln].numpy().reshape(R, -1)
            Ws = [W10.ravel()] + [traj[e][ln].numpy().reshape(R, -1).ravel()
                                  for e in eps]
            se, total = settle_epochs(grid, Ws, args.tol)
            Wf = d["W"][ln].numpy().reshape(R, -1)
            U, s, Vt = np.linalg.svd(Wf.astype(np.float64), full_matrices=False)
            K = max(1, int(0.1 * len(s)))
            lev = ((U[:, :K] ** 2).sum(1)[:, None]
                   + (Vt[:K, :] ** 2).sum(0)[None, :]).ravel()
            all_se.append(se); all_lev.append(lev)
        se_cat = np.concatenate(all_se); lev_cat = np.concatenate(all_lev)
        # f(e) curve
        fe = {e: float((se_cat <= e).mean()) for e in grid}
        active = {e: 1 - fe[e] for e in grid}
        mean_active = float(np.mean([active[e] for e in grid]))
        corr = spearman(se_cat, lev_cat)
        summary[mode] = dict(fe=fe, active=active, mean_active=mean_active,
                             corr_lev=corr, grid=grid)
        store[mode] = (se_cat, lev_cat)
        print(f"\n[{mode}]  settle-by-epoch f(e) (tol={args.tol}):")
        print("   ep:   " + "  ".join(f"{e:>4}" for e in grid))
        print("   f :   " + "  ".join(f"{fe[e]*100:4.0f}" for e in grid) + "  (%)")
        print("   act:  " + "  ".join(f"{active[e]*100:4.0f}" for e in grid)
              + "  (% still needing v-hat)")
        print(f"   mean active fraction over training = {mean_active*100:.0f}% "
              f"(=time-avg variance budget if you free on settle)")
        print(f"   corr(settle_epoch, SVD-leverage) spearman {corr:+.3f}  "
              f"(>0 => top-subspace weights stay active longest)")

    print("\n==== SUMMARY ====")
    for mode, s in summary.items():
        print(f"  {mode:6s} mean-active {s['mean_active']*100:3.0f}%   "
              f"corr(settle,lev) {s['corr_lev']:+.2f}")
    # figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for mode, s in summary.items():
            g = s["grid"]
            ax[0].plot(g, [s["active"][e] * 100 for e in g], marker="o", label=mode)
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("% weights still active")
        ax[0].set_title("variance budget over training (lower=cheaper)")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        if store:
            m0 = next(iter(store)); se, lev = store[m0]
            ax[1].hexbin(ranks(lev), ranks(se), gridsize=40, bins="log", cmap="cividis")
            ax[1].set_xlabel("SVD-leverage pctile (final W)")
            ax[1].set_ylabel("settle-epoch pctile")
            ax[1].set_title(f"{m0}: late-settlers vs top-subspace "
                            f"sp={summary[m0]['corr_lev']:+.2f}")
        fig.tight_layout(); fig.savefig(f"{args.prefix}_fig_settle.png", dpi=110)
        print(f"[fig saved: {args.prefix}_fig_settle.png]")
    except Exception as e:
        print(f"[fig skipped: {e}]")


if __name__ == "__main__":
    main()
