"""Where does v-hat change the weights -- and is the difference LOW-RANK?

Loads the forked weight sets + shared-warmup garbage factor and characterizes
the weights that diverge most between optimizers along the axes the user asked:
  - magnitude   : are the most-changed weights unusually large or small |W|?
  - garbage     : high-coh (signal) or low-coh (noise)?
  - SVD leverage: do changes load onto the layer's important singular modes?
  - LOW RANK    : is dW = W_b - W_a itself low-rank, and aligned with W's top
                  singular subspace (=> a cheap, detectable correction)?

Primary contrast: full (Adam) vs none (SGD) = the per-coordinate v-hat effect.
Also rank1-vs-full (what Adafactor's factoring loses) and rank1-vs-none.
Prints per-layer detail, pooled summary, plain-language SYNTHESIS, PNG figs.
"""
import argparse
import numpy as np
import torch


def ranks(x):
    r = np.empty_like(x, dtype=np.float64)
    r[np.argsort(x)] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def decile_table(key, val, nb=10):
    q = np.quantile(key, np.linspace(0, 1, nb + 1)); out = []
    for i in range(nb):
        lo, hi = q[i], q[i + 1]
        m = (key >= lo) & (key <= hi) if i == nb - 1 else (key >= lo) & (key < hi)
        out.append(val[m].mean() if m.any() else float("nan"))
    return out


def zcat(lst):
    return np.concatenate([(x - x.mean()) / (x.std() + 1e-30) for x in lst])


def lowrank_stats(Wref, dW, m=None):
    """Spectrum of dW itself + alignment with Wref's top-m singular subspace."""
    R, C = dW.shape; k = min(R, C)
    s = np.linalg.svd(dW.astype(np.float64), compute_uv=False)
    s2 = s ** 2; E = s2.sum() + 1e-30
    sr = float(s2.sum() / (s2[0] + 1e-30))          # stable rank
    p = s2 / E; erank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    r90 = int(np.argmax(np.cumsum(s2) / E >= 0.9)) + 1
    top1 = float(s2[0] / E); top5 = float(s2[:5].sum() / E)
    U, _, Vt = np.linalg.svd(Wref.astype(np.float64), full_matrices=False)
    if m is None:
        m = max(1, int(round(0.05 * k)))
    aL = float(np.linalg.norm(U[:, :m].T @ dW.astype(np.float64)) ** 2 / E)
    return dict(k=k, sr=sr, erank=erank, r90=r90, top1=top1, top5=top5,
                aL=aL, rand=m / R, m=m, spec=(s2 / E), R=R, C=C)


def analyze_pair(name_a, A, name_b, B, warm, layers, topfrac=0.01, store=None):
    print(f"\n################  {name_b} vs {name_a}  "
          f"(dW = W_{name_b} - W_{name_a})  ################")
    g_absdw, g_absw, g_coh, g_lev, g_ak2, g_sk, g_lr = [], [], [], [], [], [], []
    for ln in layers:
        Wa = A["W"][ln].numpy().reshape(A["W"][ln].shape[0], -1)
        Wb = B["W"][ln].numpy().reshape(B["W"][ln].shape[0], -1)
        W10 = warm["W10"][ln].numpy().reshape(Wa.shape) \
            if ln in warm["W10"] else 0.5 * (Wa + Wb)
        dW = (Wb - Wa).astype(np.float64)
        coh = warm["coh_warm"][ln].numpy().reshape(Wa.shape)
        absdw = np.abs(dW).ravel(); absw = np.abs(W10).ravel(); cohr = coh.ravel()
        U, s, Vt = np.linalg.svd(W10.astype(np.float64), full_matrices=False)
        K = max(1, int(0.1 * len(s)))
        lev = ((U[:, :K] ** 2).sum(1)[:, None]
               + (Vt[:K, :] ** 2).sum(0)[None, :]).ravel()
        a_k = np.diag(U.T @ dW @ Vt.T)
        lr = lowrank_stats(W10, dW)
        sel = absdw >= np.quantile(absdw, 1 - topfrac)
        print(f"\n  [{ln}]  shape {Wa.shape}  "
              f"||dW||/||W||={np.linalg.norm(dW)/np.linalg.norm(W10):.3f}")
        print(f"    LOW-RANK  stable-rank {lr['sr']:.1f}/{lr['k']}  "
              f"(erank {lr['erank']:.1f}, r90={lr['r90']})  "
              f"top1 {lr['top1']*100:.0f}% top5 {lr['top5']*100:.0f}%  |  "
              f"align top-{lr['m']} {lr['aL']*100:.0f}% (rand {lr['rand']*100:.0f}%)")
        print(f"    corr(|dW|,|W|) sp {spearman(absdw,absw):+.3f}   "
              f"(|dW|,coh) sp {spearman(absdw,cohr):+.3f}   "
              f"(|dW|,SVlev) sp {spearman(absdw,lev):+.3f}   "
              f"(a_k^2,s_k) sp {spearman(a_k**2, s):+.3f}")
        print(f"    top-{topfrac*100:.0f}% changed:  |W| pctile {np.median(ranks(absw)[sel]):.2f}"
              f"   coh pctile {np.median(ranks(cohr)[sel]):.2f}"
              f"   SVlev pctile {np.median(ranks(lev)[sel]):.2f}")
        g_absdw.append(absdw); g_absw.append(absw); g_coh.append(cohr)
        g_lev.append(lev); g_ak2.append(a_k ** 2); g_sk.append(s); g_lr.append(lr)
    za = zcat(g_absdw)
    pooled = {
        "pair": f"{name_b}-{name_a}",
        "w": spearman(za, zcat(g_absw)), "coh": spearman(za, zcat(g_coh)),
        "lev": spearman(za, zcat(g_lev)), "modeload": spearman(zcat(g_ak2), zcat(g_sk)),
        "sr_frac": float(np.mean([d["sr"] / d["k"] for d in g_lr])),
        "top5": float(np.mean([d["top5"] for d in g_lr])),
        "r90_frac": float(np.mean([d["r90"] / d["k"] for d in g_lr])),
        "align": float(np.mean([d["aL"] for d in g_lr])),
        "align_rand": float(np.mean([d["rand"] for d in g_lr])),
        # MISER'S BUDGET: variance-values to store a rank-r90 correction of dW
        # = r90*(R+C) per layer, vs full per-element v-hat = R*C. Ratio = how
        # many times cheaper. rank-1 (Adafactor) = (R+C) for reference.
        "r90_mean": float(np.mean([d["r90"] for d in g_lr])),
        "bits_full_over_r90": float(
            sum(d["R"] * d["C"] for d in g_lr)
            / sum(d["r90"] * (d["R"] + d["C"]) for d in g_lr)),
        "bits_full_over_rank1": float(
            sum(d["R"] * d["C"] for d in g_lr)
            / sum((d["R"] + d["C"]) for d in g_lr)),
    }
    print(f"\n  === POOLED {pooled['pair']} ===")
    print(f"    LOW-RANK  stable-rank {pooled['sr_frac']*100:.0f}% of full  "
          f"top5 {pooled['top5']*100:.0f}%  r90 {pooled['r90_frac']*100:.0f}% of rank  "
          f"align {pooled['align']*100:.0f}% vs rand {pooled['align_rand']*100:.0f}%")
    print(f"    corr(|dW|,|W|) sp {pooled['w']:+.3f}   (|dW|,coh) sp {pooled['coh']:+.3f}   "
          f"(|dW|,SVlev) sp {pooled['lev']:+.3f}   (a_k^2,s_k) sp {pooled['modeload']:+.3f}")
    if store is not None:
        store.append((pooled, g_absw, g_coh, g_lev, g_absdw, g_ak2, g_sk, g_lr))
    return pooled


def word(c, pos, neg, strong=0.20, weak=0.07):
    a = abs(c)
    if a < weak:
        return "no clear relation"
    mag = "strongly" if a >= strong else "mildly"
    return f"{mag} {pos if c > 0 else neg}"


def synthesize(pooleds):
    print("\n\n========================  SYNTHESIS  ========================")
    for p in pooleds:
        lowrank = p["sr_frac"] < 0.25 or p["top5"] > 0.5
        aligned = p["align"] > 1.8 * p["align_rand"]
        print(f"\n  {p['pair']}:")
        print(f"    LOW-RANK?  {'YES' if lowrank else 'no'} -- dW stable-rank is "
              f"{p['sr_frac']*100:.0f}% of full, top-5 modes hold "
              f"{p['top5']*100:.0f}% of the difference (r90 at {p['r90_frac']*100:.0f}% of rank)")
        print(f"    DETECTABLE? {'YES' if aligned else 'weak'} -- "
              f"{p['align']*100:.0f}% of dW sits in W's top-{{~5%}} singular subspace "
              f"(random baseline {p['align_rand']*100:.0f}%)")
        print(f"    MISER'S BUDGET: 90% of dW lives in ~rank-{p['r90_mean']:.0f}/layer "
              f"-> a rank-k v-hat is ~{p['bits_full_over_r90']:.0f}x cheaper than full "
              f"per-element v-hat (Adafactor rank-1 = {p['bits_full_over_rank1']:.0f}x cheaper)")
        print(f"    most-moved weights:  vs |W| {word(p['w'],'LARGER','SMALLER')} ({p['w']:+.2f}) |"
              f"  vs coh {word(p['coh'],'SIGNAL','GARBAGE')} ({p['coh']:+.2f}) |"
              f"  load {word(p['modeload'],'IMPORTANT modes','TAIL modes')} ({p['modeload']:+.2f})")


def make_figs(store, prefix):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[figs skipped: {e}]"); return
    for pooled, g_absw, g_coh, g_lev, g_absdw, g_ak2, g_sk, g_lr in store:
        pair = pooled["pair"]
        fig, ax = plt.subplots(1, 5, figsize=(25, 4.4))
        za = ranks(zcat(g_absdw))
        for j, (g, lab, key) in enumerate([
                (g_absw, "|W| pctile", "w"), (g_coh, "coh pctile", "coh"),
                (g_lev, "SVD-leverage pctile", "lev")]):
            ax[j].hexbin(ranks(zcat(g)), za, gridsize=40, bins="log", cmap="viridis")
            ax[j].set_xlabel(lab); ax[j].set_ylabel("|dW| pctile")
            ax[j].set_title(f"{lab}  sp={pooled[key]:+.3f}")
        ax[3].hexbin(ranks(zcat(g_sk)), ranks(zcat(g_ak2)), gridsize=40,
                     bins="log", cmap="magma")
        ax[3].set_xlabel("singular value s_k pctile")
        ax[3].set_ylabel("dW mode-loading a_k^2 pctile")
        ax[3].set_title(f"singular modes  sp={pooled['modeload']:+.3f}")
        for d in g_lr:                       # cumulative energy of dW spectrum
            c = np.cumsum(d["spec"]); xf = np.arange(1, len(c) + 1) / len(c)
            ax[4].plot(xf, c, alpha=0.8)
        ax[4].plot([0, 1], [0, 1], "k--", lw=1, label="full-rank (random)")
        ax[4].set_xlabel("fraction of singular modes")
        ax[4].set_ylabel("cumulative dW energy")
        ax[4].set_title(f"LOW-RANK: stable-rank {pooled['sr_frac']*100:.0f}% of full")
        ax[4].legend(fontsize=7)
        fig.suptitle(f"{prefix}  {pair}: where (and how low-rank) is the v-hat effect?")
        fig.tight_layout(); fn = f"{prefix}_fig_{pair}.png"
        fig.savefig(fn, dpi=110); plt.close(fig); print(f"[fig saved: {fn}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=str, default="vmode")
    args = ap.parse_args()
    warm = torch.load(f"{args.prefix}_warm.pt", map_location="cpu", weights_only=False)
    modes = {}
    for m in ("none", "rank1", "full"):
        try:
            modes[m] = torch.load(f"{args.prefix}_{m}.pt", map_location="cpu",
                                  weights_only=False)
        except FileNotFoundError:
            pass
    print("Loaded modes:", list(modes))
    for m, d in modes.items():
        print(f"  {m:6s} best {d['best']*100:.2f}% (ep {d['best_ep']})  "
              f"final {d['final']*100:.2f}%")
    layers = [ln for ln in modes[next(iter(modes))]["W"]]
    store, pooleds = [], []
    for a, b in [("none", "full"), ("rank1", "full"), ("none", "rank1")]:
        if a in modes and b in modes:
            pooleds.append(analyze_pair(a, modes[a], b, modes[b], warm,
                                        layers, store=store))
    synthesize(pooleds)
    make_figs(store, args.prefix)


if __name__ == "__main__":
    main()
