"""Where does the AdamW-vs-(unwhitened)Concord gap LIVE?

Consumes the two artifacts written by train_nanogpt.py --save_prefix P
  P_adamw.pt   {init, final, ...}      (fp32 weights; exact seed-0 init)
  P_concord.pt {init, final, ...}      (bf16-stored weights of the SAME init)
trained from the IDENTICAL init (seed) and IDENTICAL batch order (data_seed,
dedicated generator -> optimizer-invariant). So the per-Linear difference
  D = W_adamw - W_concord
isolates the optimizer effect (the missing per-coordinate v-hat) from init /
data-order noise. The init cancels in D, so the bf16-vs-fp32 init asymmetry is
irrelevant to the difference; we anchor each optimizer's *trajectory* dA/dC to
the exact fp32 init from the AdamW save.

Answers the four questions:
  1. WHICH layers carry the gap (attn vs mlp vs head; early vs late)?
  2. Is D LOW-RANK (detectable few-direction structure -- user hypothesis)?
  3. Does the gap live on UNUSUALLY LARGE or SMALL |W| coordinates?
  4. Does it live in the IMPORTANT singular subspace of the layer?
Plus: do the two optimizers move the same DIRECTION (cos(dA,dC)) at different
magnitude, or genuinely diverge; and is the gap CONCENTRATED (miserly: few
coords) or diffuse?
"""
import argparse
from pathlib import Path

import torch


def _layer_type(name):
    if "lm_head" in name:
        return "lm_head"
    if "c_attn" in name:
        return "attn.qkv"
    if "attn.c_proj" in name:
        return "attn.proj"
    if "c_fc" in name:
        return "mlp.fc"
    if "mlp.c_proj" in name:
        return "mlp.proj"
    return "other"


def _depth(name):
    # blocks.N. -> N ; lm_head -> last
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def subspace_energy(D, basis_vecs):
    """Fraction of ||D||_F^2 lying in the column space of basis_vecs (n x r,
    orthonormal). For right singular vectors V_r (input directions): D V_r."""
    proj = D @ basis_vecs               # (m x r)
    return (proj.pow(2).sum() / (D.pow(2).sum() + 1e-30)).item()


def analyze_layer(name, W0, Wa, Wc, topr=8):
    dA = Wa - W0                        # AdamW trajectory
    dC = Wc - W0                        # Concord trajectory
    D = Wa - Wc                         # the gap (init cancels)
    nA, nC, nD = dA.norm().item(), dC.norm().item(), D.norm().item()
    nW0 = W0.norm().item()
    cos_traj = (dA.flatten() @ dC.flatten() /
                (nA * nC + 1e-30)).item()
    # --- low-rank structure of the gap D ---
    Df = D.float()
    try:
        S = torch.linalg.svdvals(Df)
    except Exception:
        S = torch.linalg.svdvals(Df.cpu())
    s2 = S.pow(2)
    tot = s2.sum().item() + 1e-30
    stable_rank = (tot / (s2[0].item() + 1e-30))
    csum = torch.cumsum(s2, 0) / tot
    def topk_energy(k):
        k = min(k, len(csum))
        return csum[k - 1].item()
    # --- important-subspace leverage: top-r right singular vecs of W0 ---
    try:
        U0, S0, Vh0 = torch.linalg.svd(W0.float(), full_matrices=False)
    except Exception:
        U0, S0, Vh0 = torch.linalg.svd(W0.float().cpu(), full_matrices=False)
    r = min(topr, Vh0.shape[0])
    Vr = Vh0[:r].T.to(Df.device)        # (n x r) top input directions
    lev_top = subspace_energy(Df, Vr)   # energy of D in top-r input subspace
    rand_baseline = r / W0.shape[1]     # if D were isotropic in input space
    # --- where in |W| does the gap live? bin coords by |W0| decile ---
    aW0 = W0.abs().flatten()
    aD = D.abs().flatten()
    q = torch.quantile(aW0, torch.linspace(0, 1, 11, device=aW0.device))
    decile_mean_D = []
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (aW0 >= lo) & (aW0 <= hi if i == 9 else aW0 < hi)
        decile_mean_D.append(aD[m].mean().item() if m.any() else 0.0)
    # spearman-ish: correlation of ranks of |W0| and |D|
    def _rank(x):
        return x.argsort().argsort().float()
    rk = torch.corrcoef(torch.stack([_rank(aW0), _rank(aD)]))[0, 1].item()
    # --- concentration of the gap (miserly: few coords?) ---
    sortedD2 = torch.sort(aD.pow(2), descending=True).values
    cum = torch.cumsum(sortedD2, 0) / (sortedD2.sum() + 1e-30)
    n = len(aD)
    frac_top1pct = cum[max(0, int(0.01 * n) - 1)].item()
    frac_top10pct = cum[max(0, int(0.10 * n) - 1)].item()
    return dict(
        name=name, type=_layer_type(name), depth=_depth(name),
        shape=tuple(W0.shape), nW0=nW0, nA=nA, nC=nC, nD=nD,
        rel_gap=nD / (nA + 1e-30), cos_traj=cos_traj,
        stable_rank=stable_rank, full_rank=min(W0.shape),
        e1=topk_energy(1), e2=topk_energy(2), e4=topk_energy(4),
        e8=topk_energy(8), e16=topk_energy(16), e32=topk_energy(32),
        lev_top=lev_top, lev_rand=rand_baseline, lev_ratio=lev_top / rand_baseline,
        decile_mean_D=decile_mean_D, rank_corr=rk,
        frac_top1pct=frac_top1pct, frac_top10pct=frac_top10pct,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="path prefix; loads {prefix}_adamw.pt + {prefix}_concord.pt")
    ap.add_argument("--topr", type=int, default=8)
    ap.add_argument("--fig", default=None, help="save figure to this path")
    args = ap.parse_args()

    A = torch.load(f"{args.prefix}_adamw.pt", weights_only=False)
    C = torch.load(f"{args.prefix}_concord.pt", weights_only=False)
    print(f"AdamW  final_val={A['final_val']:.4f} best={A['best_val']:.4f} "
          f"peak_lr={A.get('peak_lr')} iters={A.get('max_iters')}")
    print(f"Concord final_val={C['final_val']:.4f} best={C['best_val']:.4f} "
          f"peak_lr={C.get('peak_lr')} iters={C.get('max_iters')}")
    print(f"loss gap (Concord - AdamW) = {C['final_val'] - A['final_val']:.4f} nats\n")

    init = A['init']                    # exact fp32 seed-0 init
    rows = []
    for name in init:
        rows.append(analyze_layer(name, init[name].float(),
                                  A['final'][name].float(),
                                  C['final'][name].float(), topr=args.topr))

    # ---- per-layer table ----
    tot_D2 = sum(r['nD'] ** 2 for r in rows)
    print(f"{'layer':<22}{'shape':>12}{'||D||':>9}{'rel_gap':>8}"
          f"{'cos':>7}{'st.rank':>8}{'/full':>7}{'e8':>6}{'levX':>7}"
          f"{'rk_c':>6}{'top1%':>7}{'%gap':>7}")
    for r in sorted(rows, key=lambda r: -r['nD'] ** 2):
        sh = f"{r['shape'][0]}x{r['shape'][1]}"
        print(f"{r['name'][:22]:<22}{sh:>12}{r['nD']:>9.3f}{r['rel_gap']:>8.2f}"
              f"{r['cos_traj']:>7.2f}{r['stable_rank']:>8.1f}"
              f"{r['stable_rank']/r['full_rank']*100:>6.0f}%{r['e8']*100:>5.0f}%"
              f"{r['lev_ratio']:>7.1f}{r['rank_corr']:>6.2f}"
              f"{r['frac_top1pct']*100:>6.0f}%"
              f"{r['nD']**2/tot_D2*100:>6.1f}%")

    # ---- aggregate by type ----
    print("\nGAP SHARE BY LAYER TYPE:")
    bytype = {}
    for r in rows:
        bytype.setdefault(r['type'], [0.0, 0, []])
        bytype[r['type']][0] += r['nD'] ** 2
        bytype[r['type']][1] += 1
        bytype[r['type']][2].append(r)
    for t, (e, c, rs) in sorted(bytype.items(), key=lambda kv: -kv[1][0]):
        sr = sum(x['stable_rank'] for x in rs) / c
        lv = sum(x['lev_ratio'] for x in rs) / c
        co = sum(x['cos_traj'] for x in rs) / c
        rc = sum(x['rank_corr'] for x in rs) / c
        print(f"  {t:<12} {e/tot_D2*100:>5.1f}% of gap  ({c} layers)  "
              f"mean stable_rank={sr:>5.1f}  levX={lv:>4.1f}  "
              f"cos(dA,dC)={co:>5.2f}  rank_corr(|W|,|D|)={rc:>5.2f}")

    # ---- aggregate by depth ----
    print("\nGAP SHARE BY DEPTH:")
    bydep = {}
    for r in rows:
        bydep.setdefault(r['depth'], 0.0)
        bydep[r['depth']] += r['nD'] ** 2
    for d in sorted(bydep):
        lbl = "lm_head" if d == -1 else f"block {d}"
        print(f"  {lbl:<10} {bydep[d]/tot_D2*100:>5.1f}%")

    # ---- verdicts ----
    mean_sr_ratio = sum(r['stable_rank'] / r['full_rank'] for r in rows) / len(rows)
    mean_lev = sum(r['lev_ratio'] for r in rows) / len(rows)
    mean_rk = sum(r['rank_corr'] for r in rows) / len(rows)
    mean_cos = sum(r['cos_traj'] for r in rows) / len(rows)
    mean_t1 = sum(r['frac_top1pct'] for r in rows) / len(rows)
    print("\nVERDICT:")
    print(f"  low-rank?       mean stable_rank = {mean_sr_ratio*100:.0f}% of full "
          f"-> {'LOW-RANK (detectable few-direction structure)' if mean_sr_ratio < 0.5 else 'NOT low-rank (diffuse)'}")
    print(f"  important sub?  mean leverage = {mean_lev:.1f}x random "
          f"-> {'concentrated in top singular subspace' if mean_lev > 1.5 else 'spread across spectrum'}")
    print(f"  large/small W?  mean rank_corr(|W|,|D|) = {mean_rk:+.2f} "
          f"-> {'gap on LARGE |W|' if mean_rk > 0.1 else ('gap on SMALL |W|' if mean_rk < -0.1 else 'no |W| preference')}")
    print(f"  same direction? mean cos(dA,dC) = {mean_cos:+.2f} "
          f"-> {'same direction, different magnitude (rescale)' if mean_cos > 0.6 else 'genuinely divergent directions'}")
    print(f"  concentrated?   top 1% of coords carry mean {mean_t1*100:.0f}% of gap energy")

    if args.fig:
        _make_fig(rows, args.fig)


def _make_fig(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: (r['depth'], r['type']))
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    names = [r['name'].replace("blocks.", "b").replace(".weight", "") for r in rows]
    # (0,0) per-layer gap energy share
    tot = sum(r['nD'] ** 2 for r in rows)
    ax[0, 0].bar(range(len(rows)), [r['nD'] ** 2 / tot * 100 for r in rows])
    ax[0, 0].set_xticks(range(len(rows))); ax[0, 0].set_xticklabels(names, rotation=90, fontsize=6)
    ax[0, 0].set_ylabel("% of total gap energy"); ax[0, 0].set_title("WHERE: gap energy by layer")
    # (0,1) stable rank fraction
    ax[0, 1].bar(range(len(rows)), [r['stable_rank'] / r['full_rank'] * 100 for r in rows], color='C1')
    ax[0, 1].set_xticks(range(len(rows))); ax[0, 1].set_xticklabels(names, rotation=90, fontsize=6)
    ax[0, 1].set_ylabel("stable rank / full rank (%)"); ax[0, 1].set_title("LOW-RANK? gap stable-rank")
    ax[0, 1].axhline(50, ls='--', c='k', lw=0.5)
    # (1,0) top-k energy curves
    ks = [1, 2, 4, 8, 16, 32]
    for r in rows:
        ax[1, 0].plot(ks, [r[f'e{k}'] for k in ks], alpha=0.4, lw=0.8)
    ax[1, 0].set_xscale('log', base=2); ax[1, 0].set_xlabel("top-k singular dirs")
    ax[1, 0].set_ylabel("cumulative energy"); ax[1, 0].set_title("LOW-RANK? energy in top-k of D")
    ax[1, 0].axhline(0.9, ls='--', c='k', lw=0.5)
    # (1,1) |D| vs |W| decile
    for r in rows:
        d = r['decile_mean_D']
        ax[1, 1].plot(range(1, 11), [x / (d[-1] + 1e-30) for x in d], alpha=0.4, lw=0.8)
    ax[1, 1].set_xlabel("|W_init| decile (1=smallest)")
    ax[1, 1].set_ylabel("mean |D| (norm. to top decile)")
    ax[1, 1].set_title("WHERE in |W|? gap vs weight magnitude")
    plt.tight_layout(); plt.savefig(path, dpi=110)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
