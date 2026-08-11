"""Layout de-risk for the graph-native gradient-SNR sketch (csnr_meter.CSNRMeter).

The eager GradSketch means the output gradient over the BATCH but keeps the TOKEN axis
([B, tokens, feat] -> mean over B -> [tokens*feat], sample K coords). The fused grad_x kernel,
however, sees the gradient already flattened to [M = B*tokens, feat] and cannot cheaply separate
B from tokens. Two candidate graph-native sketches:

  feat-only        mean over ALL M rows at K fixed feat columns.   Trivial kernel (a column
                   reduction like memgap). Collapses the token axis.
  token-preserved  mean over B keeping tokens, at K fixed (token,feat) coords. A strided gather;
                   this IS the eager sketch, already validated by test_csnr_meter.

Question: does feat-only still recover the per-timestep SNR SHAPE? If yes, the kernel is a simple
reduction; if it degrades (esp. when the per-timestep signal varies across tokens, which the
feat-only mean averages away), we need the token-preserved gather.

Simulates the meter's generative model: per bin b a signal direction mu_b over [tokens, feat] with
tunable token coherence, |mu_b|^2 ~ S[b]; per-sample noise ~ N[b]. Each step draws B samples at
random timesteps, forms both sketches, feeds two CSNRMeters, cashes out, and scores Spearman rank
correlation of recovered vs true per-bin SNR.
"""
import math

import torch

from modules.util.optimizer.csnr_meter import CSNRMeter

T = 1000
N_BINS = 8
FEAT = 320
TOKENS = 64
K = 256
WINDOW = 50


def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def true_snr():
    b = torch.arange(N_BINS).float()
    S = torch.exp(-b / 2.5) + 0.05           # signal power decays with the bin (t)
    N = 0.3 + 0.0 * b                         # ~flat noise floor
    return S, N, S / N


def run_layout(layout, coherence, B, seed, window=WINDOW):
    g = torch.Generator().manual_seed(seed)
    S, N, _ = true_snr()
    # per-bin signal directions over [tokens, feat]: a token-shared part + a token-varying part,
    # mixed by `coherence`. feat-only averages over tokens -> keeps the shared part, cancels the rest.
    shared = torch.randn(N_BINS, 1, FEAT, generator=g)
    varying = torch.randn(N_BINS, TOKENS, FEAT, generator=g)
    mu = coherence * shared + math.sqrt(1.0 - coherence ** 2) * varying      # [bins, tokens, feat]
    mu = mu / mu.reshape(N_BINS, -1).norm(dim=1).view(N_BINS, 1, 1)          # unit, then scale by sqrt(S)
    mu = mu * S.sqrt().view(N_BINS, 1, 1)

    if layout == "feat_only":
        coords = torch.randint(0, FEAT, (K,), generator=g)
    else:                                                                    # token-preserved (eager)
        ct = torch.randint(0, TOKENS, (K,), generator=g)
        cf = torch.randint(0, FEAT, (K,), generator=g)

    meter = CSNRMeter(n_bins=N_BINS)
    edges = torch.linspace(0, T, N_BINS + 1)
    for _ in range(window):
        ts = torch.randint(0, T, (B,), generator=g)
        bins = torch.bucketize(ts.float(), edges[1:-1])
        # per-sample gradient field [B, tokens, feat] = mu_bin + noise
        grad = mu[bins] + torch.randn(B, TOKENS, FEAT, generator=g) * N[bins].sqrt().view(B, 1, 1)
        if layout == "feat_only":
            gm = grad.mean(dim=(0, 1))                       # mean over B AND tokens -> [feat]
            sketch = gm[coords]
        else:
            gm = grad.mean(dim=0)                            # mean over B, keep tokens -> [tokens, feat]
            sketch = gm[ct, cf]
        meter.add(sketch, ts, T)
    curve = meter.cash_out(T)
    if curve is None:
        return None
    # collapse the per-timestep curve back to per-bin (nearest) for scoring
    bidx = torch.bucketize(torch.arange(T).float(), edges[1:-1]).clamp(max=N_BINS - 1)
    rec = torch.zeros(N_BINS)
    for b in range(N_BINS):
        m = (bidx == b)
        rec[b] = curve[m].mean() if m.any() else 0.0
    return rec


def aggregate_stretch_sweep():
    """The production fix: feed the accumulation-aggregated batch (B=4 micro -> 12 pooled) as one
    observation, and stretch the per-epoch window. Sweep both on the chosen feat-only layout."""
    _, _, snr_true = true_snr()
    print(f"=== csnr aggregate+stretch sweep (feat-only, T={T}, bins={N_BINS}, K={K}, "
          f"tokens={TOKENS}) ===")
    print("  rank-corr of recovered vs true per-bin SNR (mean over kept windows / 8 seeds); "
          "rej = rejected\n")
    for coh in (0.7, 1.0):
        print(f"  token coherence = {coh}")
        print(f"    {'eff_B':<8}" + "".join(f"win={w:<9}" for w in (50, 100, 150, 200)))
        for B in (4, 12):
            row = f"    {B:<8}"
            for w in (50, 100, 150, 200):
                vals, rej = [], 0
                for seed in range(8):
                    r = run_layout("feat_only", coh, B, seed, window=w)
                    if r is None:
                        rej += 1
                    else:
                        vals.append(_spearman(r, snr_true))
                m = sum(vals) / len(vals) if vals else float('nan')
                row += f"{m:>6.2f}({rej})   "
            print(row)
        print()
    print("  Read: does feat-only at eff_B=12 (aggregated) + a longer window clear a usable")
    print("  rank-corr (say >0.7) with few rejects, where eff_B=4 / win=50 did not?\n")


def main():
    aggregate_stretch_sweep()
    _, _, snr_true = true_snr()
    print(f"=== csnr sketch-layout de-risk (T={T}, bins={N_BINS}, K={K}, tokens={TOKENS}, "
          f"window={WINDOW}) ===")
    print(f"  true per-bin SNR (b0..b7): {[round(float(x), 3) for x in snr_true]}\n")
    for B in (4, 16, 64):
        print(f"  batch B={B}  (Spearman rank-corr of recovered vs true per-bin SNR, mean over 5 seeds)")
        print(f"    {'coherence':<12}{'feat_only':<14}{'token_preserved':<16}{'rejected(feat/tok)':<20}")
        for coh in (0.0, 0.5, 1.0):
            fo, tp, nf, nt = [], [], 0, 0
            for seed in range(5):
                r_fo = run_layout("feat_only", coh, B, seed)
                r_tp = run_layout("token_preserved", coh, B, seed + 100)
                if r_fo is None:
                    nf += 1
                else:
                    fo.append(_spearman(r_fo, snr_true))
                if r_tp is None:
                    nt += 1
                else:
                    tp.append(_spearman(r_tp, snr_true))
            mfo = sum(fo) / len(fo) if fo else float('nan')
            mtp = sum(tp) / len(tp) if tp else float('nan')
            print(f"    {coh:<12.2f}{mfo:<14.3f}{mtp:<16.3f}{f'{nf}/{nt}':<20}")
        print()
    print("  Read: if feat_only tracks token_preserved (rank-corr close, few rejects) even at low")
    print("  coherence, the simple column-reduction kernel suffices. If feat_only collapses as")
    print("  coherence -> 0 (token-varying signal averaged away), the strided token-preserved gather")
    print("  is required. eff_B (=B) too small -> rejects (degeneracy guard).")


if __name__ == "__main__":
    main()
