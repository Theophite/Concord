"""Are the gap's directions the NEXT-LOWEST-NOISE ranks?

Concord's coherence gate commits the lowest-noise (highest-SNR) directions ->
its displacement dC is low-norm + low-rank (parsimony). The AdamW-vs-Concord gap
D = W_adamw - W_concord is the motion the gate REJECTS. Hypothesis (the rank
ladder): D lives in the directions with the NEXT-lowest noise after Concord's
committed ones -- a low-rank, identifiable-by-noise tier, NOT diffuse noise.

Test: per Linear, take the top singular directions of
  dC = W_concord - init   (what Concord committed)
  D  = W_adamw  - W_concord (the gap / what AdamW committed beyond)
plus random directions, and measure the per-direction SNR of the minibatch
gradient at the init point (where both optimizers started):
  proj_k(u,v) = u^T g_k v   over K minibatches
  SNR(u,v) = mean_k(proj)^2 / var_k(proj)        # signal^2 / minibatch-noise
High SNR = a consistent (coherent) direction the chase would commit; low SNR =
noise. Ladder CONFIRMED if  SNR(dC dirs) > SNR(D dirs) >> SNR(random).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn

from nanogpt import GPT, GPTConfig, load_char_data, get_batch


def top_dirs(M, r):
    """Top-r left/right singular vectors of M (out x in) as (U_r[out,r], V_r[in,r])."""
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    r = min(r, S.numel())
    return U[:, :r], Vh[:r, :].T, S[:r]


def proj_stats(grads_uv):
    """grads_uv: (K, r) per-minibatch projections -> per-dir SNR = mean^2/var."""
    m = grads_uv.mean(0)
    v = grads_uv.var(0, unbiased=True) + 1e-30
    return (m * m / v)            # (r,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--data", default="nanogpt_data/enwik8")
    ap.add_argument("--K", type=int, default=96, help="# minibatches for SNR")
    ap.add_argument("--r", type=int, default=16, help="# top dirs per layer")
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=777)
    ap.add_argument("--at", choices=["init", "concord"], default="init",
                    help="point to evaluate gradient SNR: shared init, or "
                         "Concord's converged solution (needs aux_final in save).")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    A = torch.load(f"{args.prefix}_adamw.pt", weights_only=False)
    C = torch.load(f"{args.prefix}_concord.pt", weights_only=False)
    init = A['init']

    train, val, vocab, _ = load_char_data(args.data, device)
    # init model = GPT(seed) -- matches the saved init (verified bf16-eps).
    cfg = GPTConfig(vocab_size=vocab, block_size=args.block_size)
    model = GPT(cfg).to(device); model.train()

    # map saved layer-name -> live nn.Linear module
    name2mod = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    targets = list(init.keys())

    if args.at == "concord":
        # rebuild the full fp32 model AT Concord's converged solution: Linears
        # from its materialized weights, aux (embeddings+LN) from aux_final.
        assert 'aux_final' in C, "re-run train_nanogpt to save aux_final"
        with torch.no_grad():
            for n, m in name2mod.items():
                m.weight.copy_(C['final'][n].to(device))
            live = dict(model.named_parameters())
            for n, p in C['aux_final'].items():
                if n in live and n not in name2mod:   # don't clobber Linears
                    live[n].copy_(p.to(device))
        print(f"[probe] evaluating gradients AT Concord solution "
              f"(val={C['final_val']:.4f})")
    else:
        print("[probe] evaluating gradients at shared INIT")

    # precompute top dirs of dC and D + random dirs per layer (on device)
    dirs = {}
    for n in targets:
        W0 = init[n].float().to(device)
        dC = C['final'][n].float().to(device) - W0
        D = A['final'][n].float().to(device) - C['final'][n].float().to(device)
        Uc, Vc, _ = top_dirs(dC, args.r)
        Ud, Vd, _ = top_dirs(D, args.r)
        # random orthonormal-ish dirs (matched count)
        g = torch.Generator(device=device).manual_seed(hash(n) % 2**31)
        Ur = torch.randn(W0.shape[0], args.r, generator=g, device=device)
        Vr = torch.randn(W0.shape[1], args.r, generator=g, device=device)
        Ur /= Ur.norm(dim=0, keepdim=True); Vr /= Vr.norm(dim=0, keepdim=True)
        dirs[n] = dict(c=(Uc, Vc), d=(Ud, Vd), r=(Ur, Vr),
                       proj={k: [] for k in ("c", "d", "r")})

    # collect K minibatch gradients; project each layer's grad onto its dirs
    gen = torch.Generator(device=device).manual_seed(args.data_seed)
    for k in range(args.K):
        x, y = get_batch(train, args.bsz, args.block_size, device, generator=gen)
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        for n in targets:
            gW = name2mod[n].weight.grad           # (out, in)
            for key in ("c", "d", "r"):
                U, V = dirs[n][key]
                # proj_i = u_i^T gW v_i  = sum over out,in -> diag(U^T gW V)
                p = (U * (gW @ V)).sum(0)           # (r,)
                dirs[n]['proj'][key].append(p.detach())

    def _typ(n):
        if 'lm_head' in n: return 'lm_head'
        if 'c_attn' in n: return 'attn.qkv'
        if 'attn.c_proj' in n: return 'attn.proj'
        if 'c_fc' in n: return 'mlp.fc'
        if 'mlp.c_proj' in n: return 'mlp.proj'
        return '?'

    # per-layer SNR, aggregated by type
    import collections
    agg = collections.defaultdict(lambda: {k: [] for k in ("c", "d", "r")})
    rows = []
    for n in targets:
        snr = {}
        for key in ("c", "d", "r"):
            P = torch.stack(dirs[n]['proj'][key], 0)     # (K, r)
            s = proj_stats(P)                            # (r,)
            snr[key] = s
            agg[_typ(n)][key].append(s)
        rows.append((n, snr))

    print(f"per-direction SNR = mean_k(u^T g v)^2 / var_k   (K={args.K} minibatches, "
          f"r={args.r} dirs/layer)\n")
    print(f"{'layer':<22}{'SNR(dC)':>10}{'SNR(D-gap)':>12}{'SNR(rand)':>11}"
          f"{'dC/D':>7}{'D/rand':>8}")
    for n, snr in rows:
        mc, md, mr = snr['c'].mean().item(), snr['d'].mean().item(), snr['r'].mean().item()
        print(f"{n[:22]:<22}{mc:>10.3f}{md:>12.3f}{mr:>11.4f}"
              f"{mc/(md+1e-9):>7.2f}{md/(mr+1e-9):>8.1f}")

    print("\nBY TYPE (mean per-direction SNR):")
    print(f"{'type':<12}{'SNR(dC)':>10}{'SNR(D-gap)':>12}{'SNR(rand)':>11}{'dC/D':>7}{'D/rand':>8}")
    order = ['mlp.fc', 'mlp.proj', 'attn.qkv', 'attn.proj', 'lm_head']
    for t in order:
        if t not in agg: continue
        mc = torch.cat(agg[t]['c']).mean().item()
        md = torch.cat(agg[t]['d']).mean().item()
        mr = torch.cat(agg[t]['r']).mean().item()
        print(f"{t:<12}{mc:>10.3f}{md:>12.3f}{mr:>11.4f}{mc/(md+1e-9):>7.2f}{md/(mr+1e-9):>8.1f}")

    # global verdict
    allc = torch.cat([s['c'] for _, s in rows]); alld = torch.cat([s['d'] for _, s in rows])
    allr = torch.cat([s['r'] for _, s in rows])
    print(f"\nGLOBAL  SNR(dC)={allc.mean():.3f}  SNR(D-gap)={alld.mean():.3f}  "
          f"SNR(rand)={allr.mean():.4f}")
    ladder = allc.mean() > alld.mean() > 3 * allr.mean()
    print(f"RANK LADDER {'CONFIRMED' if ladder else 'NOT CLEAN'}: "
          f"SNR(dC) {'>' if allc.mean()>alld.mean() else '<='} SNR(gap) "
          f"{'>>' if alld.mean()>3*allr.mean() else '~'} SNR(rand)")
    print(f"  -> the gap's directions carry {alld.mean()/allr.mean():.0f}x the SNR of random "
          f"dirs: {'real next-tier signal, not noise' if alld.mean()>3*allr.mean() else 'near noise floor'}")


if __name__ == "__main__":
    main()
