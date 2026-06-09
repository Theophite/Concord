"""Exp 10: 80 epochs + augmentation on the small set — the ablation.

The 4k/25ep protocol is starved by design (the leaderboard question: MLP-class
ceiling is ~98-98.5 on full data; ~95-97 on 4k). This pushes the small set to
its limit: 80 epochs, pad-2 random-crop augmentation applied ONLY to the 4k
train subset (test set untouched), and ablates aug x optimizer x regime
against the existing 25-epoch no-aug numbers (exps 8/9/9c).

Arms at each one's best-known settings (flagged where unswept):
    adamw         lr 1e-3 (exp-8 standard; not re-swept)
    concord_vhat  lr 1e-2 (its 9c lr*), kappa: clean 0 / noisy 400
    concord_muon  lr 1e-2 (its 9c lr*), kappa: clean 0 / noisy 100, c=0
    muon_native   lr 1e-3 (exp-9; not swept), beta 0.95

kappa values are the 25-epoch oracles — possibly stale at 80ep/aug (noted).
Same-seed arms see IDENTICAL augmented batch streams (perm and crop draws come
from the same generator sequence, independent of the optimizer).

References (25ep, no aug): clean adamw 92.78 / vhat(lr*) 94.68 / muon(lr*)
96.07 / native 95.40; noisy adamw 89.15 / vhat 90.77 / muon 92.02 /
native 82.32 (100% memorized).
"""
import json
import math

import torch
import torch.nn.functional as F

from concord_ref import ConcordRef, adamw_with_winner_schedule, swap_to_deploy
from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, BATCH, SEEDS
from exp9_muon import MuonConcord, MuonNative

torch.set_num_threads(4)
EPOCHS = 80
ARMS = {
    #  name           lr     kappa(clean, noisy)
    "adamw":        (1e-3, (None, None)),
    "concord_vhat": (1e-2, (0.0, 400.0)),
    "concord_muon": (1e-2, (0.0, 100.0)),
    "muon_native":  (1e-3, (None, None)),
}


def augment(xb, gen):
    """pad-2 random crop, per-sample, vectorized over the 25 shift groups."""
    B = xb.shape[0]
    img = xb.view(B, 28, 28)
    pad = F.pad(img, (2, 2, 2, 2))
    dx = torch.randint(0, 5, (B,), generator=gen)
    dy = torch.randint(0, 5, (B,), generator=gen)
    out = torch.empty_like(img)
    for sx in range(5):
        for sy in range(5):
            m = (dx == sx) & (dy == sy)
            if m.any():
                out[m] = pad[m][:, sx:sx + 28, sy:sy + 28]
    return out.reshape(B, -1)


def run(arm, nf, seed, data, aug):
    lr, (kap_c, kap_n) = ARMS[arm]
    kappa = kap_c if nf == 0.0 else kap_n
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    if arm == "adamw":
        opt, set_lr = adamw_with_winner_schedule(net.parameters(), lr, steps)
    elif arm == "concord_vhat":
        opt = ConcordRef(net, lr=lr, total_steps=steps, gate=True, kappa=kappa,
                         noise=False, generator=torch.Generator().manual_seed(seed + 10))
    elif arm == "concord_muon":
        opt = MuonConcord(net, lr=lr, total_steps=steps, gate=True, kappa=kappa,
                          noise=False, generator=torch.Generator().manual_seed(seed + 10))
    else:
        opt = MuonNative(net, lr=lr, total_steps=steps)
    t = 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(x[idx], gen) if aug else x[idx]
            loss = F.cross_entropy(net(xb), y[idx])
            if not math.isfinite(float(loss.detach())):
                return None
            if arm == "adamw":
                set_lr(t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t += 1
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0   # un-augmented
    if arm in ("adamw", "muon_native"):
        return accuracy(net, xte, yte), fit
    with swap_to_deploy(opt):
        return accuracy(net, xte, yte), fit


if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    out = {}
    for nf in (0.0, 0.30):
        for aug in (False, True):
            print(f"rho={nf:.0%} aug={'on' if aug else 'off'} (80 epochs):")
            for arm in ARMS:
                rs = [run(arm, nf, s, data, aug) for s in SEEDS]
                if any(r is None for r in rs):
                    print(f"  {arm:13s} DIVERGED")
                    out[(arm, nf, aug)] = None
                    continue
                accs = [r[0] for r in rs]
                fits = [r[1] for r in rs]
                out[(arm, nf, aug)] = (mean(accs), spread(accs), mean(fits))
                print(f"  {arm:13s} acc={mean(accs)*100:.2f}±{spread(accs)*100:.2f}%"
                      f"  memorized={mean(fits)*100:.1f}%", flush=True)
    json.dump({f"{a}|{n}|{int(g)}": v for (a, n, g), v in out.items()},
              open("exp10_results.json", "w"), indent=1)
