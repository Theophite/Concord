"""Exp 22: the s_slow chase window — the telescope's missing middle rung.

Exp 20 pinned the ANCHOR window at one epoch (freshness law: every example
votes once before motion counts as drift). But the cascade's middle rung was
never swept: the chase runs at alpha=0.1 (a ~5-step window), so the drift
signal C*(S-A) compares ~5 steps of commits against the epoch integral — a
noisy numerator. Proposal (user, 2026-06-11): v_slow at epoch length, s_slow
at SOME FRACTION of epoch length, so coherence becomes trend agreement
between two well-sampled integrals, and deploy = S+A becomes a Polyak-style
fraction-of-epoch average of the trajectory.

Constraints this must respect:
  - rho = alpha_v/alpha < 0.5 or C* = L*2rho/(1-2rho) blows up: the chase
    window must stay SHORTER than half the anchor window. Sweep tops out at
    epoch/2.5 (rho = 0.4).
  - Init consolidation: load_weights packs the whole init into u, and at
    slow alpha it would sit friction-exposed with deploy ~ 0 for ~1/alpha
    steps. The telescope's zero-input fixed point is S = A (leak fixed
    point), so the paired fix is SPLIT-INIT packing: S = A = W0/2, u = 0 —
    the system starts AT its static equilibrium, drift signal exactly 0.
    (On MNIST from-scratch the init is small random noise, so split-init is
    predicted ~neutral here; its real payoff — killing the init-residue
    coherence artifact — is a fine-tune phenomenon. The alpha=0.1 packing
    pair is the neutrality control.)
  - Friction exposure: hypotheses live in u for ~1/alpha steps before
    committing, so at fixed F the dissipation taxes them over a longer
    window. F* should DROP as the chase slows; phase 2 sweeps F at the
    best window.

Protocol: v-hat drive (ConcordRef — the fork's drive), lr 1e-3, fluctuation
OFF (the user's SDXL operating point), gate on, min-leak 0.1, pad-2 crop
aug, 4k MNIST x 25 ep, 3 seeds. BATCH=32 (not exp-20's 128) so the epoch is
125 steps and the window ladder {ep/25=legacy, ep/16, ep/8, ep/4, ep/2.5}
spans a 10x range; exp-20/21 anchors do NOT transfer — the alpha=0.1 arms
are the internal baseline. W_anchor = 1 epoch throughout (exp-20 optimum).

Metrics per arm: DEPLOY acc (S+A), LIVE acc (u+S+A; deploy>live in noise =
the Polyak smoothing claim), memorized fraction on flipped labels, mean coh.
"""
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

from exp3_mnist import accuracy, load_mnist, make_net
from exp6_autotune import make_run, SEEDS
from exp10_aug_ablation import augment
from concord_ref import ConcordRef, swap_to_deploy

torch.set_num_threads(4)
EPOCHS, LR, BATCH = 25, 1e-3, 32
SPE = 4000 // BATCH                      # 125 steps/epoch
ALPHA_V = 1.0 / (2.0 * SPE)              # anchor window = 1 epoch
# chase windows: steps = 1/(2*alpha); labels in epoch fractions
WINDOWS = (("ep/25", 0.1), ("ep/16", 0.064), ("ep/8", 0.032),
           ("ep/4", 0.016), ("ep/2.5", 0.010))
F_DEFAULT = 0.5
F_SWEEP = (0.0, 0.25, 1.0)               # phase 2, at the best window


def split_init(opt):
    """Repack init at the telescope's zero-input fixed point: S = A = W0/2,
    u = 0. Live weight u+S+A is invariant; drift C*(S-A) starts exactly 0."""
    with torch.no_grad():
        for st in opt.swapped:
            w0 = st["u"].clone()
            st["u"].zero_()
            st["S"].copy_(w0 / 2)
            st["A"].copy_(w0 / 2)


def run(alpha, packing, nf, friction_F, seed, data):
    x, y, flip, gen, xte, yte = make_run(nf, seed, data)
    net = make_net(seed)
    steps = EPOCHS * (len(x) // BATCH)
    opt = ConcordRef(net, lr=LR, total_steps=steps, gate=True, kappa=0.0,
                     friction_F=(friction_F if friction_F > 0 else None),
                     alpha=alpha, alpha_v=ALPHA_V, noise=False,
                     generator=torch.Generator().manual_seed(seed + 10))
    if packing == "split":
        split_init(opt)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(x[idx], gen)
            loss = F.cross_entropy(net(xb), y[idx])
            if not math.isfinite(float(loss.detach())):
                return None
            opt.zero_grad()
            loss.backward()
            opt.step()
    live = accuracy(net, xte, yte)
    fit = accuracy(net, x[flip], y[flip]) if flip.any() else 0.0
    with swap_to_deploy(opt):
        dep = accuracy(net, xte, yte)
    return dep, live, fit, opt.mean_coh()


def cell(tag, alpha, packing, nf, fF, data, results, key, t0):
    rs = [run(alpha, packing, nf, fF, s, data) for s in SEEDS]
    if any(r is None for r in rs):
        results[key] = None
        print(f"{tag}: DIVERGED   [{time.time()-t0:.0f}s]", flush=True)
        return None
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
    lv, mf, mc = (mean([r[1] for r in rs]), mean([r[2] for r in rs]),
                  mean([r[3] for r in rs]))
    results[key] = (m, sp, lv, mf, mc)
    print(f"{tag}: deploy={m*100:.2f}±{sp*100:.2f}%  live={lv*100:.2f}%  "
          f"memorized={mf*100:.1f}%  coh={mc:.3f}   [{time.time()-t0:.0f}s]",
          flush=True)
    return m


def dump(results):
    json.dump({"|".join(map(str, k)): v for k, v in results.items()},
              open("exp22_results.json", "w"), indent=1)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        EPOCHS, sds = 2, (0,)
        SEEDS = sds  # noqa: F811 (smoke only)
    data = load_mnist()
    results = {}
    t0 = time.time()

    print(f"phase 1: window ladder at F={F_DEFAULT} "
          f"(SPE={SPE}, anchor=1ep, alpha_v={ALPHA_V:g})", flush=True)
    for nf in (0.0, 0.30):
        for label, alpha in WINDOWS:
            packs = ("u", "split") if alpha == 0.1 else ("split",)
            for packing in packs:
                tag = (f"W_c={label:7s} a={alpha:.3f} {packing:5s} "
                       f"F={F_DEFAULT} noise={nf:.0%}")
                cell(tag, alpha, packing, nf, F_DEFAULT, data, results,
                     ("p1", label, packing, nf, F_DEFAULT), t0)
                dump(results)
        if smoke:
            break

    if not smoke:
        # phase 2: F sweep at the best noisy window (split packing)
        noisy = {k: v for k, v in results.items()
                 if k[0] == "p1" and k[2] == "split" and k[3] == 0.30 and v}
        best_label = max(noisy, key=lambda k: noisy[k][0])[1]
        best_alpha = dict(WINDOWS)[best_label]
        print(f"\nphase 2: F sweep at best noisy window {best_label}", flush=True)
        for fF in F_SWEEP:
            for nf in (0.30, 0.0):
                tag = (f"W_c={best_label:7s} a={best_alpha:.3f} split "
                       f"F={fF} noise={nf:.0%}")
                cell(tag, best_alpha, "split", nf, fF, data, results,
                     ("p2", best_label, "split", nf, fF), t0)
                dump(results)
    print(f"\ndone [{time.time()-t0:.0f}s]", flush=True)
