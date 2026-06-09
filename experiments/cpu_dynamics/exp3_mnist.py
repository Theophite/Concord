"""Exp 3: MNIST ablation — AdamW vs Concord arms, CPU.

MLP 784-256-10, batch 128, 2 epochs over the full 60k train set (938 steps),
winner schedule (warmup 100, cosine to 0.2). No per-arm lr tuning: every arm
runs at the same peak lr (1e-3, the standard AdamW MNIST choice).

Arms:
  adamw        torch.optim.AdamW (wd=0), same schedule
  bare         Concord, gates open, kappa=0, no noise   (preconditioner+cascade only)
  dissip       + ratio-coherence gates + evaporation (kappa=50)
  winner       + rising-late isotropic gradient noise (sigma 0.6)
  winner_c2    winner with the drift-cancel coefficient doubled
               (C* refit for the mass-preserving leak: rho = 2*alpha_v)

Reported: test accuracy at the live weights W and the deploy weights P,
mean +/- spread over 3 seeds.
"""
import struct
import time

import torch

from concord_ref import (ConcordRef, adamw_with_winner_schedule,
                         compute_drift_cancel_C, swap_to_deploy)

torch.set_num_threads(4)
EPOCHS, BATCH, LR, SEEDS = 2, 128, 1e-3, (0, 1, 2)


def load_idx(path):
    with open(path, "rb") as f:
        magic = struct.unpack(">i", f.read(4))[0]
        dims = [struct.unpack(">i", f.read(4))[0] for _ in range(magic % 256)]
        data = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
    return data.reshape(dims)


def load_mnist():
    xtr = load_idx("data/train-images-idx3-ubyte").reshape(-1, 784).float() / 255.0
    ytr = load_idx("data/train-labels-idx1-ubyte").long()
    xte = load_idx("data/t10k-images-idx3-ubyte").reshape(-1, 784).float() / 255.0
    yte = load_idx("data/t10k-labels-idx1-ubyte").long()
    return xtr, ytr, xte, yte


def make_net(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))


@torch.no_grad()
def accuracy(net, x, y):
    return (net(x).argmax(1) == y).float().mean().item()


def run(arm, seed, data):
    xtr, ytr, xte, yte = data
    net = make_net(seed)
    steps = EPOCHS * (len(xtr) // BATCH)
    kw = dict(lr=LR, total_steps=steps,
              generator=torch.Generator().manual_seed(seed + 10))
    if arm == "adamw":
        opt, set_lr = adamw_with_winner_schedule(net.parameters(), LR, steps)
    elif arm == "bare":
        opt = ConcordRef(net, gate=False, kappa=0.0, noise=False, **kw)
    elif arm == "dissip":
        opt = ConcordRef(net, gate=True, kappa=50.0, noise=False, **kw)
    elif arm == "winner":
        opt = ConcordRef(net, gate=True, kappa=50.0, noise=True, **kw)
    elif arm == "winner_c2":
        opt = ConcordRef(net, gate=True, kappa=50.0, noise=True, **kw)
        opt.Cstar = 2.0 * compute_drift_cancel_C()
    gen = torch.Generator().manual_seed(seed)
    t = 0
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(xtr) - BATCH + 1, BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
            if arm == "adamw":
                set_lr(t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t += 1
    acc_live = accuracy(net, xte, yte)
    if arm == "adamw":
        return acc_live, acc_live
    with swap_to_deploy(opt):
        acc_dep = accuracy(net, xte, yte)
    return acc_live, acc_dep


if __name__ == "__main__":
    data = load_mnist()
    print(f"steps/run = {EPOCHS * (len(data[0]) // BATCH)}")
    for arm in ("adamw", "bare", "dissip", "winner", "winner_c2"):
        lives, deps = [], []
        t0 = time.time()
        for seed in SEEDS:
            al, ad = run(arm, seed, data)
            lives.append(al)
            deps.append(ad)
        mean = lambda v: sum(v) / len(v)
        spread = lambda v: (max(v) - min(v)) / 2
        print(f"{arm:10s} live={mean(lives)*100:.2f}±{spread(lives)*100:.2f}%  "
              f"deploy={mean(deps)*100:.2f}±{spread(deps)*100:.2f}%  "
              f"({time.time()-t0:.0f}s)")
