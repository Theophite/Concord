"""The miser's experiment: how many RANKS of the v-hat correction must you pay
for, measured in ACCURACY (not energy)?

Take W_base (e.g. none/SGD), add back only the rank-k truncation of
dW = W_target - W_base (e.g. full/Adam) in each weight matrix, recalibrate BN,
evaluate. Sweep k. The k where accuracy saturates near W_target is the true
number of variance-directions that matter -- everything above it is variance
you are tracking on things that don't change the loss.

BN affine + biases are kept at target's values (negligible storage) and BN
running stats are recalibrated per reconstruction so the curve isolates the
weight-matrix low-rank correction. k=-1 => full (no truncation) = validator.

Storage framing: a rank-k weight/v-hat correction costs k*(R+C) per layer vs
full per-element v-hat R*C. (Effect-rank => motivates a rank-k v-hat surrogate;
not a drop-in proof that a low-rank optimizer reproduces it.)
"""
import argparse
import numpy as np
import torch

from cifar_vmode_fork import WiderConvNet, ANALYZE, evaluate
from cifar_in_memory import get_loaders_in_memory


def rank_k(dW, k):
    if k < 0:
        return dW
    if k == 0:
        return np.zeros_like(dW)
    U, s, Vt = np.linalg.svd(dW, full_matrices=False)
    k = min(k, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


@torch.no_grad()
def recalibrate_bn(model, tl, device, nbatch):
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats(); m.momentum = None      # cumulative mean
    model.train()
    for i, (x, y) in enumerate(tl):
        if i >= nbatch:
            break
        model(x.to(device, non_blocking=True))
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="vmode")
    ap.add_argument("--base", default="none")
    ap.add_argument("--target", default="full")
    ap.add_argument("--ks", default="0,1,2,4,8,16,32,64,-1")
    ap.add_argument("--bn_batches", type=int, default=300)
    ap.add_argument("--data_dir", default="./cifar_data")
    args = ap.parse_args()
    device = "cuda"
    tl, vl = get_loaders_in_memory(16, device, data_dir=args.data_dir)
    base = torch.load(f"{args.prefix}_{args.base}.pt", map_location="cpu",
                      weights_only=False)
    tgt = torch.load(f"{args.prefix}_{args.target}.pt", map_location="cpu",
                     weights_only=False)
    acc_base, acc_tgt = base["final"], tgt["final"]
    print(f"base={args.base} {acc_base*100:.2f}%   target={args.target} "
          f"{acc_tgt*100:.2f}%   gap {(acc_tgt-acc_base)*100:+.2f}", flush=True)

    dW, shp = {}, {}
    for ln in ANALYZE:
        Wb = base["state"][ln].numpy(); shp[ln] = Wb.shape
        Wt = tgt["state"][ln].numpy()
        dW[ln] = (Wt.reshape(Wb.shape[0], -1)
                  - Wb.reshape(Wb.shape[0], -1)).astype(np.float64)

    def bits_ratio(k):                          # full per-elem v / rank-k
        if k < 0:
            return 1.0
        num = sum(int(np.prod(shp[l])) for l in ANALYZE)
        den = sum((1 if k == 0 else k) * (shp[l][0] + int(np.prod(shp[l][1:])))
                  for l in ANALYZE)
        return num / max(den, 1)

    model = WiderConvNet().to(device)
    rows = []
    for k in [int(x) for x in args.ks.split(",")]:
        state = {kk: v.clone() for kk, v in tgt["state"].items()}   # BN/bias=tgt
        for ln in ANALYZE:
            Wb = base["state"][ln].numpy().reshape(shp[ln][0], -1)
            Wk = (Wb + rank_k(dW[ln], k)).reshape(shp[ln])
            state[ln] = torch.tensor(Wk, dtype=tgt["state"][ln].dtype)
        model.load_state_dict(state)
        recalibrate_bn(model, tl, device, args.bn_batches)
        acc, _ = evaluate(model, vl, device)
        klab = "full" if k < 0 else str(k)
        frac = ((acc - acc_base) / (acc_tgt - acc_base + 1e-9)) if k >= 0 else 1.0
        rows.append((klab, k, acc, frac))
        print(f"  rank {klab:>4}: val_acc {acc*100:.2f}%   "
              f"recovers {frac*100:5.1f}% of gap   "
              f"(v-hat {bits_ratio(k):.0f}x cheaper than full)", flush=True)

    # smallest k recovering >=90% of the accuracy gap
    star = next((r for r in rows if r[1] >= 0 and r[3] >= 0.90), None)
    print("\n==== MISER'S ANSWER ====")
    if star:
        print(f"  rank-{star[0]} recovers {star[3]*100:.0f}% of the Adam gap "
              f"-> a rank-{star[0]} v-hat surrogate is ~{bits_ratio(star[1]):.0f}x "
              f"cheaper than full per-element v-hat.")
    else:
        print("  no k<full reached 90% of the gap (correction not low-rank "
              "in accuracy terms).")


if __name__ == "__main__":
    main()
