"""CIFAR side-by-side: mass-preserving chase ON vs OFF.

Runs cifar_concord_adamw.py's training loop twice with identical seeds
and config, differing only in _GLOBAL_MASS_PRESERVE_CHASE. Reports:
  - Final val accuracy (correctness check: mass-preserve shouldn't hurt)
  - Mean |s_fast| per layer (key signal: should be ~1000 baseline,
    ~10 with mass-preserve -- the latter fits int8 easily)
  - Mean |s_slow| (should be similar in both: holds the cumulative magnitude)

If both runs converge to similar val_acc AND |s_fast| drops 2+ orders
of magnitude in the mass-preserve run, we can proceed to int8 s_fast.

Run:
    python test_mass_preserve_chase.py
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(device):
    """Same architecture as cifar_concord_adamw.py uses."""
    from concord_linear_fused import ConcordLinearFused, ConcordConv2dFused

    class WiderConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConcordConv2dFused(3, 32, 3, padding=1, device=device)
            self.conv2 = ConcordConv2dFused(32, 64, 3, padding=1, device=device)
            self.conv3 = ConcordConv2dFused(64, 128, 3, padding=1, device=device)
            self.conv4 = ConcordConv2dFused(128, 256, 3, padding=1, device=device)
            self.pool = nn.MaxPool2d(2, 2)
            # 256 channels * 2x2 spatial after 4 pools (32->16->8->4->2)
            self.fc1 = ConcordLinearFused(256 * 2 * 2, 512, device=device)
            self.fc2 = ConcordLinearFused(512, 256, device=device)
            self.fc3 = ConcordLinearFused(256, 10, device=device)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    model = WiderConvNet().to(device)
    for m in model.modules():
        if isinstance(m, (ConcordLinearFused, ConcordConv2dFused)):
            m.optimizer_v_kind = 'three_accum'
            m.set_optimizer_kind('adamw', weight_decay=0.01)
            m.lr = 0.1 * 0.2  # v_lr_scale
            m.wd_sv = 1e-5
            m.wd_sf = 1e-5
    return model


def measure_state(model):
    """Mean |s_slow|, |s_fast| across all Concord layers."""
    from concord_linear_fused import ConcordLinearFused, ConcordConv2dFused
    slow_means, fast_means = [], []
    for m in model.modules():
        if isinstance(m, (ConcordLinearFused, ConcordConv2dFused)):
            slow_means.append(m.s_slow.abs().float().mean().item())
            fast_means.append(m.s_fast.abs().float().mean().item())
    return (sum(slow_means) / len(slow_means),
            sum(fast_means) / len(fast_means))


def run(mass_preserve_chase, n_epochs=2):
    import concord_triton_fused
    concord_triton_fused.set_mass_preserve_chase(mass_preserve_chase)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    model = build_model(device)

    # Tiny CIFAR-shaped synthetic for speed (real CIFAR loader is fine
    # too but adds setup cost; the dynamic we want to measure is the
    # state-magnitude convergence, not data dependence).
    N = 2048
    x = torch.randn(N, 3, 32, 32, device=device).clamp(-3, 3)
    y_true_W = torch.randn(10, 3 * 32 * 32, device=device) * 0.1
    y = (x.flatten(1) @ y_true_W.T).argmax(dim=1)

    bsz = 64
    losses = []
    correct = 0
    total = 0
    t0 = time.time()
    for ep in range(n_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bsz):
            idx = perm[i:i + bsz]
            xb = x[idx]
            yb = y[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            losses.append(loss.item())
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.numel()
    dt = time.time() - t0
    slow_mean, fast_mean = measure_state(model)
    return {
        'mass_preserve': mass_preserve_chase,
        'final_loss': sum(losses[-10:]) / 10,
        'train_acc': correct / total,
        'slow_mean': slow_mean,
        'fast_mean': fast_mean,
        'wall_s': dt,
    }


def main():
    print("=== Run A: mass_preserve_chase=False (baseline) ===")
    r_off = run(mass_preserve_chase=False, n_epochs=3)
    for k, v in r_off.items():
        print(f"  {k}: {v}")
    print()
    print("=== Run B: mass_preserve_chase=True ===")
    r_on = run(mass_preserve_chase=True, n_epochs=3)
    for k, v in r_on.items():
        print(f"  {k}: {v}")
    print()
    print("=== Comparison ===")
    print(f"  Loss: baseline {r_off['final_loss']:.4f} -> "
          f"mass-preserve {r_on['final_loss']:.4f}  "
          f"(diff {r_on['final_loss'] - r_off['final_loss']:+.4f})")
    print(f"  Train acc: baseline {r_off['train_acc']:.3f} -> "
          f"mass-preserve {r_on['train_acc']:.3f}")
    print(f"  |s_slow|: baseline {r_off['slow_mean']:.1f} -> "
          f"mass-preserve {r_on['slow_mean']:.1f}  "
          f"(should be similar)")
    print(f"  |s_fast|: baseline {r_off['fast_mean']:.1f} -> "
          f"mass-preserve {r_on['fast_mean']:.1f}  "
          f"(should drop dramatically)")
    fast_ratio = r_on['fast_mean'] / max(r_off['fast_mean'], 1e-9)
    print(f"           ratio: {fast_ratio:.4f}  "
          f"(< 0.05 = ~int8-friendly)")

    # Sanity criteria
    train_acc_dropped = r_off['train_acc'] - r_on['train_acc']
    if train_acc_dropped > 0.10:
        print(f"  [WARN] mass-preserve hurt train accuracy by "
              f"{train_acc_dropped:.3f}; revisit dynamics")
    else:
        print(f"  [OK] train accuracy within {train_acc_dropped:.3f} "
              f"of baseline")

    if r_on['fast_mean'] < 100:
        print(f"  [OK] |s_fast| = {r_on['fast_mean']:.1f} fits in int8")
    else:
        print(f"  [WARN] |s_fast| = {r_on['fast_mean']:.1f} does NOT "
              f"fit in int8; chase rate alpha may need tuning")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        sys.exit(0)
    main()
