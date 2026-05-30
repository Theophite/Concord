"""CIFAR side-by-side: int16 s_fast vs int8 s_fast (delta storage).

Two builds of the same WiderConvNet, identical seeds:
  Run A: ConcordLinearFused / ConcordConv2dFused (int16 s_fast)
  Run B: ConcordLinearFusedInt8 / ConcordConv2dFusedInt8 (int8 s_fast,
         delta storage; mass-preserve chase baked in)

We report:
  - Final loss (correctness check)
  - Train acc (correctness check; should be within ~2-3% of int16)
  - Mean |s_slow|, |s_fast|, |delta| stats per layer
  - Range pressure on s_fast int8 (% of params hitting +/-128 clamp;
    should be ~0% in normal regime)

This is a SMOKE TEST, not a converged-run accuracy report. It
validates the kernels compile, the autograd Functions plumb through,
and the dynamic produces sensible state magnitudes. For accuracy
sign-off, use the full cifar_concord_adamw.py with both class sets.

Run:
    python test_int8_path.py
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(device, use_int8):
    """Same architecture as test_mass_preserve_chase.py uses, with
    optional int8 s_fast layers."""
    if use_int8:
        from concord_linear_fused import (ConcordLinearFusedInt8 as Linear,
                                            ConcordConv2dFusedInt8 as Conv2d)
    else:
        from concord_linear_fused import (ConcordLinearFused as Linear,
                                            ConcordConv2dFused as Conv2d)

    class WiderConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(3, 32, 3, padding=1, device=device)
            self.conv2 = Conv2d(32, 64, 3, padding=1, device=device)
            self.conv3 = Conv2d(64, 128, 3, padding=1, device=device)
            self.conv4 = Conv2d(128, 256, 3, padding=1, device=device)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = Linear(256 * 2 * 2, 512, device=device)
            self.fc2 = Linear(512, 256, device=device)
            self.fc3 = Linear(256, 10, device=device)

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
    Conv2dCls = type(model.conv1)
    LinearCls = type(model.fc1)
    for m in model.modules():
        if isinstance(m, (LinearCls, Conv2dCls)):
            m.optimizer_v_kind = 'three_accum'
            m.set_optimizer_kind('adamw', weight_decay=0.01)
            m.lr = 0.1 * 0.2  # v_lr_scale
            m.wd_sv = 1e-5
            m.wd_sf = 1e-5
    return model


def measure_state(model):
    """Per-layer stats: |s_slow|, |s_fast|, plus s_fast clamp-hit rate
    (% of params at |s_fast| == 127, which signals int8 saturation
    pressure)."""
    from concord_linear_fused import (ConcordLinearFused, ConcordConv2dFused,
                                        ConcordLinearFusedInt8,
                                        ConcordConv2dFusedInt8)
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (ConcordLinearFused, ConcordConv2dFused,
                          ConcordLinearFusedInt8, ConcordConv2dFusedInt8)):
            ss = m.s_slow.to(torch.int32).abs().float().mean().item()
            sf = m.s_fast.to(torch.int32).abs().float().mean().item()
            is_int8 = m.s_fast.dtype == torch.int8
            if is_int8:
                clamp_hits = (m.s_fast.abs() >= 127).float().mean().item()
            else:
                clamp_hits = 0.0
            stats.append({
                'name': name, 'slow': ss, 'fast': sf,
                'clamp_hits_pct': clamp_hits * 100,
                'dtype': str(m.s_fast.dtype),
            })
    return stats


def run(use_int8, n_epochs=2):
    """Train the WiderConvNet on synthetic CIFAR-shaped data for n_epochs.
    For the int8 path, mass-preserve chase is baked into the kernel
    (not a separate toggle); the int16 path runs in its default
    non-mass-preserving mode for direct dynamic comparison."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    model = build_model(device, use_int8)

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
    stats = measure_state(model)
    return {
        'use_int8': use_int8,
        'final_loss': sum(losses[-10:]) / 10,
        'train_acc': correct / total,
        'stats': stats,
        'wall_s': dt,
    }


def main():
    print("=== Run A: int16 s_fast (baseline) ===")
    r_off = run(use_int8=False, n_epochs=3)
    print(f"  final_loss: {r_off['final_loss']:.4f}")
    print(f"  train_acc:  {r_off['train_acc']:.3f}")
    print(f"  wall_s:     {r_off['wall_s']:.1f}")
    for s in r_off['stats']:
        print(f"    {s['name']:>10}: |slow|={s['slow']:7.1f}  "
              f"|fast|={s['fast']:7.1f}  ({s['dtype']})")
    print()
    print("=== Run B: int8 s_fast (delta storage, mass-preserve baked in) ===")
    r_on = run(use_int8=True, n_epochs=3)
    print(f"  final_loss: {r_on['final_loss']:.4f}")
    print(f"  train_acc:  {r_on['train_acc']:.3f}")
    print(f"  wall_s:     {r_on['wall_s']:.1f}")
    for s in r_on['stats']:
        print(f"    {s['name']:>10}: |slow|={s['slow']:7.1f}  "
              f"|delta|={s['fast']:6.2f}  clamp_hits={s['clamp_hits_pct']:.2f}%  "
              f"({s['dtype']})")
    print()
    print("=== Comparison ===")
    print(f"  Loss: int16 {r_off['final_loss']:.4f} -> "
          f"int8 {r_on['final_loss']:.4f}  "
          f"(diff {r_on['final_loss'] - r_off['final_loss']:+.4f})")
    print(f"  Train acc: int16 {r_off['train_acc']:.3f} -> "
          f"int8 {r_on['train_acc']:.3f}")
    print(f"  Wall: int16 {r_off['wall_s']:.1f}s -> "
          f"int8 {r_on['wall_s']:.1f}s  "
          f"(int8 should be similar; same kernel pattern, different storage)")

    # Sanity criteria
    train_acc_dropped = r_off['train_acc'] - r_on['train_acc']
    if train_acc_dropped > 0.10:
        print(f"  [WARN] int8 hurt train accuracy by "
              f"{train_acc_dropped:.3f}; revisit dynamics")
    else:
        print(f"  [OK] train accuracy within {train_acc_dropped:.3f} "
              f"of int16 baseline")

    # Clamp pressure check
    max_clamp = max((s['clamp_hits_pct'] for s in r_on['stats']),
                     default=0.0)
    if max_clamp > 1.0:
        print(f"  [WARN] max clamp_hits = {max_clamp:.2f}% — "
              f"int8 delta is saturating; tune alpha / lr / step_cap")
    else:
        print(f"  [OK] max clamp_hits = {max_clamp:.4f}% (int8 has plenty "
              f"of headroom)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        sys.exit(0)
    main()
