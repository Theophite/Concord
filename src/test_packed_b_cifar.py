"""CIFAR-style synthetic benchmark for packed B vs existing int8 path.

Uses the same WiderConvNet shape and AdamW three_accum config as
test_int8_path.py. Three runs, identical seeds:
  Run A: existing ConcordLinearFusedInt8 / ConcordConv2dFusedInt8
         (s_slow int16, s_fast_delta int8, v_slow_i8 int8 — 3 buffers)
  Run B: new ConcordLinearPackedB / ConcordConv2dPackedB
         (one int32 packed buffer per layer with all 3 accumulators)
  Run C (optional): torch.optim.AdamW baseline (fp32, fast reference)

Reports per-layer state magnitudes + final train accuracy. The key
diagnostic is whether packed B converges similarly to the existing
int8 path at the same hyperparameters — a "swap of which accumulator
gets 16 vs 8 bits + packed-into-one-word" shouldn't change the
dynamics qualitatively.

Synthetic CIFAR-shape data (not real CIFAR-10) for speed.

Run:
    python test_packed_b_cifar.py
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model_int8(device):
    from concord_linear_fused import (ConcordLinearFusedInt8,
                                        ConcordConv2dFusedInt8)

    class WiderConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConcordConv2dFusedInt8(3, 32, 3, padding=1, device=device)
            self.conv2 = ConcordConv2dFusedInt8(32, 64, 3, padding=1, device=device)
            self.conv3 = ConcordConv2dFusedInt8(64, 128, 3, padding=1, device=device)
            self.conv4 = ConcordConv2dFusedInt8(128, 256, 3, padding=1, device=device)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = ConcordLinearFusedInt8(256 * 2 * 2, 512, device=device)
            self.fc2 = ConcordLinearFusedInt8(512, 256, device=device)
            self.fc3 = ConcordLinearFusedInt8(256, 10, device=device)

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
        if isinstance(m, (ConcordLinearFusedInt8, ConcordConv2dFusedInt8)):
            m.optimizer_v_kind = 'three_accum'
            m.set_optimizer_kind('adamw', weight_decay=0.01)
            m.lr = 0.1 * 0.2  # v_lr_scale
            m.wd_sv = 1e-5
            m.wd_sf = 1e-5
    return model


def build_model_packedB(device):
    from prototype_packed_b import (ConcordLinearPackedB,
                                      ConcordConv2dPackedB,
                                      compute_drift_cancel_C)

    class WiderConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConcordConv2dPackedB(3, 32, 3, padding=1, device=device)
            self.conv2 = ConcordConv2dPackedB(32, 64, 3, padding=1, device=device)
            self.conv3 = ConcordConv2dPackedB(64, 128, 3, padding=1, device=device)
            self.conv4 = ConcordConv2dPackedB(128, 256, 3, padding=1, device=device)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = ConcordLinearPackedB(256 * 2 * 2, 512, device=device)
            self.fc2 = ConcordLinearPackedB(512, 256, device=device)
            self.fc3 = ConcordLinearPackedB(256, 10, device=device)

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
        if isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB)):
            m.set_optimizer_kind('adamw', weight_decay=0.01,
                                  eps=1.0, step_cap=10.0)
            m.lr = 0.1 * 0.2  # v_lr_scale
            m.alpha = 0.1
            m.alpha_v_fast = 0.001
            m.drift_cancel_C = compute_drift_cancel_C(m.alpha, m.alpha_v_fast)
            m.wd_sv = 1e-5
            m.wd_sf = 1e-5
    return model


def measure_int8(model):
    from concord_linear_fused import (ConcordLinearFusedInt8,
                                        ConcordConv2dFusedInt8)
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (ConcordLinearFusedInt8, ConcordConv2dFusedInt8)):
            stats.append({
                'name': name,
                'slow_mean': m.s_slow.abs().float().mean().item(),
                'fast_mean': m.s_fast.abs().float().mean().item(),
                'fast_dtype': str(m.s_fast.dtype),
            })
    return stats


def measure_packedB(model):
    from prototype_packed_b import (ConcordLinearPackedB,
                                      ConcordConv2dPackedB)
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, (ConcordLinearPackedB, ConcordConv2dPackedB)):
            s_fast, s_slow_i8, v_slow_i8 = m.get_state()
            stats.append({
                'name': name,
                'slow_mean': (s_slow_i8.abs().float().mean().item() * 128),  # in mantissa units
                'fast_mean': s_fast.abs().float().mean().item(),
                'v_mean': (v_slow_i8.abs().float().mean().item() * 128),
                'fast_dtype': 'int16 (packed)',
            })
    return stats


def run(model_builder, measure_fn, name, n_epochs=2):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = 'cuda'
    model = model_builder(device)

    # Synthetic CIFAR-shape data.
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
    return {
        'name': name,
        'final_loss': sum(losses[-10:]) / 10,
        'train_acc': correct / total,
        'stats': measure_fn(model),
        'wall_s': dt,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    print("=== Run A: existing int8 path (s_slow i16 + s_fast i8 + v_slow_i8) ===")
    r_a = run(build_model_int8, measure_int8, 'int8 path', n_epochs=3)
    print(f"  final_loss: {r_a['final_loss']:.4f}")
    print(f"  train_acc:  {r_a['train_acc']:.3f}")
    print(f"  wall_s:     {r_a['wall_s']:.1f}")
    for s in r_a['stats']:
        print(f"    {s['name']:>10}: |slow|={s['slow_mean']:7.1f}  "
              f"|fast|={s['fast_mean']:6.1f}  ({s['fast_dtype']})")
    print()

    print("=== Run B: packed-B (i16 fast + i8×128 slow + i8×128 v, one int32) ===")
    r_b = run(build_model_packedB, measure_packedB, 'packed-B', n_epochs=3)
    print(f"  final_loss: {r_b['final_loss']:.4f}")
    print(f"  train_acc:  {r_b['train_acc']:.3f}")
    print(f"  wall_s:     {r_b['wall_s']:.1f}")
    for s in r_b['stats']:
        print(f"    {s['name']:>10}: |slow|={s['slow_mean']:7.1f}  "
              f"|fast|={s['fast_mean']:6.1f}  |v|={s['v_mean']:7.1f}  "
              f"({s['fast_dtype']})")
    print()

    print("=== Comparison ===")
    print(f"  Loss:      int8 {r_a['final_loss']:.4f} -> "
          f"packed-B {r_b['final_loss']:.4f}  "
          f"(diff {r_b['final_loss'] - r_a['final_loss']:+.4f})")
    print(f"  Train acc: int8 {r_a['train_acc']:.3f} -> "
          f"packed-B {r_b['train_acc']:.3f}  "
          f"(diff {r_b['train_acc'] - r_a['train_acc']:+.3f})")
    print(f"  Wall:      int8 {r_a['wall_s']:.1f}s -> "
          f"packed-B {r_b['wall_s']:.1f}s "
          f"({r_b['wall_s']/r_a['wall_s']:.2f}x)")

    train_acc_dropped = r_a['train_acc'] - r_b['train_acc']
    if train_acc_dropped > 0.10:
        print(f"  [WARN] packed-B hurt train acc by {train_acc_dropped:.3f}")
    else:
        print(f"  [OK] train acc within {train_acc_dropped:.3f} of int8 baseline")


if __name__ == "__main__":
    main()
