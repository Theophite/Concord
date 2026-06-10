"""CPU parity test for the shipped NS5 (concord/packed_b.py::ns5) against the
exp-9 reference (exp9_muon.py::ns5). packed_b imports triton at module top, so
we exec the shipped source (the test_autotuner_parity.py pattern). Checks:
  1. parity: shipped == reference on fp32 inputs (identical op sequence);
  2. semi-orthogonality: singular values of NS5(X) near 1 (the spectral bound
     that replaces eps/step_cap/trust region);
  3. idempotence: NS5(NS5(X)) ~ NS5(X);
  4. transpose path: NS5(X^T) == NS5(X)^T;
  5. bf16 sanity (the drive's actual regime) + the gamma=sqrt(max(N,K))
     per-element RMS ~ 0.7 claim (MUON_DRIVE.md).
Point at a different copy (e.g. the fork's prototype_packed_b.py) via NS5_SRC."""
import os
import re

import torch

SRC_PATH = os.environ.get("NS5_SRC", "../../concord/packed_b.py")
SRC = open(SRC_PATH, encoding="utf-8").read()
m = re.search(r"^def ns5.*?(?=^\S)", SRC, re.M | re.S)
assert m, "shipped ns5 not found"
ns = {"torch": torch}
exec(m.group(0), ns)
ns5_shipped = ns["ns5"]

REF = open("exp9_muon.py", encoding="utf-8").read()
mr = re.search(r"^def ns5.*?(?=^\S)", REF, re.M | re.S)
assert mr, "reference ns5 not found"
nr = {"torch": torch, "NS_STEPS": 5}
exec(mr.group(0), nr)
ns5_ref = nr["ns5"]

torch.manual_seed(0)
ok = True

def check(name, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

# 1. parity vs reference, several shapes (square / wide / tall)
worst = 0.0
for shape in [(64, 64), (64, 96), (96, 64), (128, 32)]:
    X = torch.randn(*shape)
    d = (ns5_shipped(X) - ns5_ref(X)).abs().max().item()
    worst = max(worst, d)
check("1. shipped == reference (fp32, 4 shapes)", worst < 1e-5, f"max|diff|={worst:.2e}")

# 2. semi-orthogonality: singular values near 1 (use the reference as the truth bar)
X = torch.randn(64, 96)
sv_s = torch.linalg.svdvals(ns5_shipped(X).float())
sv_r = torch.linalg.svdvals(ns5_ref(X).float())
dev_s = (sv_s - 1).abs().max().item()
dev_r = (sv_r - 1).abs().max().item()
check("2. semi-orthogonal: max|sv-1| <= ref + 1e-4 and < 0.5",
      dev_s <= dev_r + 1e-4 and dev_s < 0.5, f"shipped={dev_s:.3f} ref={dev_r:.3f}")

# 3. idempotence: NS5 of a semi-orthogonal matrix stays put (loose)
Y = ns5_shipped(X)
rel = ((ns5_shipped(Y) - Y).norm() / Y.norm()).item()
check("3. idempotence: |NS5(Y)-Y|/|Y| < 0.2", rel < 0.2, f"rel={rel:.3f}")

# 4. transpose path
d = (ns5_shipped(X.T) - ns5_shipped(X).T).abs().max().item()
check("4. transpose path: NS5(X^T) == NS5(X)^T", d < 1e-5, f"max|diff|={d:.2e}")

# 5. bf16 regime + the gamma RMS claim
Xb = torch.randn(256, 320).to(torch.bfloat16)
Yb = ns5_shipped(Xb)
fin = bool(torch.isfinite(Yb.float()).all())
sv_b = torch.linalg.svdvals(Yb.float())
gamma = float(max(Xb.shape)) ** 0.5
rms = (gamma * Yb.float()).pow(2).mean().sqrt().item()
check("5a. bf16: finite, sv in (0.3, 1.7)",
      fin and 0.3 < sv_b.min().item() and sv_b.max().item() < 1.7,
      f"sv=[{sv_b.min():.2f},{sv_b.max():.2f}]")
check("5b. gamma*NS5 per-element RMS ~ 0.7 (0.4..1.0)", 0.4 < rms < 1.0, f"rms={rms:.3f}")

print("\nALL NS5 PARITY TESTS PASSED" if ok else "\nNS5 PARITY FAILURES")
raise SystemExit(0 if ok else 1)
