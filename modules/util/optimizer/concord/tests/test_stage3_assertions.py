"""Tests for the Stage-3 (CUDA-graph) design assertions, so the build stands on
verified ground. Run with the OneTrainer venv.

  A1  bf16 => OneTrainer creates NO GradScaler (so the plain loss.backward() path is
      taken, which is the capturable one); fp16 => scaler is on. -> gate the graph on bf16.
  A2  torch.randn on the DEFAULT generator ADVANCES across CUDA-graph replays (fresh
      noise), but a CUSTOM torch.Generator does NOT (frozen or not capturable). Since
      OneTrainer's predict() seeds a custom per-step generator for the diffusion noise,
      that noise must be computed EAGERLY and fed via static buffers (eager-feed split).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))   # OneTrainer repo root

import torch
import torch.nn as nn

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# --- A1: bf16 => no GradScaler, fp16 => GradScaler ---
print("A1: grad-scaler gate (bf16 vs fp16)")
from modules.util.dtype_util import enable_grad_scaling
from modules.util.enum.DataType import DataType
p = [nn.Parameter(torch.zeros(2, device="cuda"))]
bf16_scale = bool(enable_grad_scaling(DataType.BFLOAT_16, p))
fp16_scale = bool(enable_grad_scaling(DataType.FLOAT_16, p))
check("bf16 -> no grad scaling (scaler is None)", bf16_scale is False, f"enable_grad_scaling(bf16)={bf16_scale}")
check("fp16 -> grad scaling on", fp16_scale is True, f"enable_grad_scaling(fp16)={fp16_scale}")


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


# --- A2a: default generator advances across replays ---
print("\nA2: noise generator under graph replay")
x = torch.zeros(8, device="cuda")
gd = capture(lambda: x.copy_(torch.randn(8, device="cuda")))
gd.replay(); a = x.clone(); gd.replay(); b = x.clone()
check("default generator ADVANCES per replay (fresh noise)", not torch.equal(a, b))

# --- A2b: custom generator does NOT advance (frozen) or is not capturable ---
gen = torch.Generator(device="cuda").manual_seed(0)
y = torch.zeros(8, device="cuda")
custom_ok = True
try:
    gc = capture(lambda: y.copy_(torch.randn(8, device="cuda", generator=gen)))
    gc.replay(); c = y.clone(); gc.replay(); d = y.clone()
    frozen = torch.equal(c, d)
    check("custom generator does NOT advance (frozen) -> must feed eagerly", frozen,
          "captured but identical draws" if frozen else "ADVANCED (unexpected!)")
except Exception as e:
    check("custom generator not capturable -> must feed eagerly", True,
          f"capture raised {type(e).__name__}")

print(f"\n[RESULT] {sum(results)}/{len(results)} assertions hold "
      + ("-> Stage-3 design (gate bf16 + eager-feed noise) is sound" if all(results)
         else "-> REVISIT design"))
