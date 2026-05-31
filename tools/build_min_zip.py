"""Build the LEAN Concord package: just the elements that work -- the validated
recipe (rank-1 v-hat AdamW + fixed coherence gate) as the bare default, the 32-bit
EVAPORATE/CONSOLIDATE recipe (ratio-coh + floor + (1-coh) evaporation), and
deploy-slow (consolidated_weight). Strips the experimental dead-ends that did NOT
pan out: coh-weighted-v (cwv, neutral) and fast_gain forward anneal (no overfit win).

Reproducible + atomic: reads canonical src/prototype_packed_b.py, transforms IN
MEMORY with assert-guarded strips (fails loudly, writes nothing on mismatch), then
writes the staging package under dist/concord_min/. Zip the result with
Compress-Archive (no `zip` binary on this box).

    python tools/build_min_zip.py
    # then: Compress-Archive dist/concord_min/concord_optimizer_min -> concord_optimizer_min.zip
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "prototype_packed_b.py")
OUT = os.path.join(ROOT, "dist", "concord_min", "concord_optimizer_min")
PKG = os.path.join(OUT, "concord")
LOG = os.path.join(ROOT, "compare_out", "min_build.log")

log_lines = []
def log(m): log_lines.append(str(m))


def strip_if_block(text, marker, expect=1):
    """Remove a marker line + all following MORE-indented (non-blank) lines."""
    lines = text.split("\n")
    tgt = [i for i, l in enumerate(lines) if marker in l]
    assert len(tgt) == expect, ("strip_if_block", marker, len(tgt), expect)
    for i in sorted(tgt, reverse=True):
        indent = len(lines[i]) - len(lines[i].lstrip())
        j = i + 1
        while j < len(lines):
            l = lines[j]
            if l.strip() == "":
                break
            if (len(l) - len(l.lstrip())) <= indent:
                break
            j += 1
        del lines[i:j]
    log("strip_if_block %r -> removed at %s" % (marker, tgt))
    return "\n".join(lines)


def strip_line_plus_comments(text, marker, expect=1):
    """Remove a marker line + immediately-following pure-comment continuation lines."""
    lines = text.split("\n")
    tgt = [i for i, l in enumerate(lines) if marker in l]
    assert len(tgt) == expect, ("strip_line_plus_comments", marker, len(tgt), expect)
    for i in sorted(tgt, reverse=True):
        j = i + 1
        while j < len(lines) and lines[j].lstrip().startswith("#"):
            j += 1
        del lines[i:j]
    log("strip_line_plus_comments %r -> removed at %s" % (marker, tgt))
    return "\n".join(lines)


def strip_exact_block(text, block, expect=1):
    n = text.count(block)
    assert n == expect, ("strip_exact_block", repr(block[:40]), n, expect)
    log("strip_exact_block %r x%d" % (block.split("\n")[0][:50], n))
    return text.replace(block, "")


def collapse_blanks(text):
    out, blanks = [], 0
    for l in text.split("\n"):
        if l.strip() == "":
            blanks += 1
            if blanks <= 2:
                out.append(l)
        else:
            blanks = 0
            out.append(l)
    return "\n".join(out)


def main():
    src = open(SRC, encoding="utf-8").read()
    log("read %s (%d lines)" % (SRC, src.count("\n") + 1))

    # ---- strip coh-weighted-v (cwv): neutral on enwik8, no decisive win ----
    src = strip_exact_block(src,
        "def set_coh_weighted_v(enabled):\n"
        "    global _COH_WEIGHTED_V\n"
        "    _COH_WEIGHTED_V = bool(enabled)\n")
    src = strip_line_plus_comments(src, "_COH_WEIGHTED_V = False")
    src = strip_if_block(src, "if _COH_WEIGHTED_V and ctx.coh_pre is not None:", expect=2)

    # ---- strip fast_gain forward anneal: did NOT reduce overfit ----
    src = strip_if_block(src, "if fg < 1.0:", expect=1)
    src = strip_exact_block(src, "        fg = self.fast_gain\n")
    src = strip_line_plus_comments(src, "self.fast_gain = 1.0")

    # guard: the things we KEEP must survive
    for keep in ("def consolidated_weight", "def set_ratio_coh(", "def set_ratio_coh_floors",
                 "def set_fixed_coh", "def disable_cohpre", "self.gf_consol = 0.0",
                 "class ConcordLinearPackedB", "class ConcordConv2dPackedB"):
        assert keep in src, ("LOST a required symbol", keep)
    for gone in ("_COH_WEIGHTED_V", "fast_gain"):
        assert gone not in src, ("FAILED to fully strip", gone)

    src = collapse_blanks(src)

    os.makedirs(PKG, exist_ok=True)
    open(os.path.join(PKG, "packed_b.py"), "w", encoding="utf-8").write(src)
    open(os.path.join(PKG, "__init__.py"), "w", encoding="utf-8").write(INIT_PY)
    open(os.path.join(PKG, "recipe.py"), "w", encoding="utf-8").write(RECIPE_PY)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(README_MD)
    open(os.path.join(OUT, "test_min.py"), "w", encoding="utf-8").write(TEST_MIN_PY)
    open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8").write(REQS_TXT)
    log("wrote package to %s (%d lines in packed_b.py)" % (OUT, src.count("\n") + 1))

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    open(LOG, "w", encoding="utf-8").write("\n".join(log_lines) + "\nBUILD_OK\n")
    print("BUILD_OK ->", OUT)


INIT_PY = '''"""Concord -- packed-int storage optimizer (32 bits/param), lean build.

A bare ``ConcordLinearPackedB`` / ``ConcordConv2dPackedB`` is the validated
optimizer (rank-1 v-hat AdamW + fixed coherence gate); the update is fused into
the backward pass. For the 32-bit evaporate/consolidate recipe use
``enable_evaporate_consolidate``; deploy with ``layer.consolidated_weight()``.
"""
from .packed_b import (
    ConcordLinearPackedB,
    ConcordConv2dPackedB,
    compute_drift_cancel_C,
    set_fixed_coh,
    set_ratio_coh,
    set_ratio_coh_floors,
    S_SLOW_FACTOR,
    V_SLOW_FACTOR,
    MANTISSA_BIAS,
)
from .recipe import (
    enable_evaporate_consolidate,
    evaporate_consolidate_floor_schedule,
    deploy_weights,
)

__all__ = [
    "ConcordLinearPackedB",
    "ConcordConv2dPackedB",
    "compute_drift_cancel_C",
    "set_fixed_coh",
    "set_ratio_coh",
    "set_ratio_coh_floors",
    "enable_evaporate_consolidate",
    "evaporate_consolidate_floor_schedule",
    "deploy_weights",
    "S_SLOW_FACTOR",
    "V_SLOW_FACTOR",
    "MANTISSA_BIAS",
]
'''


RECIPE_PY = '''"""Concord recipes: the 32-bit evaporate/consolidate path + deploy-slow helper.

The bare layer is already the validated recipe (rank-1 v-hat AdamW + fixed
coherence gate, 64 b/param with the fp32 coh_pre EMA). evaporate/consolidate is
the 32-bit alternative: drop coh_pre, gate the cascade by the live coherence
ratio, hold a minimum chase/leak floor (consolidate coherent mass), and evaporate
incoherent s_fast by (1-coh). It matches the 64-bit gate's deployed-weight quality
at 32 b/param (tiny-shakespeare overfit, 10.8M + 49M).
"""
import math
from .packed_b import set_ratio_coh, set_fixed_coh, set_ratio_coh_floors


def enable_evaporate_consolidate(layers, chase_floor_min=0.1, leak_floor_min=0.1,
                                 gf_consol=50.0):
    """Switch ConcordLinear/Conv2dPackedB layers to the 32-bit evaporate/consolidate
    recipe. Drops the fp32 coh_pre buffer (-> 32 b/param), gates chase+leak by the
    live coherence, sets the steady-state minimum floors, and turns on (1-coh)
    evaporation of incoherent s_fast. Call once after building the layers. For the
    bootstrap ramp (recommended), also call set_ratio_coh_floors(
    *evaporate_consolidate_floor_schedule(it, total)) each step."""
    set_fixed_coh(True)
    set_ratio_coh(True)
    set_ratio_coh_floors(chase_floor_min, leak_floor_min)
    for m in layers:
        m.disable_cohpre()             # drop fp32 coh_pre EMA -> 32 bits/param
        m.gf_consol = float(gf_consol)  # (1-coh) evaporation rate (rho_eff = lr*gf_consol)
    return layers


def evaporate_consolidate_floor_schedule(it, total_iters, chase_start=0.9,
                                         chase_min=0.1, leak_start=0.999,
                                         leak_min=0.1, ramp_frac=1.0):
    """Per-step bootstrap ramp: cosine-decay the chase/leak floors from an ungated
    start to the steady-state minimum over ramp_frac of training, then hold. Ignites
    the cascade from the all-s_fast init, then tightens to coherence gating. Returns
    (chase_floor, leak_floor) to pass to set_ratio_coh_floors()."""
    h = max(1, int(ramp_frac * total_iters))
    def c(start, mn):
        if it >= h:
            return mn
        return mn + (start - mn) * 0.5 * (1.0 + math.cos(math.pi * it / h))
    return c(chase_start, chase_min), c(leak_start, leak_min)


def deploy_weights(layers):
    """Return the DEPLOYABLE weights: [layer.consolidated_weight() for layer in
    layers] -- the consolidated slow weight (s_slow + v_slow), dropping the
    transient/overfit-prone s_fast. Use these for inference / export, NOT the live
    training weight."""
    return [m.consolidated_weight() for m in layers]
'''


README_MD = '''# Concord packed-B optimizer -- lean build (32 bits/param)

One **int32 per parameter** holds the entire optimizer state -- no fp32 master
weight, no separate momentum/variance. A fused Triton kernel runs the step inside
the backward pass. This is the lean build: the validated recipe + the 32-bit
evaporate/consolidate path + deploy-slow, with the experimental dead-ends removed.

## The int32 word

```
bits 31:16   s_fast   int16   velocity / recent update   (fast accumulator)
bits 15:8    s_slow   int8    position bearer            (chase target, alpha~0.1)
bits  7:0    v_slow   int8    long-time anchor           (leak target, alpha_v~0.001)
```

Live weight = (s_slow*128 + s_fast + v_slow*128) * 2^(row_exp + col_exp - bias).
The cascade: each step lands in s_fast; a chase commits s_fast into s_slow; a slow
leak trickles s_slow into v_slow.

## Usage -- the bare layer is the validated optimizer

```python
from concord import ConcordLinearPackedB

layer = ConcordLinearPackedB(in_features, out_features, bias=True)  # drop-in nn.Linear
layer.lr = 5e-4

y = layer(x)
loss = criterion(y, target)
loss.backward()        # the Concord step is FUSED here (no optimizer.step())
layer.rebalance()      # per-step block-float envelope retune
```

Non-Linear params (embeddings, norms) take their own tiny optimizer (e.g. AdamW).
The bare default = rank-1 v-hat AdamW (v_scale=0, gf_trust_delta_sq=1, eps=1e-10,
precond_p=0.5) + fixed Wiener coherence gate.

## Deploy the consolidated (slow) weight -- drop s_fast

**For inference / export, deploy `s_slow + v_slow` (drop `s_fast`), not the live
weight.** s_fast holds the newest, noisiest, most overfit-prone velocity; the slow
accumulators are the denoised position.

```python
W_deploy = layer.consolidated_weight()   # s_slow*128 + v_slow*128   <- ship this
# or, for a list of layers:
from concord import deploy_weights
weights = deploy_weights(layers)
```

Validated on tiny-shakespeare (overfit regime, capacity >> data) at **10.8M and
49M params**: the consolidated weight beats the live weight by **~0.04-0.05 val
nats**, stable across scale (s_fast settles to ~4-7% of the weight mass). The plain
sum is correct -- doubling the anchor (s_slow + 2*v_slow) overshoots and is worse.

## The 32-bit evaporate/consolidate recipe

The fixed coherence gate keeps a per-element fp32 coh_pre EMA -- an extra 32
bits/param (so it is really 64 b/param). The evaporate/consolidate recipe restores
32 b/param: drop coh_pre, gate the cascade by the live coherence ratio, hold a
minimum chase/leak floor so coherent mass keeps consolidating, and evaporate
incoherent s_fast by a (1 - coh) term. It matches the 64-bit gate's deployed-weight
quality at 32 bits/param (49M overfit: best LIVE weight of any arm, TIES the gate
on the deployed weight).

```python
from concord import (ConcordLinearPackedB, enable_evaporate_consolidate,
                     evaporate_consolidate_floor_schedule, set_ratio_coh_floors,
                     deploy_weights)

layers = [m for m in model.modules() if isinstance(m, ConcordLinearPackedB)]
enable_evaporate_consolidate(layers, chase_floor_min=0.1, leak_floor_min=0.1,
                             gf_consol=50.0)

for it in range(total_iters):
    # recommended: bootstrap ramp (ungated start -> coherence gating)
    set_ratio_coh_floors(*evaporate_consolidate_floor_schedule(it, total_iters))
    ...  # forward / backward / rebalance as usual

weights = deploy_weights(layers)   # consolidated slow weights
```

## Files

- `concord/packed_b.py` -- the optimizer (Linear + Conv2d) + fused Triton kernels.
- `concord/recipe.py`   -- evaporate/consolidate + deploy helpers.
- `concord/__init__.py` -- the public surface.
- `test_min.py`         -- baked default + evaporate/consolidate + deploy checks.
- `requirements.txt`    -- torch + triton (CUDA).

## Requirements

PyTorch >= 2.1 + Triton, CUDA GPU. Run `python test_min.py` to verify.
'''


TEST_MIN_PY = '''"""Lean-build smoke test: the bare default IS the validated recipe and trains; the
32-bit evaporate/consolidate recipe trains and is 32 b/param; deploy-slow works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn.functional as F
import concord.packed_b as ppb
from concord import (ConcordLinearPackedB, ConcordConv2dPackedB,
                     enable_evaporate_consolidate, deploy_weights)

dev = "cuda"
torch.manual_seed(0); torch.cuda.manual_seed_all(0)


def check_cfg(m, name):
    assert m.optimizer_kind == "adamw", (name, m.optimizer_kind)
    assert abs(m._eps_value - 1e-10) < 1e-20, (name, m._eps_value)
    assert m.v_scale == 0.0, (name, m.v_scale)
    assert m.gf_trust_delta_sq == 1.0, (name, m.gf_trust_delta_sq)
    assert m.precond_p == 0.5, (name, m.precond_p)
    assert m._coh_pre is not None, name      # gate ON by default (64 b/param)
    print("  [%-7s] validated default OK" % name)


assert ppb._USE_FIXED_COH is True, "fixed coherence gate not the default"
# the stripped dead-ends are gone:
assert not hasattr(ppb, "_COH_WEIGHTED_V"), "cwv not stripped"
print("global _USE_FIXED_COH=True, cwv stripped  OK")
check_cfg(ConcordLinearPackedB(64, 128, bias=False, device=dev), "Linear")
check_cfg(ConcordConv2dPackedB(3, 16, 3, padding=1, bias=False, device=dev), "Conv2d")
assert not hasattr(ConcordLinearPackedB(8, 8, device=dev), "fast_gain"), "fast_gain not stripped"
print("fast_gain stripped  OK")


def fit(layer, steps=80):
    torch.manual_seed(1)
    layer.lr = 0.02
    tgt = torch.randn(32, 32, device=dev) * 0.3
    x = torch.randn(256, 32, device=dev)
    y = x @ tgt.T
    first = last = None
    for _ in range(steps):
        loss = F.mse_loss(layer(x.detach().requires_grad_(True)), y)
        loss.backward(); layer.rebalance()
        if first is None: first = loss.item()
        last = loss.item()
    return first, last


# 1) bare default trains
W = ConcordLinearPackedB(32, 32, bias=False, device=dev)
f, l = fit(W)
print("  validated default fit: %.4f -> %.4f" % (f, l))
assert l < 0.9 * f, "validated default not training"

# 2) evaporate/consolidate trains AND is 32 b/param (coh_pre dropped)
W2 = ConcordLinearPackedB(32, 32, bias=False, device=dev)
enable_evaporate_consolidate([W2], chase_floor_min=0.1, leak_floor_min=0.1, gf_consol=50.0)
assert W2._coh_pre is None, "evaporate/consolidate must drop coh_pre (32 b/param)"
assert W2.gf_consol == 50.0
f2, l2 = fit(W2)
print("  evaporate/consolidate fit (32 b/param): %.4f -> %.4f" % (f2, l2))
assert l2 < 0.9 * f2, "evaporate/consolidate not training"

# 3) deploy-slow: consolidated_weight = slow path (s_slow + v_slow), drops s_fast.
#    Check the exact construction bit-for-bit -- robust; differencing two separately
#    bf16-rounded tensors is noisy after hard convergence (s_fast shrinks).
s_slow = ((W.packed_w << 16) >> 24).to(torch.float32)
v_slow = ((W.packed_w << 24) >> 24).to(torch.float32)
exp = (W.row_exp[:, None].to(torch.int32) + W.col_exp[None, :].to(torch.int32)
       - W.MANTISSA_BIAS).float()
expect = ((s_slow * 128.0 + v_slow * 128.0) * torch.pow(2.0, exp)).to(torch.bfloat16)
Wd = deploy_weights([W])[0]
assert torch.equal(Wd, expect), "consolidated_weight must be the slow path (s_slow+v_slow)"
sf = (W.packed_w >> 16).to(torch.float32)
print("  deploy-slow OK: consolidated = slow path, s_fast (mean|.|=%.1f) dropped"
      % sf.abs().mean().item())

print("ALL LEAN-BUILD CHECKS PASSED")
'''


REQS_TXT = "torch>=2.1\ntriton\n"


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        open(LOG, "w", encoding="utf-8").write("\n".join(log_lines) +
                                                "\nBUILD_FAILED: %r\n" % (e,))
        print("BUILD_FAILED:", e)
        sys.exit(1)
