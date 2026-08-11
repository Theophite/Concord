"""nn.Module wrappers for the dual-dissipation packed-B fine accumulator.

Mirrors the ConcordConv2dPackedB / ConcordLinearPackedB pattern from
prototype_packed_b.py: the packed weight is a *buffer* (the int32 OFFSET word
lives inside DualDissipationLayer), and the per-layer `bias` is an nn.Parameter
that anchors the autograd graph. The forward DECODES live_weight() into a fresh
grad-requiring proxy tensor, runs F.conv2d / F.linear with it, and retains the
proxy's .grad so the training loop can read the per-layer WEIGHT-SPACE gradient
back out. Deploy-eval reads deploy_weight().

THE TWO-GRADIENT (RANDOM MICROBATCH SPLIT) MECHANISM lives in the optimizer step,
not here: the training loop runs TWO forward+backward passes (one per random half
of the batch), reads each layer's weight-proxy .grad for each half, and calls
  layer.dd.step(grad_L, lr, grad_W_H=grad_H, **kw)
so arm L integrates data-half-1's FULL gradient and arm H data-half-2's FULL
gradient. Each arm gets its OWN half's whole gradient — NOT half of a shared one.

The packed step is NOT fused into backward here (the reference step() is pure
Python, not a Triton kernel), so the loop calls layer.dd.step(...) explicitly
after collecting the two half-gradients. CPU-capable; GPU-capable via the
reference's device-following internal tensors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dual_dissipation_ref import (
    make_substrate, SUBSTRATE_XAVIER, SUBSTRATE_KAIMING)
# Same-scale (CARRY=1) IS the dual-dissipation by default (architect ruling 2026-06-28,
# memory concord-no-x128; CIFAR-confirmed flat 0.595 > x128 0.533 @ 6k/40ep). The x128
# DualDissipationLayer in dual_dissipation_ref is the DEPRECATED artifact, kept only for the
# legacy test suite. Not a flag, not a variant -- this is how it works.
from dual_dissipation_flat_ref import DualDissipationFlatLayer as DualDissipationLayer


class _DualDissipationBase(nn.Module):
    """Shared plumbing: a DualDissipationLayer over an (out, in*kh*kw) matrix, a bias
    Parameter (graph anchor), live/deploy decode, and the weight-proxy grad capture."""

    def __init__(self, N, K, *, bias=True, device="cpu", grad_accum_M=8,
                 bracket_d=0.5, seed=0, substrate_seed=None,
                 substrate_mode=SUBSTRATE_KAIMING, substrate_scale=1.0):
        super().__init__()
        self.N, self.K = N, K
        self.device_ = torch.device(device)
        # FROM-SCRATCH substrate scale. The substrate is a FIXED random prior (a frozen
        # random projection); from-scratch its only jobs are symmetry-breaking and setting
        # the per-row/col block-float scale. At full Kaiming magnitude it DOMINATES the
        # deploy weight (substrate + small consolidated offset), pinning deploy-eval at
        # chance. A small substrate (<<1x) lets the TRAINED OFFSET dominate the deploy
        # weight while still breaking symmetry, so the deploy path actually learns.
        self.substrate_scale = float(substrate_scale)
        # The packed OFFSET word + all accumulator state live in the reference Layer.
        self.dd = DualDissipationLayer(
            N, K, enabled=True, grad_accum_M=grad_accum_M, bracket_d=bracket_d,
            seed=seed, substrate_seed=substrate_seed, substrate_mode=substrate_mode)
        # bias is a real nn.Parameter -> it anchors the autograd graph so backward fires
        # even though the decoded weight proxy is created fresh each forward.
        if bias:
            self.bias = nn.Parameter(torch.zeros(N, device=self.device_))
        else:
            self.register_parameter("bias", None)
        # Set after each forward: the grad-requiring decoded weight (shape [N, K]).
        # The training loop reads ._wproxy.grad to get the per-layer weight-space gradient.
        self._wproxy = None

    @torch.no_grad()
    def load_weights(self, W=None, base=None):
        """Initialize the packed accumulator (zeroed) + substrate. ENABLED from-scratch:
        substrate is a seeded draw scaled by substrate_scale (W only fixes device/shape).
        Finetune: pass base (the pretrained weight) -> substrate = base, scale ignored."""
        if W is None:
            W = torch.zeros(self.N, self.K, device=self.device_)
        if base is None and self.substrate_scale != 1.0:
            # from-scratch with a SMALL prior: draw the seeded substrate, scale it down, and
            # feed it as `base` so the reference uses it verbatim (and derives row/col exps
            # from the scaled magnitude -> a finer block-float scale that the offset can fill).
            base = self.substrate_scale * make_substrate(
                self.N, self.K, self.dd.substrate_seed, self.dd.substrate_mode)
        self.dd.load_weights(W.to(self.device_), base=None if base is None else base.to(self.device_))

    def _live_matrix(self):
        """Decode the live weight [N, K] as a fresh grad-requiring leaf, retain its grad."""
        w = self.dd.live_weight().detach().to(self.device_)
        w.requires_grad_(True)
        w.retain_grad()
        self._wproxy = w
        return w

    @torch.no_grad()
    def deploy_matrix(self, step_salt=0, deterministic=False):
        return self.dd.deploy_weight(step_salt=step_salt, deterministic=deterministic)

    def weight_grad(self):
        """The per-layer weight-space gradient from the last backward ([N, K]), or None."""
        if self._wproxy is None or self._wproxy.grad is None:
            return None
        return self._wproxy.grad.detach().reshape(self.N, self.K)

    def zero_weight_grad(self):
        if self._wproxy is not None and self._wproxy.grad is not None:
            self._wproxy.grad = None

    @torch.no_grad()
    def step_two_grad(self, grad_L, grad_H, lr, **kw):
        """Route data-half-1's gradient into arm L and data-half-2's into arm H (the
        two-gradient step). grad_{L,H} are weight-space [N, K]."""
        return self.dd.step(grad_L.reshape(self.N, self.K), lr,
                            grad_W_H=grad_H.reshape(self.N, self.K), **kw)


class DualDissipationLinear(_DualDissipationBase):
    """Linear layer backed by a dual-dissipation packed accumulator. forward decodes
    live_weight() [out, in] and runs F.linear; the bias Parameter anchors autograd."""

    def __init__(self, in_features, out_features, *, bias=True, device="cpu", **kw):
        super().__init__(out_features, in_features, bias=bias, device=device, **kw)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        w = self._live_matrix()                       # [out, in], grad-tracked
        return F.linear(x, w, self.bias)


class DualDissipationConv2d(_DualDissipationBase):
    """Conv2d layer backed by a dual-dissipation packed accumulator. forward decodes
    live_weight() and reshapes [out, in*kh*kw] -> [out, in, kh, kw] for F.conv2d; the
    bias Parameter anchors autograd. The packed accumulator is the (out, in*kh*kw) matrix,
    so the per-layer weight gradient is reshaped back to that matrix for the step."""

    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1, padding=0,
                 bias=True, device="cpu", **kw):
        if isinstance(kernel_size, int):
            kh = kw_ = kernel_size
        else:
            kh, kw_ = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = kh, kw_
        self.stride = stride
        self.padding = padding
        super().__init__(out_channels, in_channels * kh * kw_, bias=bias, device=device, **kw)

    def forward(self, x):
        w = self._live_matrix()                       # [out, in*kh*kw], grad-tracked
        w4 = w.reshape(self.out_channels, self.in_channels, self.kh, self.kw)
        return F.conv2d(x, w4, self.bias, stride=self.stride, padding=self.padding)
