"""Polyak-leak hypothesis selector for Concord SGD.

Three-level ancestral sampling on the variational posterior implicit in
the concord format (q(W) = N(mu, tau^2 * Sigma), with mu from
(s_slow + s_fast) * scale and Sigma from the slow/fast gap):

    1. BoxVelocityMean  (CPU fp16 ring of K snapshots + per-layer
            |   velocity anchor from s_fast - s_slow)
            |   box mean -> bounded-memory Polyak average over the past K
            |   observations; velocity term projects the lagged mean
            |   forward to align with the current step.
            v
    2. H state (per-layer bf16 on GPU, "hypothesis selector")
            |  H <- (1-beta) * H + beta * BoxVelocity   (EMA low-pass,
            |  arithmetic in fp32 registers, stored as bf16)
            v   probe forward with H swapped in vs current concord.
                Accept gate:
                  - greedy (T == 0): accept iff loss_H < loss_current.
                  - MH (T > 0): also admit some bad-loss proposals,
                    weighted by exp(-(loss_H - loss_current) / T).
                T(t) = temperature / max(rings[0].n_blocks, 1),
                so it anneals to greedy as observations accumulate (the
                Polyak target becomes more reliable).
    3. Slow accumulator (s_slow of each concord layer)
                scatter commit_strength * (H - reconstruct_W(layer)) onto
                s_slow ONLY (s_fast left alone) via SR rounding.

The point is to recover sub-LSB precision that single-step stochastic rounding
washes out. The box filter averages K nearby SR realisations; its per-element
noise floor is roughly LSB/sqrt(K). The velocity anchor uses the implicit
momentum (s_fast - s_slow) at zero marginal storage cost to keep the candidate
aligned with the trajectory rather than lagging behind it.

Why scatter into s_slow only?
    The weight used in matmul is s_slow + s_fast, so moving s_slow by D moves
    the live weight by D and changes the implicit velocity v = s_fast - s_slow
    by -D. The next SGD step's gradient then has two options:
        - If the Polyak direction agreed with what the gradient wants
          (good leak), the gradient happily pushes s_fast onward; velocity
          decays back toward zero; the leak survives.
        - If the Polyak direction disagreed (bad leak), the gradient pushes
          s_fast back the other way; velocity grows in the un-leak direction;
          the next chase step pulls s_slow back. The leak gets undone.
    So the gradient signal stays sovereign; Polyak only proposes a re-centering
    of where the chase is heading. Combined with the accept/reject probe,
    bad Polyak proposals are gated out before they enter s_slow at all.

The probe acceptance gate is also what makes the mechanism safe during the
descent phase of training: if the trajectory is still moving toward better
loss, the Polyak mean is BEHIND the current iterate (README SST-2 finding 5:
"the centre is behind the frontier"), the probe always favours current, and
no leak fires. The mechanism only becomes active on plateau / oscillation,
where it actually helps.

Memory footprint:
    Cascade: n_levels * capacity * n_fol * 2 bytes (fp16 CPU). For
        BaselineConvNet (~60k concord params): ~6 MB total. Scales linearly
        with concord parameter count.
    H state: n_fol * 2 bytes (bf16 GPU). For BaselineConvNet: ~120 KB.
    Both released after end-of-training; the saved checkpoint is concord-int
    only.

    Note: for very large models (e.g. SDXL UNet ~2.5B concord params) the
    cascade does not fit in CPU RAM at the default sizes. Reduce capacity
    or n_levels, or use selective layer-set wrapping.
"""
import contextlib

import torch
import torch.nn as nn

# Imported on first call to avoid a hard dependency at import time.
# gauge_anneal.py provides the SR scatter primitive we customise below.


# ---------------------------------------------------------------------------
# SR scatter into s_slow only
# ---------------------------------------------------------------------------

_INT16_MIN, _INT16_MAX = -32768, 32767


@torch.no_grad()
def sr_scatter_delta_to_slow(layer, dW, gen):
    """Add dW (fp32, shape matching reconstructed weight) into a concord
    layer's s_slow buffer via stochastic rounding. s_fast is left untouched
    so the implicit velocity v = s_fast - s_slow shifts by -dW/scale,
    giving the next SGD step authority to ratify or reject the leak.

    Mirrors gauge_anneal.sr_scatter_delta but feeds the WHOLE dW into s_slow
    instead of splitting it across both accumulators.
    """
    # MANTISSA_BIAS = 15 in concord_linear_fused.py
    bias = getattr(layer, "MANTISSA_BIAS", 15)
    exp = (layer.row_exp[:, None] + layer.col_exp[None, :] - bias).float()
    scale = torch.exp2(exp)
    full = dW / scale
    u = torch.rand(full.shape, generator=gen, device=full.device,
                   dtype=torch.float32)
    a_slow = torch.floor(full + u).to(torch.int32)
    s_slow_new = (layer.s_slow.to(torch.int32) + a_slow).clamp(
        _INT16_MIN, _INT16_MAX).to(torch.int16)
    layer.s_slow.copy_(s_slow_new)
    if getattr(layer, "vsign", None) is not None:
        layer.vsign.copy_(layer.s_fast >= layer.s_slow)


# ---------------------------------------------------------------------------
# Concord weight reconstruction + flattening for the cascade
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reconstruct_W(layer):
    """fp32 reconstruction of a concord layer's stored weight."""
    bias = getattr(layer, "MANTISSA_BIAS", 15)
    exp = (layer.row_exp[:, None] + layer.col_exp[None, :] - bias).float()
    return ((layer.s_slow.to(torch.int32) + layer.s_fast.to(torch.int32))
            .float() * torch.exp2(exp))


@torch.no_grad()
def flatten_concord(concord_layers, dtype=torch.float32):
    """Return a single 1D tensor concatenating each layer's reconstructed
    weight (out, in) flattened. Output is on the same device as the
    concord buffers."""
    parts = []
    for m in concord_layers:
        parts.append(_reconstruct_W(m).flatten().to(dtype))
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# Forward-swap context manager (for probe forward with H weights)
# ---------------------------------------------------------------------------


def _named_modules_safe(model):
    """Yield (name, module) for every nn.Module reachable from `model`,
    whether `model` is an nn.Module or an OneTrainer container holding
    nn.Module attributes (StableDiffusionXLModel etc.). Duplicates
    concord_optimizer._named_modules_safe to avoid an import cycle."""
    if isinstance(model, nn.Module):
        yield from model.named_modules()
        return
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        try:
            val = getattr(model, attr_name)
        except Exception:
            continue
        if isinstance(val, nn.Module):
            yield attr_name, val
            for sub_name, sub_mod in val.named_modules():
                if sub_name:
                    yield f"{attr_name}.{sub_name}", sub_mod


@contextlib.contextmanager
def _swap_to_fp32(model, concord_layers, H_per_layer):
    """Temporarily replace each concord module on `model` with an fp32
    nn.Linear / nn.Conv2d backed by the matching H tensor. Reverts on exit.

    H_per_layer[i] must be a (out, in) fp32 tensor for layer i.

    Used to run a probe forward pass with hypothesis weights so we can
    compare loss_H to loss_current without disturbing the concord state.
    """
    # Locate each concord layer's (parent, attr) so we can swap in place.
    fol_set = {id(m): i for i, m in enumerate(concord_layers)}
    loc = {}
    for _pname, parent in _named_modules_safe(model):
        for attr, child in parent.named_children():
            if id(child) in fol_set:
                loc[id(child)] = (parent, attr)

    # ConcordConv2dFused is a subclass of ConcordLinearFused; detect it
    # via the conv-specific attributes rather than an explicit isinstance
    # so this module stays decoupled from concord_linear_fused.
    swaps = []
    device = next(iter(concord_layers)).s_slow.device
    for m in concord_layers:
        if id(m) not in loc:
            continue
        i = fol_set[id(m)]
        parent, attr = loc[id(m)]
        h = H_per_layer[i]
        has_bias = getattr(m, "bias", None) is not None
        if hasattr(m, "kh") and hasattr(m, "kw"):
            w4d = h.reshape(m.out_channels, m.in_channels, m.kh, m.kw)
            new_mod = nn.Conv2d(
                m.in_channels, m.out_channels, (m.kh, m.kw),
                stride=m.stride, padding=m.padding, bias=has_bias,
                device=device, dtype=torch.float32)
            new_mod.weight.data.copy_(w4d)
        else:
            new_mod = nn.Linear(m.in_features, m.out_features, bias=has_bias,
                                 device=device, dtype=torch.float32)
            new_mod.weight.data.copy_(h)
        if has_bias:
            new_mod.bias.data.copy_(m.bias.data.float())
        swaps.append((parent, attr, m, new_mod))
        setattr(parent, attr, new_mod)

    try:
        yield
    finally:
        # Restore. Note: swaps in reverse to be defensive about nested
        # parents (a swap might have set an attr that's later read as a
        # parent for another swap -- in practice they're siblings, but
        # reverse order is harmless).
        for parent, attr, original, _new_mod in reversed(swaps):
            setattr(parent, attr, original)


# ---------------------------------------------------------------------------
# The hypothesis selector
# ---------------------------------------------------------------------------


class _BoxVelocityRingProxy:
    """Mirrors SWAGRing's interface for PolyakHypothesis. Counts total
    observations seen (n_blocks) and how many of those are currently in
    the box-filter window (filled, bounded by K)."""

    def __init__(self, parent):
        self._parent = parent

    @property
    def n_blocks(self):
        # Total observations ever — drives the MH temperature anneal so
        # T(t) -> 0 as the buffer fills.
        return self._parent.n_observe_total

    @property
    def filled(self):
        # Bounded by K. Gates update_H on `filled >= polyak_warmup`.
        return self._parent.filled_count


class BoxVelocityMean:
    """Box-filter + per-layer velocity-anchored Polyak hypothesis.

    Maintains a fixed K-sample ring of fp16 live-weight snapshots on
    CPU. polyak() returns the box-filter mean ADJUSTED by the implicit
    velocity (s_fast - s_slow) * scale_fwd, projected forward by
    extrap_steps to undo the K/2-step lag inherent in a window mean.

    Memory: K * n_fol_total * 2 bytes. For K=5 at SDXL ~25 GB CPU --
    midway between SimplePolyakMean (~10 GB, no temporal structure)
    and full SWAGCascade (~240 GB, multi-resolution).

    Why this design:
      * box filter -> bounded memory, doesn't drag in early-training
        iterates the way SimplePolyakMean does after long runs.
      * velocity anchor -> uses information that already exists in the
        concord state (the implicit momentum s_fast - s_slow) at zero
        marginal storage cost. The lagged box mean gets shifted along
        the current trajectory direction to align with the present.
      * per-layer slicing -> the velocity computation reads from each
        concord layer's tensors directly, no flat-vector reshuffling.

    Interface-compatible with SWAGCascade for PolyakHypothesis use:
      observe(w):    push one flat (n_fol,) tensor of weights.
      polyak(lvl):   return the box-mean + velocity extrapolation as
                     a (n_fol,) fp32 CPU tensor; lvl is ignored.
      rings[lvl]:    proxy with .n_blocks, .filled for the MH gate.
    """

    def __init__(self, concord_layers, K=5, extrap_steps=None):
        """concord_layers: list of ConcordLinearFused / ConcordConv2dFused
        modules. K: ring depth (snapshots retained). extrap_steps: how far
        forward in steps to project the box mean using the velocity;
        default K/2 (the natural lag of a uniform K-sample window mean)."""
        self.concord = list(concord_layers)
        self.K = int(K)
        if self.K < 1:
            raise ValueError("K must be >= 1")
        self.extrap_steps = float(extrap_steps if extrap_steps is not None
                                   else self.K / 2.0)

        # Per-layer offsets into the flat n_fol_total vector, so observe()
        # / polyak() can slice without rebuilding the layout each call.
        offsets = [0]
        for m in self.concord:
            offsets.append(offsets[-1] +
                           int(m.out_features) * int(m.in_features))
        self.offsets = offsets
        self.n_fol_total = offsets[-1]

        # Ring of K fp16 snapshots on CPU. The running box_sum is fp32
        # so observe()/polyak() are O(n_fol) per call rather than
        # O(K * n_fol) for a from-scratch mean.
        self.ring = torch.zeros(self.K, self.n_fol_total,
                                dtype=torch.float16)
        self.box_sum = torch.zeros(self.n_fol_total, dtype=torch.float32)
        self.pos = 0
        self.filled_count = 0
        self.n_observe_total = 0

        self.rings = [_BoxVelocityRingProxy(self)]

    @torch.no_grad()
    def observe(self, w):
        """w: (n_fol_total,) CPU fp16 or fp32 tensor — current flattened
        concord live weight. Incrementally updates box_sum so polyak()
        stays O(n_fol)."""
        if w.dtype != torch.float32:
            w_fp32 = w.float()
        else:
            w_fp32 = w
        if self.filled_count == self.K:
            # Subtract the oldest (about to be overwritten) entry.
            self.box_sum.sub_(self.ring[self.pos].float())
        self.box_sum.add_(w_fp32)
        self.ring[self.pos].copy_(w_fp32.to(torch.float16))
        self.pos = (self.pos + 1) % self.K
        self.filled_count = min(self.filled_count + 1, self.K)
        self.n_observe_total += 1

    @torch.no_grad()
    def polyak(self, level=0):
        """Return box mean + per-layer velocity extrapolation as a
        (n_fol_total,) fp32 CPU tensor. `level` is ignored — there are
        no levels in this backend."""
        del level
        if self.filled_count == 0:
            return torch.zeros(self.n_fol_total, dtype=torch.float32)
        # Box mean.
        out = self.box_sum / float(self.filled_count)
        # Per-layer velocity extrapolation: H(t=now) = mean + v * extrap.
        # v_ij in live-weight units = (s_fast - s_slow) * 2^(r+c-B), per
        # element, summed to give the layer's velocity field.
        extrap = self.extrap_steps
        if extrap == 0.0:
            return out
        for i, m in enumerate(self.concord):
            start = self.offsets[i]
            end = self.offsets[i + 1]
            bias = getattr(m, "MANTISSA_BIAS", 15)
            exp = (m.row_exp[:, None].float()
                   + m.col_exp[None, :].float()
                   - bias)
            scale = torch.exp2(exp)
            v_mantissa = (m.s_fast.to(torch.int32)
                          - m.s_slow.to(torch.int32)).float()
            v_live = (v_mantissa * scale).flatten().cpu()
            out[start:end].add_(v_live, alpha=extrap)
        return out

    def laplacian_bands(self):
        """No multi-scale bands here. Returns []."""
        return []


class PolyakHypothesis:
    """Two-stage Polyak-leak hypothesis selector.

    Construction:
        concord_layers -- list of ConcordLinearFused / ConcordConv2dFused
            modules to track. Order is the layout order used to slice the
            backend's flat n_fol vector when one is exchanged.
        cascade -- a Polyak-mean backend object. Today this is always
            BoxVelocityMean (the argument name is retained for historical
            reasons; the constructor doesn't require a specific concrete
            type, only the {observe, polyak, rings} interface).
        polyak_leak    -- EMA rate Polyak -> H per update_H call (default 0.05).
        commit_strength -- fraction of (H - current) scattered into s_slow on
            an accepted probe (default 0.1).
        probe_every    -- training-loop steps between probe_and_commit calls.
            The caller (training loop) is responsible for honouring this --
            this class only exposes the probe operation. (default 200).
        polyak_level   -- cascade level whose Polyak mean drives H. Higher
            level => coarser timescale => more averaging => less responsive.
            (default 1).
        polyak_warmup  -- minimum cascade fill at `polyak_level` before the
            mechanism activates. Until reached, update_H is a no-op and
            probe_and_commit always reports "skipped". (default 2).
        seed           -- SR RNG seed for the s_slow scatter.

    Per-step contract (called by the optimizer wrapper):
        observe()         -- once per step (or per K), feeds flattened
                              current concord weights into the cascade.
        update_H()        -- once per step (cheap), EMA-pulls H toward
                              cascade.polyak(polyak_level).

    Per-probe-period contract (called by the training loop):
        probe_and_commit(model, probe_x, probe_y, criterion)
            Forward both current concord and H-swapped, compare losses.
            If H is better, sr_scatter_delta_to_slow(commit_strength * (H - W)).
            Returns a dict with diagnostics.

    Diagnostics: self.n_observe, n_update_H, n_probe, n_accept,
        last_probe_loss_current, last_probe_loss_H, accept_rate.
    """

    def __init__(self, concord_layers, cascade,
                 polyak_leak=0.05,
                 commit_strength=0.1,
                 probe_every=200,
                 polyak_level=1,
                 polyak_warmup=2,
                 temperature=0.0,
                 seed=0):
        self.concord = list(concord_layers)
        self.cascade = cascade
        self.polyak_leak = float(polyak_leak)
        self.commit_strength = float(commit_strength)
        self.probe_every = int(probe_every)
        self.polyak_level = int(polyak_level)
        self.polyak_warmup = int(polyak_warmup)
        # MH temperature scale T_0. The effective per-probe temperature is
        #     T(t) = temperature / max(cascade.rings[polyak_level].n_blocks, 1)
        # so the gate is greedy when the cascade has stabilised (n_blocks
        # large, T -> 0) and permissive when it hasn't (n_blocks small,
        # T large -> chain explores). At temperature == 0 the gate is the
        # vanilla greedy 'accept iff loss_H < loss_current' and the chain
        # is a hill-climber instead of a sampler.
        self.temperature = float(temperature)
        # Per-layer slab offsets into the cascade's flat (n_fol,) layout.
        # Must match the order concord_layers is passed; flatten_concord
        # uses the same iteration order.
        self._layout = []
        off = 0
        for m in self.concord:
            n = m.out_features * m.in_features
            self._layout.append((off, n, m.out_features, m.in_features))
            off += n
        self._n_fol_total = off
        # H state: per-layer bf16 (out, in) tensors initialised to the
        # current concord reconstruction. They will EMA-track the cascade
        # Polyak mean over time. Storage is bf16 (half the fp32 memory);
        # the EMA update in update_H() promotes to fp32 in registers, does
        # the arithmetic, and stores back as bf16.
        dev = self.concord[0].s_slow.device
        self._H = [_reconstruct_W(m).to(torch.bfloat16) for m in self.concord]
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(int(seed))
        # Diagnostics
        self.n_observe = 0
        self.n_update_H = 0
        self.n_probe = 0
        self.n_accept = 0
        self.n_accept_mh = 0     # accepts that ONLY survived because T > 0
        self.last_probe_loss_current = float("nan")
        self.last_probe_loss_H = float("nan")
        self.last_probe_T = float("nan")
        self._accept_window = []   # rolling list of recent accept (0/1)s
        self._window_size = 32

    # ---------------------- step-frequency hooks ---------------------- #

    @torch.no_grad()
    def observe(self):
        """Feed one flattened concord snapshot into the cascade. Cheap
        (one cat + one CPU transfer); the cascade itself amortises blocks."""
        flat = flatten_concord(self.concord, dtype=torch.float32)
        self.cascade.observe(flat.half().cpu())
        self.n_observe += 1

    @torch.no_grad()
    def update_H(self):
        """EMA-pull H toward the cascade's level-l Polyak mean.

        No-op until the cascade has observed at least `polyak_warmup` total
        snapshots at the target level -- before that the mean is undefined
        or noisy. (Was previously gated on `ring.filled`, which is capped at
        K; making the gate scale with `ring.n_blocks` lets warmup > K
        actually delay activation for as long as the user wants.)
        """
        if self.polyak_level >= len(self.cascade.rings):
            return
        ring = self.cascade.rings[self.polyak_level]
        if ring.n_blocks < self.polyak_warmup:
            return
        polyak_cpu = self.cascade.polyak(self.polyak_level)  # (n_fol,) fp32 CPU
        dev = self._H[0].device
        polyak = polyak_cpu.to(dev, non_blocking=True)
        leak = self.polyak_leak
        for i, (off, n, of_, if_) in enumerate(self._layout):
            slab = polyak[off:off + n].view(of_, if_)            # fp32
            # EMA in fp32 registers; store back as bf16. We pay one cast
            # round-trip per layer per update_H, which is cheap relative
            # to the per-element work; in exchange H storage is half.
            h_fp32 = self._H[i].float()
            h_fp32.mul_(1.0 - leak).add_(slab, alpha=leak)
            self._H[i].copy_(h_fp32)
        self.n_update_H += 1

    @property
    def H_active(self):
        """True once H has been updated at least once from the cascade."""
        return self.n_update_H > 0

    # ---------------------- probe & commit ---------------------- #

    @torch.no_grad()
    def probe_and_commit(self, model, probe_x, probe_y, criterion,
                          commit_scale=1.0):
        """Forward `model(probe_x)` twice: once with current concord weights,
        once with the concord layers swapped for fp32 nn.Modules backed by H.
        Compare `criterion(logits, probe_y)`. If H wins (lower loss), scatter
        `commit_strength * commit_scale * (H_i - reconstruct_W(layer_i))` into
        each layer's s_slow via SR (s_fast untouched).

        commit_scale (default 1.0) is an optional per-call multiplier on
        commit_strength. Caller passes a fraction in [0, 1] to suppress
        leaks during the high-LR phase (e.g. 1 - lr/lr_init) so the chase
        trajectory isn't perturbed before the box mean is meaningful.
        commit_scale == 0 makes accepts a pure no-op (probe still runs
        and increments diagnostics).

        Returns:
            {'skipped': bool, 'accepted': bool,
             'loss_current': float, 'loss_H': float,
             'accept_rate': float}
        """
        self.n_probe += 1

        # Defer the test until H has been updated at least once -- before then
        # H is just the initial weight clone and the probe is a no-op.
        if not self.H_active:
            return {"skipped": True, "accepted": False,
                    "loss_current": float("nan"), "loss_H": float("nan"),
                    "accept_rate": self._accept_rate()}

        # 1. Loss with current concord weights (model is already in this state).
        was_training = model.training
        model.eval()
        try:
            logits_current = model(probe_x)
            loss_current = float(criterion(logits_current, probe_y).item())

            # 2. Swap to H, eval, swap back via context manager.
            with _swap_to_fp32(model, self.concord, self._H):
                logits_H = model(probe_x)
                loss_H = float(criterion(logits_H, probe_y).item())
        finally:
            if was_training:
                model.train()

        self.last_probe_loss_current = loss_current
        self.last_probe_loss_H = loss_H

        # Acceptance gate. Greedy (T == 0) accepts iff loss_H < loss_current.
        # MH (T > 0) admits some bad-loss proposals weighted by their loss
        # gap, with the per-probe temperature decaying as the cascade fills:
        #     T(t) = self.temperature / max(n_blocks_at_level, 1)
        # so early in training (cascade young, n_blocks small) T is large
        # and the chain explores; once n_blocks is large T -> 0 and the gate
        # is effectively greedy.
        greedy = (loss_H < loss_current)
        accepted = greedy
        mh_only = False
        T = 0.0
        if not greedy and self.temperature > 0.0:
            n_blocks = max(
                self.cascade.rings[self.polyak_level].n_blocks, 1)
            T = self.temperature / float(n_blocks)
            # accept with prob exp(-(loss_H - loss_current) / T)
            log_ratio = -(loss_H - loss_current) / max(T, 1e-30)
            u = float(torch.rand(1, generator=self._gen,
                                  device=self._gen.device).item())
            # rand() < exp(log_ratio)  <=>  log(rand()) < log_ratio
            import math
            if math.log(max(u, 1e-30)) < log_ratio:
                accepted = True
                mh_only = True
        self.last_probe_T = T

        self._accept_window.append(1 if accepted else 0)
        if len(self._accept_window) > self._window_size:
            self._accept_window.pop(0)

        if accepted:
            # Scatter commit_strength * commit_scale * (H - reconstruct_W(layer))
            # into s_slow for each concord layer.
            effective_strength = self.commit_strength * float(commit_scale)
            if effective_strength > 0.0:
                for i, m in enumerate(self.concord):
                    current_W = _reconstruct_W(m)
                    dW = effective_strength * (self._H[i] - current_W)
                    sr_scatter_delta_to_slow(m, dW, self._gen)
            self.n_accept += 1
            if mh_only:
                self.n_accept_mh += 1

        return {"skipped": False, "accepted": accepted, "mh_only": mh_only,
                "loss_current": loss_current, "loss_H": loss_H, "T": T,
                "accept_rate": self._accept_rate()}

    def _accept_rate(self):
        if not self._accept_window:
            return float("nan")
        return sum(self._accept_window) / len(self._accept_window)

    # ---------------------- diagnostics ---------------------- #

    def summary(self):
        """Compact summary string for logging."""
        ar = self._accept_rate()
        ar_str = f"{ar * 100:.0f}%" if ar == ar else "NA"
        return (f"polyak[obs={self.n_observe} updH={self.n_update_H} "
                f"probes={self.n_probe} accepts={self.n_accept} "
                f"(mh={self.n_accept_mh}) "
                f"rate={ar_str} "
                f"last lossC={self.last_probe_loss_current:.4f} "
                f"lossH={self.last_probe_loss_H:.4f} "
                f"T={self.last_probe_T:.4g}]")
