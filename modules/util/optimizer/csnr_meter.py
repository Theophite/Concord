"""Passive per-timestep gradient-SNR meter for the CONSTANT_SNR timestep sampler.

Concord's CONSTANT_SNR draw (ModelSetupNoiseMixin) wants the RESOLVABLE per-timestep
gradient SNR so it can hold the accumulated SNR constant across timesteps. The schedule
SNR abar/(1-abar) is a static proxy; this meter measures the LIVE gradient SNR so the
draw self-adapts (mastered timesteps decay -- exp 36).

At SDXL scale there are no per-sample gradients, and the graph-captured self-stepping
kernel hides the flat gradient, so we CANNOT measure per-bin SNR directly. Instead we
DECONVOLVE it from mixed-timestep batches (exp 39): over a short window each step gives
a batch-mean gradient SKETCH u_s in R^K (a random projection/subsample of the gradient)
and its timestep histogram w_{b,s} = n_{b,s}/B. With per-bin signal dirs ~orthonormal,

    E<u_s,u_s'> = sum_b w_{b,s} w_{b,s'} S_b       (independent steps: noise cancels)  -> S_b
    E||u_s||^2  = sum_b w_{b,s}^2 S_b + (1/B) sum_b w_{b,s} N_b                         -> N_b
    SNR_b = S_b / N_b     (the projection scale K/P cancels in the ratio)

two nonneg least-squares. exp 39 + port smoke give the operating envelope this meter
assumes, and it is TIGHT -- three conditions all matter:
  (a) the cross-dot VECTOR sketch (scalar ||g||^2 alone is near-collinear, tops ~0.5);
  (b) a SMOOTHNESS prior across bins (the SNR curve is smooth in t) -- without it 50
      steps recovers nothing (~0.5), with it ~0.98;
  (c) COARSE bins (8) and effective batch (micro-batch x grad-accum) >= ~64 samples/step.
At n_bins=8/eff_B>=64 recovery is clean (all-seed ~1.0 over 50 steps). At eff_B=32, or at
12-16 bins, a single window occasionally INVERTS (rank-corr < 0) -- a confident-but-wrong
curve that would make the draw oversample exactly the wrong timesteps. So one window is
NOT trustworthy on its own: the collector MUST robustify -- EMA the per-bin SNR across
successive windows and blend toward the schedule-SNR prior (measured = a*deconv +
(1-a)*schedule), so a bad window is outvoted and the curve adapts slowly. cash_out()
here rejects only DEGENERATE (unresolved-flat) windows; it cannot catch a confident
inversion -- that is what the EMA + prior blend are for.

Meter-only: this module computes a curve; it never touches training.

Sketch source -- eager reference vs kernel port. GradSketch below is the EAGER reference
(backward hooks read k_each fixed random coordinates of each hooked module's BATCH-MEAN
output-gradient into a per-step K-vector). Hooks do not fire under CUDA-graph replay, so
the production port moves the SAME read into the fused apply kernel:
  - allocate a device buffer u[K] (zeroed at window start, outside capture);
  - in the per-layer fused backward, for each of this layer's slot coordinates, atomically
    add the batch-mean gradient entry at a fixed hashed index into u[slot*k_each + c]
    (the kernel already forms the gradient tick; this is one extra indexed add per sketched
    coord, gated by a WRITE_SKETCH constexpr -> zero cost when off);
  - the host reads u after replay (device->host copy of K floats) and calls CSNRMeter.add.
The coordinate set and slot layout are identical to GradSketch, so the eager test validates
the kernel path's math. No stochastic rounding is involved (a plain read, not a weight
write), so no new SR salt is needed. The buffer fill/zero crosses the graph boundary as a
device tensor, per CUDA-graph discipline.
"""
import torch


def _nnls(A, b, iters=200, smooth=0.0):
    """Nonneg least squares by projected gradient. `smooth` adds smooth*||D2 x||^2, a
    Tikhonov penalty on the second difference across bins (the SNR curve is smooth) --
    exp 39: this is what lets 50 steps resolve the ordering instead of ~200."""
    At = A.t()
    AtA = At @ A
    if smooth > 0.0:
        m = A.shape[1]
        D = torch.zeros(m - 2, m, dtype=A.dtype, device=A.device)
        idx = torch.arange(m - 2)
        D[idx, idx], D[idx, idx + 1], D[idx, idx + 2] = 1.0, -2.0, 1.0
        AtA = AtA + smooth * (D.t() @ D)
    Atb = At @ b
    x = torch.zeros(A.shape[1], dtype=A.dtype, device=A.device)
    L = torch.linalg.eigvalsh(AtA)[-1].clamp_min(1e-12)
    for _ in range(iters):
        x = (x - (AtA @ x - Atb) / L).clamp_min(0.0)
    return x


class CSNRMeter:
    """Accumulate per-step (sketch, timestep-histogram) over a window, then cash out a
    per-timestep SNR curve. Usage: reset(); add(sketch, hist) each step; when count()
    reaches the window, curve = cash_out(num_train_timesteps)."""

    def __init__(self, n_bins=8, smooth=3.0, min_range=1.3):
        # n_bins=8 is the reliable default (finer bins invert at 50 steps -- see module
        # docstring); min_range is the degeneracy guard (reject flat/unresolved windows).
        self.n_bins = int(n_bins)
        self.smooth = float(smooth)
        self.min_range = float(min_range)
        self.reset()

    def reset(self):
        self._U = []          # list of sketch vectors [K]
        self._W = []          # list of histograms [n_bins] (fractions, sum to 1)
        self._B = []          # list of effective batch sizes per step

    def count(self):
        return len(self._U)

    @torch.no_grad()
    def add(self, sketch, timesteps, num_train_timesteps):
        """sketch: [K] batch-mean gradient projection for this step. timesteps: [B] ints."""
        edges = torch.linspace(0, num_train_timesteps, self.n_bins + 1, device=timesteps.device)
        b = torch.bucketize(timesteps.float(), edges[1:-1])
        w = torch.bincount(b, minlength=self.n_bins).float()
        eff_b = float(w.sum())
        self._U.append(sketch.detach().flatten().float().cpu())
        self._W.append((w / max(eff_b, 1.0)).cpu())
        self._B.append(eff_b)

    @torch.no_grad()
    def cash_out(self, num_train_timesteps):
        """Deconvolve S_b, N_b -> per-bin SNR, then expand to a per-timestep curve
        [num_train_timesteps]. Returns None if the window is too thin to trust."""
        n = self.count()
        if n < 8:
            return None
        U = torch.stack(self._U)                              # [n, K]
        W = torch.stack(self._W)                              # [n, n_bins]
        eff_b = sum(self._B) / n
        i, j = torch.triu_indices(n, n, offset=1)
        dots = (U[i] * U[j]).sum(1)
        S = _nnls(W[i] * W[j], dots, smooth=self.smooth)      # cross-step dots -> signal
        resid = (U * U).sum(1) - (W ** 2 * S).sum(1)
        N = _nnls(W / eff_b, resid, smooth=self.smooth)       # residual energy -> noise
        snr_bin = S / (N + 1e-12)
        # guard: bins that were never sampled or came back degenerate -> fill from neighbours
        seen = (W.sum(0) > 0)
        if not seen.any():
            return None
        fill = float(snr_bin[seen].median())
        snr_bin = torch.where(seen & (snr_bin > 0), snr_bin, torch.full_like(snr_bin, fill))
        # degeneracy guard: an unresolved window comes back nearly FLAT (deconvolution found
        # no per-bin structure). Reject it so the caller keeps its previous curve rather than
        # overwriting with noise. (Cannot catch a confident inversion -- EMA + prior blend do.)
        pos = snr_bin[snr_bin > 0]
        if pos.numel() < 2 or float(pos.max() / pos.min()) < self.min_range:
            return None
        # expand per-bin -> per-timestep (nearest bin), on CPU; the sampler clamps into
        # [floor*gamma, gamma] and inverts, so only the SHAPE matters here.
        t = torch.arange(num_train_timesteps).float()
        edges = torch.linspace(0, num_train_timesteps, self.n_bins + 1)
        bidx = torch.bucketize(t, edges[1:-1]).clamp(max=self.n_bins - 1)
        return snr_bin[bidx].contiguous()


def select_sketch_modules(model, n=8):
    """Pick n evenly-spaced weight-bearing modules (Linear/Conv2d) to hook for the sketch.
    A representative spread through the net; the sketch is a random projection, so exactly
    which layers matter little as long as they carry gradient."""
    import torch.nn as nn
    elig = [m for _, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv2d)) and getattr(m, "weight", None) is not None]
    if not elig:
        return []
    if len(elig) <= n:
        return elig
    step = len(elig) / n
    return [elig[int(i * step)] for i in range(n)]


class GradSketch:
    """Random-coordinate gradient sketch via backward hooks. EAGER REFERENCE for the
    graph-native kernel projection buffer: each hooked module contributes k_each fixed
    random coordinates of its BATCH-MEAN output-gradient to a per-step K-vector u_s. The
    batch-mean over samples (which sit at different timesteps) is exactly the mixed
    observation the deconvolution expects; concatenating per-module slots keeps a
    high-magnitude layer from swamping the sketch.

    This runs eager (hooks do not fire under CUDA-graph replay); the production port moves
    the same fixed-coordinate read into the fused apply kernel, writing u_s to a device
    buffer the host reads after replay. Same observable, graph-safe. Concord's self-stepping
    does not populate .grad, but the module output-gradient DOES flow to a full backward
    hook, so this reference works on the real packed layers too (in an eager window)."""

    def __init__(self, modules, K=256, seed=1234, device="cpu"):
        self.modules = list(modules)
        self.n = max(1, len(self.modules))
        self.k_each = max(1, K // self.n)
        self.K = self.k_each * self.n
        self.seed = seed
        self.device = device
        self._idx = {}                                   # (slot, feat) -> coord indices
        self._slot = {id(m): s for s, m in enumerate(self.modules)}
        self._acc = None
        self._handles = []

    def _coords(self, slot, feat):
        key = (slot, feat)
        if key not in self._idx:
            g = torch.Generator().manual_seed(self.seed + slot * 7919 + feat)
            self._idx[key] = torch.randint(0, feat, (self.k_each,), generator=g)
        return self._idx[key]

    def _hook(self, m, grad_input, grad_output):
        g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        if g is None:
            return
        gm = g.detach().reshape(g.shape[0], -1).mean(0).float()   # batch-mean output grad [feat]
        slot = self._slot[id(m)]
        idx = self._coords(slot, gm.numel()).to(gm.device)
        if self._acc is None:
            self._acc = torch.zeros(self.K, device=self.device)
        self._acc[slot * self.k_each:(slot + 1) * self.k_each] = gm[idx].to(self.device)

    def register(self):
        self._acc = None
        for m in self.modules:
            self._handles.append(m.register_full_backward_hook(self._hook))

    def pop(self):
        """Return the accumulated sketch for the step just finished and reset. None if no
        hook fired (e.g. a step that did not touch the hooked modules)."""
        u, self._acc = self._acc, None
        return u

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class CSNRCollector:
    """Periodic 50-step epilogue collector: arms the sketch, feeds (sketch, timesteps) to a
    CSNRMeter each step, and on window completion cashes out a per-timestep SNR curve,
    integrating it with EMA + a schedule-prior blend so one bad window cannot flip the draw.
    `emit()` returns (curve, gen) for model._concord_csnr_curve; gen bumps only on a kept
    window, invalidating the sampler CDF cache. Meter-only: never touches training."""

    def __init__(self, modules, num_train_timesteps, alphas_cumprod=None, window=50,
                 K=256, n_bins=8, ema=0.7, prior_w=0.3, seed=1234, device="cpu",
                 source="eager"):
        # 'eager': own backward hooks (CPU reference; hooks die under CUDA-graph replay).
        # 'kernel': the fused grad_x kernel writes the sketch to device buffers, which the
        # host assembles and passes to observe() -- the graph-native production path.
        self.source = str(source)
        self.sketch = GradSketch(modules, K=K, seed=seed, device=device) if self.source == "eager" else None
        self.meter = CSNRMeter(n_bins=n_bins)
        self.T = int(num_train_timesteps)
        self.window = int(window)
        self.ema = float(ema)
        self.prior_w = float(prior_w)
        self._log_prior = None
        if alphas_cumprod is not None:
            ac = alphas_cumprod.detach().float().cpu()[:self.T]
            self._log_prior = torch.log((ac / (1.0 - ac)).clamp_min(1e-8))
        self._log_curve = None
        self.gen = 0
        self._active = False
        self._seen = 0

    def arm(self):
        """Open a measurement window (call at the epilogue / per-epoch boundary)."""
        if self.sketch is not None:
            self.sketch.register()
        self.meter.reset()
        self._active = True
        self._seen = 0

    def active(self):
        return self._active

    @torch.no_grad()
    def observe(self, timesteps, sketch=None):
        """Call once per OPTIMIZER step while a window is open. In 'kernel' mode pass the
        host-assembled sketch (feat-only batch-mean read from the grad_x buffers, pooled across
        the accumulation micro-batches, with its timesteps pooled to match); in 'eager' mode it
        pops the hook sketch. Feeds the meter and finishes the window at `window` observations."""
        if not self._active:
            return
        u = sketch if sketch is not None else (self.sketch.pop() if self.sketch is not None else None)
        if u is not None and timesteps is not None and timesteps.numel() > 0:
            self.meter.add(u, timesteps.detach(), self.T)
            self._seen += 1
        if self._seen >= self.window:
            self._finish()

    def _finish(self):
        if self.sketch is not None:
            self.sketch.remove()
        self._active = False
        curve = self.meter.cash_out(self.T)              # None if degenerate -> keep previous
        if curve is None:
            return
        log_new = torch.log(curve.clamp_min(1e-8))
        if self._log_prior is not None:
            log_new = self.prior_w * self._log_prior + (1.0 - self.prior_w) * log_new
        if self._log_curve is None:
            self._log_curve = log_new
        else:
            self._log_curve = self.ema * self._log_curve + (1.0 - self.ema) * log_new
        self.gen += 1

    def emit(self):
        """(per-timestep SNR curve, gen) for the sampler, or None until the first kept window."""
        if self._log_curve is None:
            return None
        return self._log_curve.exp(), self.gen
