"""Per-image-shape cache of the Concord Adafactor v_hat (v_row, v_col).

Motivation: randomizing aspect-ratio buckets across an epoch degrades two things.
 (1) MEMORY -- the CUDA caching allocator sees a stream of differently-sized
     activation/graph-workspace blocks and FRAGMENTS; reserved memory creeps and you
     OOM on nominally-free memory. CUDA-graph pools amplify it (each capture pins a
     shape-sized pool). Bucketing whole epochs (or long contiguous blocks) by shape
     gives one allocation footprint per run -> the allocator reaches steady state.
 (2) PRECONDITIONER -- the gradient-statistic EMA v_hat gets yanked between each
     bucket's gradient scale and never settles.

Bucketing fixes (1). This class fixes (2): at a shape boundary, snapshot the outgoing
shape's v_hat and restore the incoming shape's, so each shape keeps its own
preconditioner across the cycling.

Scope / why it's cheap:
 * The packed WEIGHTS (s_fast/s_slow/v_slow) are SHARED across shapes -- they are the
   model, not a per-shape statistic. Only the rank-1 v_hat (v_row[out], v_col[in]) is
   cached; sum_v_inv is derived per-apply (1/v_row.sum()) and never stored. A snapshot
   is a few KB/layer, so ~hundreds of layers x a handful of shapes is a few MB.
 * Restore is IN-PLACE (copy_) so a captured CUDA graph's v_row/v_col pointers stay
   valid -- swapping v_hat forces no recapture beyond the shape's own geometry change.
 * coh_pre is also shape-sensitive but per-element [N,K] (weight-sized), so caching it
   per shape would ~double optimizer memory per shape; it is deliberately left to
   re-warm. (In frozen-anchor TE mode coh_pre is pinned anyway.)

Boundary order (the caller's job, e.g. at an epoch/shape switch):
    graph.release()            # drop the captured graph (frees its pool)
    torch.cuda.empty_cache()   # hand the arena back so the next shape starts clean
    buckets.switch_to(shape)   # save outgoing v_hat, restore incoming (survives the
                               #   empty_cache -- snapshots are live tensors)
    # ... next step recaptures the graph for `shape`'s geometry
"""
import torch


class ConcordVHatBuckets:
    """modules: any iterable of packed layers / embedding cores carrying .v_row/.v_col
    (e.g. ConcordController.layers + each control_plane.cp.trainable.core). v_hat is
    resolved lazily per module and keyed by id(), so it is robust to lazy buffer
    allocation (v_row may be None until the first step) and to module-list order."""

    def __init__(self, modules):
        self.modules = list(modules)
        self.cache = {}        # shape_key -> {id(module): (v_row_clone, v_col_clone)}
        self.active = None

    def _live(self):
        return [m for m in self.modules
                if getattr(m, "v_row", None) is not None
                and getattr(m, "v_col", None) is not None]

    @torch.no_grad()
    def _snapshot(self):
        return {id(m): (m.v_row.detach().clone(), m.v_col.detach().clone())
                for m in self._live()}

    @torch.no_grad()
    def switch_to(self, shape_key):
        """Save the active shape's v_hat, restore shape_key's (in-place). Returns True
        if v_hat was restored from cache, False if this shape is new (warm-started from
        the current v_hat -- a related shape's stats beat a cold reset). No-op if
        shape_key is already active."""
        if shape_key == self.active:
            return None
        if self.active is not None:
            self.cache[self.active] = self._snapshot()
        snap = self.cache.get(shape_key)
        restored = snap is not None
        if restored:
            for m in self._live():
                t = snap.get(id(m))
                if t is not None:
                    m.v_row.copy_(t[0])
                    m.v_col.copy_(t[1])
        self.active = shape_key
        return restored

    def seen_shapes(self):
        s = set(self.cache)
        if self.active is not None:
            s.add(self.active)
        return s

    def memory_bytes(self):
        return sum(vr.numel() * vr.element_size() + vc.numel() * vc.element_size()
                   for snap in self.cache.values() for (vr, vc) in snap.values())
