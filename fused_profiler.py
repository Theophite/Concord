"""CUDA-event-based timing for fused training. Accumulates time per
category over many calls, prints summary on demand."""
import torch
from collections import defaultdict


class FusedProfiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.totals = defaultdict(float)  # category -> total seconds
        self.counts = defaultdict(int)
        self._pool = []   # reusable event pool

    def _get_pair(self):
        if len(self._pool) >= 2:
            e2 = self._pool.pop()
            e1 = self._pool.pop()
        else:
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
        return e1, e2

    def _release(self, e1, e2):
        self._pool.extend([e1, e2])

    class _Ctx:
        def __init__(self, prof, name):
            self.prof = prof
            self.name = name
            self.e1 = self.e2 = None
        def __enter__(self):
            if not self.prof.enabled:
                return self
            self.e1, self.e2 = self.prof._get_pair()
            self.e1.record()
            return self
        def __exit__(self, *args):
            if not self.prof.enabled:
                return
            self.e2.record()
            # Defer time-elapsed query to summarize() to avoid sync per call
            self.prof._pending.append((self.name, self.e1, self.e2))

    def time(self, name):
        if not hasattr(self, '_pending'):
            self._pending = []
        return self._Ctx(self, name)

    def flush(self):
        if not self.enabled:
            return
        if not hasattr(self, '_pending'):
            return
        torch.cuda.synchronize()
        for name, e1, e2 in self._pending:
            ms = e1.elapsed_time(e2)
            self.totals[name] += ms / 1000.0
            self.counts[name] += 1
            self._release(e1, e2)
        self._pending = []

    def summarize(self, total_time=None):
        self.flush()
        if not self.totals:
            return ''
        total = sum(self.totals.values())
        lines = [f'  --- Profile (kernel time, total = {total:.2f}s) ---']
        items = sorted(self.totals.items(), key=lambda kv: -kv[1])
        for name, t in items:
            cnt = self.counts[name]
            pct = 100 * t / max(total, 1e-9)
            per_call_us = (t / max(cnt, 1)) * 1e6
            lines.append(f'    {name:<30} {t:6.2f}s ({pct:4.1f}%) '
                         f'over {cnt:6d} calls ({per_call_us:6.0f} us/call)')
        if total_time is not None and total_time > 0:
            wall_pct = 100 * total / total_time
            lines.append(f'  kernel time / wall time: {wall_pct:.1f}%')
        return '\n'.join(lines)

    def reset(self):
        self.totals.clear()
        self.counts.clear()
        if hasattr(self, '_pending'):
            self._pending = []


# Module-level singleton; layers grab it lazily.
PROFILER = FusedProfiler(enabled=False)


def enable(enabled=True):
    PROFILER.enabled = enabled
