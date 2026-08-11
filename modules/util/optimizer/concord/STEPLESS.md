# The stepless lineage

This branch carries `prototype_packed_stepless.py`,
`prototype_packed_stepless_2fast.py` and `kernel_select.py` — the stepless
clone of the Concord kernel. Until this commit those three files were
**untracked in any repository**: they were not in the research repo (production
source does not go there) and had never been added here, so the entire lineage
existed only as files on one disk.

## What "stepless" means

Concord's optimizer step currently does three separate jobs, and "stepless"
means removing all three:

1. **The step is a barrier.** Every packed word is read, modified and written
   every iteration, whether or not that coordinate did anything.
2. **The step is a clock.** The host advances lr, σ, the floors and the SR salt
   once per step, and everything downstream is timed off that tick.
3. **The step is the unit of accounting.** Schedules, budgets and the whole
   same-seed A/B methodology are denominated in steps.

The enabling fact is that **between gradient events the per-coordinate dynamics
are linear and time-invariant.** Chase, leak and evaporation are each a fixed
rate applied to a stored quantity, so Δ elapsed steps can be applied *exactly*
in O(1) with a compounded factor — `1 − (1 − a)^Δ` instead of Δ separate
applications of `a`. Nothing needs to be simulated step by step; a coordinate
that has been idle for 400 steps can be brought fully up to date in one visit.

Once consolidation can catch up lazily like that, the hot path collapses to the
**tick** alone. And the bit layout already supports the tick as a bare
`atomic_add`: the fast field is the high half of the int32 word, so an addend
of `tick << 16` reaches it without touching the deploy fields below.

So the arc is: make consolidation catch-up-able (compounded closed forms) →
split the tick away from consolidation (producer/visit) → let consolidation run
on its own schedule (waves, or a daemon) → stop denominating anything in steps.

## Where the lineage sits

`kernel_select.bind_kernel("stepless")` aliases **all four import spellings** to
the two stepless module objects, so `import prototype_packed_b` and
`import prototype_packed_2fast` resolve here after binding. That is why the
experiment rigs import the ordinary names and still exercise this code.

The clone diverges from the shipped kernel deliberately, and its banner carries
a divergence ledger. As of this commit the ledger covers: the module-identity
self-alias, the hybrid gate (v1.1 → v1.3), the Phase-1 drained visit, and the
Phase-2 work described below.

## What this branch adds beyond the drained visit

Everything is flag-gated and **off by default**; flag-off is byte-identical to
the previous behaviour.

- `set_bare_atomic(on, atomic=)` — the producer/visit split, as two constexpr
  projections (`SPLIT_ROLE`) of one kernel source. `atomic=False` deposits with
  an ordinary read-modify-write, which is legal whenever there is a single
  writer and ~3× cheaper than the atomic. **98.38% MNIST deploy at three seeds
  and +32.6% graphed** — the campaign's main result.
- `set_waves(K)` — K phase-shifted slices: consolidation becomes continuous
  (1/K of the parameters per micro) and the producer needs an atomic only for
  the slice currently under visit. **Do not enable without also disabling the
  producer spill** (see below).
- `set_producer_spill(on)` — the producer's pre-emptive relief valve. Default
  ON, and it is the known defect: its transfer is proportional to accumulated
  arm content and is *ungated*, so under a wave window it becomes the main road
  into `s_slow`, bypassing the coherence gate. Deploy accuracy collapses to
  14–34% with it on under waves and recovers to 96.2% with it off. The real fix
  is to move that relief into the visit, where it passes the gate.
- `set_leak_every(N)` — the leak on its own window with its own `q^Δ` compound.
  α_v is ~100× smaller than α, so the leak's natural timescale is ~1000 steps
  and a per-micro visit resolves nothing it cares about. **+0.11p at 64×,
  three seeds** — batching it also lifts the per-step increment above the int8
  resolution floor.
- `set_gate_every(N)` — the coherence gate on its own cadence, with a cached
  per-element `coh`. Accuracy-free at 8× and 16×, but buys ~0% because the gate
  was never the expensive part.
- `set_apply_block(tile, num_warps, num_stages)` — launch-config override.
  32×16 is **+9.7% paired median and bit-identical** on MNIST; it should
  eventually be an autotune key rather than a constant, since the best tile is
  shape-dependent.
- `set_grad_hook(fn)` / `set_count_hook(fn)` — the data-parallel seams. The
  optimizer step is fused into the backward, so the gradient is consumed the
  instant it exists and the v̂ EMAs read it *before* the apply; a reduction at
  the usual optimizer boundary is already too late. With the hook at the right
  place, five ranks stay **bit-identical** in words and all four exponent
  planes.
- `set_cstar_precond(on)`, `set_arrival_averaged(on)` — two Δ-corrections that
  are right in principle and measured null. Kept because invariant 3 requires
  the two C\* consumers to agree, and because the arrival-averaged retention is
  simply the better-founded compound.

## Reading order for someone picking this up

1. `docs/HANDOFF.md` in the research repo — what travels where, the method
   rules, the owed-work ledger.
2. `docs/STEPLESS_PLAN.md` — the phase plan this lineage implements.
3. `docs/PHASE2_BARE_ATOMIC_DESIGN.md` — the producer/visit design and its
   thirteen stop rules.
4. `experiments/cpu_dynamics/EXPERIMENTS.md`, entries exp82 onward — the lab
   log, including the retractions.

## Caveat carried with this commit

`concord_embedding_packed.py` also contains a change from this work (the
embedding calibration seam: `_accum`, `_seen` and `_power` need a **sum**
reduction, not the gradient hook's mean, because counts do not average). It is
**not** committed here: that file is tracked on the integration branch and its
working-tree diff mixes this change with other uncommitted work, so it needs
merging rather than overwriting.
