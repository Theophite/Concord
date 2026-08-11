"""kernel_select: bind the Concord kernel lineage ONCE per process.

Default (Optimizer.CONCORD): this module is NEVER CALLED -- production keeps
its exact module topology. HISTORY NOTE (stale claim corrected 2026-07-26):
an earlier revision of this docstring said production "keeps the documented
two-copy seam" -- that seam (bare `prototype_packed_b` vs package-path
imports as DISTINCT module objects, the set_consolidate/set_arm_sel
dead-knob bug) was CLOSED by prototype_packed_b's 2026-07-24 EOF
self-aliasing fix; both spellings now resolve to one module object on the
production lineage too. The fused-matmul block's "verified: bare is pkg ->
False" record predates that fix.

Optimizer.CONCORD_STEPLESS: bind_kernel("stepless") imports
prototype_packed_stepless ONCE and installs that single object under BOTH
sys.modules spellings plus the parent-package attribute -- every existing
import site (bare or package path, `import ... as ppb` or `from ... import
set_consolidate`) then resolves to the clone, byte-unchanged. Deliberate
consequences, on the record:
  - the stepless lineage has NO module seam: set_consolidate/set_arm_sel fill
    the same _CONSOLIDATE_FLAGS/_ARM_SEL_FLAGS device tensors the apply
    launcher reads, and layers + packed embeddings share ONE SR step-counter
    stream;
  - therefore a same-seed accum>1 A/B against CONCORD may differ BECAUSE of
    the production seam, not because of stepless math -- compare at accum=1
    (seam-free) until the seam investigation resolves.

Guard semantics: binding is set-once per process. Re-binding the same
selection is a no-op; a different selection raises (the GUI process persists
across Start presses -- restart the trainer to switch lineage; the restart
wrapper's fresh process per segment re-binds cleanly from the config). If
either kernel spelling was already imported before bind_kernel runs, we raise
rather than half-alias a live kernel.
"""
import importlib
import os
import sys

_BOUND = None
_BARE = "prototype_packed_b"
_PKG = "modules.util.optimizer.concord." + _BARE
_BARE2 = "prototype_packed_2fast"
_PKG2 = "modules.util.optimizer.concord." + _BARE2
_FILES = {"stepless": "prototype_packed_stepless"}
# The 2-fast kernel is a lineage member too: under "stepless" both
# prototype_packed_2fast spellings resolve to the stepless-2fast clone
# (which carries the fused hybrid gate). Claiming the production 2fast
# keys is exclusively THIS module's act -- the clone's EOF self-alias
# only ever claims its own stepless_2fast spellings.
_FILES2 = {"stepless": "prototype_packed_stepless_2fast"}


def bind_kernel(selection: str):
    global _BOUND
    if selection not in _FILES:
        raise ValueError(f"unknown Concord kernel lineage {selection!r} "
                         f"(known: {sorted(_FILES)})")
    if _BOUND is not None:
        if _BOUND != selection:
            raise RuntimeError(
                f"Concord kernel already bound to {_BOUND!r}; cannot rebind to "
                f"{selection!r} in this process. Restart the trainer to switch "
                f"lineage.")
        return sys.modules[_BARE]
    for key in (_BARE, _PKG, _BARE2, _PKG2):
        if key in sys.modules:
            raise RuntimeError(
                f"{key} was imported before bind_kernel() -- the lineage can no "
                f"longer be selected in this process. Restart the trainer (the "
                f"bind must run before the first kernel import; it is placed at "
                f"the top of the Concord block in setup_model).")
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)                 # bare-name imports resolve here
    mod = importlib.import_module(_FILES[selection])
    sys.modules[_BARE] = mod
    sys.modules[_PKG] = mod
    import modules.util.optimizer.concord as _pkg
    setattr(_pkg, _BARE, mod)                    # `from ...concord import prototype_packed_b`
    # 2fast member: imported AFTER the base aliasing so its own
    # `import prototype_packed_b` lands on the stepless base above.
    mod2 = importlib.import_module(_FILES2[selection])
    sys.modules[_BARE2] = mod2
    sys.modules[_PKG2] = mod2
    setattr(_pkg, _BARE2, mod2)
    os.environ["CONCORD_KERNEL"] = selection     # belt: visible to any subprocess
    _BOUND = selection
    print(f"[concord] KERNEL LINEAGE: {selection!r} ({_FILES[selection]}.py + "
          f"{_FILES2[selection]}.py) bound for this process -- all four import "
          f"spellings resolve to the two lineage module objects", flush=True)
    return mod
