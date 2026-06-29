"""concord_core._smoke — standalone smoke / diagnose harness (NOT imported in prod).

RESPONSIBILITY
    The hand-run sanity harness: _run_one / diagnose / smoke_test / main plus the
    packed-b and reference torch MLP factories. Never imported by production or by
    the test suite -> trivially safe to move. Imports the full stack (kernels,
    layers) so it only runs on the GPU path.

    *** MIGRATION FIX (required) ***  The monolith tail (PB:4148-4150) ends with a
    bare top-level ``sys.exit(1)`` and has NO ``if __name__ == "__main__":`` guard.
    When this code moves to its own module, ADD the guard so a plain
    ``import concord_core._smoke`` stays INERT (no exit, no GPU work):
        if __name__ == "__main__":
            main()
    Also drop the now-unused top-level ``import sys`` / ``import time`` from the
    monolith IF (verify first) only the smoke code used them.

MIGRATES FROM prototype_packed_b.py (PB) — DO NOT MOVE CODE YET; line-range map only:
    PB:3948-4150  _run_one, _packed_b_mlp*, _torch_mlp*, diagnose, smoke_test, main
                  (+ ADD the missing ``if __name__ == "__main__": main()`` guard)

RE-EXPORT: none (no external consumer; not re-exported by the shim).

MIGRATION: STEP 3 (fully independent; can run first). Gate: L0 + L1
    (smoke is never imported in prod/tests -> trivially safe; just confirm import
    is inert after the guard is added).
"""
