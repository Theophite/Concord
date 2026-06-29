"""
Concord checkpoint-restart wrapper.

On a card the model nearly fills (24 GB), the Concord CUDA-graph boundary work fragments the VRAM
heap irreversibly: empty_cache / reset / model-defrag cannot reclaim the fragmented-but-committed
reserved memory, so the in-process recommit + graph recapture at each boundary overflows the
dedicated ceiling and WDDM demotes the tail to shared memory. The demotion is STICKY and COMPOUNDS
across boundaries (observed 1.08 -> 2.60 s/it over a few epochs). The graph recapture itself is
sound -- proven: forced recaptures with NO boundary churn never hang. The fix is to give each
boundary recommit a FRESH process with a clean allocator.

This wrapper does that. It runs scripts/train.py with both segment triggers set; whenever the
trainer checkpoints and exits with code 42 (after a sample OR after a per-epoch backup -- whichever
the config produces), it relaunches a fresh process that resumes from that backup. The resume is
bit-faithful: the controller clock (concord_clock.json in the backup) restores the exact update-step
and the drive sidecar restores the per-token calibration, so divot / fill-ramp / drives all continue
seamlessly. Net effect: full graph speedup during training, a clean allocator every segment, no
demotion compounding -- at the cost of one model reload (~1-2 min) per segment (per epoch in the
standard sampling-off config).

Bounded crash-retry: the sample/backup boundary work runs at the 24 GB ceiling, and once in a while
the recommit/recapture there dies on a native fault (a CUDA-graph/WDDM access violation, exit code
0xC0000005) instead of exiting 42 cleanly. Rather than let one such fault end an unattended run, this
wrapper RESUMES from the last backup on an abnormal termination too -- but caps CONSECUTIVE crashes
(CONCORD_MAX_CRASH_RETRIES, default 5) so a persistently broken state can't loop forever. The cap
resets on every clean segment boundary (exit 42), so a run that keeps progressing is never starved.
Intentional exits (0 = done; a small non-zero = a real Python error; Ctrl+C) are trusted -> stop.

Usage (drop-in for scripts/train.py):
    python scripts/concord_train_restart.py --config-path path/to/config.json [--secrets-path ...]
    Env: CONCORD_MAX_CRASH_RETRIES=N (default 5) -- consecutive native-crash resumes before giving up.

Only relevant when the Concord CUDA graph is active (CONCORD optimizer + concord_cuda_graph gate)
AND sampling is enabled. Plain `python scripts/train.py ...` is completely unaffected.
"""
import os
import subprocess
import sys

RESTART_EXIT_CODE = 42

# Bounded crash-retry budget: consecutive native crashes we'll resume-through before giving up.
# Resets on every clean segment boundary (exit 42), so only crashes with NO intervening progress
# count against it. Tunable via env without editing the script.
MAX_CRASH_RETRIES = int(os.environ.get("CONCORD_MAX_CRASH_RETRIES", "5"))

# STATUS_CONTROL_C_EXIT: a Ctrl+C / GUI Stop that hard-terminates the child lands in the
# 0xC0000000 NTSTATUS range like a real fault, but it's user-initiated -- never auto-restart it.
_STATUS_CONTROL_C_EXIT = 0xC000013A


def _is_crash(code):
    """True only for an ABNORMAL termination the OS killed: a Windows NTSTATUS fault
    (0xC0000005 access-violation, 0xC00000FD stack-overflow, 0xC0000374 heap-corruption, ...) or
    a POSIX signal (negative code). Intentional Python exits -- 0 (done) and small positive codes
    (sys.exit(1) tracebacks, deliberate stops) -- are NOT crashes: a resume won't change their
    outcome, so we trust them and stop. Ctrl+C (STATUS_CONTROL_C_EXIT) is excluded explicitly."""
    if code == _STATUS_CONTROL_C_EXIT:
        return False
    return code < 0 or code >= 0xC0000000


def main():
    # CTRL_BREAK (GUI Stop) reaches this whole process group. Let the train.py CHILD handle it
    # (it turns SIGBREAK into a graceful KeyboardInterrupt + final save); IGNORE it here so the
    # wrapper isn't hard-terminated mid-wait -- subprocess.run() then blocks until the child
    # finishes saving and we exit with the child's (clean, non-42) code.
    import signal as _signal
    if hasattr(_signal, "SIGBREAK"):
        _signal.signal(_signal.SIGBREAK, _signal.SIG_IGN)

    here = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(here, "train.py")
    train_args = sys.argv[1:]

    env = dict(os.environ)
    # Tell the trainer to checkpoint + exit(42) at each segment boundary (instead of recapturing
    # in-process and wedging on fragmented/demoted VRAM). Both triggers are set so the wrapper works
    # whether the run samples, backs up, or both -- whichever boundary fires first ends the segment.
    # With sampling off (the standard config), the per-epoch BACKUP is the boundary that matters.
    env["CONCORD_RESTART_ON_SAMPLE"] = "1"
    env["CONCORD_RESTART_ON_BACKUP"] = "1"

    segment = 0
    consecutive_crashes = 0
    while True:
        if segment == 0:
            print(f"[concord-restart] launching training (segment {segment})", flush=True)
        else:
            print(f"[concord-restart] relaunching fresh process (segment {segment}) -> "
                  f"resume from last backup", flush=True)

        ret = subprocess.run([sys.executable, train_py] + train_args, env=env)
        code = ret.returncode

        if code == RESTART_EXIT_CODE:
            # Clean segment boundary: the prior process checkpointed. Resume from it, and reset
            # the crash budget -- we made forward progress.
            env["CONCORD_RESUMING"] = "1"
            segment += 1
            consecutive_crashes = 0
            continue

        if _is_crash(code):
            # Native crash (e.g. the 0xC0000005 CUDA-graph/WDDM fault at the VRAM ceiling). The
            # crashed process wrote no checkpoint this segment, so resume from the last good backup.
            consecutive_crashes += 1
            shown = f"{code} / 0x{code:08X}" if code >= 0xC0000000 else str(code)
            if consecutive_crashes > MAX_CRASH_RETRIES:
                print(f"[concord-restart] segment {segment} crashed (code {shown}); crash-retry "
                      f"budget exhausted ({MAX_CRASH_RETRIES} consecutive) -> stopping.", flush=True)
                sys.exit(code)
            env["CONCORD_RESUMING"] = "1"
            print(f"[concord-restart] segment {segment} crashed (code {shown}); "
                  f"crash-retry {consecutive_crashes}/{MAX_CRASH_RETRIES} -> resume from last backup.",
                  flush=True)
            continue

        # 0 = training finished normally; a small non-zero = a deliberate/clean error exit. Both
        # are intentional (a resume won't change the outcome) -- stop here.
        print(f"[concord-restart] training exited with code {code}; stopping "
              f"(after {segment} restart(s)).", flush=True)
        sys.exit(code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # GUI Stop / Ctrl+C delivers CTRL_BREAK to the whole process group: the train.py child
        # already caught its own KeyboardInterrupt and saved gracefully, and subprocess.run
        # re-raises it here. Exit cleanly -- no relaunch, no scary traceback.
        print("[concord-restart] interrupted -> stopping (child saved on its own KeyboardInterrupt)",
              flush=True)
        sys.exit(0)
