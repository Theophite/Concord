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

Usage (drop-in for scripts/train.py):
    python scripts/concord_train_restart.py --config-path path/to/config.json [--secrets-path ...]

Only relevant when the Concord CUDA graph is active (CONCORD optimizer + concord_cuda_graph gate)
AND sampling is enabled. Plain `python scripts/train.py ...` is completely unaffected.
"""
import os
import subprocess
import sys

RESTART_EXIT_CODE = 42


def main():
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
    while True:
        if segment == 0:
            print(f"[concord-restart] launching training (segment {segment})", flush=True)
        else:
            print(f"[concord-restart] relaunching fresh process (segment {segment}) -> "
                  f"resume from last backup", flush=True)

        ret = subprocess.run([sys.executable, train_py] + train_args, env=env)

        if ret.returncode == RESTART_EXIT_CODE:
            # Every relaunch after the first resumes from the backup the prior process just wrote.
            env["CONCORD_RESUMING"] = "1"
            segment += 1
            continue

        # 0 = training finished normally; anything else = a real error. Either way, stop here.
        print(f"[concord-restart] training exited with code {ret.returncode}; stopping "
              f"(after {segment} restart(s)).", flush=True)
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
