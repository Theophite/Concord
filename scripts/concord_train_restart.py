"""
Concord checkpoint-restart wrapper.

The Concord CUDA-graph path and periodic sampling cannot coexist in one process on Windows: each
sample irreversibly fragments the VRAM heap (empty_cache / reset / model-defrag cannot reclaim the
fragmented-but-not-live reserved memory), so within a few samples the post-sample graph recapture's
warmup thrashes at the 24 GB ceiling and the run appears to "hang on resume". The graph recapture
itself is sound -- proven: forced recaptures with NO sampling never hang. The fix is to give each
post-sample recapture a FRESH process with a clean allocator.

This wrapper does that. It runs scripts/train.py, and whenever the trainer checkpoints and exits
with code 42 (right after a sample), it relaunches a fresh process that resumes from that backup.
Net effect: full graph speedup during training, clean sampling, no fragmentation wedge -- at the
cost of one model reload (~1-2 min) per sample.

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
    # Tell the trainer to checkpoint + exit(42) after each sample (instead of recapturing in-process
    # and wedging on fragmented VRAM).
    env["CONCORD_RESTART_ON_SAMPLE"] = "1"

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
