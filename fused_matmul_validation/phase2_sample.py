"""Phase 2: sample one Concord checkpoint in a fresh clean-memory process.

For a single backup (epoch E): isolate it so `continue_last_backup` loads exactly it,
launch train.py (resume -> the start-of-run sample fires on E's PURE weights BEFORE any
training), kill the whole process tree the instant the 12 sample PNGs land (so no training
step runs -> the CUDA graph never captures -> no fragmentation wedge, no eager training),
collect the images to phase2_out/epochNN/, and restore all backups.

Usage:  python phase2_sample.py <epoch:int> <backup_dirname>
"""
import os, sys, time, glob, shutil, subprocess

WS         = r"C:/Concord/overfit/workspace"
BACKUP     = os.path.join(WS, "backup")
HELD       = os.path.join(WS, "backup_held")
SAMPLE_DIR = os.path.join(WS, "samples")   # OneTrainer writes <idx> - <prompt>/<ts>.jpg here
OUT        = r"C:/Concord/overfit/phase2_out"
CONFIG     = r"C:/Concord/overfit/phase2_sample.json"
OT_DIR     = r"C:/fisher/OneTrainer-clean"
PY         = r"C:/fisher/OneTrainer-clean/venv/Scripts/python.exe"
N_PROMPTS  = 12
TIMEOUT    = 1800   # 30 min hard cap (model load + 12 samples should be ~2-3 min)


def backups_in(d):
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d)
                  if os.path.isdir(os.path.join(d, x)) and "backup-" in x)


def isolate(target):
    """Leave exactly `target` in BACKUP; move every other backup to HELD."""
    os.makedirs(HELD, exist_ok=True)
    for x in backups_in(BACKUP):
        if x != target:
            shutil.move(os.path.join(BACKUP, x), os.path.join(HELD, x))
    # if target was already moved aside in a prior run, bring it back
    if target in backups_in(HELD) and target not in backups_in(BACKUP):
        shutil.move(os.path.join(HELD, target), os.path.join(BACKUP, target))


def restore_all():
    """Move everything from HELD back into BACKUP."""
    for x in backups_in(HELD):
        dst = os.path.join(BACKUP, x)
        if not os.path.exists(dst):
            shutil.move(os.path.join(HELD, x), dst)


def imgs():
    s = set()
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        s |= set(glob.glob(os.path.join(SAMPLE_DIR, "**", ext), recursive=True))
    return s


def kill_tree(p):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        p.wait(timeout=30)
    except Exception:
        pass


def run_one(epoch, target):
    isolate(target)
    before = imgs()
    logpath = rf"C:/Concord/overfit/phase2_ep{epoch}.log"
    log = open(logpath, "w")
    env = dict(os.environ)
    env.pop("CONCORD_RESUMING", None)
    env.pop("CONCORD_RESTART_ON_SAMPLE", None)
    print(f"[phase2] ep{epoch}: launching resume+sample from {target}", flush=True)
    p = subprocess.Popen([PY, "scripts/train.py", "--config-path", CONFIG],
                         cwd=OT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    new = set()
    while time.time() - t0 < TIMEOUT:
        if p.poll() is not None:
            print(f"[phase2] ep{epoch}: process exited early (code {p.returncode})", flush=True)
            break
        new = imgs() - before
        if len(new) >= N_PROMPTS:
            time.sleep(3)                 # let the last image flush
            new = imgs() - before
            print(f"[phase2] ep{epoch}: {len(new)} images landed -> killing before train step", flush=True)
            break
        time.sleep(2)
    kill_tree(p)

    outdir = os.path.join(OUT, f"epoch{epoch:02d}")
    os.makedirs(outdir, exist_ok=True)
    # first N_PROMPTS by mtime = this checkpoint's single pre-training sample event
    for i, src in enumerate(sorted(new, key=os.path.getmtime)[:N_PROMPTS]):
        shutil.copy(src, os.path.join(outdir, f"{i:02d}_{os.path.basename(src)}"))
    print(f"[phase2] ep{epoch}: collected {len(new)} -> {outdir}", flush=True)
    return len(new), outdir


if __name__ == "__main__":
    epoch = int(sys.argv[1]); target = sys.argv[2]
    try:
        n, outdir = run_one(epoch, target)
    finally:
        restore_all()
        print(f"[phase2] backups restored ({len(backups_in(BACKUP))} in place)", flush=True)
    print(f"PHASE2_RESULT epoch={epoch} images={n} out={outdir}", flush=True)
