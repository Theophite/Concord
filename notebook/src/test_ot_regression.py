"""Regression net for the Concord OneTrainer integration -- locks in the validated
Stage 1/2 behaviour BEFORE the Stage-3 v2 trainer surgery. Runs a real default-path
(graph OFF) SDXL fine-tune via OneTrainer and asserts the invariants the v2 cuts must
NOT break:

  A  the UNet swaps to Concord (794 layers)
  B  training does not go NaN
  C  the deploy consolidation fires (packed -> standard)
  D  the saved checkpoint is a real SDXL model: no packed_w, standard conv_in shape
  E  sanitize zeroed the configured single-token words, left controls intact

Run with the OneTrainer venv python. Heavy (a full ~2-min training run) -- it's an
integration regression, not a unit test. Re-run after the v2 trainer wiring to confirm
the default path is unregressed.
"""
import subprocess
import sys
from pathlib import Path

OT = r"C:\fisher\OneTrainer-clean"
VENV = OT + r"\venv\Scripts\python.exe"
CONFIG = r"C:\Concord\ot_val\config.json"
CKPT = r"C:\Concord\ot_val\output\concord_val.safetensors"
LOG = r"C:\Concord\ot_val\regression.log"

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

print("=== running default-path (graph OFF) Concord fine-tune ===")
with open(LOG, "w") as f:
    r = subprocess.run([VENV, "-u", "scripts/train.py", "--config-path", CONFIG],
                       cwd=OT, stdout=f, stderr=subprocess.STDOUT)
log = Path(LOG).read_text(errors="ignore")
print(f"[run] exit {r.returncode}")

check("A. UNet swapped to Concord (794 layers)", "swapped 794 UNet layers" in log)
check("B. training did not go NaN", "became NaN" not in log, "NaN seen" if "became NaN" in log else "")
check("C. deploy consolidation fired", "consolidated 794 layers" in log)

# D + E: inspect the saved checkpoint
sys.path.insert(0, OT)
try:
    from safetensors import safe_open
    from transformers import CLIPTokenizer
    f = safe_open(CKPT, "pt")
    keys = list(f.keys())
    no_packed = not any("packed_w" in k for k in keys)
    ek = [k for k in keys if "embedders.0" in k and "token_embedding.weight" in k]
    conv = [k for k in keys if "input_blocks.0.0.weight" in k]
    conv_ok = bool(conv) and tuple(f.get_tensor(conv[0]).shape) == (320, 4, 3, 3)
    check("D. checkpoint is standard SDXL (no packed_w, conv_in 320x4x3x3)", no_packed and conv_ok,
          f"no_packed={no_packed} conv_ok={conv_ok}")
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    emb = f.get_tensor(ek[0])
    def rownorm(w):
        ids = tok(w, add_special_tokens=False).input_ids
        return emb[ids[0]].float().norm().item() if len(ids) == 1 else None  # None = multi-token
    san = {w: rownorm(w) for w in ["dog", "guitar", "banana"]}
    ctl = {w: rownorm(w) for w in ["cat", "mountain"]}
    zeroed = all(n is not None and n < 1e-6 for n in san.values())   # NB: 0.0 is falsy -> test
    intact = all(n is not None and n > 0.1 for n in ctl.values())    # explicitly, never `or`
    check("E. sanitize zeroed dog/guitar/banana, kept cat/mountain", zeroed and intact,
          f"sanitized={san} controls={ctl}")
except Exception as e:
    check("D+E. checkpoint inspection", False, f"{type(e).__name__}: {e}")

print(f"\n[REGRESSION] {sum(results)}/{len(results)} invariants hold "
      + ("-> Stage 1/2 baseline locked" if all(results) else "-> BASELINE BROKEN"))
sys.exit(0 if all(results) else 1)
