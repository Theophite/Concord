"""Write a OneTrainer config with `additional_embeddings` PRE-LOADED from the token
vocabulary in token_init.LIST, so the GUI's "additional embeddings" tab opens already
populated -- one trainable embedding per token (placeholder <name>, subword-mean init from
the surface form). Edit token_init.LIST to add/remove tokens.

There is NO cap on the number of embeddings (config / training / the Concord packed path
are all dynamic) -- the GUI's "additional embeddings" tab just bogs down past ~100 heavy
rows, so adding the Nth via the button fails to render. This script is the bypass: add the
token to token_list.local.txt and run it.

Run with the OneTrainer venv:

  # MERGE into your existing config -- keeps every current embedding AND its uuid (so
  # trained state still maps on resume) and APPENDS only tokens not already present.
  # This is how you add the 101st (and beyond). Writes the config in place by default.
  set CONCORD_BASE_CONFIG=path/to/your_config.json
  venv/Scripts/python.exe modules/util/optimizer/concord/preinit_embeddings.py

  # or build a fresh default config with just the embeddings + optimizer=CONCORD:
  venv/Scripts/python.exe modules/util/optimizer/concord/preinit_embeddings.py

Output: CONCORD_OUT_CONFIG. Default = the base config itself when injecting (in-place update,
written atomically), else training_presets/concord_embeddings.json (so a fresh build appears
in the GUI dropdown). Adding an embedding changes the model (a new token), so RESTART training
to pick it up -- it takes effect on a fresh setup, not a hot reload.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # OneTrainer repo root
sys.path.insert(0, str(Path(__file__).parent))                 # this dir (token_init)

from token_init import additional_embeddings_from_list
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer

config = TrainConfig.default_values()
base = os.environ.get("CONCORD_BASE_CONFIG")
new_embs = additional_embeddings_from_list()
if base:
    # MERGE: keep the existing embeddings (and their uuids -> trained state still maps on
    # resume) and APPEND only tokens whose placeholder isn't already present. Dedup by
    # placeholder, so re-running after adding a line to token_list.local.txt only adds the
    # new one(s). (The old behavior REPLACED the list with fresh uuids, which would orphan
    # every trained embedding -- never what you want when adding the Nth.)
    config.from_dict(json.load(open(base, "r")))
    existing = list(config.additional_embeddings or [])
    have = {e.placeholder for e in existing}
    to_add = [e for e in new_embs if e.placeholder not in have]
    config.additional_embeddings = existing + to_add
    print(f"[merge] {base}: {len(existing)} kept (uuids preserved) + {len(to_add)} new "
          f"appended ({len(new_embs) - len(to_add)} list token(s) already present)")
else:
    config.optimizer.optimizer = Optimizer.CONCORD             # sensible default for a fresh config
    config.additional_embeddings = new_embs
    print(f"[base] fresh default config (optimizer=CONCORD): {len(new_embs)} embedding(s)")

# Injecting -> update the base config IN PLACE (the intent of "add to my config"); a fresh
# build goes to the preset dir so it shows in the GUI dropdown. Override with CONCORD_OUT_CONFIG.
default_out = base if base else str(
    Path(__file__).resolve().parents[4] / "training_presets" / "concord_embeddings.json")
out = Path(os.environ.get("CONCORD_OUT_CONFIG", default_out))
out.parent.mkdir(parents=True, exist_ok=True)
# Atomic write (temp + replace) so an interrupted run can't corrupt an in-place config.
tmp = out.with_suffix(out.suffix + ".tmp")
with open(tmp, "w") as f:
    json.dump(config.to_dict(), f, indent=1, default=str)
os.replace(tmp, out)
print(f"[done] {len(config.additional_embeddings)} embedding(s) total -> {out}")
if base:
    print("       Updated your config in place. RESTART training to pick up the new token(s)")
    print("       (a new embedding changes the model -> fresh setup, not a hot reload).")
else:
    print(f"       In the OneTrainer GUI top bar, pick '{out.stem}' from the config dropdown")
    print("       (RESTART the GUI first if it's already open -- the dropdown is scanned at startup).")
