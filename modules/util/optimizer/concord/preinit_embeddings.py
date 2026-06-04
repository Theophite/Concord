"""Write a OneTrainer config with `additional_embeddings` PRE-LOADED from the token
vocabulary in token_init.LIST, so the GUI's "additional embeddings" tab opens already
populated -- one trainable embedding per token (placeholder <name>, subword-mean init from
the surface form). Edit token_init.LIST to add/remove tokens.

Run with the OneTrainer venv:

  # inject into your existing config (keeps your model / dataset / settings):
  set CONCORD_BASE_CONFIG=path/to/your_config.json
  venv/Scripts/python.exe modules/util/optimizer/concord/preinit_embeddings.py

  # or build a fresh default config with just the embeddings + optimizer=CONCORD:
  venv/Scripts/python.exe modules/util/optimizer/concord/preinit_embeddings.py

Output: CONCORD_OUT_CONFIG (default training_presets/concord_embeddings.json, so it appears
in the GUI's config dropdown). In the top bar, pick it from the dropdown -- RESTART the GUI
first if it's already open (the dropdown is scanned at startup). The "additional embeddings"
tab will then show the list.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # OneTrainer repo root
sys.path.insert(0, str(Path(__file__).parent))                 # this dir (token_init)

from token_init import additional_embeddings_from_list, parse_list
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer

config = TrainConfig.default_values()
base = os.environ.get("CONCORD_BASE_CONFIG")
if base:
    config.from_dict(json.load(open(base, "r")))
    print(f"[base] injecting embeddings into {base}")
else:
    config.optimizer.optimizer = Optimizer.CONCORD             # sensible default for a fresh config
    print("[base] fresh default config (optimizer=CONCORD)")

config.additional_embeddings = additional_embeddings_from_list()

# Default to OneTrainer's preset dir so it shows in the GUI's config dropdown (the top bar
# lists training_presets/*.json; selecting a config there fires load_preset -> the
# additional-embeddings tab refreshes). Writing it anywhere else won't appear in the GUI.
default_out = Path(__file__).resolve().parents[4] / "training_presets" / "concord_embeddings.json"
out = Path(os.environ.get("CONCORD_OUT_CONFIG", str(default_out)))
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(config.to_dict(), f, indent=1, default=str)
print(f"[done] {len(config.additional_embeddings)} embeddings ({len(parse_list())} tokens) "
      f"pre-loaded -> {out}")
print(f"       In the OneTrainer GUI top bar, pick '{out.stem}' from the config dropdown")
print("       (RESTART the GUI first if it's already open -- the dropdown is scanned at")
print("       startup). The 'additional embeddings' tab will then show the list.")
