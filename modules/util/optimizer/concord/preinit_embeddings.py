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

Output: CONCORD_OUT_CONFIG (default ./concord_embeddings_config.json). Load it in the GUI
(or pass to scripts/train.py) -- the "additional embeddings" tab will show the list.
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

out = os.environ.get("CONCORD_OUT_CONFIG", "concord_embeddings_config.json")
with open(out, "w") as f:
    json.dump(config.to_dict(), f, indent=1, default=str)
print(f"[done] {len(config.additional_embeddings)} embeddings ({len(parse_list())} tokens) "
      f"pre-loaded -> {out}")
print("       Load it in the OneTrainer GUI; the 'additional embeddings' tab will be populated.")
