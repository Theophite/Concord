"""Build a minimal OneTrainer validation run for optimizer=CONCORD: a tiny synthetic
dataset (filenames are the captions) + a concepts JSON + a fine-tune config JSON.
Run with the OneTrainer venv. Then: scripts/train.py --config-path <config>."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

OT = str(Path(__file__).resolve().parents[5])                    # OneTrainer repo root
sys.path.insert(0, OT)

VAL = Path(os.environ.get("CONCORD_TEST_WORKDIR",
                          str(Path(tempfile.gettempdir()) / "concord_ot_test")))
for d in (VAL / "workspace", VAL / "cache", VAL / "output"):
    d.mkdir(parents=True, exist_ok=True)

# 1. CONCEPT (dataset): point CONCORD_TEST_DATASET at your own image directory; otherwise a
#    tiny synthetic one is generated (filenames are the captions, prompt_source='filename').
_user_ds = os.environ.get("CONCORD_TEST_DATASET")
DS = Path(_user_ds) if _user_ds else VAL / "dataset"
if _user_ds:
    print(f"[concept] using your dataset -> {DS}")
else:
    DS.mkdir(parents=True, exist_ok=True)
    captions = ["a photo of a red square", "a photo of a blue circle", "a photo of a green field",
                "a photo of a corgi", "a photo of a mountain", "a photo of a sunset",
                "a photo of a cat", "a photo of a wooden house"]
    rng = np.random.RandomState(0)
    for i, cap in enumerate(captions):
        base = np.array([(i * 31) % 256, (i * 61) % 256, (i * 91) % 256], dtype=np.float32)
        arr = (rng.rand(512, 512, 3) * 70 + base * 0.7).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(DS / f"{cap}.png")
    print(f"[concept] synthesized {len(captions)} images -> {DS}")

# 2. concepts.json (a list of concept dicts)
from modules.util.config.ConceptConfig import ConceptConfig
cc = ConceptConfig.default_values()
cc.name = "val"
cc.path = str(DS)
cc.enabled = True
cc.text.prompt_source = os.environ.get("CONCORD_TEST_PROMPT_SOURCE", "filename")  # filename|sample|concept
concepts_path = VAL / "concepts.json"
concepts_path.write_text(json.dumps([cc.to_dict()], indent=1, default=str))
print(f"[concepts] -> {concepts_path}")

# 3. config.json (defaults + overrides)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.enum.DataType import DataType
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
c = TrainConfig.default_values()
c.model_type = ModelType.STABLE_DIFFUSION_XL_10_BASE
_model = os.environ.get("CONCORD_TEST_MODEL")          # INPUT model
if not _model:
    raise SystemExit("Set CONCORD_TEST_MODEL to an SDXL .safetensors checkpoint (the input model).")
c.base_model_name = _model
c.training_method = TrainingMethod.FINE_TUNE
c.optimizer.optimizer = Optimizer.CONCORD
c.learning_rate = 1e-5
c.resolution = "512"
c.batch_size = 1
c.gradient_accumulation_steps = 1
c.epochs = 3
c.gradient_checkpointing = GradientCheckpointingMethod.ON
c.train_dtype = DataType.BFLOAT_16
c.concept_file_name = str(concepts_path)
c.workspace_dir = str(VAL / "workspace")
c.cache_dir = str(VAL / "cache")
c.output_model_destination = os.environ.get(           # OUTPUT model
    "CONCORD_TEST_OUTPUT", str(VAL / "output" / "concord_val.safetensors"))
c.unet.train = True
c.text_encoder.train = False
c.text_encoder_2.train = False
c.text_encoder.train_embedding = False
c.text_encoder_2.train_embedding = False
c.concord_sanitize_tokens = "dog,guitar,banana"   # Stage 2: zero these single-token words
# config packing reads these referenced files even when disabled -> point at empty ones
samples_path = VAL / "samples.json"
samples_path.write_text("[]")
c.sample_definition_file_name = str(samples_path)
# keep the smoke run focused on training: no sampling / backup / mid-run save
try:
    from modules.util.enum.TimeUnit import TimeUnit
    for fld in ("sample_after_unit", "backup_after_unit", "save_after_unit"):
        if hasattr(c, fld):
            setattr(c, fld, TimeUnit.NEVER)
    print("[config] disabled sample/backup/save via TimeUnit.NEVER")
except Exception as e:
    print("[warn] TimeUnit.NEVER not set:", repr(e))
config_path = VAL / "config.json"
config_path.write_text(json.dumps(c.to_dict(), indent=1, default=str))
print(f"[config] -> {config_path}")
print(f"[run] {OT}\\venv\\Scripts\\python.exe scripts/train.py --config-path {config_path}")
