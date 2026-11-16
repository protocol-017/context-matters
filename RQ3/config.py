# configuration for experiment 3
# this module defines paths, dataset columns, model settings, and frozen selector state.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MTCSD_DIR", ROOT.parent / "snmpaper" / "data" / "processed"))
OUT_DIR = Path(os.environ.get("EXP3_OUT", ROOT / "outputs"))
FROZEN_DIR = OUT_DIR / "frozen"
CACHE_DIR = OUT_DIR / "cache"
RESULTS_DIR = OUT_DIR / "results"
TABLES_DIR = OUT_DIR / "tables"

SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
EXPECTED_THREADS = {"train": 150, "validation": 33, "test": 35}
COLUMNS = {
    "id": "message_id", "text": "text", "target": "target", "label": "label",
    "parent_id": "parent_id", "thread_id": "thread_id", "depth": "depth",
}
DROP_COLUMNS = ("logical_relationship", "dialogue_act")
LABEL_MAP = {"FAVOR": "Favor", "AGAINST": "Against", "NONE": "None",
             "favor": "Favor", "against": "Against", "none": "None"}
MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
BUDGET_TOKENIZER = MODELS["qwen"]
QUANTIZATION = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "float16"}
MAX_NEW_TOKENS = 8
BATCH_SIZE = 8
TOKEN_BUDGET = 512
MAX_MSG_TOKENS = 200
MAX_TARGET_TOKENS = 512
ROLE_OVERHEAD_TOKENS = 12
K_VALUES = (3, 5)
K_DEFAULT = 3
EMBED_MODEL = "all-MiniLM-L6-v2"
RECENCY_DECAY = 0.5
GRID_STEP = 0.25
PERTURBATIONS = ("clean", "insert_start", "insert_before_target", "remove_parent", "remove_all")
SEEDED = ("insert_start", "insert_before_target")
INSERT_N = 2
TEST_SEEDS = (11, 22, 33)
VAL_SEED = 7
SUBSET_SIZE = 1000
SUBSET_SEED = 42
SUBSET_MIN_DEPTH = 2
PROMPT_VERSION = "v1"

def save_frozen(values: dict[str, Any]) -> Path:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    path = FROZEN_DIR / "selector_config.json"
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    return path

def load_frozen() -> dict[str, Any]:
    path = FROZEN_DIR / "selector_config.json"
    if not path.exists():
        raise FileNotFoundError(f"frozen selector configuration is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
