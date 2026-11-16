# cached sentence embeddings
# this module computes l2-normalised embeddings for comments and stores them per split.

from __future__ import annotations

from pathlib import Path

import numpy as np

from snmpaper.RQ3.config import CACHE_DIR, EMBED_MODEL
from snmpaper.RQ3.data import Split

def load_embeddings(split_name: str, split: Split, model_name: str = EMBED_MODEL) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{split_name}_{model_name.rsplit('/', 1)[-1]}.npz"
    if path.exists():
        loaded = np.load(path)
        return {key: loaded[key] for key in loaded.files}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    ids = list(split.comments)
    vectors = model.encode([split.comments[cid].text for cid in ids], normalize_embeddings=True,
                           show_progress_bar=False)
    result = {cid: np.asarray(vector, dtype=np.float32) for cid, vector in zip(ids, vectors)}
    np.savez_compressed(path, **result)
    return result
