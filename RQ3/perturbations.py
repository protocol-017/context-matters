# validation perturbations for context selection
# this module inserts controlled cross-thread noise or removes ancestors before selection.

from __future__ import annotations

import random
from dataclasses import replace

from snmpaper.RQ3.config import INSERT_N
from snmpaper.RQ3.data import Comment

def perturb_chain(chain: list[Comment], comment: Comment, kind: str, seed: int,
                  pool: list[Comment], noise_ctx: dict[str, list[Comment]] | None = None) -> list[Comment]:
    if kind == "clean":
        return list(chain)
    if kind == "remove_parent":
        return chain[:-1]
    if kind == "remove_all":
        return []
    if kind not in {"insert_start", "insert_before_target"}:
        raise ValueError(f"unknown perturbation: {kind}")
    if noise_ctx is not None and comment.id in noise_ctx:
        noise = noise_ctx[comment.id]
    else:
        candidates = [item for item in pool if item.thread_id != comment.thread_id]
        rng = random.Random(f"noise-{seed}-{comment.id}")
        noise = [replace(item, is_noise=True) for item in rng.sample(candidates, min(INSERT_N, len(candidates)))]
        if noise_ctx is not None:
            noise_ctx[comment.id] = noise
    return list(noise) + list(chain) if kind == "insert_start" else list(chain) + list(noise)
