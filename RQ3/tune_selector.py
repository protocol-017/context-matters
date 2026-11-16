# validation-only selector tuning
# this module freezes selector weights, k, budget, and validation search evidence.

from __future__ import annotations

import argparse
import json
from itertools import product

import numpy as np

from snmpaper.RQ3.config import BATCH_SIZE, K_DEFAULT, K_VALUES, GRID_STEP, TOKEN_BUDGET, VAL_SEED, save_frozen, FROZEN_DIR
from snmpaper.RQ3.context_selection import SelectorConfig, TokenCounter
from snmpaper.RQ3.data import load_split, sample_ids
from snmpaper.RQ3.embeddings import load_embeddings
from snmpaper.RQ3.llm_runner import LLMRunner
from snmpaper.RQ3.metrics import confusion, macro_f1
from snmpaper.RQ3.pipeline import Pipeline

def _score(pipeline: Pipeline, runner: LLMRunner, ids: list[str], config: SelectorConfig,
           kind: str) -> float:
    records = []
    for start in range(0, len(ids), BATCH_SIZE):
        examples = pipeline.build_examples(ids[start:start + BATCH_SIZE], "combined", kind, VAL_SEED, config)
        records.extend(pipeline.run_condition(runner, examples, {"kind": kind}))
    matrix = confusion([record["gold"] for record in records], [record["pred"] for record in records])
    return macro_f1(matrix)

def _objective(scores: dict[str, float]) -> float:
    return float(np.mean([scores["clean"], scores["insert_start"], scores["insert_before_target"]]))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    frozen_path = FROZEN_DIR / "selector_config.json"
    if frozen_path.exists() and not args.overwrite:
        raise FileExistsError(f"{frozen_path} already exists; pass --overwrite to retune")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("qwen selector tuning requires torch and a cuda gpu") from error
    if not torch.cuda.is_available():
        raise RuntimeError("qwen selector tuning requires a cuda gpu; no gpu is available")
    split = load_split("validation")
    ids = sample_ids(split, args.n_val, VAL_SEED)
    counter = TokenCounter()
    embeddings = load_embeddings("validation", split)
    pipeline = Pipeline("validation", split, embeddings, counter)
    runner = LLMRunner("qwen")
    values = [round(i * GRID_STEP, 2) for i in range(round(1 / GRID_STEP) + 1)]
    stage_one = []
    for alpha, beta in product(values, repeat=2):
        gamma = round(1 - alpha - beta, 2)
        if gamma < 0 or gamma > 1:
            continue
        config = SelectorConfig(K_DEFAULT, TOKEN_BUDGET, alpha, beta, gamma)
        scores = {kind: _score(pipeline, runner, ids, config, kind)
                  for kind in ("clean", "insert_start", "insert_before_target")}
        stage_one.append({"alpha": alpha, "beta": beta, "gamma": gamma,
                          "k": K_DEFAULT, "budget": TOKEN_BUDGET,
                          "scores": scores, "objective": _objective(scores)})
    best_weights = max(stage_one, key=lambda item: item["objective"])
    stage_two = []
    for k in K_VALUES:
        config = SelectorConfig(k, TOKEN_BUDGET, best_weights["alpha"], best_weights["beta"], best_weights["gamma"])
        scores = {kind: _score(pipeline, runner, ids, config, kind)
                  for kind in ("clean", "insert_start", "insert_before_target")}
        stage_two.append({"alpha": config.alpha, "beta": config.beta, "gamma": config.gamma,
                          "k": k, "budget": TOKEN_BUDGET, "scores": scores,
                          "objective": _objective(scores)})
    best = max(stage_two, key=lambda item: item["objective"])
    save_frozen({"alpha": best["alpha"], "beta": best["beta"], "gamma": best["gamma"],
                 "k": best["k"], "budget": best["budget"], "seed": VAL_SEED,
                 "validation_ids": ids, "objective": "mean clean and insertion macro-f1",
                 "search_log": {"stage_one": stage_one, "stage_two": stage_two}})
    print(json.dumps({"selected": best, "validation_ids": ids}, indent=2))

if __name__ == "__main__":
    main()
