# experiment command-line runner
# this module loads one split, enforces frozen test settings, and writes resumable condition logs.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snmpaper.RQ3.config import BATCH_SIZE, K_DEFAULT, K_VALUES, MODELS, RESULTS_DIR, TEST_SEEDS, load_frozen
from snmpaper.RQ3.context_selection import SelectorConfig, TokenCounter
from snmpaper.RQ3.data import load_or_build_subset, load_split
from snmpaper.RQ3.embeddings import load_embeddings
from snmpaper.RQ3.llm_runner import LLMRunner
from snmpaper.RQ3.pipeline import Pipeline, save_jsonl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--scope", choices=("full", "subset"), default="subset")
    parser.add_argument("--perturb", choices=("clean", "insert_start", "insert_before_target", "remove_parent", "remove_all"), default="clean")
    parser.add_argument("--methods", nargs="+", default=("none", "parent", "nearest_k", "full", "semantic_only", "combined"))
    parser.add_argument("--ablation", nargs="+", default=())
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-unfrozen", action="store_true")
    args = parser.parse_args()
    try:
        frozen = load_frozen()
    except FileNotFoundError:
        if not args.allow_unfrozen:
            raise RuntimeError("frozen selector configuration is required; run tune_selector.py first")
        frozen = {"alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3, "k": K_DEFAULT, "budget": 512}
    if args.ablation and not (args.model == "qwen" and args.scope == "subset"):
        raise ValueError("ablations require qwen and subset")
    split = load_split(args.split)
    ids = list(split.comments) if args.scope == "full" else load_or_build_subset(split)
    ids = ids[:args.limit] if args.limit else ids
    emb = load_embeddings(args.split, split)
    counter = TokenCounter()
    pipeline = Pipeline(args.split, split, emb, counter)
    runner = LLMRunner(args.model)
    seeds = TEST_SEEDS if args.perturb in {"insert_start", "insert_before_target"} else (0,)
    methods = list(args.methods) + list(args.ablation)
    selector = SelectorConfig(k=int(frozen["k"]), budget=int(frozen["budget"]), alpha=float(frozen["alpha"]),
                              beta=float(frozen["beta"]), gamma=float(frozen["gamma"]))
    output_dir = RESULTS_DIR / args.model / f"{args.split}_{args.scope}"
    all_meta = {"model": args.model, "split": args.split, "scope": args.scope, "perturb": args.perturb}
    for seed in seeds:
        for method in methods:
            suffix = args.perturb if seed == 0 else f"{args.perturb}_s{seed}"
            path = output_dir / f"{suffix}__{method}.jsonl"
            if path.exists() and not args.overwrite:
                continue
            records = []
            for start in range(0, len(ids), args.batch_size):
                examples = pipeline.build_examples(ids[start:start + args.batch_size], method, args.perturb, seed, selector)
                records.extend(pipeline.run_condition(runner, examples, {**all_meta, "method": method, "seed": seed}))
            save_jsonl(path, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_meta.json").write_text(json.dumps({**all_meta, "runner": runner.meta()}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
