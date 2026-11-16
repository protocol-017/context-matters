# result aggregation
# this module reads experiment logs and writes metric, confusion, and depth tables.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from snmpaper.RQ3.config import RESULTS_DIR, TABLES_DIR
from snmpaper.RQ3.metrics import confusion, depth_bucket, per_class, summarize

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--scope", default="subset")
    args = parser.parse_args()
    root = RESULTS_DIR / args.model / f"{args.split}_{args.scope}"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in root.glob("*.jsonl"):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        matrix = confusion([r["gold"] for r in records], [r["pred"] for r in records])
        rows.append({"file": path.name, **summarize(matrix), "mean_input_tokens": sum(r["n_in"] for r in records) / max(1, len(records)),
                     "mean_output_tokens": sum(r["n_out"] for r in records) / max(1, len(records))})
        (TABLES_DIR / f"{path.stem}__confusion.json").write_text(json.dumps(matrix.tolist(), indent=2), encoding="utf-8")
        classes = per_class(matrix)
        (TABLES_DIR / f"{path.stem}__per_class.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    if rows:
        with (TABLES_DIR / f"{args.model}_{args.split}_{args.scope}_main.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    main()
