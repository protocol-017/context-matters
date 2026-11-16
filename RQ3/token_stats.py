# validation context token statistics
# this module measures capped ancestor-chain lengths with the fixed budget tokenizer.

from __future__ import annotations

import numpy as np
from transformers import AutoTokenizer

from snmpaper.RQ3.config import BUDGET_TOKENIZER, MAX_MSG_TOKENS, ROLE_OVERHEAD_TOKENS
from snmpaper.RQ3.data import load_split

def main() -> None:
    split = load_split("validation")
    tokenizer = AutoTokenizer.from_pretrained(BUDGET_TOKENIZER, use_fast=True)
    lengths = []
    for comment in split.comments.values():
        if comment.depth < 1:
            continue
        total = 0
        for ancestor in split.ancestors(comment.id):
            token_count = len(tokenizer.encode(ancestor.text, add_special_tokens=False))
            total += min(token_count, MAX_MSG_TOKENS) + ROLE_OVERHEAD_TOKENS
        lengths.append(total)
    values = np.asarray(lengths, dtype=float)
    print(f"count: {len(values)}")
    print(f"mean: {values.mean():.2f}")
    print(f"median: {np.median(values):.2f}")
    print(f"p95: {np.percentile(values, 95):.2f}")
    print(f"max: {values.max():.2f}")
    print(f"share_over_512: {np.mean(values > 512):.4f}")
    print(f"share_over_768: {np.mean(values > 768):.4f}")

if __name__ == "__main__":
    main()
