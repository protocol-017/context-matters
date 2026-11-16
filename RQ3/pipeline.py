# example construction, inference records, and jsonl persistence
# this module keeps target text fixed while applying perturbation and selection conditions.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from snmpaper.RQ3.context_selection import SelectorConfig, TokenCounter, select
from snmpaper.RQ3.data import Comment, Split
from snmpaper.RQ3.perturbations import perturb_chain
from snmpaper.RQ3.prompts import build_prompt, parse_label

@dataclass
class Example:
    comment: Comment
    prompt: str
    context_ids: list[str]
    n_ancestors: int
    n_chain: int
    n_selected: int
    n_noise_selected: int

class Pipeline:
    def __init__(self, split_name: str, split: Split, emb: dict, counter: TokenCounter) -> None:
        self.split_name, self.split, self.emb, self.counter = split_name, split, emb, counter

    def base_chain(self, cid: str) -> list[Comment]:
        return self.split.ancestors(cid)

    def build_examples(self, ids: list[str], method: str, kind: str, seed: int,
                       selcfg: SelectorConfig) -> list[Example]:
        pool = list(self.split.comments.values())
        noise_ctx: dict[str, list[Comment]] = {}
        examples = []
        for cid in ids:
            comment = self.split.comments[cid]
            chain = perturb_chain(self.base_chain(cid), comment, kind, seed, pool, noise_ctx)
            chosen = select(method, chain, cid, self.emb, selcfg, self.counter)
            examples.append(Example(comment, build_prompt(comment.target, comment.text, [self.split.comments[x.id] if x.id in self.split.comments else x for x in chosen]),
                                    [x.id for x in chosen], len(self.base_chain(cid)), len(chain), len(chosen),
                                    sum(x.is_noise for x in chosen)))
        return examples

    def run_condition(self, runner, examples: list[Example], meta: dict) -> list[dict]:
        outputs = runner.generate([example.prompt for example in examples])
        records = []
        for example, output in zip(examples, outputs):
            c = example.comment
            records.append({**meta, "id": c.id, "thread_id": c.thread_id, "depth": c.depth, "target": c.target,
                            "gold": c.label, "pred": parse_label(output["raw"]), "raw": output["raw"],
                            "context_ids": example.context_ids, "n_ancestors": example.n_ancestors,
                            "n_chain": example.n_chain, "n_selected": example.n_selected,
                            "n_noise_selected": example.n_noise_selected, "n_in": output["n_in"],
                            "n_out": output["n_out"], "seconds": output["seconds"]})
        return records

def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
