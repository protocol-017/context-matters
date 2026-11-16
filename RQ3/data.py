# dataset loading and reply-tree utilities
# this module normalises mt-csd rows and exposes ancestry and fixed subset helpers.

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from snmpaper.RQ3.config import COLUMNS, DATA_DIR, EXPECTED_THREADS, FROZEN_DIR, LABEL_MAP, SPLIT_FILES, SUBSET_MIN_DEPTH, SUBSET_SEED, SUBSET_SIZE

@dataclass(frozen=True)
class Comment:
    id: str
    text: str
    target: str
    label: str
    parent_id: str | None
    thread_id: str
    depth: int
    parent_missing: bool = False
    is_noise: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent_id is None and not self.parent_missing

@dataclass
class Split:
    comments: dict[str, Comment]
    threads: dict[str, list[str]]
    by_thread: dict[str, dict[str, Comment]]

    def ancestors(self, cid: str) -> list[Comment]:
        chain: list[Comment] = []
        current = self.comments[cid].parent_id
        seen: set[str] = set()
        while current and current not in seen and current in self.comments:
            seen.add(current)
            chain.append(self.comments[current])
            current = self.comments[current].parent_id
        return list(reversed(chain))

def _parse_parent(value: object) -> str | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value)

def load_split(name: str) -> Split:
    if name not in SPLIT_FILES:
        raise ValueError(f"unknown split: {name}")
    frame = pd.read_csv(DATA_DIR / SPLIT_FILES[name], dtype=str).fillna("")
    missing = set(COLUMNS.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    comments: dict[str, Comment] = {}
    depth_differences = 0
    missing_parent_rows = 0
    for row in frame.to_dict("records"):
        label = LABEL_MAP.get(str(row[COLUMNS["label"]]).strip(), str(row[COLUMNS["label"]]).strip())
        if label not in {"Favor", "Against", "None"}:
            raise ValueError(f"unmapped label: {label}")
        cid = str(row[COLUMNS["id"]])
        parent_id = _parse_parent(row[COLUMNS["parent_id"]])
        parent_missing = str(row.get("parent_missing", "")).strip().lower() == "true"
        comments[cid] = Comment(cid, str(row[COLUMNS["text"]]), str(row[COLUMNS["target"]]), label,
                                parent_id, str(row[COLUMNS["thread_id"]]), int(row[COLUMNS["depth"]]),
                                parent_missing)
        missing_parent_rows += int(parent_missing)
    threads: dict[str, list[str]] = {}
    for comment in comments.values():
        threads.setdefault(comment.thread_id, []).append(comment.id)
    actual_depths: dict[str, int] = {}
    for comment in comments.values():
        current = comment.parent_id
        seen: set[str] = set()
        actual_depth = 0
        while current and current not in seen and current in comments:
            seen.add(current)
            actual_depth += 1
            current = comments[current].parent_id
        depth_differences += int(actual_depth != comment.depth)
        actual_depths[comment.id] = actual_depth
    comments = {cid: replace(comment, depth=actual_depths[cid]) for cid, comment in comments.items()}
    print(f"{name}: {depth_differences} depth differences; {missing_parent_rows} parent_missing rows")
    for ids in threads.values():
        ids.sort(key=lambda cid: (comments[cid].depth, cid))
    if len(threads) != EXPECTED_THREADS[name]:
        raise ValueError(f"{name} has {len(threads)} threads, expected {EXPECTED_THREADS[name]}")
    return Split(comments, threads, {tid: {cid: comments[cid] for cid in ids} for tid, ids in threads.items()})

def build_subset(split: Split, path: Path | None = None) -> list[str]:
    path = path or FROZEN_DIR / "exp3_subset_ids.json"
    candidates = [c for c in split.comments.values() if c.depth >= SUBSET_MIN_DEPTH]
    rng = random.Random(SUBSET_SEED)
    groups: dict[tuple[str, str], list[Comment]] = {}
    for c in candidates:
        groups.setdefault((c.target, c.label), []).append(c)
    selected: list[Comment] = []
    for group in groups.values():
        rng.shuffle(group)
    while len(selected) < min(SUBSET_SIZE, len(candidates)):
        available = [group for group in groups.values() if group]
        if not available:
            break
        for group in available:
            selected.append(group.pop())
            if len(selected) >= min(SUBSET_SIZE, len(candidates)):
                break
    ids = sorted(c.id for c in selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    return ids

def load_or_build_subset(split: Split) -> list[str]:
    path = FROZEN_DIR / "exp3_subset_ids.json"
    if path.exists():
        ids = json.loads(path.read_text(encoding="utf-8"))
        if all(cid in split.comments for cid in ids):
            return ids
    return build_subset(split, path)

def sample_ids(split: Split, n: int, seed: int = 7) -> list[str]:
    ids = [c.id for c in split.comments.values() if c.depth >= SUBSET_MIN_DEPTH]
    random.Random(seed).shuffle(ids)
    return sorted(ids[:n])
