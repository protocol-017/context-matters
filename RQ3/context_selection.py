# context selection methods and token budgeting
# this module provides one tokenizer, ancestor truncation, scoring, and ablations.

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from snmpaper.RQ3.config import K_DEFAULT, MAX_MSG_TOKENS, MODELS, RECENCY_DECAY, ROLE_OVERHEAD_TOKENS, TOKEN_BUDGET
from snmpaper.RQ3.data import Comment

@dataclass(frozen=True)
class CtxMsg:
    id: str
    text: str
    cost: int
    is_root: bool
    is_parent: bool
    is_noise: bool = False

@dataclass(frozen=True)
class SelectorConfig:
    k: int = K_DEFAULT
    budget: int = TOKEN_BUDGET
    alpha: float = 1 / 3
    beta: float = 1 / 3
    gamma: float = 1 / 3
    decay: float = RECENCY_DECAY

class TokenCounter:
    def __init__(self) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS["qwen"], use_fast=True)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, limit: int = MAX_MSG_TOKENS) -> tuple[str, int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= limit:
            return text, len(tokens) + ROLE_OVERHEAD_TOKENS
        kept = max(0, limit - 1)
        return self.tokenizer.decode(tokens[:kept], skip_special_tokens=True) + "…", limit + ROLE_OVERHEAD_TOKENS

def _messages(chain: list[Comment], target_id: str, counter: TokenCounter) -> list[CtxMsg]:
    parent = chain[-1].id if chain else None
    result = []
    for comment in chain:
        text, cost = counter.truncate(comment.text)
        result.append(CtxMsg(comment.id, text, cost, comment.is_root, comment.id == parent, comment.is_noise))
    return result

def _normalise(values: list[float]) -> list[float]:
    low, high = min(values, default=0), max(values, default=0)
    return [0.0] * len(values) if high == low else [(v - low) / (high - low) for v in values]

def select(method: str, chain: list[Comment], target_id: str, emb: dict[str, np.ndarray],
           cfg: SelectorConfig, counter: TokenCounter) -> list[CtxMsg]:
    messages = _messages(chain, target_id, counter)
    if method == "none":
        return []
    if method == "parent":
        return messages[-1:] if messages and messages[-1].cost <= cfg.budget else []
    if method in {"full", "nearest_k"}:
        chosen: list[CtxMsg] = []
        for message in reversed(messages):
            if method == "nearest_k" and len(chosen) >= cfg.k:
                break
            if sum(item.cost for item in chosen) + message.cost > cfg.budget:
                break
            chosen.append(message)
        return list(reversed(chosen))
    if method not in {"semantic_only", "combined", "combined_no_semantic", "combined_no_structural", "combined_no_recency"}:
        raise ValueError(f"unknown selection method: {method}")
    target = emb[target_id]
    semantic = _normalise([float(np.dot(target, emb[m.id])) for m in messages])
    structural = [1.0 if m.is_root or m.is_parent else 0.0 for m in messages]
    recency = _normalise([math.exp(-cfg.decay * (len(messages) - i - 1)) for i in range(len(messages))])
    if method == "semantic_only":
        weights = (1.0, 0.0, 0.0)
    else:
        weights = (cfg.alpha, cfg.beta, cfg.gamma)
        if method == "combined_no_semantic":
            weights = (0.0, cfg.beta, cfg.gamma)
        elif method == "combined_no_structural":
            weights = (cfg.alpha, 0.0, cfg.gamma)
        elif method == "combined_no_recency":
            weights = (cfg.alpha, cfg.beta, 0.0)
        total = sum(weights)
        weights = tuple(w / total for w in weights) if total else (1.0, 0.0, 0.0)
    ranked = sorted(range(len(messages)),
                    key=lambda i: (-(weights[0] * semantic[i] + weights[1] * structural[i] + weights[2] * recency[i]),
                                   -(len(messages) - i)))
    chosen = []
    for index in ranked:
        if len(chosen) >= cfg.k:
            break
        if sum(item.cost for item in chosen) + messages[index].cost <= cfg.budget:
            chosen.append(messages[index])
    return sorted(chosen, key=lambda message: next(i for i, item in enumerate(messages) if item.id == message.id))
