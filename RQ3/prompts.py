# prompt construction and model output parsing
# this module owns the shared prompt, text cleanup, context formatting, and label parser.

from __future__ import annotations

import re
import string
from typing import Iterable

from snmpaper.exp3.data import Comment

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def format_context(chain: Iterable[Comment]) -> str:
    items = list(chain)
    if not items:
        return ""
    lines = ["\nContext (oldest to newest):"]
    for index, comment in enumerate(items, 1):
        role = "[ROOT]" if index == 1 else "[PARENT]" if index == len(items) else "[REPLY]"
        lines.append(f"[{index}] {role:<8} {clean_text(comment.text)}")
    return "\n".join(lines)

def build_prompt(target: str, comment: str, chain: Iterable[Comment]) -> str:
    context = format_context(chain)
    return (f"You classify the stance of one Reddit comment toward a target.\nTarget: {clean_text(target)}\n\n"
            "Labels:\nFavor   - the author supports the target.\nAgainst - the author opposes the target.\n"
            "None    - the author takes no clear supportive or opposing position.\n\n"
            "Judge the author's stance toward the TARGET, not agreement or\n"
            "disagreement with the previous message. Earlier messages are\n"
            f"context only.{context}\n\nComment to classify:\n{clean_text(comment)}\n\n"
            "Answer with exactly one word: Favor, Against, or None.")

def parse_label(raw: str) -> str:
    line = next((line.strip() for line in str(raw).splitlines() if line.strip()), "")
    line = line.strip(string.whitespace + string.punctuation).lower()
    return {"favor": "Favor", "against": "Against", "none": "None"}.get(line, "UNPARSABLE")
