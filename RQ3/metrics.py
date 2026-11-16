# classification metrics
# this module computes all metrics from a three-by-four confusion matrix.

from __future__ import annotations

import numpy as np

LABELS = ("Favor", "Against", "None")

def confusion(gold: list[str], pred: list[str]) -> np.ndarray:
    matrix = np.zeros((3, 4), dtype=int)
    columns = {label: i for i, label in enumerate((*LABELS, "UNPARSABLE"))}
    for actual, predicted in zip(gold, pred):
        if actual in LABELS:
            matrix[LABELS.index(actual), columns.get(predicted, 3)] += 1
    return matrix

def per_class(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for i, label in enumerate(LABELS):
        tp, fn = matrix[i, i], matrix[i].sum() - matrix[i, i]
        fp = matrix[:, i].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1}
    return result

def summarize(matrix: np.ndarray) -> dict[str, float]:
    values = per_class(matrix)
    return {"macro_f1": float(np.mean([v["f1"] for v in values.values()])),
            "accuracy": float(np.trace(matrix[:, :3]) / matrix.sum()) if matrix.sum() else 0.0,
            "favor_against_f1": float(np.mean([values[x]["f1"] for x in ("Favor", "Against")])),
            "unparsable_rate": float(matrix[:, 3].sum() / matrix.sum()) if matrix.sum() else 0.0}

def macro_f1(matrix: np.ndarray) -> float: return summarize(matrix)["macro_f1"]
def accuracy(matrix: np.ndarray) -> float: return summarize(matrix)["accuracy"]
def favor_against_f1(matrix: np.ndarray) -> float: return summarize(matrix)["favor_against_f1"]
def flip_rate(a: list[str], b: list[str]) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b))) if a else 0.0
def depth_bucket(depth: int) -> str:
    return "0" if depth == 0 else "1" if depth == 1 else "2" if depth == 2 else "3-4" if depth <= 4 else "5+"
