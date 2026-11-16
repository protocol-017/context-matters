# paired statistical tests
# this module computes thread-level bootstrap intervals and holm correction.

from __future__ import annotations

import numpy as np

def _mean_system(tensor: np.ndarray) -> np.ndarray:
    return tensor.mean(axis=0) if tensor.ndim == 4 else tensor

def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_resamples: int = 10000,
                 seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n_resamples)]
    return tuple(np.percentile(samples, [2.5, 97.5]).tolist())

def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                     seed: int = 7) -> dict[str, float]:
    a, b = _mean_system(a), _mean_system(b)
    observed = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        indexes = rng.integers(0, len(a), len(a))
        draws[i] = np.mean(a[indexes] - b[indexes])
    ci = np.percentile(draws, [2.5, 97.5])
    p = 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))
    return {"delta": observed, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(min(1.0, p))}

def per_thread_tensors(records: list[dict], labels: tuple[str, ...] = ("Favor", "Against", "None")) -> np.ndarray:
    result = {}
    for record in records:
        key = record["thread_id"]
        result.setdefault(key, np.zeros((3, 4), dtype=int))
        if record["gold"] in labels:
            result[key][labels.index(record["gold"]), labels.index(record["pred"]) if record["pred"] in labels else 3] += 1
    return np.stack(list(result.values()))

def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    return {name: min(1.0, p_values[name] * (len(ordered) - i)) for i, name in enumerate(ordered)}
