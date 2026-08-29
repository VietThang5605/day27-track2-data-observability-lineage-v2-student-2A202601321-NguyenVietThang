from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import mad_detector, zscore_detector


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    # Deliberately simple proxy; no tokenizer/model download needed.
    return [len(str(t).split()) for t in texts]


def detect_text_length_shift(
    current_texts: Iterable[str],
    baseline_batch_means: Iterable[float],
    *,
    threshold: float = 3.0,
) -> dict[str, Any]:
    lengths = approximate_token_lengths(current_texts)
    current_mean = float(np.mean(lengths)) if lengths else 0.0
    result = zscore_detector(current_mean, baseline_batch_means, threshold=threshold)
    result["metric"] = "mean_text_length"
    result["current_mean"] = current_mean
    return result


def detect_embedding_norm_shift(
    current_norms: Iterable[float], baseline_norms: Iterable[float]
) -> dict[str, Any]:
    """Embedding-space drift signal using the norm of embeddings.

    No embedding model is required: hidden evaluation feeds precomputed
    norm vectors. The mean norm of the current batch is compared against the
    baseline norms with the robust MAD detector; mean-norm collapse or growth
    indicates embedding/source drift in the RAG index.
    """
    cur = np.asarray(list(current_norms), dtype=float)
    base = np.asarray(list(baseline_norms), dtype=float)
    if cur.size == 0 or base.size < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "mad",
            "metric": "embedding_norm",
            "reason": "insufficient_input",
        }
    current_mean = float(np.mean(cur))
    result = mad_detector(current_mean, base, threshold=3.5)
    result["metric"] = "embedding_norm"
    result["current_mean"] = current_mean
    result["baseline_mean"] = float(np.mean(base))
    return result
