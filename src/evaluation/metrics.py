"""Evaluation metrics: recall@k, QPS, memory usage."""

import numpy as np
import psutil


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int | None = None) -> float:
    """Compute mean recall@k.

    Args:
        predicted: Predicted neighbor indices, shape (n_q, k_pred).
        ground_truth: True neighbor indices, shape (n_q, k_true).
        k: Number of neighbors to consider. If None, uses min(k_pred, k_true).

    Returns:
        Mean recall@k across all queries.
    """
    n_q = predicted.shape[0]
    if k is None:
        k = min(predicted.shape[1], ground_truth.shape[1])

    pred = predicted[:, :k]
    gt = ground_truth[:, :k]

    recalls = np.zeros(n_q)
    for i in range(n_q):
        pred_set = set(pred[i].tolist())
        gt_set = set(gt[i].tolist())
        recalls[i] = len(pred_set & gt_set) / len(gt_set)

    return float(np.mean(recalls))


def queries_per_second(n_queries: int, elapsed_seconds: float) -> float:
    """Compute queries per second.

    Args:
        n_queries: Number of queries processed.
        elapsed_seconds: Wall-clock time in seconds.

    Returns:
        QPS value.
    """
    if elapsed_seconds <= 0:
        return float("inf")
    return n_queries / elapsed_seconds


def memory_usage_bytes() -> int:
    """Return current process RSS memory in bytes."""
    return psutil.Process().memory_info().rss
