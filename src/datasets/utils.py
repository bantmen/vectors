"""Dataset utility functions: stats, dimension padding, normalization."""

import numpy as np


def dataset_stats(X: np.ndarray) -> dict:
    """Compute basic statistics for a dataset.

    Args:
        X: Data matrix of shape (n, D).

    Returns:
        Dict with keys: n, D, mean_norm, std_norm, min_norm, max_norm.
    """
    norms = np.linalg.norm(X, axis=1)
    return {
        "n": X.shape[0],
        "D": X.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
    }


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row. For angular datasets (e.g. GloVe),
    this makes Euclidean distance correspond to cosine distance.

    Args:
        X: Data matrix of shape (n, D).

    Returns:
        Normalized array of same shape.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms
