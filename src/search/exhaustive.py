"""Brute-force exact k-NN search baseline."""

import numpy as np


def exhaustive_search(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k-nearest neighbor search using brute-force squared Euclidean distance.

    Args:
        queries: Query vectors of shape (n_q, D).
        database: Database vectors of shape (n_db, D).
        k: Number of neighbors to return.

    Returns:
        indices: int32 array of shape (n_q, k) with neighbor indices.
        distances: float32 array of shape (n_q, k) with squared L2 distances.
    """
    n_q = queries.shape[0]
    n_db = database.shape[0]
    k = min(k, n_db)

    # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 * q . x
    q_norms = np.sum(queries ** 2, axis=1, keepdims=True)  # (n_q, 1)
    db_norms = np.sum(database ** 2, axis=1)  # (n_db,)

    # Process in batches to limit memory usage
    batch_size = max(1, min(1000, 2 ** 30 // (n_db * 4)))  # ~4GB limit
    all_indices = np.empty((n_q, k), dtype=np.int32)
    all_distances = np.empty((n_q, k), dtype=np.float32)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        batch_q = queries[start:end]
        batch_q_norms = q_norms[start:end]

        # (batch, n_db)
        dists = batch_q_norms + db_norms - 2.0 * (batch_q @ database.T)
        np.maximum(dists, 0, out=dists)  # clamp numerical noise

        # argpartition is faster than full sort for large n_db
        if k < n_db:
            top_k_idx = np.argpartition(dists, k, axis=1)[:, :k]
            # Sort the top-k by distance
            rows = np.arange(end - start)[:, np.newaxis]
            top_k_dists = dists[rows, top_k_idx]
            sorted_order = np.argsort(top_k_dists, axis=1)
            top_k_idx = top_k_idx[rows, sorted_order]
            top_k_dists = top_k_dists[rows, sorted_order]
        else:
            sorted_order = np.argsort(dists, axis=1)
            top_k_idx = sorted_order[:, :k]
            rows = np.arange(end - start)[:, np.newaxis]
            top_k_dists = dists[rows, top_k_idx]

        all_indices[start:end] = top_k_idx.astype(np.int32)
        all_distances[start:end] = top_k_dists.astype(np.float32)

    return all_indices, all_distances
