"""Re-ranked search wrapper for any Quantizer.

Retrieves a shortlist of candidates using approximate distances from the base
quantizer, then re-scores with exact squared L2 distances and returns the
top-k from that refined list.  Standard practice for improving recall when
approximate distance estimates are noisy.
"""

import numpy as np

from .base import Quantizer


class RerankedQuantizer(Quantizer):
    """Wrapper that adds exact-distance re-ranking on top of a base quantizer.

    Args:
        base: Any Quantizer instance (e.g. ProductQuantizer or RaBitQ).
        n_shortlist: Number of candidates to retrieve with approximate
            distances before re-ranking with exact L2.  Must be >= k
            at search time.
    """

    def __init__(self, base: Quantizer, n_shortlist: int = 100):
        self.base = base
        self.n_shortlist = n_shortlist
        self._database: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RerankedQuantizer":
        self.base.fit(X)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.base.encode(X)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.base.decode(codes)

    def build_database(self, X: np.ndarray) -> None:
        self.base.build_database(X)
        self._database = np.array(X, dtype=np.float32, copy=True)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._database is None:
            raise RuntimeError("Call build_database() before search()")

        n_q = queries.shape[0]
        n_db = self._database.shape[0]
        shortlist = min(self.n_shortlist, n_db)
        k = min(k, n_db)

        # Stage 1: approximate distances from base quantizer
        approx_dists = self.base.estimate_distances(queries)  # (n_q, n_db)

        all_indices = np.empty((n_q, k), dtype=np.int32)
        all_distances = np.empty((n_q, k), dtype=np.float32)

        for i in range(n_q):
            # Pick shortlist candidates
            if shortlist < n_db:
                cand_idx = np.argpartition(approx_dists[i], shortlist)[:shortlist]
            else:
                cand_idx = np.arange(n_db)

            # Stage 2: exact squared L2 on candidates
            diff = self._database[cand_idx] - queries[i]  # (shortlist, D)
            exact_dists = np.sum(diff * diff, axis=1)  # (shortlist,)

            # Top-k from exact distances
            if k < shortlist:
                top_k_local = np.argpartition(exact_dists, k)[:k]
                sorted_order = np.argsort(exact_dists[top_k_local])
                top_k_local = top_k_local[sorted_order]
            else:
                top_k_local = np.argsort(exact_dists)[:k]

            all_indices[i] = cand_idx[top_k_local].astype(np.int32)
            all_distances[i] = exact_dists[top_k_local].astype(np.float32)

        return all_indices, all_distances

    def memory_usage(self) -> int:
        base_mem = self.base.memory_usage()
        raw_mem = 0
        if self._database is not None:
            raw_mem = self._database.nbytes  # n * D * 4
        return base_mem + raw_mem
