"""Product Quantization (Jegou et al., 2011).

Splits D-dim vectors into M subspaces, learns k-means codebook per subspace
(k=256 centroids), and encodes each subvector as a 1-byte centroid index.
Distance computation uses Asymmetric Distance Computation (ADC) lookup tables.
"""

import numpy as np
from scipy.cluster.vq import kmeans2

from .base import Quantizer


class ProductQuantizer(Quantizer):
    """Product Quantization for approximate nearest neighbor search.

    Args:
        M: Number of subspaces (D must be divisible by M).
        Ks: Number of centroids per subspace (default 256, max 256 for uint8 codes).
        n_iter: Number of k-means iterations.
        seed: Random seed for k-means initialization.
    """

    def __init__(self, M: int = 8, Ks: int = 256, n_iter: int = 20, seed: int = 42):
        if Ks > 256:
            raise ValueError("Ks must be <= 256 for uint8 codes")
        self.M = M
        self.Ks = Ks
        self.n_iter = n_iter
        self.seed = seed

        self.D: int = 0
        self.Ds: int = 0  # sub-dimension = D // M
        self.codebooks: np.ndarray | None = None  # (M, Ks, Ds)
        self.codes: np.ndarray | None = None  # (n, M) uint8

    def fit(self, X: np.ndarray) -> "ProductQuantizer":
        """Learn k-means codebook for each subspace.

        Args:
            X: Training vectors of shape (n, D), dtype float32.
        """
        n, D = X.shape
        if D % self.M != 0:
            raise ValueError(f"D={D} must be divisible by M={self.M}")

        self.D = D
        self.Ds = D // self.M
        self.codebooks = np.empty((self.M, self.Ks, self.Ds), dtype=np.float32)

        for m in range(self.M):
            sub = X[:, m * self.Ds : (m + 1) * self.Ds]
            centroids, _ = kmeans2(
                sub, self.Ks, iter=self.n_iter, minit="points", seed=self.seed + m
            )
            self.codebooks[m] = centroids

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Assign each subvector to its nearest centroid.

        Args:
            X: Vectors of shape (n, D).

        Returns:
            Codes of shape (n, M), dtype uint8.
        """
        n = X.shape[0]
        codes = np.empty((n, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub = X[:, m * self.Ds : (m + 1) * self.Ds]  # (n, Ds)
            centroids = self.codebooks[m]  # (Ks, Ds)
            # Squared distances: ||sub - c||^2 = ||sub||^2 + ||c||^2 - 2*sub.c
            sub_norms = np.sum(sub ** 2, axis=1, keepdims=True)  # (n, 1)
            c_norms = np.sum(centroids ** 2, axis=1)  # (Ks,)
            dists = sub_norms + c_norms - 2.0 * (sub @ centroids.T)  # (n, Ks)
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from centroid codes.

        Args:
            codes: Array of shape (n, M), dtype uint8.

        Returns:
            Reconstructed vectors of shape (n, D), dtype float32.
        """
        n = codes.shape[0]
        X_hat = np.empty((n, self.D), dtype=np.float32)

        for m in range(self.M):
            X_hat[:, m * self.Ds : (m + 1) * self.Ds] = self.codebooks[m][codes[:, m]]

        return X_hat

    def build_database(self, X: np.ndarray) -> None:
        """Encode and store database vectors for search.

        Args:
            X: Database vectors of shape (n, D).
        """
        self.codes = self.encode(X)

    def _build_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Pre-compute ADC lookup table for a single query.

        For each subspace m and centroid k, compute ||q_m - c_{m,k}||^2.

        Args:
            query: Single query vector of shape (D,).

        Returns:
            Distance table of shape (M, Ks).
        """
        table = np.empty((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            q_sub = query[m * self.Ds : (m + 1) * self.Ds]  # (Ds,)
            centroids = self.codebooks[m]  # (Ks, Ds)
            # ||q_sub - c||^2
            table[m] = np.sum((centroids - q_sub) ** 2, axis=1)
        return table

    def estimate_distances(self, queries: np.ndarray) -> np.ndarray:
        """Estimate squared L2 distances from queries to all database vectors.

        Uses ADC: distance = sum over m of table[m, code[m]]

        Args:
            queries: Query vectors of shape (n_q, D).

        Returns:
            Distance matrix of shape (n_q, n_db), dtype float32.
        """
        n_q = queries.shape[0]
        n_db = self.codes.shape[0]
        distances = np.zeros((n_q, n_db), dtype=np.float32)

        for i in range(n_q):
            table = self._build_distance_table(queries[i])
            # Vectorized table lookup: for each database vector, sum M lookups
            for m in range(self.M):
                distances[i] += table[m, self.codes[:, m]]

        return distances

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN search using ADC.

        Args:
            queries: Query vectors of shape (n_q, D).
            k: Number of neighbors.

        Returns:
            indices: int32 array of shape (n_q, k).
            distances: float32 array of shape (n_q, k).
        """
        if self.codes is None:
            raise RuntimeError("Call build_database() before search()")

        n_q = queries.shape[0]
        n_db = self.codes.shape[0]
        k = min(k, n_db)

        all_indices = np.empty((n_q, k), dtype=np.int32)
        all_distances = np.empty((n_q, k), dtype=np.float32)

        for i in range(n_q):
            table = self._build_distance_table(queries[i])
            dists = np.zeros(n_db, dtype=np.float32)
            for m in range(self.M):
                dists += table[m, self.codes[:, m]]

            if k < n_db:
                top_k = np.argpartition(dists, k)[:k]
                sorted_order = np.argsort(dists[top_k])
                top_k = top_k[sorted_order]
            else:
                top_k = np.argsort(dists)[:k]

            all_indices[i] = top_k.astype(np.int32)
            all_distances[i] = dists[top_k].astype(np.float32)

        return all_indices, all_distances

    def memory_usage(self) -> int:
        """Memory for stored codes + codebooks."""
        if self.codes is None:
            return 0
        code_bytes = self.codes.nbytes  # n * M bytes
        codebook_bytes = self.codebooks.nbytes  # M * Ks * Ds * 4 bytes
        return code_bytes + codebook_bytes
