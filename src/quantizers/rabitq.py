"""RaBitQ (Gao & Long, SIGMOD 2024).

Randomized Binary Quantization for approximate nearest neighbor search.
Applies a random orthogonal transform, binarizes to D-bit codes, and stores
two scalar corrections per vector. Provides an unbiased distance estimator
with O(1/sqrt(D)) error bound.

Encoding pipeline:
  1. Center vectors: x' = x - centroid
  2. Store ||x'|| per vector
  3. Normalize: x_bar = x' / ||x'||
  4. Project: x_proj = P^T x_bar  (random orthogonal P)
  5. Binarize: b = (x_proj > 0)  -> {0, 1}^D
  6. Compute o_bar = (2b - 1) / sqrt(D)  (the "binary unit vector")
  7. Store correction scalar: <o_bar, x_bar_proj> = <o_bar, x_proj>
     where x_proj is the projected normalized vector

Distance estimation for query q:
  1. q' = q - centroid
  2. q_proj = P^T q'
  3. For each database vector i:
     <x'_i, q'> ≈ ||x'_i|| * <o_bar_i, q_proj> / <o_bar_i, x_bar_proj_i>
  4. ||x_i - q||^2 = ||x'_i||^2 + ||q'||^2 - 2 * <x'_i, q'>
"""

import numpy as np

from .base import Quantizer
from ..utils.binary import pack_bits, binary_dot_product
from ..utils.math import random_orthogonal_matrix, pad_to_multiple


class RaBitQ(Quantizer):
    """RaBitQ: Randomized Binary Quantization.

    Args:
        seed: Random seed for the orthogonal matrix.
    """

    def __init__(self, seed: int = 42, use_correction: bool = True):
        self.seed = seed
        self.use_correction = use_correction

        self.D: int = 0  # original dimension
        self.D_padded: int = 0  # padded to multiple of 64
        self.centroid: np.ndarray | None = None  # (D,)
        self.P: np.ndarray | None = None  # (D_padded, D_padded) orthogonal matrix

        # Per-vector stored data
        self.norms: np.ndarray | None = None  # (n,) ||x - centroid||
        self.codes_packed: np.ndarray | None = None  # (n, D_padded // 64) uint64
        self.obar_dot_xbar: np.ndarray | None = None  # (n,) correction scalars

    def fit(self, X: np.ndarray) -> "RaBitQ":
        """Compute centroid, generate random orthogonal matrix.

        Args:
            X: Training vectors of shape (n, D), dtype float32.
        """
        _, D = X.shape
        self.D = D
        self.D_padded = int(np.ceil(D / 64) * 64)
        self.centroid = np.mean(X, axis=0).astype(np.float32)
        self.P = random_orthogonal_matrix(self.D_padded, seed=self.seed)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode vectors to binary codes with scalar corrections.

        Args:
            X: Vectors of shape (n, D), dtype float32.

        Returns:
            Packed binary codes of shape (n, D_padded // 64), dtype uint64.
        """
        n = X.shape[0]

        # 1. Center
        centered = X - self.centroid  # (n, D)

        # 2. Pad to D_padded
        centered_padded = pad_to_multiple(centered, 64)  # (n, D_padded)

        # 3. Compute and store norms
        self.norms = np.linalg.norm(centered_padded, axis=1).astype(np.float32)

        # 4. Normalize (avoid division by zero)
        safe_norms = np.maximum(self.norms, 1e-10)
        x_bar = centered_padded / safe_norms[:, np.newaxis]

        # 5. Project: x_proj = P^T @ x_bar^T -> columns are projected vectors
        x_proj = (x_bar @ self.P).astype(np.float32)  # (n, D_padded) since P is orthogonal

        # 6. Binarize
        bits = (x_proj > 0).astype(np.uint8)  # (n, D_padded)

        # 7. Compute correction: <o_bar, x_proj> where o_bar = (2*bits - 1) / sqrt(D_padded)
        o_bar = (2.0 * bits.astype(np.float32) - 1.0) / np.sqrt(self.D_padded)
        self.obar_dot_xbar = np.sum(o_bar * x_proj, axis=1).astype(np.float32)

        # Clamp correction away from zero for numerical stability
        self.obar_dot_xbar = np.where(
            np.abs(self.obar_dot_xbar) < np.float32(0.1),
            np.sign(self.obar_dot_xbar) * np.float32(0.1) + (self.obar_dot_xbar == 0) * np.float32(0.1),
            self.obar_dot_xbar,
        ).astype(np.float32)

        # 8. Pack bits
        self.codes_packed = pack_bits(bits)

        return self.codes_packed

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Approximate reconstruction from binary codes.

        This is lossy: we reconstruct the binary approximation scaled by norms.
        Not typically used for RaBitQ (distance estimation is done directly),
        but provided for interface compatibility.

        Args:
            codes: Packed codes of shape (n, D_padded // 64), dtype uint64.

        Returns:
            Approximate reconstructed vectors of shape (n, D), dtype float32.
        """
        from ..utils.binary import unpack_bits
        bits = unpack_bits(codes, self.D_padded)
        o_bar = (2.0 * bits.astype(np.float32) - 1.0) / np.sqrt(self.D_padded)
        # Invert projection: x_bar ≈ P @ o_bar^T
        x_bar_approx = (o_bar @ self.P.T).astype(np.float32)
        # Scale by norms and add centroid
        x_approx = x_bar_approx * self.norms[:, np.newaxis]
        return (x_approx[:, :self.D] + self.centroid).astype(np.float32)

    def build_database(self, X: np.ndarray) -> None:
        """Encode and store database vectors for search.

        Args:
            X: Database vectors of shape (n, D).
        """
        self.encode(X)

    def estimate_distances(self, queries: np.ndarray) -> np.ndarray:
        """Estimate squared Euclidean distances from queries to database.

        Args:
            queries: Query vectors of shape (n_q, D), dtype float32.

        Returns:
            Distance matrix of shape (n_q, n_db), dtype float32.
        """
        n_q = queries.shape[0]
        n_db = self.codes_packed.shape[0]

        # Center and pad queries
        q_centered = queries - self.centroid
        q_padded = pad_to_multiple(q_centered, 64)
        q_norms_sq = np.sum(q_padded ** 2, axis=1)  # (n_q,)

        # Project queries
        q_proj = (q_padded @ self.P).astype(np.float32)  # (n_q, D_padded)

        # Database norms squared
        db_norms_sq = self.norms ** 2  # (n_db,)

        # Binary dot product: popcount(q_bits AND db_bits)
        # <o_bar, q_proj> = (2 * popcount(q_bits AND db_bits) - popcount(q_bits) - popcount(db_bits) + D) / sqrt(D) ... no
        # Actually: o_bar_i = (2*b_i - 1)/sqrt(D), so <o_bar_i, q_proj> needs to be computed differently
        #
        # <o_bar, q_proj> = sum_j (2*b_j - 1) * q_proj_j / sqrt(D)
        #                 = (2 * sum_j b_j * q_proj_j - sum_j q_proj_j) / sqrt(D)
        #
        # sum_j b_j * q_proj_j = dot product of bits with q_proj values
        # For efficiency, we compute this using the sign of q_proj and binary operations

        # We need <o_bar_i, q_proj> for each db vector i and each query
        # This requires the actual float values of q_proj, not just bits

        distances = np.empty((n_q, n_db), dtype=np.float32)

        for qi in range(n_q):
            qp = q_proj[qi]  # (D_padded,)

            # For each database vector, compute <o_bar, q_proj>
            # o_bar = (2*bits - 1) / sqrt(D_padded)
            # Use packed bits for efficiency:
            # <o_bar, q_proj> = (2 * sum(bits * qp) - sum(qp)) / sqrt(D_padded)
            #
            # Split qp into positive and negative parts for binary computation:
            # sum(bits * qp) = sum(bits * qp_pos) - sum(bits * |qp_neg|)
            #   where qp_pos = max(qp, 0), qp_neg = min(qp, 0)
            #
            # But we need to use the packed binary codes efficiently.
            # The key insight: we can separate q_proj by the bits of the database vectors.
            #
            # Actually, for moderate n_db, unpacking is simpler and still fast.

            from ..utils.binary import unpack_bits
            bits = unpack_bits(self.codes_packed, self.D_padded)  # (n_db, D_padded)

            # <o_bar_i, q_proj> = (2 * bits_i . qp - sum(qp)) / sqrt(D_padded)
            sum_qp = np.sum(qp)
            bits_dot_qp = bits.astype(np.float32) @ qp  # (n_db,)
            obar_dot_q = (2.0 * bits_dot_qp - sum_qp) / np.sqrt(self.D_padded)

            # Estimated inner product: <x'_i, q'> ≈ ||x'_i|| * <o_bar_i, q_proj> / <o_bar_i, x_bar_proj_i>
            if self.use_correction:
                ip_est = self.norms * obar_dot_q / self.obar_dot_xbar
            else:
                ip_est = self.norms * obar_dot_q

            # ||x_i - q||^2 = ||x'_i||^2 + ||q'||^2 - 2 * <x'_i, q'>
            distances[qi] = db_norms_sq + q_norms_sq[qi] - 2.0 * ip_est

        # Clamp negative distances from numerical noise
        np.maximum(distances, 0, out=distances)
        return distances

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN search using RaBitQ distance estimation.

        Args:
            queries: Query vectors of shape (n_q, D).
            k: Number of neighbors.

        Returns:
            indices: int32 array of shape (n_q, k).
            distances: float32 array of shape (n_q, k).
        """
        if self.codes_packed is None:
            raise RuntimeError("Call build_database() before search()")

        distances = self.estimate_distances(queries)
        n_q = queries.shape[0]
        n_db = distances.shape[1]
        k = min(k, n_db)

        all_indices = np.empty((n_q, k), dtype=np.int32)
        all_distances = np.empty((n_q, k), dtype=np.float32)

        for i in range(n_q):
            if k < n_db:
                top_k = np.argpartition(distances[i], k)[:k]
                sorted_order = np.argsort(distances[i, top_k])
                top_k = top_k[sorted_order]
            else:
                top_k = np.argsort(distances[i])[:k]
            all_indices[i] = top_k.astype(np.int32)
            all_distances[i] = distances[i, top_k]

        return all_indices, all_distances

    def memory_usage(self) -> int:
        """Memory for codes + norms + corrections."""
        if self.codes_packed is None:
            return 0
        code_bytes = self.codes_packed.nbytes  # n * D_padded/64 * 8
        norm_bytes = self.norms.nbytes  # n * 4
        corr_bytes = self.obar_dot_xbar.nbytes  # n * 4
        matrix_bytes = self.P.nbytes  # D_padded^2 * 4
        return code_bytes + norm_bytes + corr_bytes + matrix_bytes
