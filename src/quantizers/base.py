"""Abstract base class for vector quantizers."""

from abc import ABC, abstractmethod

import numpy as np


class Quantizer(ABC):
    """Abstract interface for vector quantization methods.

    All quantizers support:
      - fit(): Learn codebook/parameters from training data
      - encode(): Compress vectors to compact codes
      - decode(): Reconstruct approximate vectors from codes
      - search(): Approximate k-NN search using compressed representations
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Quantizer":
        """Learn quantization parameters from training data.

        Args:
            X: Training vectors of shape (n, D), dtype float32.

        Returns:
            self
        """
        ...

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode vectors to compact codes.

        Args:
            X: Vectors of shape (n, D), dtype float32.

        Returns:
            Codes array (shape and dtype depend on method).
        """
        ...

    @abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to approximate vectors.

        Args:
            codes: Compact codes from encode().

        Returns:
            Reconstructed vectors of shape (n, D), dtype float32.
        """
        ...

    @abstractmethod
    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k-NN search.

        Args:
            queries: Query vectors of shape (n_q, D), dtype float32.
            k: Number of neighbors to return.

        Returns:
            indices: int32 array of shape (n_q, k).
            distances: float32 array of shape (n_q, k) with estimated squared L2 distances.
        """
        ...

    @abstractmethod
    def memory_usage(self) -> int:
        """Return approximate memory usage in bytes for the encoded database."""
        ...
