"""Math utilities: random orthogonal matrix generation."""

import numpy as np


def random_orthogonal_matrix(D: int, seed: int = 42) -> np.ndarray:
    """Generate a random orthogonal matrix of size D x D.

    Uses QR decomposition of a random Gaussian matrix, with sign
    correction to ensure uniform distribution over O(D).

    Args:
        D: Matrix dimension.
        seed: Random seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (D, D) with dtype float32.
    """
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((D, D)).astype(np.float64)
    Q, R = np.linalg.qr(H)
    # Ensure uniform distribution (Mezzadri 2007)
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1
    Q = Q * sign[np.newaxis, :]
    return Q.astype(np.float32)


def pad_to_multiple(x: np.ndarray, multiple: int) -> np.ndarray:
    """Pad vectors to nearest multiple of `multiple` by appending zeros.

    Args:
        x: Array of shape (n, D).
        multiple: Target multiple for dimension.

    Returns:
        Padded array of shape (n, D_padded).
    """
    n, D = x.shape
    remainder = D % multiple
    if remainder == 0:
        return x
    pad_width = multiple - remainder
    return np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
