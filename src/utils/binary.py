"""Bit packing/unpacking utilities for binary codes."""

import numpy as np


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack a boolean/0-1 array into uint64 words.

    Args:
        bits: Array of shape (n, D) with values in {0, 1}.
              D must be a multiple of 64.

    Returns:
        Packed array of shape (n, D // 64) with dtype uint64.
    """
    n, D = bits.shape
    if D % 64 != 0:
        raise ValueError(f"Dimension {D} must be a multiple of 64")
    bits = bits.astype(np.uint64)
    packed = bits.reshape(n, D // 64, 64)
    # Bit 0 is most significant within each 64-bit word
    powers = np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)
    return (packed * powers).sum(axis=2)


def unpack_bits(packed: np.ndarray, D: int) -> np.ndarray:
    """Unpack uint64 words back to a boolean array.

    Args:
        packed: Array of shape (n, D // 64) with dtype uint64.
        D: Original dimension (must be a multiple of 64).

    Returns:
        Array of shape (n, D) with values in {0, 1}, dtype uint8.
    """
    n = packed.shape[0]
    if D % 64 != 0:
        raise ValueError(f"Dimension {D} must be a multiple of 64")
    powers = np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)
    # Broadcast: (n, D//64, 1) & (64,) -> (n, D//64, 64)
    unpacked = ((packed[:, :, np.newaxis] & powers) > 0).astype(np.uint8)
    return unpacked.reshape(n, D)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Hamming distance between packed bit arrays using popcount.

    Args:
        a: Packed array of shape (n, W) with dtype uint64.
        b: Packed array of shape (m, W) with dtype uint64.

    Returns:
        Distance matrix of shape (n, m).
    """
    # XOR and popcount via lookup table
    # For each pair, XOR the uint64 words and count set bits
    n, W = a.shape
    m = b.shape[0]

    # Build popcount lookup for uint16 chunks
    lut = np.zeros(65536, dtype=np.int32)
    for i in range(65536):
        lut[i] = bin(i).count("1")

    distances = np.zeros((n, m), dtype=np.int32)
    for w in range(W):
        xor = np.bitwise_xor(a[:, w:w+1], b[:, w:w+1].T)  # (n, m) uint64
        # Split each uint64 into 4 uint16 chunks and sum popcount
        for shift in range(0, 64, 16):
            chunk = ((xor >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.int64)
            distances += lut[chunk]
    return distances


def binary_dot_product(a: np.ndarray, b: np.ndarray, D: int) -> np.ndarray:
    """Compute dot product between packed binary vectors.

    For binary vectors x, y in {0,1}^D:
        <x, y> = popcount(x AND y)

    Also useful: <2x-1, 2y-1> = 4*popcount(x AND y) - 2*popcount(x) - 2*popcount(y) + D
    which maps {0,1} -> {-1,+1}.

    Args:
        a: Packed array of shape (n, W) with dtype uint64.
        b: Packed array of shape (m, W) with dtype uint64.
        D: Original dimension.

    Returns:
        Dot product matrix of shape (n, m).
    """
    n, W = a.shape
    m = b.shape[0]

    lut = np.zeros(65536, dtype=np.int32)
    for i in range(65536):
        lut[i] = bin(i).count("1")

    result = np.zeros((n, m), dtype=np.int32)
    for w in range(W):
        and_bits = np.bitwise_and(a[:, w:w+1], b[:, w:w+1].T)
        for shift in range(0, 64, 16):
            chunk = ((and_bits >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.int64)
            result += lut[chunk]
    return result
