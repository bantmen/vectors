"""Tests for utility modules: binary packing, math, timer."""

import numpy as np
import pytest

from src.utils.binary import pack_bits, unpack_bits, hamming_distance, binary_dot_product
from src.utils.math import random_orthogonal_matrix, pad_to_multiple
from src.utils.timer import timer


class TestBitPacking:
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=(100, 128)).astype(np.uint8)
        packed = pack_bits(bits)
        assert packed.shape == (100, 2)
        assert packed.dtype == np.uint64
        recovered = unpack_bits(packed, 128)
        np.testing.assert_array_equal(bits, recovered)

    def test_round_trip_large_dim(self):
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, size=(10, 960)).astype(np.uint8)
        packed = pack_bits(bits)
        assert packed.shape == (10, 15)
        recovered = unpack_bits(packed, 960)
        np.testing.assert_array_equal(bits, recovered)

    def test_all_zeros(self):
        bits = np.zeros((5, 64), dtype=np.uint8)
        packed = pack_bits(bits)
        assert np.all(packed == 0)
        recovered = unpack_bits(packed, 64)
        np.testing.assert_array_equal(bits, recovered)

    def test_all_ones(self):
        bits = np.ones((5, 64), dtype=np.uint8)
        packed = pack_bits(bits)
        assert np.all(packed == np.uint64(0xFFFFFFFFFFFFFFFF))
        recovered = unpack_bits(packed, 64)
        np.testing.assert_array_equal(bits, recovered)

    def test_invalid_dimension(self):
        bits = np.zeros((5, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="multiple of 64"):
            pack_bits(bits)


class TestHammingDistance:
    def test_self_distance_zero(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, size=(10, 128)).astype(np.uint8)
        packed = pack_bits(bits)
        dists = hamming_distance(packed, packed)
        np.testing.assert_array_equal(np.diag(dists), 0)

    def test_known_distance(self):
        a = np.zeros((1, 64), dtype=np.uint8)
        b = np.ones((1, 64), dtype=np.uint8)
        pa = pack_bits(a)
        pb = pack_bits(b)
        dists = hamming_distance(pa, pb)
        assert dists[0, 0] == 64


class TestBinaryDotProduct:
    def test_self_dot_product(self):
        bits = np.ones((1, 64), dtype=np.uint8)
        packed = pack_bits(bits)
        dp = binary_dot_product(packed, packed, 64)
        assert dp[0, 0] == 64

    def test_orthogonal(self):
        a = np.zeros((1, 128), dtype=np.uint8)
        a[0, :64] = 1
        b = np.zeros((1, 128), dtype=np.uint8)
        b[0, 64:] = 1
        pa = pack_bits(a)
        pb = pack_bits(b)
        dp = binary_dot_product(pa, pb, 128)
        assert dp[0, 0] == 0


class TestOrthogonalMatrix:
    def test_orthogonality(self):
        Q = random_orthogonal_matrix(64)
        product = Q.astype(np.float64) @ Q.astype(np.float64).T
        np.testing.assert_allclose(product, np.eye(64), atol=1e-5)

    def test_deterministic(self):
        Q1 = random_orthogonal_matrix(32, seed=123)
        Q2 = random_orthogonal_matrix(32, seed=123)
        np.testing.assert_array_equal(Q1, Q2)

    def test_different_seeds(self):
        Q1 = random_orthogonal_matrix(32, seed=1)
        Q2 = random_orthogonal_matrix(32, seed=2)
        assert not np.allclose(Q1, Q2)

    def test_det_plus_minus_one(self):
        Q = random_orthogonal_matrix(64)
        det = np.linalg.det(Q.astype(np.float64))
        assert abs(abs(det) - 1.0) < 1e-4


class TestPadToMultiple:
    def test_no_pad_needed(self):
        x = np.ones((5, 128))
        padded = pad_to_multiple(x, 64)
        assert padded.shape == (5, 128)

    def test_pad_needed(self):
        x = np.ones((5, 100))
        padded = pad_to_multiple(x, 64)
        assert padded.shape == (5, 128)
        np.testing.assert_array_equal(padded[:, :100], 1.0)
        np.testing.assert_array_equal(padded[:, 100:], 0.0)


class TestTimer:
    def test_measures_time(self):
        with timer() as t:
            _ = sum(range(10000))
        assert t.elapsed > 0
        assert t.elapsed < 5.0  # sanity
