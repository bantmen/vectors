"""Tests for dataset loading and exhaustive search."""

import numpy as np
import pytest

from src.datasets.utils import dataset_stats, normalize_rows
from src.search.exhaustive import exhaustive_search


class TestDatasetUtils:
    def test_dataset_stats(self):
        X = np.random.randn(100, 10).astype(np.float32)
        stats = dataset_stats(X)
        assert stats["n"] == 100
        assert stats["D"] == 10
        assert stats["mean_norm"] > 0
        assert stats["min_norm"] <= stats["mean_norm"] <= stats["max_norm"]

    def test_normalize_rows(self):
        X = np.random.randn(50, 10).astype(np.float32)
        Xn = normalize_rows(X)
        norms = np.linalg.norm(Xn, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_normalize_zero_row(self):
        X = np.zeros((1, 10), dtype=np.float32)
        Xn = normalize_rows(X)
        assert np.all(np.isfinite(Xn))


class TestExhaustiveSearch:
    def test_self_search(self):
        """Searching a dataset against itself should return identity."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 16)).astype(np.float32)
        indices, distances = exhaustive_search(X, X, k=1)
        np.testing.assert_array_equal(indices[:, 0], np.arange(100))
        np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-5)

    def test_k_neighbors(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 32)).astype(np.float32)
        queries = X[:10]
        indices, distances = exhaustive_search(queries, X, k=5)
        assert indices.shape == (10, 5)
        assert distances.shape == (10, 5)
        # First neighbor should be the query itself
        np.testing.assert_array_equal(indices[:, 0], np.arange(10))
        # Distances should be sorted
        for i in range(10):
            assert np.all(np.diff(distances[i]) >= -1e-6)

    def test_small_database(self):
        """k > n_db should work."""
        db = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        q = np.array([[0, 0]], dtype=np.float32)
        indices, distances = exhaustive_search(q, db, k=10)
        assert indices.shape == (1, 3)
        assert indices[0, 0] == 0
        np.testing.assert_allclose(distances[0, 0], 0.0, atol=1e-6)
