"""Tests for Product Quantization."""

import numpy as np
import pytest

from src.quantizers.pq import ProductQuantizer
from src.search.exhaustive import exhaustive_search


class TestProductQuantizer:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((1000, 32)).astype(np.float32)
        X_test = rng.standard_normal((50, 32)).astype(np.float32)
        return X_train, X_test

    def test_fit_encode_decode_shapes(self, sample_data):
        X_train, _ = sample_data
        pq = ProductQuantizer(M=4, Ks=16)
        pq.fit(X_train)

        codes = pq.encode(X_train[:10])
        assert codes.shape == (10, 4)
        assert codes.dtype == np.uint8

        decoded = pq.decode(codes)
        assert decoded.shape == (10, 32)
        assert decoded.dtype == np.float32

    def test_reconstruction_error_decreases_with_M(self, sample_data):
        X_train, _ = sample_data
        errors = []
        for M in [4, 8, 16]:
            pq = ProductQuantizer(M=M, Ks=64)
            pq.fit(X_train)
            codes = pq.encode(X_train)
            X_hat = pq.decode(codes)
            mse = np.mean((X_train - X_hat) ** 2)
            errors.append(mse)
        # More subspaces -> lower reconstruction error
        assert errors[0] > errors[1] > errors[2]

    def test_reconstruction_error_reasonable(self, sample_data):
        X_train, _ = sample_data
        pq = ProductQuantizer(M=8, Ks=256)
        pq.fit(X_train)
        codes = pq.encode(X_train)
        X_hat = pq.decode(codes)
        mse = np.mean((X_train - X_hat) ** 2)
        # MSE should be significantly less than data variance
        data_var = np.var(X_train)
        assert mse < data_var * 0.5

    def test_distance_estimation(self, sample_data):
        X_train, X_test = sample_data
        pq = ProductQuantizer(M=8, Ks=256)
        pq.fit(X_train)
        pq.build_database(X_train[:200])

        # Compare PQ distances to true distances for a small subset
        q = X_test[:5]
        db = X_train[:200]
        pq_dists = pq.estimate_distances(q)
        true_dists = np.sum((q[:, np.newaxis] - db[np.newaxis]) ** 2, axis=2)

        # PQ distances should be correlated with true distances
        for i in range(5):
            corr = np.corrcoef(pq_dists[i], true_dists[i])[0, 1]
            assert corr > 0.8, f"Query {i}: correlation {corr:.3f} too low"

    def test_search_recall(self, sample_data):
        X_train, X_test = sample_data
        db = X_train[:500]
        queries = X_test[:20]
        k = 10

        pq = ProductQuantizer(M=8, Ks=256)
        pq.fit(X_train)
        pq.build_database(db)

        pq_indices, _ = pq.search(queries, k=k)
        true_indices, _ = exhaustive_search(queries, db, k=k)

        # Compute recall@10
        recalls = []
        for i in range(queries.shape[0]):
            pq_set = set(pq_indices[i])
            true_set = set(true_indices[i])
            recalls.append(len(pq_set & true_set) / k)
        mean_recall = np.mean(recalls)
        assert mean_recall > 0.3, f"Recall@{k} = {mean_recall:.3f} is too low"

    def test_search_distances_sorted(self, sample_data):
        X_train, X_test = sample_data
        pq = ProductQuantizer(M=4, Ks=64)
        pq.fit(X_train)
        pq.build_database(X_train[:200])
        _, dists = pq.search(X_test[:5], k=10)
        for i in range(5):
            assert np.all(np.diff(dists[i]) >= -1e-6)

    def test_invalid_D_M(self):
        pq = ProductQuantizer(M=7)
        X = np.random.randn(100, 32).astype(np.float32)
        with pytest.raises(ValueError, match="divisible"):
            pq.fit(X)

    def test_memory_usage(self, sample_data):
        X_train, _ = sample_data
        pq = ProductQuantizer(M=8, Ks=256)
        pq.fit(X_train)
        pq.build_database(X_train[:1000])
        mem = pq.memory_usage()
        # codes: 1000 * 8 = 8000 bytes
        # codebooks: 8 * 256 * 4 * 4 = 32768 bytes
        assert mem == 1000 * 8 + 8 * 256 * 4 * 4
