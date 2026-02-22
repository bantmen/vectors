"""Tests for RaBitQ."""

import numpy as np
import pytest

from src.quantizers.rabitq import RaBitQ
from src.search.exhaustive import exhaustive_search


class TestRaBitQ:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((1000, 128)).astype(np.float32)
        X_test = rng.standard_normal((50, 128)).astype(np.float32)
        return X_train, X_test

    def test_fit_encode_shapes(self, sample_data):
        X_train, _ = sample_data
        rq = RaBitQ()
        rq.fit(X_train)
        assert rq.D == 128
        assert rq.D_padded == 128
        assert rq.P.shape == (128, 128)
        assert rq.centroid.shape == (128,)

        codes = rq.encode(X_train[:10])
        assert codes.shape == (10, 2)  # 128 / 64 = 2
        assert codes.dtype == np.uint64
        assert rq.norms.shape == (10,)
        assert rq.obar_dot_xbar.shape == (10,)

    def test_fit_encode_non_multiple_of_64(self):
        """Test dimension padding for D not a multiple of 64."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 100)).astype(np.float32)
        rq = RaBitQ()
        rq.fit(X)
        assert rq.D == 100
        assert rq.D_padded == 128
        codes = rq.encode(X[:10])
        assert codes.shape == (10, 2)

    def test_correction_scalars_positive(self, sample_data):
        """<o_bar, x_bar_proj> should generally be positive (binary approx correlates with original)."""
        X_train, _ = sample_data
        rq = RaBitQ()
        rq.fit(X_train)
        rq.encode(X_train)
        # Most correction scalars should be positive
        assert np.mean(rq.obar_dot_xbar > 0) > 0.9

    def test_unbiasedness(self, sample_data):
        """RaBitQ distance estimator should be approximately unbiased."""
        X_train, X_test = sample_data
        db = X_train[:500]
        queries = X_test[:20]

        rq = RaBitQ()
        rq.fit(X_train)
        rq.build_database(db)

        est_dists = rq.estimate_distances(queries)
        true_dists = np.sum((queries[:, np.newaxis] - db[np.newaxis]) ** 2, axis=2)

        # Mean relative error should be near zero (unbiased)
        # Use relative error: (est - true) / true
        mask = true_dists > 1.0  # avoid near-zero distances
        if np.any(mask):
            rel_errors = (est_dists[mask] - true_dists[mask]) / true_dists[mask]
            mean_rel_error = np.mean(rel_errors)
            # Unbiased means mean error ≈ 0 (allow some tolerance for finite sample)
            assert abs(mean_rel_error) < 0.15, f"Mean relative error {mean_rel_error:.4f} too large"

    def test_error_scaling_with_dimension(self):
        """Error should scale as O(1/sqrt(D)) - lower D means higher variance."""
        rng = np.random.default_rng(42)
        variances = []
        for D in [64, 256]:
            D_padded = int(np.ceil(D / 64) * 64)
            X = rng.standard_normal((500, D)).astype(np.float32)
            queries = rng.standard_normal((20, D)).astype(np.float32)

            rq = RaBitQ(seed=42)
            rq.fit(X)
            rq.build_database(X)

            est = rq.estimate_distances(queries)
            true = np.sum((queries[:, np.newaxis] - X[np.newaxis]) ** 2, axis=2)

            mask = true > 1.0
            if np.any(mask):
                rel_errors = (est[mask] - true[mask]) / true[mask]
                variances.append(np.var(rel_errors))

        # Higher dimension should have lower error variance
        if len(variances) == 2:
            assert variances[1] < variances[0], \
                f"Variance did not decrease: D=64 -> {variances[0]:.4f}, D=256 -> {variances[1]:.4f}"

    def test_search_recall(self, sample_data):
        X_train, X_test = sample_data
        db = X_train[:500]
        queries = X_test[:20]
        k = 10

        rq = RaBitQ()
        rq.fit(X_train)
        rq.build_database(db)

        rq_indices, _ = rq.search(queries, k=k)
        true_indices, _ = exhaustive_search(queries, db, k=k)

        recalls = []
        for i in range(queries.shape[0]):
            rq_set = set(rq_indices[i])
            true_set = set(true_indices[i])
            recalls.append(len(rq_set & true_set) / k)
        mean_recall = np.mean(recalls)
        assert mean_recall > 0.3, f"Recall@{k} = {mean_recall:.3f} is too low"

    def test_search_distances_sorted(self, sample_data):
        X_train, X_test = sample_data
        rq = RaBitQ()
        rq.fit(X_train)
        rq.build_database(X_train[:200])
        _, dists = rq.search(X_test[:5], k=10)
        for i in range(5):
            assert np.all(np.diff(dists[i]) >= -1e-6)

    def test_decode_approximate(self, sample_data):
        """Decode should produce vectors roughly similar to originals."""
        X_train, _ = sample_data
        rq = RaBitQ()
        rq.fit(X_train)
        codes = rq.encode(X_train[:10])
        X_hat = rq.decode(codes)
        assert X_hat.shape == (10, 128)
        # Reconstruction is lossy but should be in the right ballpark
        mse = np.mean((X_train[:10] - X_hat) ** 2)
        data_var = np.var(X_train[:10])
        assert mse < data_var * 2.0  # very generous bound for 1-bit quantization

    def test_memory_usage(self, sample_data):
        X_train, _ = sample_data
        rq = RaBitQ()
        rq.fit(X_train)
        rq.build_database(X_train[:100])
        mem = rq.memory_usage()
        # codes: 100 * 2 * 8 = 1600 bytes
        # norms: 100 * 4 = 400 bytes
        # corrections: 100 * 4 = 400 bytes
        # matrix: 128 * 128 * 4 = 65536 bytes
        expected = 1600 + 400 + 400 + 65536
        assert mem == expected
