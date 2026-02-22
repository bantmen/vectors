"""Tests for RerankedQuantizer wrapper."""

import numpy as np
import pytest

from src.quantizers.pq import ProductQuantizer
from src.quantizers.rabitq import RaBitQ
from src.quantizers.reranked import RerankedQuantizer
from src.search.exhaustive import exhaustive_search


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((1000, 32)).astype(np.float32)
    X_test = rng.standard_normal((50, 32)).astype(np.float32)
    return X_train, X_test


class TestRerankedQuantizerPQ:
    def test_search_shapes(self, sample_data):
        X_train, X_test = sample_data
        base = ProductQuantizer(M=8, Ks=256)
        rq = RerankedQuantizer(base, n_shortlist=50)
        rq.fit(X_train)
        rq.build_database(X_train[:500])

        indices, distances = rq.search(X_test[:10], k=10)
        assert indices.shape == (10, 10)
        assert distances.shape == (10, 10)
        assert indices.dtype == np.int32
        assert distances.dtype == np.float32

    def test_distances_are_exact(self, sample_data):
        X_train, X_test = sample_data
        db = X_train[:500]
        queries = X_test[:10]

        base = ProductQuantizer(M=8, Ks=256)
        rq = RerankedQuantizer(base, n_shortlist=50)
        rq.fit(X_train)
        rq.build_database(db)

        indices, distances = rq.search(queries, k=5)
        # Returned distances should be exact squared L2
        for i in range(queries.shape[0]):
            expected = np.sum((db[indices[i]] - queries[i]) ** 2, axis=1)
            np.testing.assert_allclose(distances[i], expected, rtol=1e-5)

    def test_distances_sorted(self, sample_data):
        X_train, X_test = sample_data
        base = ProductQuantizer(M=8, Ks=256)
        rq = RerankedQuantizer(base, n_shortlist=50)
        rq.fit(X_train)
        rq.build_database(X_train[:500])

        _, dists = rq.search(X_test[:10], k=10)
        for i in range(10):
            assert np.all(np.diff(dists[i]) >= -1e-6)

    def test_recall_improves_over_base(self, sample_data):
        X_train, X_test = sample_data
        db = X_train[:500]
        queries = X_test[:20]
        k = 10

        true_indices, _ = exhaustive_search(queries, db, k=k)

        # Base PQ
        base = ProductQuantizer(M=8, Ks=256)
        base.fit(X_train)
        base.build_database(db)
        base_indices, _ = base.search(queries, k=k)

        # Reranked PQ
        rq = RerankedQuantizer(ProductQuantizer(M=8, Ks=256), n_shortlist=100)
        rq.fit(X_train)
        rq.build_database(db)
        rr_indices, _ = rq.search(queries, k=k)

        def mean_recall(pred, gt):
            recalls = []
            for i in range(pred.shape[0]):
                recalls.append(len(set(pred[i]) & set(gt[i])) / k)
            return np.mean(recalls)

        base_recall = mean_recall(base_indices, true_indices)
        rr_recall = mean_recall(rr_indices, true_indices)
        assert rr_recall >= base_recall, (
            f"Reranked recall {rr_recall:.3f} < base recall {base_recall:.3f}"
        )

    def test_memory_includes_raw_vectors(self, sample_data):
        X_train, _ = sample_data
        db = X_train[:500]

        base = ProductQuantizer(M=8, Ks=256)
        rq = RerankedQuantizer(base, n_shortlist=50)
        rq.fit(X_train)
        rq.build_database(db)

        base_only = ProductQuantizer(M=8, Ks=256)
        base_only.fit(X_train)
        base_only.build_database(db)

        raw_bytes = db.shape[0] * db.shape[1] * 4
        assert rq.memory_usage() == base_only.memory_usage() + raw_bytes


class TestRerankedQuantizerRaBitQ:
    def test_search_shapes(self, sample_data):
        X_train, X_test = sample_data
        base = RaBitQ()
        rq = RerankedQuantizer(base, n_shortlist=50)
        rq.fit(X_train)
        rq.build_database(X_train[:500])

        indices, distances = rq.search(X_test[:10], k=10)
        assert indices.shape == (10, 10)
        assert distances.shape == (10, 10)
        assert indices.dtype == np.int32
        assert distances.dtype == np.float32

    def test_recall_improves_over_base(self, sample_data):
        X_train, X_test = sample_data
        db = X_train[:500]
        queries = X_test[:20]
        k = 10

        true_indices, _ = exhaustive_search(queries, db, k=k)

        # Base RaBitQ
        base = RaBitQ()
        base.fit(X_train)
        base.build_database(db)
        base_indices, _ = base.search(queries, k=k)

        # Reranked RaBitQ
        rq = RerankedQuantizer(RaBitQ(), n_shortlist=100)
        rq.fit(X_train)
        rq.build_database(db)
        rr_indices, _ = rq.search(queries, k=k)

        def mean_recall(pred, gt):
            recalls = []
            for i in range(pred.shape[0]):
                recalls.append(len(set(pred[i]) & set(gt[i])) / k)
            return np.mean(recalls)

        base_recall = mean_recall(base_indices, true_indices)
        rr_recall = mean_recall(rr_indices, true_indices)
        assert rr_recall >= base_recall, (
            f"Reranked recall {rr_recall:.3f} < base recall {base_recall:.3f}"
        )

    def test_shortlist_larger_than_db(self, sample_data):
        """When n_shortlist > n_db, should still work (degrades to exact search)."""
        X_train, X_test = sample_data
        db = X_train[:50]
        rq = RerankedQuantizer(RaBitQ(), n_shortlist=200)
        rq.fit(X_train)
        rq.build_database(db)

        indices, distances = rq.search(X_test[:5], k=10)
        assert indices.shape == (5, 10)

    def test_error_before_build_database(self, sample_data):
        X_train, X_test = sample_data
        rq = RerankedQuantizer(RaBitQ(), n_shortlist=50)
        rq.fit(X_train)
        with pytest.raises(RuntimeError, match="build_database"):
            rq.search(X_test[:5], k=10)
