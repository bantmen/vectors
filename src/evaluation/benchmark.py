"""Automated benchmark runner for comparing quantizers."""

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from ..quantizers.pq import ProductQuantizer
from ..quantizers.rabitq import RaBitQ
from ..quantizers.reranked import RerankedQuantizer
from ..search.exhaustive import exhaustive_search
from ..utils.timer import timer
from .metrics import recall_at_k, queries_per_second


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    method: str
    params: dict
    recall_at_1: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    qps: float = 0.0
    memory_bytes: int = 0
    build_time: float = 0.0
    mean_distance_error: float = 0.0
    std_distance_error: float = 0.0


def run_single_benchmark(
    method: str,
    params: dict,
    train: np.ndarray,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    warmup_queries: int = 5,
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        method: "pq" or "rabitq".
        params: Method-specific parameters (e.g. {"M": 8} for PQ).
        train: Training vectors for fit().
        database: Database vectors for search.
        queries: Query vectors.
        ground_truth: Ground truth neighbor indices, shape (n_q, k_gt).
        k: Number of neighbors to retrieve.
        warmup_queries: Number of warmup queries before timing.

    Returns:
        BenchmarkResult with metrics.
    """
    # Create and train quantizer
    n_shortlist = params.pop("n_shortlist", None)
    result = BenchmarkResult(method=method, params={**params, **({"n_shortlist": n_shortlist} if n_shortlist else {})})
    if method == "pq":
        quantizer = ProductQuantizer(**params)
    elif method == "rabitq":
        quantizer = RaBitQ(**params)
    elif method == "pq+rerank":
        quantizer = RerankedQuantizer(
            ProductQuantizer(**params), n_shortlist=n_shortlist or 100
        )
    elif method == "rabitq+rerank":
        quantizer = RerankedQuantizer(
            RaBitQ(**params), n_shortlist=n_shortlist or 100
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    with timer() as t_build:
        quantizer.fit(train)
        quantizer.build_database(database)
    result.build_time = t_build.elapsed
    result.memory_bytes = quantizer.memory_usage()

    # Warmup
    if warmup_queries > 0 and queries.shape[0] > warmup_queries:
        quantizer.search(queries[:warmup_queries], k=k)

    # Timed search
    with timer() as t_search:
        pred_indices, pred_distances = quantizer.search(queries, k=k)
    result.qps = queries_per_second(queries.shape[0], t_search.elapsed)

    # Recall
    result.recall_at_1 = recall_at_k(pred_indices, ground_truth, k=1)
    k10 = min(10, ground_truth.shape[1], pred_indices.shape[1])
    result.recall_at_10 = recall_at_k(pred_indices, ground_truth, k=k10)
    k100 = min(100, ground_truth.shape[1], pred_indices.shape[1])
    result.recall_at_100 = recall_at_k(pred_indices, ground_truth, k=k100)

    # Distance estimation error
    true_dists_sq = np.sum(
        (queries[:, np.newaxis] - database[pred_indices]) ** 2, axis=2
    )  # (n_q, k)
    errors = pred_distances - true_dists_sq
    result.mean_distance_error = float(np.mean(errors))
    result.std_distance_error = float(np.std(errors))

    return result


def run_benchmark_sweep(
    train: np.ndarray,
    database: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    pq_ms: list[int] | None = None,
    n_shortlist: int = 100,
) -> list[BenchmarkResult]:
    """Run full benchmark sweep: multiple PQ configs + RaBitQ.

    Args:
        train: Training data.
        database: Database vectors.
        queries: Query vectors.
        ground_truth: Ground truth neighbor indices.
        k: Number of neighbors.
        pq_ms: List of M values for PQ. Defaults to [4, 8, 16, 32].

    Returns:
        List of BenchmarkResults.
    """
    D = train.shape[1]
    if pq_ms is None:
        pq_ms = [m for m in [4, 8, 16, 32] if D % m == 0]

    results = []

    # PQ sweep
    for M in tqdm(pq_ms, desc="PQ sweep"):
        try:
            r = run_single_benchmark(
                "pq", {"M": M}, train, database, queries, ground_truth, k=k
            )
            results.append(r)
            print(f"  PQ(M={M}): recall@{k}={r.recall_at_10:.3f}, QPS={r.qps:.0f}, "
                  f"mem={r.memory_bytes/1024:.1f}KB")
        except Exception as e:
            print(f"  PQ(M={M}): FAILED - {e}")

    # RaBitQ
    try:
        r = run_single_benchmark(
            "rabitq", {}, train, database, queries, ground_truth, k=k
        )
        results.append(r)
        print(f"  RaBitQ: recall@{k}={r.recall_at_10:.3f}, QPS={r.qps:.0f}, "
              f"mem={r.memory_bytes/1024:.1f}KB")
    except Exception as e:
        print(f"  RaBitQ: FAILED - {e}")

    # Reranked variants
    # PQ(M=32)+rerank (use largest valid M)
    rerank_M = max(m for m in pq_ms if D % m == 0) if pq_ms else 32
    if D % rerank_M == 0:
        try:
            r = run_single_benchmark(
                "pq+rerank",
                {"M": rerank_M, "n_shortlist": n_shortlist},
                train, database, queries, ground_truth, k=k,
            )
            results.append(r)
            print(f"  PQ(M={rerank_M})+rerank: recall@{k}={r.recall_at_10:.3f}, "
                  f"QPS={r.qps:.0f}, mem={r.memory_bytes/1024:.1f}KB")
        except Exception as e:
            print(f"  PQ(M={rerank_M})+rerank: FAILED - {e}")

    # RaBitQ+rerank
    try:
        r = run_single_benchmark(
            "rabitq+rerank",
            {"n_shortlist": n_shortlist},
            train, database, queries, ground_truth, k=k,
        )
        results.append(r)
        print(f"  RaBitQ+rerank: recall@{k}={r.recall_at_10:.3f}, QPS={r.qps:.0f}, "
              f"mem={r.memory_bytes/1024:.1f}KB")
    except Exception as e:
        print(f"  RaBitQ+rerank: FAILED - {e}")

    return results
