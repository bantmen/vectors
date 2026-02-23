#!/usr/bin/env python
"""Run benchmark comparisons between PQ and RaBitQ."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from src.datasets.loader import load_dataset, list_datasets
from src.datasets.utils import normalize_rows
from src.evaluation.benchmark import run_benchmark_sweep
from src.evaluation.plotting import (
    plot_recall_vs_qps,
    plot_memory_comparison,
    plot_error_histogram,
)
from src.search.exhaustive import exhaustive_search


def main():
    parser = argparse.ArgumentParser(description="Run PQ vs RaBitQ benchmark")
    parser.add_argument(
        "--dataset", default="sift-128",
        choices=list_datasets(),
        help="Dataset to benchmark on",
    )
    parser.add_argument("--n-db", type=int, default=10000, help="Number of database vectors")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--pq-ms", type=int, nargs="+", default=None, help="PQ M values")
    parser.add_argument("--n-shortlist", type=int, default=100, help="Rerank shortlist size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading {args.dataset}...")
    data = load_dataset(args.dataset)
    train = data["train"]
    database = train[:args.n_db]
    queries = data["test"][:args.n_queries]

    # Normalize angular datasets
    if data["distance_metric"] == "angular":
        print("Angular dataset: normalizing to unit vectors")
        train = normalize_rows(train)
        database = normalize_rows(database)
        queries = normalize_rows(queries)

    print(f"Database: {database.shape}, Queries: {queries.shape}")

    # Compute ground truth
    print("Computing ground truth (exact k-NN)...")
    gt_indices, gt_distances = exhaustive_search(queries, database, k=args.k)
    print(f"Ground truth computed: {gt_indices.shape}")

    # Run benchmarks
    print("\nRunning benchmarks...")
    results = run_benchmark_sweep(
        train=train,
        database=database,
        queries=queries,
        ground_truth=gt_indices,
        k=args.k,
        pq_ms=args.pq_ms,
        n_shortlist=args.n_shortlist,
    )

    # Save results
    results_data = []
    for r in results:
        results_data.append({
            "method": r.method,
            "params": r.params,
            "recall@1": r.recall_at_1,
            "recall@10": r.recall_at_10,
            "recall@100": r.recall_at_100,
            "qps": r.qps,
            "memory_bytes": r.memory_bytes,
            "build_time": r.build_time,
            "mean_distance_error": r.mean_distance_error,
            "std_distance_error": r.std_distance_error,
        })

    json_path = output_dir / f"{args.dataset}_results.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_recall_vs_qps(
        results, k=args.k,
        save_path=output_dir / f"{args.dataset}_recall_qps.png",
        title=f"{args.dataset}: Recall@{args.k} vs QPS",
    )
    plot_memory_comparison(
        results,
        save_path=output_dir / f"{args.dataset}_memory.png",
        title=f"{args.dataset}: Memory Usage",
    )
    plot_error_histogram(
        results,
        save_path=output_dir / f"{args.dataset}_error.png",
        title=f"{args.dataset}: Distance Estimation Error",
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary for {args.dataset} (n_db={args.n_db}, n_q={args.n_queries}, k={args.k})")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Recall@10':>10} {'QPS':>10} {'Memory':>10} {'Bias':>12}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for r in results:
        if r.method == "pq":
            name = f"PQ(M={r.params.get('M', '')})"
        elif r.method == "pq+rerank":
            name = f"PQ(M={r.params.get('M', '')})+rerank({r.params.get('n_shortlist', '')})"
        elif r.method == "rabitq+rerank":
            name = f"RaBitQ+rerank({r.params.get('n_shortlist', '')})"
        else:
            name = "RaBitQ"
        print(f"{name:<20} {r.recall_at_10:>10.3f} {r.qps:>10.0f} "
              f"{r.memory_bytes/1024:>9.1f}K {r.mean_distance_error:>12.2f}")


if __name__ == "__main__":
    main()
