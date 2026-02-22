#!/usr/bin/env python
"""Download all ANN benchmark datasets."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.loader import download_dataset, list_datasets
from src.datasets.utils import dataset_stats
from src.datasets.loader import load_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download ANN benchmark datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_datasets(),
        choices=list_datasets(),
        help="Datasets to download",
    )
    args = parser.parse_args()

    for name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        try:
            data = load_dataset(name)
            stats = dataset_stats(data["train"])
            print(f"  Train: {data['train'].shape}")
            print(f"  Test:  {data['test'].shape}")
            print(f"  Ground truth neighbors: {data['neighbors'].shape}")
            print(f"  Distance metric: {data['distance_metric']}")
            print(f"  Mean norm: {stats['mean_norm']:.2f}")
            print(f"  Std norm:  {stats['std_norm']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
