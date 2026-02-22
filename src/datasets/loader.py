"""Download and load ANN benchmark datasets in HDF5 format."""

import os
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# ann-benchmarks dataset URLs and metadata
DATASETS = {
    "sift-128": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "filename": "sift-128-euclidean.hdf5",
        "distance": "euclidean",
    },
    "glove-100": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "filename": "glove-100-angular.hdf5",
        "distance": "angular",
    },
    "gist-960": {
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "filename": "gist-960-euclidean.hdf5",
        "distance": "euclidean",
    },
}


def download_dataset(name: str, data_dir: Path | None = None) -> Path:
    """Download a dataset if not already cached.

    Args:
        name: Dataset name (e.g. "sift-128").
        data_dir: Directory to store files. Defaults to project data/.

    Returns:
        Path to the downloaded HDF5 file.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")

    info = DATASETS[name]
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / info["filename"]

    if filepath.exists():
        return filepath

    print(f"Downloading {name} from {info['url']}...")
    response = requests.get(info["url"], stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return filepath


def load_dataset(name: str, data_dir: Path | None = None) -> dict:
    """Load a dataset, downloading if needed.

    Args:
        name: Dataset name (e.g. "sift-128").
        data_dir: Directory to store/find files.

    Returns:
        Dict with keys:
          - train: float32 array (n_train, D)
          - test: float32 array (n_test, D)
          - neighbors: int32 array (n_test, k) of ground-truth neighbor indices
          - distances: float32 array (n_test, k) of ground-truth distances
          - distance_metric: "euclidean" or "angular"
    """
    filepath = download_dataset(name, data_dir)
    info = DATASETS[name]

    with h5py.File(filepath, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
        distances = np.array(f["distances"], dtype=np.float32)

    return {
        "train": train,
        "test": test,
        "neighbors": neighbors,
        "distances": distances,
        "distance_metric": info["distance"],
    }


def list_datasets() -> list[str]:
    """Return available dataset names."""
    return list(DATASETS.keys())
