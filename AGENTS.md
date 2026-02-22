# PQ vs RaBitQ: Educational ANN Comparison

## Purpose
Compare Product Quantization and RaBitQ for Approximate Nearest Neighbor search.
Prioritizes clarity and correctness over raw performance.

## Architecture
- `src/quantizers/base.py` defines the abstract `Quantizer` interface (fit/encode/decode/search)
- `src/quantizers/pq.py` and `src/quantizers/rabitq.py` implement the two methods
- `src/quantizers/reranked.py` wraps any Quantizer with exact-distance re-ranking (shortlist + exact L2)
- `src/datasets/loader.py` downloads/caches HDF5 from ann-benchmarks
- `src/search/exhaustive.py` provides brute-force exact k-NN baseline
- `src/evaluation/` contains metrics, benchmarking, and plotting
- `src/utils/` has shared math, binary, and timing utilities

## Key Conventions
- NumPy-vectorized implementations throughout
- All vectors are float32 arrays of shape (n, D)
- Distance metric is squared Euclidean unless noted
- Angular datasets (GloVe) are L2-normalized first
- RaBitQ pads dimensions to nearest multiple of 64

## Datasets
Downloaded from ann-benchmarks.com to `data/` directory:
- SIFT-128 (128-dim, 1M vectors)
- GloVe-100 (100-dim, 1.2M vectors)
- GIST-960 (960-dim, 1M vectors)
