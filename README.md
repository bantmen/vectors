# vectors

Educational comparison of Product Quantization (PQ) and RaBitQ for Approximate Nearest Neighbor search.

Implemented with Claude Code + Opus 4.6.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Download data

```bash
python scripts/download_datasets.py # all datasets
python scripts/download_datasets.py --datasets sift-128-euclidean # just SIFT-128
```

## Run notebooks

```bash
jupyter notebook notebooks/
```

- `01_pq_walkthrough.ipynb` — Product Quantization: subspace splitting, k-means codebooks, ADC, and recall
- `02_rabitq_walkthrough.ipynb` — RaBitQ: random projection, binarization, correction scalars, and error scaling
- `03_comparison.ipynb` — PQ vs RaBitQ on synthetic and SIFT-128 data: recall, bias, memory, speed

## Run benchmarks

```bash
python scripts/run_benchmark.py --dataset sift-128 --n-db 10000 --n-queries 100 --k 10
```

Results on SIFT-128 (10k database, 100 queries, k=10):

| Method | Recall@10 | QPS | Memory | Dist. Bias |
|---|---|---|---|---|
| PQ(M=4) | 0.401 | 6911 | 167 KB | -19221 |
| PQ(M=8) | 0.575 | 3989 | 206 KB | -8140 |
| PQ(M=16) | 0.727 | 2291 | 284 KB | -2181 |
| PQ(M=32) | 0.823 | 1217 | 441 KB | -661 |
| RaBitQ | 0.535 | 799 | 298 KB | -15221 |
| PQ(M=32)+rerank(100) | 1.000 | 1189 | 5441 KB | 0 |
| RaBitQ+rerank(100) | 0.976 | 815 | 5298 KB | 0 |

Results on GloVe-100 (10k database, 100 queries, k=10, angular/cosine):

| Method | Recall@10 | QPS | Memory | Dist. Bias |
|---|---|---|---|---|
| PQ(M=4) | 0.213 | 6033 | 139 KB | -0.48 |
| PQ(M=5) | 0.227 | 5422 | 149 KB | -0.44 |
| PQ(M=10) | 0.341 | 3242 | 198 KB | -0.25 |
| PQ(M=20) | 0.533 | 1925 | 295 KB | -0.10 |
| PQ(M=25) | 0.618 | 1561 | 344 KB | -0.06 |
| RaBitQ | 0.457 | 791 | 298 KB | -0.16 |
| PQ(M=25)+rerank(100) | 0.990 | 1523 | 4250 KB | 0.00 |
| RaBitQ+rerank(100) | 0.905 | 785 | 4205 KB | 0.00 |

Results on GIST-960 (10k database, 100 queries, k=10):

| Method | Recall@10 | QPS | Memory | Dist. Bias |
|---|---|---|---|---|
| PQ(M=4) | 0.167 | 4288 | 999 KB | -0.74 |
| PQ(M=8) | 0.242 | 2924 | 1038 KB | -0.59 |
| PQ(M=16) | 0.356 | 1822 | 1116 KB | -0.43 |
| PQ(M=32) | 0.467 | 1070 | 1273 KB | -0.29 |
| RaBitQ | 0.741 | 53 | 4850 KB | -0.03 |
| PQ(M=32)+rerank(100) | 0.958 | 1005 | 38773 KB | 0.00 |
| RaBitQ+rerank(100) | 0.999 | 50 | 42350 KB | 0.00 |

## Run tests

```bash
pytest
```

## Project structure

```
├── src/
│   ├── quantizers/    PQ, RaBitQ, and re-ranked wrapper implementations
│   ├── datasets/      dataset download and caching
│   ├── evaluation/    metrics, benchmarks, plots
│   ├── search/        brute-force exact k-NN baseline
│   └── utils/         shared math, binary, timing helpers
├── notebooks/         walkthrough and comparison notebooks
├── scripts/           CLI tools for download and benchmarking
└── tests/
```

## References

- [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462v2) — Jégou et al. 2011
- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound](https://arxiv.org/abs/2405.12497) — Gao & Long 2024
- [ann-benchmarks](http://ann-benchmarks.com/) — standard ANN evaluation datasets
