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

## Run tests

```bash
pytest
```

## Project structure

```
├── src/
│   ├── quantizers/    PQ and RaBitQ implementations
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
