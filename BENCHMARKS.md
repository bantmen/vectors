# Benchmark Results

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

Results on DBpedia-OpenAI-100k (10k database, 10k queries, k=10, 1536-d angular/cosine):

| Method | Recall@10 | QPS | Memory | Dist. Bias |
|---|---|---|---|---|
| PQ(M=4) | 0.226 | 3584 | 1575 KB | -0.20 |
| PQ(M=8) | 0.275 | 2817 | 1614 KB | -0.18 |
| PQ(M=16) | 0.336 | 1876 | 1692 KB | -0.17 |
| PQ(M=32) | 0.427 | 1122 | 1849 KB | -0.14 |
| RaBitQ | 0.821 | 14 | 11169 KB | -0.00 |
| PQ(M=32)+rerank(100) | 0.894 | 963 | 61849 KB | 0.00 |
| RaBitQ+rerank(100) | 1.000 | 15 | 71169 KB | 0.00 |

