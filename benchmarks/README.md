# Benchmarks

Model-specific benchmark summaries live here so the top-level README can stay
focused on usage and reproducible commands.

All stable reports should follow the shared
[benchmark standard](STANDARD.md): invalid output is a failure, cosine drift is
a reported tolerance metric, and published results use deterministic realistic
randomized inputs.

| Model | Report | Current best CPU artifact | Notes |
|---|---|---|---|
| `BAAI/bge-m3` | [BGE-M3](bge-m3.md) | `q8_0+repack_on` | Best conservative baseline; `q4_k` variants are useful only when larger cosine drift is acceptable. |

Generated one-off reports remain under `scripts/output/` and are intentionally
not treated as stable documentation.
