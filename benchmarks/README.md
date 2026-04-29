# Benchmarks

Model-specific benchmark summaries live here so the top-level README can stay
focused on usage and reproducible commands.

| Model | Report | Current best CPU artifact | Notes |
|---|---|---|---|
| `BAAI/bge-m3` | [BGE-M3](bge-m3.md) | `q8_0+repack_on` | Best conservative baseline; `q4_k` variants are useful only when larger cosine drift is acceptable. |

Generated one-off reports remain under `scripts/output/` and are intentionally
not treated as stable documentation.
