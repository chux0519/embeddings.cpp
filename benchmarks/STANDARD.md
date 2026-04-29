# Benchmark Standard

This repository uses one benchmark protocol for model reports so results remain
comparable across Snowflake, BGE-M3, and future embedding models.

## Result Semantics

Benchmark failure is reserved for invalid execution:

- the runner crashes or times out;
- the embedding shape is wrong;
- an embedding contains `NaN` or non-finite values;
- required single or batch outputs are missing.

Cosine drift is reported as a metric, not as a hard failure by default. Scripts
may expose `--fail-on-threshold` for CI jobs that intentionally gate on a chosen
tolerance.

## Tolerance Tiers

Every report should include raw `min_cos`, `mean_cos`, `mse`, and `max_abs`.
Tolerance tiers are labels for reading the drift:

| Tier | Min cosine | Intended use |
|---|---:|---|
| `strict` | `>= 0.999` | fp16, q8, or default publish candidates where near-Python alignment is expected. |
| `practical` | `>= 0.99` | Quantized candidates that should preserve retrieval behavior for most workloads. |
| `relaxed` | `>= 0.95` | Aggressive compression candidates; acceptable only when product owners explicitly accept this drift. |
| `outside_relaxed` | `< 0.95` | Usually not a publish candidate unless a task-specific evaluation says otherwise. |

Model registry thresholds remain the model's default reporting tolerance. The
current registry tolerances are:

| Model | Python CPU tolerance | Batch-vs-single tolerance | Why |
|---|---:|---:|---|
| `Snowflake/snowflake-arctic-embed-m-v2.0` | `0.96` | `0.999` | Historical production artifact is an optimized mixed-quant GGUF; cross-runner alignment is looser, but batch behavior should remain stable. |
| `BAAI/bge-m3` | `0.999` | `0.999999` | Current conservative contract for the fp16/q8 BGE-M3 path. |

Reports should still show the same tier table even when the registry tolerance
is looser or stricter than a tier boundary.

## Reference Comparisons

Each model benchmark should compare:

1. Python CPU from the source Hugging Face model as the correctness and speed
   baseline.
2. `embeddings.cpp` GGUF variants under test.
3. `embeddings.cpp` batch output against its own single-request output.
4. TEI only when that model has explicit TEI support.

Required correctness columns:

- `Min Cos vs Python`
- `Batch vs Single Min Cos`
- `Mean Cos`
- `MSE`
- `Max Abs`
- tolerance tier/status

Required performance columns:

- batch size
- mean, p50, and p95 latency
- texts/second
- peak RSS MB
- load RSS MB when measured
- `C++/Python TPS` ratio
- `C++ - Python RSS` delta

## Inputs

Published model reports should use deterministic realistic randomized inputs
from `scripts.embedding_eval_common.REALISTIC_TEXT_POOL` instead of repeated
toy strings.

Default published seed: `20260429`.

Required correctness cases:

| Case | Purpose |
|---|---|
| `single_en` | single English request |
| `single_zh` | single Chinese request |
| `mixed_batch_4` | multilingual production-like batch |
| `random_batch_8` | randomized batch throughput and drift |
| `length_skew_batch_4` | short/long mixed batch regression |

Model-specific regression cases are allowed, but they must be marked as
additional cases. Snowflake's `["A cute cat....", "A cute cat."]` batch
regression is one such case.

## Published Reports

Stable reports belong in `benchmarks/<model-slug>.md`. Raw generated outputs
belong in `scripts/output/` and are not stable documentation.

Each stable model report should include:

- test date, host class if known, seed, iterations, warmup, and batch sizes;
- exact source report filenames or commands;
- Python CPU baseline table;
- kquant and repack summary;
- layered quantization summary when tested;
- current recommended artifact and why;
- Hugging Face upload status when the model has a publishable artifact.

The README should link to `benchmarks/README.md` and avoid embedding every
model's full result table inline.

## Artifact Recommendation Rule

Default published GGUF artifacts should favor the strongest quality baseline
that is still materially faster or smaller than Python CPU.

For current BGE-M3 results this is `q8_0+repack_on`: it is in the `strict` tier,
uses much less RSS than Python CPU, and is the safest first Hugging Face upload.
Aggressive variants such as `q4_k` or `q4_k_mlp_q8_attn` should be published as
separate higher-drift variants only when the report clearly marks their relaxed
tolerance profile.
