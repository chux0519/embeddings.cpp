# Snowflake on ggml b8833

Date: 2026-04-25

## What changed

- Switched vendored `ggml/` to llama.cpp `b8833` and restored the local GTE/Snowflake overlay.
- Fixed `CPU_REPACK` integration for model loading:
  - skip repacking `embeddings.word_embeddings.weight`
  - fall back to `memcpy` when a tensor has no repack traits
  - size the weights buffer with `ggml_backend_buft_get_alloc_size()`
- Kept `EMBEDDINGS_CPP_GTE_LENGTH_SORT=1` as the default.

## Correctness

Latest alignment report:

- [alignment_20260425_003040.md](/home/yongsheng/repos/embeddings.cpp/scripts/output/alignment_20260425_003040.md)

All Snowflake cases pass, including batch-vs-single regression coverage.

## Realistic Batch Benchmark

Platform:

- Host: this workstation
- Threads: `12`
- Batch size: `8`
- Warmup: `3`
- Iterations: `12`
- Input set: randomized realistic multilingual texts from `scripts/profile_snowflake.py`

| Runner | Mean ms | P50 ms | P95 ms | Text/s | RSS MB |
|---|---:|---:|---:|---:|---:|
| embeddings.cpp `q4_k_mlp_q8_attn` | 90.02 | 92.43 | 94.23 | 88.87 | 543.09 |
| Python CPU | 91.01 | 84.91 | 110.71 | 87.90 | 1156.98 |
| TEI ORT | 96.89 | 97.89 | 107.98 | 82.57 | 1965.34 |

Artifacts:

- [snowflake_profile_20260425_080033.json](/home/yongsheng/repos/embeddings.cpp/scripts/output/snowflake_profile_20260425_080033.json)
- [snowflake_profile_20260425_080050.json](/home/yongsheng/repos/embeddings.cpp/scripts/output/snowflake_profile_20260425_080050.json)
- [snowflake_profile_20260425_080129.json](/home/yongsheng/repos/embeddings.cpp/scripts/output/snowflake_profile_20260425_080129.json)

## A/B findings from new ggml

| Variant | Mean ms | Text/s | RSS MB | Takeaway |
|---|---:|---:|---:|---|
| default (`CPU_REPACK=1`, length sort on) | 99.37 | 80.51 | 546.95 | current best |
| `EMBEDDINGS_CPP_CPU_REPACK=0` | 140.04 | 57.13 | 543.02 | repack is a clear win |
| `GGML_REPACK_Q8_AVX2=1` | 278.10 | 28.77 | 556.72 | bad on this model/CPU |
| `EMBEDDINGS_CPP_GTE_LENGTH_SORT=0` | 101.28 | 78.99 | 552.20 | default length sort still helps |

Artifacts:

- [snowflake_profile_20260425_003708.json](/home/yongsheng/repos/embeddings.cpp/scripts/output/snowflake_profile_20260425_003708.json)
- [snowflake_profile_20260425_003721.json](/home/yongsheng/repos/embeddings.cpp/scripts/output/snowflake_profile_20260425_003721.json)
- [snowflake_profile_20260425_003732.json](/home/yongsheng/repos/embeddings.cpp/scripts/output/snowflake_profile_20260425_003732.json)

## Optimization directions worth keeping

1. Keep `CPU_REPACK` enabled by default for quantized Snowflake.
2. Keep GTE length sorting enabled by default.
3. Do not enable AVX2 `Q8_0` repack by default on this host for `q4_k_mlp_q8_attn`.
4. Keep the primary published benchmark in `end_to_end` scope and use `engine_only` only as a secondary diagnostic view.
5. Focus the next round on preserving repack-friendly row grouping and reducing small-group fallbacks in attention/linear hot paths.

## Optimization directions not worth chasing first

1. Repacking the embedding table.
2. Enabling `Q8_0` AVX2 repack globally.
3. More scalar micro-tweaks before measuring GEMM/GEMV mix and grouped-attention fragmentation.
