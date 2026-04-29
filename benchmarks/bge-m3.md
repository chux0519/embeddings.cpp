# BGE-M3 Benchmark

This report summarizes the BGE-M3 CPU evaluation from 2026-04-29. The runs use
deterministic randomized realistic long-text inputs shared with the Snowflake
profiling path, seed `20260429`, and batch sizes `1`, `4`, and `8`.

This report follows the shared [benchmark standard](STANDARD.md).

Cosine thresholds are product tolerances, not process-failure gates. A lower
cosine reports larger drift; it is only an execution failure when embeddings are
not produced or contain invalid values. Use `--fail-on-threshold` for CI gates.

## Recommendation

The current conservative CPU baseline is `q8_0+repack_on`.

- It keeps the strongest alignment observed in this sweep: min cosine vs Python
  CPU `0.999215` and batch-vs-single min cosine `0.999814`.
- It uses about `915 MB` peak RSS, roughly `1.35 GB` less than Python CPU on the
  measured host.
- It is the safest artifact to publish first because quality is materially above
  the loose `0.95` tolerance and above the practical `0.99` tolerance.

If the product tolerance is truly near `0.95`, `q4_k+repack_on` and
`q4_k_mlp_q8_attn+repack_on` become interesting size/speed candidates, but they
should be documented as higher-drift variants rather than the default baseline.

## Baseline: Python CPU vs q8

Source report: `scripts/output/bge_m3_eval_20260429_015937.md`

Tolerance settings: `--min-cos 0.99 --batch-min-cos 0.99
--quantized-batch-min-cos 0.99`.

| Runner | Variant | Batch | Mean ms | P50 ms | P95 ms | Text/s | Peak RSS MB |
|---|---|---:|---:|---:|---:|---:|---:|
| Python CPU | `python_cpu` | 1 | 67.549 | 67.954 | 68.173 | 14.804 | 2260.59 |
| embeddings.cpp | `q8_0+repack_on` | 1 | 43.102 | 43.173 | 47.188 | 23.201 | 914.90 |
| Python CPU | `python_cpu` | 4 | 203.521 | 203.741 | 203.758 | 19.654 | 2261.32 |
| embeddings.cpp | `q8_0+repack_on` | 4 | 170.255 | 169.332 | 172.781 | 23.494 | 915.09 |
| Python CPU | `python_cpu` | 8 | 331.912 | 331.719 | 332.443 | 24.103 | 2264.09 |
| embeddings.cpp | `q8_0+repack_on` | 8 | 322.557 | 324.403 | 324.596 | 24.802 | 915.34 |

| Variant | Tolerance | Within Tolerance | Min Cos vs Python | Batch vs Single Min Cos | B1 C++/Python TPS | B4 C++/Python TPS | B8 C++/Python TPS | Peak RSS MB | RSS Delta MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q8_0+repack_on` | within_tolerance | 10/10 | 0.999215 | 0.999814 | 1.567 | 1.195 | 1.029 | 915.34 | -1345.68 |

## Kquant Sweep

Source report: `scripts/output/bge_m3_eval_20260429_020038.md`

| Variant | Tolerance at 0.99 | Within Tolerance | Min Cos vs Python | Batch vs Single Min Cos | B1 TPS Ratio | B4 TPS Ratio | B8 TPS Ratio | Peak RSS MB | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q8_0+repack_on` | within_tolerance | 10/10 | 0.999215 | 0.999814 | 1.645 | 1.234 | 0.908 | 915.11 | Best quality baseline; batch 8 has run-to-run variance. |
| `q6_k+repack_on` | within_tolerance | 10/10 | 0.992955 | 0.998489 | 1.056 | 0.716 | 0.619 | 783.97 | Smaller, but too slow for batch. |
| `q4_k+repack_on` | outside_tolerance | 5/10 | 0.951258 | 0.997025 | 2.060 | 1.094 | 1.236 | 644.78 | Strong size/speed, acceptable only for loose tolerance near 0.95. |

## Layered Quantization Sweep

Sources:

- `scripts/output/bge_m3_eval_20260429_090446.md`
- `scripts/output/bge_m3_eval_20260429_090616.md`

| Variant | Tolerance at 0.99 | Within Tolerance | Min Cos vs Python | Batch vs Single Min Cos | B1 TPS Ratio | B4 TPS Ratio | B8 TPS Ratio | Peak RSS MB | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `q8_0+repack_on` | within_tolerance | 10/10 | 0.999215 | 0.999814 | 1.687 | 0.959 | 1.024 | 914.95 | Conservative publishing baseline. |
| `q6_k_mlp_q8_attn+repack_on` | within_tolerance | 10/10 | 0.995855 | 0.998878 | 1.149 | 0.831 | 0.682 | 807.73 | Good quality/RSS, but slower than Python for batch. |
| `q4_k_mlp_q8_attn+repack_on` | outside_tolerance | 7/10 | 0.983726 | 0.998757 | 1.736 | 1.391 | 0.989 | 693.45 | Better drift than full `q4_k`, useful for relaxed tolerance. |
| `q4_k_mlp_q8_attn_embf16+repack_on` | outside_tolerance | 7/10 | 0.983893 | 0.998760 | 1.718 | 1.364 | 1.232 | 1044.34 | Faster at batch 8 in this run, but larger than `q8_0`. |

## Hugging Face Artifact Status

We should publish a BGE-M3 GGUF after the registry and upload workflow are
prepared for this model. Recommended first artifact: `bge-m3.q8_0.gguf`.

Current blockers before running `.github/workflows/upload-gguf-to-hf.yml` for
`BAAI/bge-m3`:

- `embeddings_cpp/registry.json` has no `hf_repo_id`, `artifact_file`, or
  `quantization_steps` for BGE-M3 yet.
- The upload workflow renders the Snowflake Hugging Face README template
  unconditionally, so a BGE-M3 model card/template should be added or selected
  dynamically before upload.

Until those are fixed, running the action for BGE-M3 would either lack artifact
metadata or publish with the wrong model-card content.
