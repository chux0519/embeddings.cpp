# Browser Benchmark

This document tracks browser-side performance for
`Snowflake/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`.

Package status: `@embeddings-cpp/web` currently defaults to stable
single-thread `wasm`. `webgpu` is exposed as experimental because
Snowflake-specific ggml ops fall back to CPU until dedicated WebGPU kernels are
implemented. `pthread` is not exposed by the npm package until it is redesigned
as a worker/proxy runner.

Scope:

- browser only
- `engine-only` model forward
- tokenizer excluded
- same pre-tokenized batch across runtimes

## Platform

- Host: `macmini`
- Model: `Mac16,10`
- CPU: `Apple M4`
- Memory: `16 GiB`
- OS: `macOS 26.3.1`
- Browser: `Google Chrome`
- Browser flags: `--enable-unsafe-webgpu --ignore-gpu-blocklist`

## Runtimes

- `WASM single-thread`
- `WASM pthread x8`
- `WebGPU`

The WebGPU build uses `ggml-webgpu` via an `emdawnwebgpu` local port package.

## Fixtures

- [`scripts/data/snowflake_wasm_batch1.txt`](/home/yongsheng/repos/embeddings.cpp/scripts/data/snowflake_wasm_batch1.txt):
  one short support-style question
- [`scripts/data/snowflake_wasm_batch8.txt`](/home/yongsheng/repos/embeddings.cpp/scripts/data/snowflake_wasm_batch8.txt):
  original mixed multilingual batch
- [`scripts/data/snowflake_wasm_short8.txt`](/home/yongsheng/repos/embeddings.cpp/scripts/data/snowflake_wasm_short8.txt):
  eight short support-style questions

All runs used `warmup=1` and `iterations=3`.

## Results

### Batch 1, Short Sentence

| Runtime | Mean ms | P50 ms | P95 ms | item/s |
|---|---:|---:|---:|---:|
| WASM single-thread | 165.24 | 164.99 | 167.95 | 6.05 |
| WASM pthread x8 | 56.24 | 55.20 | 66.43 | 17.78 |
| WebGPU | 35.20 | 34.77 | 36.21 | 28.41 |

### Batch 8, Mixed Multilingual Batch

| Runtime | Mean ms | P50 ms | P95 ms | item/s |
|---|---:|---:|---:|---:|
| WASM single-thread | 1298.39 | 1296.81 | 1301.82 | 6.16 |
| WASM pthread x8 | 342.91 | 332.14 | 362.95 | 23.33 |
| WebGPU | 51.31 | 51.23 | 51.62 | 155.90 |

### Batch 8, Short Question Set

| Runtime | Mean ms | P50 ms | P95 ms | item/s |
|---|---:|---:|---:|---:|
| WASM single-thread | 1458.59 | 1457.80 | 1460.52 | 5.48 |
| WASM pthread x8 | 390.97 | 391.56 | 392.43 | 20.46 |
| WebGPU | 50.85 | 51.52 | 51.66 | 157.34 |

## Interpretation

- On this Apple M4 host, `WebGPU` is the best browser runtime in every measured
  Snowflake scenario.
- For `batch=1`, `WebGPU` is about `1.6x` faster than `pthread x8` and about
  `4.7x` faster than single-thread WASM.
- For `batch=8`, `WebGPU` is much further ahead:
  about `6.7x` faster than `pthread x8` on the mixed batch and about `7.7x`
  faster on the short-question batch.
- The benefit is not limited to one short sentence. It holds for both mixed
  multilingual input and a more realistic short-query batch.

## Reproduction

Local browser demo:

```bash
python3 scripts/browser_wasm_bench_server.py --host 127.0.0.1 --port 18081 --root "$PWD"
```

Then open:

```text
http://127.0.0.1:18081/demo/browser-wasm/index.html
```

Mac browser matrix runner:

```bash
scripts/run_macmini_browser_bench.sh
scripts/run_macmini_browser_matrix.sh
```

Browser case runner:

```bash
node scripts/run_browser_cases.mjs
```

Dynamic browser bundles:

```bash
scripts/build_browser_dynamic.sh
```

## Packaging and Caching

For a productized browser path, the useful minimum is:

1. Ship one prebuilt browser artifact per runtime:
   - `build-wasm-web`
   - `build-wasm-web-pthread`
   - `build-wasm-webgpu-browser`
   - and `*-dyn` variants for dynamic model loading:
     - `build-wasm-web-dyn`
     - `build-wasm-web-pthread-dyn`
     - `build-wasm-webgpu-browser-dyn`
2. Version the model and browser bundle together.
3. Cache `.data`, `.js`, and `.wasm` with immutable URLs.
4. Store large artifacts in browser persistent storage:
   - `Cache Storage` for fetched bundles
   - `IndexedDB` or `OPFS` for model files when moving away from preload
5. Keep runtime selection simple:
   - use single-thread WASM by default
   - allow explicit experimental `WebGPU`
   - keep `pthread` out of the public package until the runner is redesigned

The demo now supports both paths:

- `Bundled`: benchmark-friendly preload artifacts with `.data`
- `Downloaded`: `*-dyn` artifacts that fetch the GGUF and batch fixture at
  runtime, write them into MEMFS, and reuse the fetched bytes from browser
  cache on later runs

The demo now includes a minimal browser cache implementation:

- a service worker for browser bundle requests
- explicit bundle prefetch into `Cache Storage`
- explicit GGUF URL prefetch into `Cache Storage`
- browser storage estimates in the UI

This is still a simple whole-file cache, not a final chunked GGUF loader backed
by `IndexedDB` or `OPFS`. But it is already enough to validate the product path:
serve `COOP/COEP`, pick a runtime, download the published GGUF, cache it, and
reuse it across reloads without rebuilding the page bundle.
