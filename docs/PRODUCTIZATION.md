# embeddings.cpp Productization Plan

This document describes the open workflow for shipping optimized GGUF
embedding artifacts and replacing TEI-style CPU deployments with
`embeddings.cpp`.

## Model Registry

`embeddings_cpp/registry.json` is the source of truth for supported optimized
artifacts.

Each model entry defines:

- upstream Hugging Face model id
- target GGUF artifact repository and filename
- default pooling method and embedding dimension
- recommended runtime environment
- recommended CMake flags
- quantization pipeline
- correctness thresholds
- benchmark defaults

To add a new optimized model:

1. Add a registry entry.
2. Add or reuse a Hugging Face model-card template under `docs/huggingface`.
3. Run `.github/workflows/upload-gguf-to-hf.yml` with the model id.
4. Run correctness locally or in CI.
5. Run performance benchmarks on a stable machine before publishing claims.

## Vendored GGML

`ggml/` is vendored source, not a submodule. Track its upstream source in
`.vendor/ggml-upstream.json` and follow `docs/GGML_UPSTREAM.md` when importing
updates from llama.cpp.

The default maintenance strategy is upstream-first: refresh from a pinned
llama.cpp release tag, then keep only the local embedding-specific patches that
are still justified by correctness or measured performance.

Local patches on top of upstream `ggml/` are tracked in
`docs/GGML_OVERLAY_PATCHES.md`.

Before changing or refreshing `ggml/`, check the recorded provenance:

```bash
uv run scripts/ggml_upstream.py status
```

Every upstream refresh should record the exact llama.cpp tag or commit and
explain which local embedding-specific patches were preserved, replaced, or
dropped.

## Publishing GGUF Artifacts

`.github/workflows/upload-gguf-to-hf.yml` is manually triggered. It:

1. checks out submodules
2. installs Rust, uv, Python dependencies, and `huggingface_hub`
3. builds `embeddings.cpp` and the quantizer
4. builds the registry-defined GGUF artifact
5. smoke tests the artifact
6. renders the Hugging Face README
7. uploads the GGUF and README to the target HF model repository

`.github/workflows/upload-web-assets-to-hf.yml` publishes browser package
assets to the same HF repository under:

```text
browser/<web_asset_version>/
```

The npm package default for Snowflake points at the `v0.1.1` directory. The
workflow also writes `web-assets.json` at the repo root and inside the versioned
browser directory so later clients can discover the exact runtime files from a
manifest instead of hard-coded paths.

Required GitHub secret:

- `HF_TOKEN`: Hugging Face token with write access to the target model repo.

For the current default namespace, the Snowflake artifact is published to:

```text
chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp
```

## Python Loading

Users can load registered artifacts directly:

```python
from embeddings_cpp import load

model = load("Snowflake/snowflake-arctic-embed-m-v2.0")
vectors = model.batch_encode(["hello", "world"])
```

For local files:

```python
model = load(
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    gguf_path="models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf",
)
```

## Python Package Publishing

The Python package is buildable through `setup.py`, and release workflows are
in place for wheel validation and PyPI publishing.

Planned release path:

1. Run `python scripts/set_version.py <version>` and
   `python scripts/check_version.py`.
2. Build validation wheels with `.github/workflows/build-python-wheels.yml`.
3. Publish with `.github/workflows/publish-python-package.yml` on `v*` tags or
   manual dispatch.
4. Configure PyPI Trusted Publishing for the `pypi` GitHub environment.
5. Keep `GGML_NATIVE=OFF` for PyPI wheels. Use `EMBEDDINGS_CPP_NATIVE=1` only
   for local source builds or machine-specific images.
6. Linux `riscv64` is a tier-1 wheel target alongside Linux `x86_64`, macOS
   `arm64`, and Windows `x86_64`.

Required PyPI setup:

- Create or reserve the `embeddings-cpp` project name.
- Configure PyPI Trusted Publishing for:
  - repository: `chux0519/embeddings.cpp`
  - workflow: `publish-python-package.yml`
  - environment: `pypi`

No long-lived PyPI token is required when Trusted Publishing is configured.

## npm Package Publishing

`.github/workflows/publish-web-package.yml` publishes `@embeddings-cpp/web` on
`web-v*` tags or manual dispatch.

Use the same version as the Python package. Browser assets are stored under
`browser/v<version>/`, and `scripts/check_version.py` enforces that the npm
package, Python package, browser asset URL, and demo cache-busting query all
match.

Required npm setup:

- Own or create the npm scope `@embeddings-cpp`.
- Add a GitHub secret `NPM_TOKEN` with npm publish permission, or migrate the
  workflow to npm Trusted Publishing after the package exists.
- Run `.github/workflows/upload-web-assets-to-hf.yml` first for the matching
  `web_asset_version` so the package defaults resolve on first install.

## HTTP Server

The server is intended as the TEI replacement surface:

```bash
python -m embeddings_cpp.server \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --port 8080
```

Supported endpoints:

- `GET /health`
- `POST /embed`
- `POST /v1/embeddings`

The server can download a registered GGUF from Hugging Face or load a mounted
local GGUF via `--gguf-path`.

## Container Image

`.github/workflows/publish-server-image.yml` publishes:

```text
ghcr.io/<owner>/embeddings-cpp-server:<tag>
```

The image defaults to:

```text
EMBEDDINGS_CPP_CPU_REPACK=1
EMBEDDINGS_CPP_FLASH_ATTN=1
```

The inference thread count defaults to detected CPU concurrency. Override
`--threads` or `EMBEDDINGS_CPP_THREADS` only when deploying to a measured CPU
quota or host shape.

## Correctness

Use the registry wrapper:

```bash
uv run scripts/correctness.py \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --benchmark
```

Coverage should include:

- Python CPU cosine alignment
- batch-vs-single alignment
- tokenizer alignment
- empty and fully masked inputs
- duplicate texts
- mixed language inputs
- skewed sequence lengths
- short and long text buckets

## Benchmarking

Use the registry benchmark helper for local or HTTP server runs:

```bash
uv run scripts/benchmark.py \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --gguf-path models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf

uv run scripts/benchmark.py \
  --runner http \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --url http://127.0.0.1:8080
```

GitHub-hosted runners are acceptable for build and smoke correctness. They are
not stable enough for final performance claims. Use a fixed local host or a
self-hosted runner for release benchmark numbers.

## Backlog

Tracked follow-up items:

- publish `embeddings-cpp` Python wheels to PyPI;
- publish and version the server container image for TEI replacement;
- document the Kubernetes migration from TEI to `embeddings.cpp`;
- keep `docs/GGML_OVERLAY_PATCHES.md` current before each ggml upstream refresh;
- implement missing Snowflake WebGPU kernels in the documented order;
- redesign browser `pthread` as a worker/proxy runner before exposing it again.
