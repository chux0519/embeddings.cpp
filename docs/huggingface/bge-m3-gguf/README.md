---
base_model: {{MODEL_ID}}
library_name: embeddings.cpp
tags:
- gguf
- embeddings
- embeddings.cpp
- bge
- bge-m3
- cpu
pipeline_tag: sentence-similarity
---

# {{MODEL_ID}} GGUF for embeddings.cpp

This repository contains an optimized GGUF artifact for running
`{{MODEL_ID}}` with
[`embeddings.cpp`](https://github.com/daandtu/embeddings.cpp).

The GGUF is intended for embedding inference. It is not a llama.cpp
text-generation model.

## File

| File | Quantization | Size | SHA256 |
|---|---|---:|---|
| `{{GGUF_FILE}}` | `q8_0` | `{{GGUF_SIZE}}` | `{{GGUF_SHA256}}` |

This is the conservative BGE-M3 CPU artifact. The benchmarked baseline keeps
strict alignment with Python CPU while using substantially less memory than the
Python `transformers` path.

## Python Usage

```python
from embeddings_cpp import load

model = load("{{MODEL_ID}}")
vectors = model.batch_encode(["hello world", "你好，世界"])
```

Install the Hugging Face downloader extra when loading directly from this repo:

```bash
pip install "embeddings-cpp[hub]"
```

## Recommended embeddings.cpp Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON \
  -DEMBEDDINGS_CPP_BUILD_WASM_TOOLS=OFF \
  -DGGML_CPU_REPACK=ON \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_METAL=OFF

cmake --build build -j "$(nproc)"
```

## Recommended CPU Runtime

The registry applies the recommended runtime environment automatically when
using `embeddings_cpp.load()`:

```bash
{{RUNTIME_ENV}}
```

By default, `embeddings.cpp` uses the detected CPU concurrency for model
inference. Set `EMBEDDINGS_CPP_THREADS=N` only when pinning a deployment to a
measured value for a specific CPU quota or host.

## Reproducing The GGUF

From an `embeddings.cpp` checkout:

```bash
uv pip install -r scripts/requirements.txt

uv run scripts/convert.py \
  {{MODEL_ID}} \
  models/bge-m3.f16.gguf \
  f16

./build/quantize \
  models/bge-m3.f16.gguf \
  models/{{GGUF_FILE}} \
  q8_0
```

The registry-driven artifact builder performs the same conversion and
quantization pipeline:

```bash
uv run scripts/build_gguf_artifact.py --model-id {{MODEL_ID}}
```

## Notes

This model is derived from `{{MODEL_ID}}`. Use the upstream model card and
license terms when deciding whether this artifact is appropriate for your use
case.
