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
| `{{GGUF_FILE}}` | `q4_k` MLP and vocabulary, `q8_0` attention | `{{GGUF_SIZE}}` | `{{GGUF_SHA256}}` |

Mixed quantization: the feed-forward weights and the 250k-token vocabulary
table are `q4_k`, attention stays `q8_0`. Compared with a plain `q8_0` file
(612 MB):

- Retrieval quality is unchanged on a Chinese/English agent-history
  benchmark (2040 documents: top-1 66%, recall@5 78–83%, both files).
- Cosine to the fp16 model: mean 0.990, min 0.988 (`q8_0`: 0.999).
- About 40% faster on x86 CPUs with AVX2 (Xeon E5-2673 v3, 12 threads:
  2.0 vs 1.4 docs/s) and ~230 MB less RAM.
- On Apple silicon CPUs `q8_0` is ~18% faster (i8mm repack); with Metal both
  run at the same speed. Build `q8_0` yourself (below) if you need strict
  alignment with Python or run on Apple CPUs only.

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

EMBEDDINGS_CPP_SKIP_QUANT_PATTERNS='attention.,position_embeddings' \
./build/quantize \
  models/bge-m3.f16.gguf \
  models/bge-m3.q4_k_mlp_emb_attnf16.gguf \
  q4_k

./build/quantize \
  models/bge-m3.q4_k_mlp_emb_attnf16.gguf \
  models/{{GGUF_FILE}} \
  q8_0
```

For a plain `q8_0` file, quantize `models/bge-m3.f16.gguf` with `q8_0` in one
step.

The registry-driven artifact builder performs the same conversion and
quantization pipeline:

```bash
uv run scripts/build_gguf_artifact.py --model-id {{MODEL_ID}}
```

## Notes

This model is derived from `{{MODEL_ID}}`. Use the upstream model card and
license terms when deciding whether this artifact is appropriate for your use
case.
