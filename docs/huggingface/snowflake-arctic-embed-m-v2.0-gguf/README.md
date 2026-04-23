---
base_model: {{MODEL_ID}}
library_name: embeddings.cpp
tags:
- gguf
- embeddings
- embeddings.cpp
- snowflake
- gte
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
| `{{GGUF_FILE}}` | mixed `q4_K` MLP + `q8_0` attention | `{{GGUF_SIZE}}` | `{{GGUF_SHA256}}` |

The mixed quantization policy is:

- `mlp.up_gate_proj.weight`: `q4_K`
- `mlp.down_proj.weight`: `q4_K`
- `attention.qkv_proj.weight`: `q8_0`
- `attention.o_proj.weight`: `q8_0`

## Recommended embeddings.cpp Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON \
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

The optimized CPU runtime is now the default for this artifact. You can run:

```bash
python your_script.py
```

Set `EMBEDDINGS_CPP_CPU_REPACK=0` or `EMBEDDINGS_CPP_FLASH_ATTN=0` only when
debugging or checking regressions.

By default, `embeddings.cpp` uses the detected CPU concurrency for model
inference. Set `EMBEDDINGS_CPP_THREADS=N` only when pinning a deployment to a
measured value for a specific CPU quota or host.

Do not enable the experimental `GGML_REPACK_Q8_AVX2=1` path for this artifact;
it was slower on the tuning host.

## Reproducing The GGUF

From an `embeddings.cpp` checkout:

```bash
uv pip install -r scripts/requirements.txt

uv run scripts/convert.py \
  {{MODEL_ID}} \
  models/snowflake-arctic-embed-m-v2.0.fp16.gguf \
  f16

EMBEDDINGS_CPP_SKIP_QUANT_PATTERNS='attention.qkv_proj.weight,attention.o_proj.weight' \
./build/quantize \
  models/snowflake-arctic-embed-m-v2.0.fp16.gguf \
  models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_attnf16.gguf \
  q4_k

EMBEDDINGS_CPP_SKIP_QUANT_PATTERNS='mlp.up_gate_proj.weight,mlp.down_proj.weight' \
./build/quantize \
  models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_attnf16.gguf \
  models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf \
  q8_0
```

## Notes

This model is derived from `{{MODEL_ID}}`. Use the
upstream model card and license terms when deciding whether this artifact is
appropriate for your use case.
