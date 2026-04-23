# embeddings.cpp

A C++ library for text (and maybe image) embeddings, focusing on efficient inference of BERT-like (and maybe clip-like) models.

## Overview

Many existing GGML-based text embedding libraries have limited support for Chinese text processing due to their custom tokenizer implementations. This project addresses this limitation by leveraging Hugging Face's Rust tokenizer implementation, wrapped with a C++ API to ensure consistency with the Python transformers library while providing native performance.

While currently focused on BERT-like text embedding models, the project aims to support image embedding models in the future (Work in Progress).

> **Note**: This is an experimental and educational project. It is not recommended for production use at this time.

## Supported Models

The following models have been tested and verified:
- BAAI/bge-m3
- BAAI/bge-base-zh-v1.5
- shibing624/text2vec-base-multilingual
- Snowflake/snowflake-arctic-embed-m-v2.0
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

The C++ implementation is checked against Python `transformers` CPU output. Models also supported by Hugging Face `text-embeddings-inference` can be checked against TEI as a third implementation. For repeatable correctness and performance runs, see `scripts/ALIGNMENT_README.md`.

## Model Preparation

First, install the required dependencies:
```bash
uv pip install --torch-backend cpu -r scripts/requirements.txt
```

Then convert the models to GGUF format:
```bash
# Convert BGE-M3 model
uv run scripts/convert.py BAAI/bge-m3 ./models/bge-m3.fp16.gguf f16

# Convert BGE-Base Chinese v1.5 model
uv run scripts/convert.py BAAI/bge-base-zh-v1.5 ./models/bge-base-zh-v1.5.fp16.gguf f16

uv run scripts/convert.py Snowflake/snowflake-arctic-embed-m-v2.0 ./models/snowflake-arctic-embed-m-v2.0.fp16.gguf f16

# Convert Text2Vec multilingual model
uv run scripts/convert.py shibing624/text2vec-base-multilingual ./models/text2vec-base-multilingual.fp16.gguf f16

uv run scripts/convert.py sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ./models/paraphrase-multilingual-MiniLM-L12-v2.fp16.gguf f16
```

## Model Quantization

After converting models to GGUF format, you can quantize them to reduce memory usage and improve inference speed:

```bash
# Build the quantization tool
cmake --build build --target quantize

# Quantize a model (example with different quantization types)
./build/quantize ./models/bge-m3.fp16.gguf ./models/bge-m3.q4_k.gguf q4_k
./build/quantize ./models/bge-m3.fp16.gguf ./models/bge-m3.q6_k.gguf q6_k
./build/quantize ./models/bge-m3.fp16.gguf ./models/bge-m3.q8_0.gguf q8_0

# On Windows
.\build\Release\quantize.exe .\models\bge-m3.fp16.gguf .\models\bge-m3.q4_k.gguf q4_k
```

### Supported Quantization Types

- `q4_k`: 4-bit quantization with K-means clustering (good balance of size and quality)
- `q6_k`: 6-bit quantization with K-means clustering (higher quality, larger size)
- `q8_0`: 8-bit quantization (minimal quality loss, moderate size reduction)
- Other GGML quantization types as supported by the library

### Usage

```
quantize <input_model.gguf> <output_model.gguf> <qtype>
```

The quantization tool will:
1. Load the input GGUF model
2. Quantize eligible tensors (typically weight matrices)
3. Preserve metadata and non-quantizable tensors
4. Output size comparison and compression statistics

## Running Tests

Before running, install embeddings.cpp:
```bash
# use CMAKE_ARGS to add more cmake settings
$env:CMAKE_ARGS="-DGGML_VULKAN=ON"

# Install the package
pip install .

# Generate Python stub files
cd build && make stub

# on Windows
pip install pybind11-stubgen
# then
pybind11-stubgen embeddings_cpp -o .

python tests/test_tokenizer.py
```

## Alignment and Benchmarking

Run correctness checks for every model mentioned in this README:

```bash
uv run scripts/alignment.py --convert-missing
```

Include CPU performance comparisons:

```bash
uv run scripts/alignment.py --convert-missing --benchmark
```

The benchmark report compares Python `transformers` CPU, `embeddings.cpp`, and
TEI when enabled for the model.

Measured on this PC:

- CPU: Intel Xeon E5-2673 v3 @ 2.40GHz
- Cores: 12 vCPU, 1 socket, SMT off
- Memory: 62 GiB RAM
- OS: Ubuntu Linux 5.15
- Model: `Snowflake/snowflake-arctic-embed-m-v2.0`
- GGUF: `models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`

Batch-8 result table:

| Runner | Config | Batch | Mean ms | P50 ms | P95 ms | Text/s | RSS MB |
|---|---|---:|---:|---:|---:|---:|---:|
| `python_cpu` | `threads=10`, fp32 HF model | 8 | 62.57 | 61.86 | 66.07 | 127.86 | 1156.5 |
| `embeddings.cpp` | `threads=12`, `q4_k_mlp_q8_attn.gguf` | 8 | 99.98 | 91.30 | 147.46 | 80.02 | 520.4 |
| `tei` | `cpu-1.9`, `--max-batch-tokens 8192` | 8 | 90.90 | 94.00 | 118.24 | 88.01 | 11100.2 |

The local Python and `embeddings.cpp` rows above were measured serially with
`warmup=2` and `iterations=10`. The TEI row is from the same machine with the
same batch size; RSS is read from `docker stats`.

Standalone benchmark runs also write JSON and Markdown reports under
`scripts/output/`:

```bash
uv run scripts/benchmark.py \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --gguf-path models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf
```

Pin the C++ CPU thread count while tuning:

```bash
uv run scripts/alignment.py --benchmark --cpp-threads 8
```

For models also supported by `text-embeddings-inference`, start TEI as an additional comparator:

```bash
uv run scripts/alignment.py \
  --models Snowflake/snowflake-arctic-embed-m-v2.0 \
  --convert-missing \
  --tei-start \
  --benchmark
```

For registry-driven Snowflake checks against the optimized mixed GGUF:

```bash
uv run scripts/correctness.py --model-id Snowflake/snowflake-arctic-embed-m-v2.0 --benchmark
uv run scripts/benchmark.py \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --gguf-path models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf
```

## Loading Published GGUF Models

Known optimized GGUF artifacts are listed in `embeddings_cpp/registry.json`.
The default Snowflake artifact is published under the `chux0519` Hugging Face
namespace.

- Model repository:
  `https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp`
- Direct GGUF file:
  `https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`

```python
from embeddings_cpp import load

model = load("Snowflake/snowflake-arctic-embed-m-v2.0")
vectors = model.batch_encode(["hello world", "你好，世界"])
```

By default, CPU inference uses the detected CPU concurrency. Pin
`EMBEDDINGS_CPP_THREADS=N` only after measuring a specific host or container CPU
quota.

Install the optional Hugging Face dependency when downloading from the Hub:

```bash
pip install "embeddings-cpp[hub]"
```

## HTTP Server

The server can load a registered model from Hugging Face or a local GGUF path.
For a Snowflake deployment, `embeddings.cpp` is intended to replace a TEI CPU
setup.

### TEI Comparison

For `Snowflake/snowflake-arctic-embed-m-v2.0`, the deployment mapping is:

| Concern | TEI | embeddings.cpp |
|---|---|---|
| Container image | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9` | `ghcr.io/<owner>/embeddings-cpp-server:<tag>` or a locally built image |
| Model source | Hugging Face model repo | Registered optimized GGUF from `chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp` or `--gguf-path` |
| Main request path | `POST /embed` | `POST /embed` |
| OpenAI-style path | not the primary TEI path | `POST /v1/embeddings` |
| Batch token guard | `--max-batch-tokens` | `--max-batch-tokens` |
| Thread control | TEI runtime defaults | detected CPU concurrency by default, override with `--threads` or `EMBEDDINGS_CPP_THREADS` only after measurement |
| Health probes | `/health` | `/health`, `/ready`, `/info` |

The TEI Snowflake command:

```bash
docker run --rm -p 8081:80 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --max-batch-tokens 8192
```

The equivalent `embeddings.cpp` server run is:

```bash
python -m embeddings_cpp.server \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --port 8080 \
  --max-batch-tokens 8192
```

Build and run the Docker image locally:

```bash
docker build -t embeddings-cpp-server:local .

docker run --rm -p 8080:80 \
  embeddings-cpp-server:local \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --max-batch-tokens 8192
```

Endpoints:

- `GET /health`
- `GET /ready`
- `GET /info`
- `POST /embed` with `{"inputs": ["hello", "world"]}`
- `POST /v1/embeddings` with an OpenAI-compatible embeddings request

For client compatibility, the main request surfaces are:

- TEI: `POST /embed`
- embeddings.cpp: `POST /embed`
- OpenAI-style clients: `POST /v1/embeddings`

For correctness work, the Snowflake path is checked against both Python
`transformers` CPU output and TEI. See `docs/TEST_MATRIX.md` and
`scripts/server_compare.py`. For performance work, `scripts/alignment.py` and
`scripts/benchmark.py` report both inference speed and RSS memory.

Container images can be published to GHCR with
`.github/workflows/publish-server-image.yml`, which publishes tags in the form
`ghcr.io/<owner>/embeddings-cpp-server:<tag>`.

## Building from Source

### macOS (ARM)

Configure and build with Metal support:
```bash
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DGGML_METAL=ON \
      -DGGML_METAL_EMBED_LIBRARY=ON \
      -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON ..
```

If you encountered openmp's bug, try

> brew install libomp
>
> export OpenMP_ROOT=$(brew --prefix)/opt/libomp

### Windows

build with vulkan support:

```powershell
cmake -DGGML_VULKAN=ON -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON ..
# If you encounter any issues, ensure that your graphics driver and Vulkan SDK versions are compatible.
# You can also add -DGGML_VULKAN_DEBUG=ON -DGGML_VULKAN_VALIDATE=ON for debuging
```

## Debugging

GGML debug support is now enabled by default in the vendored version. This provides better debugging capabilities for CPU backend operations without requiring additional patches.

For more information about GGML debugging features, see: https://github.com/ggml-org/ggml/discussions/655
