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

The C++ implementation shows high accuracy compared to the Python implementation, with differences in the order of 10^-9. For detailed comparison results, please refer to `alignment.ipynb`.

## Model Preparation

First, install the required dependencies:
```bash
pip install -r scripts/requirements.txt
```

Then convert the models to GGUF format:
```bash
# Convert BGE-M3 model
python scripts/convert.py BAAI/bge-m3 ./models/bge-m3.fp16.gguf f16

# Convert BGE-Base Chinese v1.5 model
python scripts/convert.py BAAI/bge-base-zh-v1.5 ./models/bge-base-zh-v1.5.fp16.gguf f16

uv run scripts/convert.py Snowflake/snowflake-arctic-embed-m-v2.0 ./models/snowflake-arctic-embed-m-v2.0.fp16.gguf f16

# Convert Text2Vec multilingual model
python scripts/convert.py shibing624/text2vec-base-multilingual ./models/text2vec-base-multilingual.fp16.gguf f16

python scripts/convert.py sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ./models/paraphrase-multilingual-MiniLM-L12-v2.fp16.gguf f16
```

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

## Building from Source

### macOS (ARM)

Configure and build with Metal support:
```bash
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DGGML_METAL=ON \
      -DGGML_METAL_EMBED_LIBRARY=ON \
      -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON ..

### Windows

build with vulkan support:

```powershell
cmake -DGGML_VULKAN=ON -DEMBEDDINGS_CPP_ENABLE_PYBIND=ON ..
# If you encounter any issues, ensure that your graphics driver and Vulkan SDK versions are compatible.
# You can also add -DGGML_VULKAN_DEBUG=ON -DGGML_VULKAN_VALIDATE=ON for debuging
```

## debugging

for better debugging ggml, we could use 0001-feat-add-debug-support-for-only-cpu-backend.patch

check: https://github.com/ggml-org/ggml/discussions/655 for details