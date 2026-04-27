# Server Replacement Test Matrix

This matrix covers the legacy TEI-style deployment shape:

```yaml
args:
  - --model-id
  - Snowflake/snowflake-arctic-embed-m-v2.0
  - --max-batch-tokens
  - "8192"
resources:
  requests:
    cpu: "2"
    memory: "16Gi"
  limits:
    cpu: "4"
    memory: "32Gi"
```

## Compatibility Surface

Required endpoints:

- `GET /health`
- `POST /embed`
- `POST /v1/embeddings`

Operational flags:

- `--model-id`
- `--gguf-path`
- `--threads`
- `--max-batch-size`
- `--max-batch-tokens`

## Functional Cases

The comparison suite in `scripts/server_compare.py` covers:

- English single input
- Chinese single input
- whitespace input
- mixed Chinese/English batch
- duplicate input batch
- skewed sequence length batch
- batch-vs-single consistency
- OpenAI-compatible `/v1/embeddings` response shape
- missing input field
- empty input list
- empty string input
- invalid input element type
- unsupported OpenAI `encoding_format`
- optional max-batch-token rejection

For TEI comparison, pass `--tei-url`. The suite checks embeddings.cpp output
shape, normalization, internal batch-vs-single cosine, and optional
embeddings.cpp-vs-TEI cosine.

## Commands

Run against an embeddings.cpp server:

```bash
uv run scripts/server_compare.py \
  --cpp-url http://127.0.0.1:8080
```

Run against embeddings.cpp and TEI:

```bash
uv run scripts/server_compare.py \
  --cpp-url http://127.0.0.1:8080 \
  --tei-url http://127.0.0.1:8081
```

Run token-limit coverage with a deliberately small server limit:

```bash
python -m embeddings_cpp.server \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0 \
  --port 8080 \
  --max-batch-tokens 32

uv run scripts/server_compare.py \
  --cpp-url http://127.0.0.1:8080 \
  --expect-token-limit
```

For benchmark comparison, use `scripts/benchmark.py` after functional tests pass.

## Browser Runtime Cases

Browser correctness should cover:

- `wasm` persistent runner
- `webgpu` persistent runner
- two or more different inputs in the same persistent runner instance
- signature changes between different inputs
- cosine against the first successful baseline runtime

Current regression command on a WebGPU-capable macOS host:

```bash
EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
BASE_URL=http://127.0.0.1:18081 \
PAGE_URL=http://127.0.0.1:18081/packages/web/examples/basic-browser.html \
MODEL_URL=http://127.0.0.1:18081/models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf \
RUNTIMES=wasm,webgpu \
RUNNER_MODE=persistent \
MIN_COSINE=0.995 \
node scripts/browser_runtime_regression.mjs
```

`pthread` browser runtime remains queued behind WebGPU.
