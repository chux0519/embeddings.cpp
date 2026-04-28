# Snowflake npm Package

## Goal

Ship a browser-first npm package for `Snowflake/snowflake-arctic-embed-m-v2.0`
that wraps the existing browser runtime behind a small API.

Version 1 should optimize for:

- one model only
- browser only
- short query workloads
- good defaults
- no framework requirement

It should **not** ship the GGUF inside the npm tarball. The package should ship
the TypeScript wrapper only, while the model and browser runtime artifacts are
loaded from URLs and cached in the browser.

## Package Shape

Suggested package name:

- `@embeddings-cpp/web`

Version 1 scope:

- runtime: browser only
- model: `snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`
- pooling: `cls`
- output: normalized `768`-dimensional vectors

Out of scope for v1:

- generic multi-model registry
- Node.js runtime
- worker pools
- pthread browser runtime until it is redesigned as a worker/proxy runner
- reranking / cross-encoder APIs
- custom tokenizer replacement

## Friendly API

The public API should stay narrow:

```ts
import { createSnowflakeEmbedder } from "@embeddings-cpp/web";

const embedder = await createSnowflakeEmbedder({
  cache: true,
});

const one = await embedder.embed("你好，世界");
const many = await embedder.embedAll([
  "How do I reset my password?",
  "请帮我总结这个工单",
]);

console.log(one.vector.length); // 768
console.log(one.runtime);       // "wasm" | "webgpu"

await embedder.dispose();
```

## Proposed API Surface

```ts
export type SnowflakeRuntime = "auto" | "webgpu" | "wasm";

export interface SnowflakeEmbedderOptions {
  modelUrl?: string;
  runtime?: SnowflakeRuntime;
  runtimeBaseUrl?: string;
  tokenizerUrl?: string;
  tokenizerScriptUrl?: string;
  cache?: boolean;
}

export interface SnowflakeEmbedding {
  vector: Float32Array;
  tokenCount: number;
  runtime: Exclude<SnowflakeRuntime, "auto">;
}

export interface SnowflakeEmbedderInfo {
  modelUrl: string;
  runtime: Exclude<SnowflakeRuntime, "auto"> | "pending";
  dim: 768;
  normalized: true;
}

export interface SnowflakeEmbedder {
  prefetch(): Promise<void>;
  embed(text: string): Promise<SnowflakeEmbedding>;
  embedAll(texts: string[]): Promise<SnowflakeEmbedding[]>;
  info(): SnowflakeEmbedderInfo;
  dispose(): Promise<void>;
}

export declare function createSnowflakeEmbedder(
  options: SnowflakeEmbedderOptions,
): Promise<SnowflakeEmbedder>;
```

## Why This Shape

`embed()` and `embedAll()` are the only methods most users need. That keeps the
package easy to adopt and leaves batching as an implementation detail.

`runtime: "auto"` should be the default:

- use stable single-thread `wasm`
- allow explicit `webgpu` for experimental testing

`prefetch()` should let applications warm the browser cache before the first
interactive request.

Current runtime status:

| Runtime | Status | Notes |
|---|---|---|
| `wasm` | Recommended default | Best current browser path for short Snowflake queries and the least fragile browser integration. |
| `webgpu` | Experimental | Correctness is covered, but Snowflake custom ggml ops still fall back to CPU, so it can be slower until dedicated WebGPU kernels are implemented. |
| `pthread` | Not exposed in v1 | The exported-function iframe runner can block the page. Re-enable only after a worker/proxy runner passes browser regression tests. |

By default, `modelUrl`, `runtimeBaseUrl`, `tokenizerUrl`, and
`tokenizerScriptUrl` point to the Snowflake Hugging Face artifact repository.

`runtimeBaseUrl` should point to a hosted directory that contains:

- `embedding_wasm_model_encode.js`
- `embedding_wasm_model_encode.wasm`
- `snowflake-tokenizer.json`
- helper assets if needed

That makes publishing simpler than bundling tens or hundreds of megabytes into
the npm artifact.

## Hosting Strategy

The npm package should stay small. Publish assets separately:

1. npm package:
   - wrapper code
   - TypeScript types
   - small helper glue
2. Hugging Face or GitHub Releases:
   - GGUF model
   - browser wasm/webgpu runtime artifacts
   - tokenizer JSON

Version 1 can assume these defaults:

- model on Hugging Face
- runtime assets on Hugging Face under `browser/<web_asset_version>/`

Current default asset base:

```text
https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/browser/v0.1.1/
```

The upload workflow writes:

- `web-assets.json`
- `browser/<web_asset_version>/scripts/wasm_encode_page.html`
- `browser/<web_asset_version>/scripts/wasm_persistent_encode_page.html`
- `browser/<web_asset_version>/build-wasm-web-dyn/*`
- `browser/<web_asset_version>/build-wasm-webgpu-browser-dyn/*`
- tokenizer JSON and `web-tokenizers.js`

Current repo layout:

- package source: `packages/web/`

## Default Behavior

`createSnowflakeEmbedder()` should behave like this:

1. detect best runtime
2. lazy-load tokenizer
3. lazy-load runtime assets
4. download or reuse cached GGUF
5. add `[CLS]` and `[SEP]` automatically
6. return normalized `Float32Array`

The user should not need to know:

- special tokens
- batch file formats
- iframe messaging
- wasm build directories

## Browser Cache Behavior

Version 1 should support:

- `Cache Storage` for runtime files and GGUF
- explicit `prefetch()`
- reuse on page reload

Later, if the package becomes a primary product surface, move model storage to:

- `OPFS`, or
- `IndexedDB`

## Packaging Notes

Keep the implementation browser-first:

- ESM only
- no framework adapters in v1
- avoid Node-only APIs

Recommended package files:

- `src/index.ts`
- `src/types.ts`
- `src/defaults.ts`
- `src/runtime.ts`
- `src/tokenizer.ts`
- `src/cache.ts`
- `src/embedder.ts`

## Release Plan

Recommended order:

1. land the package scaffold and API types
2. implement `embed()` for one text
3. implement `embedAll()` with small internal batching
4. implement `prefetch()`
5. publish assets
6. publish npm package

## Compatibility Promise for v1

Keep the first release strict:

- one model
- one output dimension
- one pooling mode
- one normalization behavior

That gives a stable contract. Multi-model support should come later as a
separate registry layer, not by widening the first API too early.
