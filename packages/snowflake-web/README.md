# @embeddings-cpp/snowflake-web

Browser-first Snowflake embedding package for
`Snowflake/snowflake-arctic-embed-m-v2.0`.

Current status:

- browser only
- one model only
- normalized `768`-dimensional output
- runtime auto-selects `webgpu`, then `pthread`, then `wasm`
- model and runtime assets load from URLs and reuse browser cache

Minimal usage:

```ts
import { createSnowflakeEmbedder } from "@embeddings-cpp/snowflake-web";

const embedder = await createSnowflakeEmbedder({
  modelUrl:
    "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf",
  runtimeBaseUrl: window.location.origin,
});

const result = await embedder.embed("你好，世界");
console.log(result.vector.length); // 768
console.log(result.runtime);       // "webgpu" | "pthread" | "wasm"

await embedder.dispose();
```

Batch usage:

```ts
const results = await embedder.embedAll([
  "How do I reset my password?",
  "请帮我总结这个工单",
]);
```

Warm the browser cache before the first interactive request:

```ts
await embedder.prefetch();
```

See [docs/SNOWFLAKE_NPM_PACKAGE.md](/home/yongsheng/repos/embeddings.cpp/docs/SNOWFLAKE_NPM_PACKAGE.md)
for the full design.
