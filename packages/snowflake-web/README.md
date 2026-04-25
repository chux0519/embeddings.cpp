# @embeddings-cpp/snowflake-web

Browser-first Snowflake embedding package scaffold.

The intended public API is:

```ts
import { createSnowflakeEmbedder } from "@embeddings-cpp/snowflake-web";

const embedder = await createSnowflakeEmbedder({
  modelUrl:
    "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf",
});

const result = await embedder.embed("你好，世界");
console.log(result.vector.length); // 768
```

Version 1 design constraints:

- browser only
- one model only
- runtime auto-selects `webgpu`, then `pthread`, then wasm
- normalized `768`-dimensional output

See [docs/SNOWFLAKE_NPM_PACKAGE.md](/home/yongsheng/repos/embeddings.cpp/docs/SNOWFLAKE_NPM_PACKAGE.md)
for the full design.
