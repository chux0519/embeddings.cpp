# @embeddings-cpp/web

Browser-first Snowflake embedding package for
`Snowflake/snowflake-arctic-embed-m-v2.0`.

Current status:

- browser only
- one model only
- normalized `768`-dimensional output
- runtime auto-selects stable single-thread `wasm`
- explicit `webgpu` is experimental for Snowflake
- model and runtime assets load from Hugging Face by default and reuse browser cache

Runtime status:

| Runtime | Status | Notes |
|---|---|---|
| `wasm` | Recommended default | Best current browser path for short Snowflake queries. |
| `webgpu` | Experimental | The Snowflake custom ggml ops still fall back to CPU, so it can be slower until dedicated kernels land. |
| `pthread` | Not exposed | The current pthread runner can block the browser page and needs a worker/proxy redesign before release. |

Repo example:

- [packages/web/examples/demo.html](/home/yongsheng/repos/embeddings.cpp/packages/web/examples/demo.html)
- [packages/web/examples/basic-browser.html](/home/yongsheng/repos/embeddings.cpp/packages/web/examples/basic-browser.html)
- [packages/web/examples/mobile-diagnostics.html](/home/yongsheng/repos/embeddings.cpp/packages/web/examples/mobile-diagnostics.html)

Minimal usage:

```ts
import { createSnowflakeEmbedder } from "@embeddings-cpp/web";

const embedder = await createSnowflakeEmbedder({
  // Optional. Defaults to the published Snowflake GGUF and browser assets on Hugging Face.
  cache: true,
});

const result = await embedder.embed("你好，世界");
console.log(result.vector.length); // 768
console.log(result.runtime);       // "wasm" | "webgpu"

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

The default assets are expected at:

```text
https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/browser/webpkg22/
```

The repo workflow `.github/workflows/upload-web-assets-to-hf.yml` publishes
that directory and writes `web-assets.json` alongside it.

Local smoke test:

```bash
python3 scripts/browser_wasm_bench_server.py --host 127.0.0.1 --port 18081 --root "$PWD"
npm install --no-save playwright
node scripts/smoke_web_package.mjs
```

Usable browser demo:

```text
http://<host>:18081/packages/web/examples/demo.html
```

Mobile diagnostics page:

```text
http://<host>:18081/packages/web/examples/mobile-diagnostics.html
```

Add `?autorun=1` to run one embedding request on load and emit a compact JSON report.

See [docs/SNOWFLAKE_NPM_PACKAGE.md](/home/yongsheng/repos/embeddings.cpp/docs/SNOWFLAKE_NPM_PACKAGE.md)
for the full design.
