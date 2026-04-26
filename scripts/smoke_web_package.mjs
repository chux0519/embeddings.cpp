import { chromium } from "playwright";

const baseUrl = process.env.BASE_URL || "http://127.0.0.1:18081";

async function main() {
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-features=SharedArrayBuffer"],
  });

  const context = await browser.newContext({ serviceWorkers: "block" });
  const page = await context.newPage();

  await page.goto(`${baseUrl}/packages/web/examples/basic-browser.html?smoke=1`, {
    waitUntil: "load",
    timeout: 120000,
  });

  const result = await page.evaluate(async () => {
    const mod = await import("/packages/web/dist/index.js");
    const embedder = await mod.createSnowflakeEmbedder({
      modelUrl: `${window.location.origin}/models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`,
      runtimeBaseUrl: window.location.origin,
      runtime: window.crossOriginIsolated ? "pthread" : "wasm",
      threads: 8,
      cache: true,
    });
    const output = await embedder.embed("你好，世界。请把这句话编码成 embedding 向量。");
    await embedder.dispose();
    return {
      status: "Done",
      runtime: output.runtime,
      tokenCount: String(output.tokenCount),
      dimension: String(output.vector.length),
      preview: Array.from(output.vector.slice(0, 16)).map((value) => value.toFixed(6)).join(", "),
    };
  });

  console.log(JSON.stringify(result, null, 2));

  if (result.status !== "Done") {
    throw new Error(`unexpected smoke status: ${result.status}`);
  }
  if (result.dimension !== "768") {
    throw new Error(`unexpected embedding dimension: ${result.dimension}`);
  }
  if (!result.preview.trim()) {
    throw new Error("embedding preview is empty");
  }

  await browser.close();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
