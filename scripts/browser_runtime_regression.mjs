import { chromium } from "playwright";

const baseUrl = process.env.BASE_URL || "https://emb.potafree.net";
const packageUrl =
  process.env.PACKAGE_URL || `${baseUrl}/packages/web/dist/index.js?v=webpkg20`;
const pageUrl =
  process.env.PAGE_URL || `${baseUrl}/packages/web/examples/basic-browser.html`;
const modelUrl =
  process.env.MODEL_URL ||
  `${baseUrl}/models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf`;
const executablePath = process.env.EXECUTABLE_PATH || undefined;
const runnerMode = process.env.RUNNER_MODE || "ephemeral";
const timeoutMs = Number.parseInt(process.env.TIMEOUT_MS || "180000", 10);
const runtimes = (process.env.RUNTIMES || "wasm,webgpu,pthread")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const texts = [
  process.env.TEXT_A || "hello world",
  process.env.TEXT_B || "你好世界",
];

function signature(values) {
  let acc = 2166136261 >>> 0;
  for (const value of values) {
    const scaled = Math.round(value * 1e6);
    acc ^= scaled >>> 0;
    acc = Math.imul(acc, 16777619) >>> 0;
  }
  return acc.toString(16).padStart(8, "0");
}

async function runRuntime(page, runtime) {
  return page.evaluate(
    async ({ packageUrl, modelUrl, runtime, runnerMode, texts, timeoutMs }) => {
      const mod = await import(packageUrl);
      const trail = [];
      const embedder = await mod.createSnowflakeEmbedder({
        modelUrl,
        runtime,
        runnerMode,
        runtimeBaseUrl: new URL(modelUrl).origin,
        cache: true,
        onStatus(event) {
          trail.push({
            stage: event.stage,
            detail: event.detail || event.runtime || "",
            atMs: event.atMs || performance.now(),
          });
        },
      });

      function sig(values) {
        let acc = 2166136261 >>> 0;
        for (const value of values) {
          const scaled = Math.round(value * 1e6);
          acc ^= scaled >>> 0;
          acc = Math.imul(acc, 16777619) >>> 0;
        }
        return acc.toString(16).padStart(8, "0");
      }

      async function withTimeout(promise, label) {
        return Promise.race([
          promise,
          new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`${label} timed out`)), timeoutMs);
          }),
        ]);
      }

      try {
        const outputs = [];
        for (const text of texts) {
          const result = await withTimeout(embedder.embed(text), `embed ${runtime}`);
          const vector = Array.from(result.vector);
          outputs.push({
            text,
            runtime: result.runtime,
            runnerMode: result.runnerMode,
            tokenCount: result.tokenCount,
            dimension: vector.length,
            sig: sig(vector),
            preview: vector.slice(0, 8),
          });
        }
        await embedder.dispose();
        return {
          ok: true,
          runtime,
          runnerMode,
          outputs,
          sigChanged: outputs.length >= 2 && outputs[0].sig !== outputs[1].sig,
          lastStatuses: trail.slice(-20),
        };
      } catch (error) {
        await embedder.dispose().catch(() => {});
        return {
          ok: false,
          runtime,
          runnerMode,
          error: String(error && error.message ? error.message : error),
          lastStatuses: trail.slice(-30),
        };
      }
    },
    { packageUrl, modelUrl, runtime, runnerMode, texts, timeoutMs },
  );
}

async function runRuntimeWithOuterTimeout(runtime) {
  const browser = await chromium.launch({
    headless: true,
    executablePath,
    args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
  });

  try {
    const context = await browser.newContext({ serviceWorkers: "block" });
    const page = await context.newPage();
    const consoleMessages = [];
    page.on("console", (message) => {
      consoleMessages.push({
        type: message.type(),
        text: message.text(),
      });
      if (consoleMessages.length > 80) {
        consoleMessages.shift();
      }
    });
    page.on("pageerror", (error) => {
      consoleMessages.push({
        type: "pageerror",
        text: String(error && error.stack ? error.stack : error),
      });
      if (consoleMessages.length > 80) {
        consoleMessages.shift();
      }
    });
    await page.goto(pageUrl, { waitUntil: "load", timeout: timeoutMs });

    const result = await Promise.race([
      runRuntime(page, runtime),
      new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            ok: false,
            runtime,
            runnerMode,
            error: `playwright outer timeout after ${timeoutMs}ms`,
            lastStatuses: [],
          });
        }, timeoutMs);
      }),
    ]);
    return {
      ...result,
      consoleMessages: consoleMessages.slice(-30),
    };
  } finally {
    await browser.close().catch(() => {});
  }
}

const results = [];
for (const runtime of runtimes) {
  const result = await runRuntimeWithOuterTimeout(runtime);
  results.push(result);
  console.log(JSON.stringify(result, null, 2));
}

const failed = results.filter(
  (result) =>
    !result.ok ||
    !result.outputs?.every((output) => output.dimension === 768) ||
    !result.sigChanged,
);
if (failed.length > 0) {
  process.exitCode = 1;
}
