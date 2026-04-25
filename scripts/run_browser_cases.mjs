import { chromium } from 'playwright';

const executablePath = process.env.EXECUTABLE_PATH || undefined;
const base = process.env.BASE_URL || 'http://127.0.0.1:18081/scripts/wasm_bench_page.html';
const mode = process.env.MODE || 'bundled';
const threads = process.env.THREADS || '8';
const modelUrl = process.env.MODEL_URL || 'http://127.0.0.1:18081/models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf';
const batchUrl = process.env.BATCH_URL || 'http://127.0.0.1:18081/scripts/data/snowflake_wasm_batch1.txt';
const defaultArgs = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];

function buildDir(runtime) {
  if (mode === 'downloaded') {
    if (runtime === 'single') return 'build-wasm-web-dyn';
    if (runtime === 'pthread') return 'build-wasm-web-pthread-dyn';
    if (runtime === 'webgpu') return 'build-wasm-webgpu-browser-dyn';
  }
  if (runtime === 'single') return 'build-wasm-web';
  if (runtime === 'pthread') return 'build-wasm-web-pthread';
  if (runtime === 'webgpu') return 'build-wasm-webgpu-browser';
  throw new Error(`unknown runtime: ${runtime}`);
}

function caseUrl(runtime) {
  const params = new URLSearchParams({
    build: buildDir(runtime),
    batch: '/snowflake_wasm_batch1.txt',
    warmup: '1',
    iterations: '3',
  });
  if (runtime === 'pthread') {
    params.set('threads', threads);
  }
  if (runtime === 'webgpu') {
    params.set('backend', 'webgpu');
  }
  if (mode === 'downloaded') {
    params.set('model_url', modelUrl);
    params.set('batch_url', batchUrl);
  }
  return `${base}?${params.toString()}`;
}

const cases = [
  [`${mode}-single`, caseUrl('single')],
  [`${mode}-pthread${threads}`, caseUrl('pthread')],
  [`${mode}-webgpu`, caseUrl('webgpu')],
];

async function run(url, args = []) {
  const browser = await chromium.launch({
    headless: true,
    executablePath,
    args,
  });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  await page.waitForFunction(
    () => window.__benchResult || window.__benchError || document.querySelector('#status')?.textContent === 'abort',
    null,
    { timeout: 120000 },
  );
  const payload = await page.evaluate(() => ({
    status: document.getElementById('status')?.textContent || '',
    result: window.__benchResult || null,
    error: window.__benchError || null,
    log: document.getElementById('log')?.textContent || '',
  }));
  await browser.close();
  return payload;
}

for (const [name, url] of cases) {
  const payload = await run(url, defaultArgs);
  console.log(JSON.stringify({ name, ...payload }, null, 2));
}
