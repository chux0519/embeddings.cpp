import { chromium } from 'playwright';

const executablePath = process.env.EXECUTABLE_PATH || undefined;
const base = process.env.BASE_URL || 'http://127.0.0.1:18081/scripts/wasm_bench_page.html';
const defaultArgs = ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'];

const cases = [
  ['single', `${base}?build=build-wasm-web&batch=/snowflake_wasm_batch1.txt&warmup=1&iterations=3`],
  ['pthread8', `${base}?build=build-wasm-web-pthread&batch=/snowflake_wasm_batch1.txt&warmup=1&iterations=3&threads=8`],
  ['webgpu', `${base}?build=build-wasm-webgpu-browser&backend=webgpu&batch=/snowflake_wasm_batch1.txt&warmup=1&iterations=3`],
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
