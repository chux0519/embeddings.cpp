#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/Users/yongsheng/repos/embeddings.cpp-macmini}"
PORT="${PORT:-18081}"

source /tmp/emsdk/emsdk_env.sh >/dev/null 2>&1

cd "$ROOT"
python3 scripts/browser_wasm_bench_server.py --host 127.0.0.1 --port "$PORT" --root "$ROOT" >/tmp/emb-browser.log 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT
sleep 2

TMP_SCRIPT="$ROOT/.tmp-run-browser-matrix.mjs"
cat > "$TMP_SCRIPT" <<'JS'
import { chromium } from 'playwright';
const executablePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function run(url) {
  const browser = await chromium.launch({
    headless: true,
    executablePath,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
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

(async () => {
  const base = 'http://127.0.0.1:18081/scripts/wasm_bench_page.html';
  const cases = [
    ['mixed8-single', `${base}?build=build-wasm-web&batch=/snowflake_wasm_batch.txt&warmup=1&iterations=3`],
    ['mixed8-pthread8', `${base}?build=build-wasm-web-pthread&batch=/snowflake_wasm_batch.txt&warmup=1&iterations=3&threads=8`],
    ['mixed8-webgpu', `${base}?build=build-wasm-webgpu-browser&backend=webgpu&batch=/snowflake_wasm_batch.txt&warmup=1&iterations=3`],
    ['short8-single', `${base}?build=build-wasm-web&batch=/snowflake_wasm_short8.txt&warmup=1&iterations=3`],
    ['short8-pthread8', `${base}?build=build-wasm-web-pthread&batch=/snowflake_wasm_short8.txt&warmup=1&iterations=3&threads=8`],
    ['short8-webgpu', `${base}?build=build-wasm-webgpu-browser&backend=webgpu&batch=/snowflake_wasm_short8.txt&warmup=1&iterations=3`],
  ];
  for (const [name, url] of cases) {
    const payload = await run(url);
    console.log(JSON.stringify({ name, ...payload }, null, 2));
  }
})();
JS

"$EMSDK_NODE" "$TMP_SCRIPT"
rm -f "$TMP_SCRIPT"
