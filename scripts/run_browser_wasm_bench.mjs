import { chromium } from 'playwright';

const url = process.argv[2] || 'http://127.0.0.1:18081/scripts/wasm_bench_page.html';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

page.on('console', (msg) => {
  console.log(`[browser:${msg.type()}] ${msg.text()}`);
});

await page.goto(url, { waitUntil: 'load', timeout: 0 });

await page.waitForFunction(() => window.__benchResult || window.__benchError, null, { timeout: 0 });

const payload = await page.evaluate(() => ({
  result: window.__benchResult || null,
  error: window.__benchError || null,
  status: document.getElementById('status')?.textContent || '',
  log: document.getElementById('log')?.textContent || '',
}));

console.log(JSON.stringify(payload, null, 2));

await browser.close();
