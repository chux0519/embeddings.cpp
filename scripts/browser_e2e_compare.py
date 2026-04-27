#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf")
DEFAULT_URL = "http://127.0.0.1:18081"
DEFAULT_TEXTS = [
    "你好，世界。请把这句话编码成 embedding 向量。",
    "Please encode this sentence into an embedding vector for retrieval.",
    "请帮我 summarize this support ticket and return an embedding.",
]


def run_node(js: str, env: dict[str, str], timeout: int = 120) -> str:
    result = subprocess.run(
        ["node", "-e", js],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=timeout,
        check=True,
    )
    return result.stdout.strip()


def browser_token_ids(base_url: str, text: str) -> list[int]:
    js = r"""
const { chromium } = require('playwright');
(async() => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ serviceWorkers: 'block' });
  const page = await context.newPage();
  await page.goto(process.env.BASE_URL + '/demo/browser-wasm/index.html?e2e=tokenizer', { waitUntil: 'load', timeout: 120000 });
  const ids = await page.evaluate(async (text) => {
    const response = await fetch('/demo/browser-wasm/assets/snowflake-tokenizer.json', { cache: 'reload' });
    const json = await response.arrayBuffer();
    const tok = await window.tokenizers.Tokenizer.fromJSON(json);
    return Array.from(tok.encode(text));
  }, process.env.EMBED_TEXT);
  console.log(JSON.stringify(ids));
  await browser.close();
})();
"""
    return json.loads(run_node(js, {"BASE_URL": base_url, "EMBED_TEXT": text}).splitlines()[-1])


def browser_vector(base_url: str, batch_line: str) -> tuple[list[float], str]:
    js = r"""
const { chromium } = require('playwright');
(async() => {
  const browser = await chromium.launch({ headless: true, args: ['--enable-features=SharedArrayBuffer'] });
  const context = await browser.newContext({ serviceWorkers: 'block' });
  const page = await context.newPage();
  const url = process.env.BASE_URL + '/scripts/wasm_encode_page.html?build=build-wasm-web-dyn&model_url=' +
    encodeURIComponent(process.env.BASE_URL + '/models/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf') +
    '&pooling=cls';
  await page.goto(url, { waitUntil: 'load', timeout: 120000 });
  await page.evaluate((line) => { window.postMessage({ type: 'encode-request', batchLine: line }, '*'); }, process.env.BATCH_LINE);
  await page.waitForFunction(() => document.querySelector('#log')?.textContent.includes('\"vectors\": ['), { timeout: 120000 });
  const payload = await page.evaluate(() => {
    const text = document.querySelector('#log')?.textContent || '';
    const start = text.indexOf('{\n');
    return text.slice(start);
  });
  console.log(payload);
  await browser.close();
})();
"""
    payload = json.loads(run_node(js, {"BASE_URL": base_url, "BATCH_LINE": batch_line}))
    return payload["vectors"][0], payload["backend"]


def native_vector(model_path: str, text: str) -> list[float]:
    sys.path.insert(0, ROOT)
    from embeddings_cpp import load

    model = load(
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        gguf_path=model_path,
    )
    return model.encode(text)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def compare_text(base_url: str, model_path: str, text: str) -> dict[str, object]:
    ids = browser_token_ids(base_url, text)
    full_ids = [0, *ids, 2]
    batch_line = ",".join(map(str, full_ids)) + "\t" + ",".join(["1"] * len(full_ids))

    browser_vec, backend = browser_vector(base_url, batch_line)
    native_vec = native_vector(model_path, text)

    abs_err = [abs(x - y) for x, y in zip(browser_vec, native_vec)]
    return {
        "text": text,
        "browser_tokens_without_special": len(ids),
        "browser_tokens_with_special": len(full_ids),
        "dim": len(browser_vec),
        "browser_backend": backend,
        "cosine_browser_vs_native": cosine(browser_vec, native_vec),
        "max_abs_err": max(abs_err),
        "mean_abs_err": sum(abs_err) / len(abs_err),
        "browser_head": browser_vec[:8],
        "native_head": native_vec[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare browser Snowflake embeddings against local embeddings.cpp output.")
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", action="append", dest="texts", help="Repeat to add explicit browser e2e test inputs.")
    parser.add_argument("--min-cos", type=float, default=0.995)
    args = parser.parse_args()

    texts: Sequence[str] = args.texts or DEFAULT_TEXTS
    reports = [compare_text(args.base_url, args.model, text) for text in texts]
    print(json.dumps(reports, ensure_ascii=False, indent=2))

    failed = [r for r in reports if r["cosine_browser_vs_native"] < args.min_cos]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
