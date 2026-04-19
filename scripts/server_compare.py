#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "scripts" / "output"
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


@dataclass(frozen=True)
class Case:
    name: str
    inputs: str | list[str]
    min_cos: float = 0.96


CASES = [
    Case("single_en", "The quick brown fox jumps over the lazy dog."),
    Case("single_zh", "机器学习是人工智能的一个重要分支。"),
    Case("whitespace", "   \t  "),
    Case("mixed_batch_4", [
        "你好，今天天气怎么样？",
        "What's the weather like today?",
        "Embedding alignment should be stable in batch mode.",
        "今天天气很好，适合出去散步。",
    ]),
    Case("duplicates_batch_8", [
        "same sentence",
        "same sentence",
        "same sentence.",
        "same sentence!",
        "A cute cat....",
        "A cute cat.",
        "短文本",
        "短文本",
    ]),
    Case("length_skew_batch_4", [
        "short",
        " ".join(["long-context"] * 64),
        "符号、数字 12345 mixed tokens.",
        "newline separated text\nstill one embedding input",
    ]),
]


def as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def finite_vector(vector: list[float]) -> bool:
    return bool(vector) and all(math.isfinite(x) for x in vector)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def cosine(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def wait_ready(url: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(url.rstrip() + "/health", timeout=5)
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    return {"status": "ok"}
            last_error = response.text[-500:]
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"{url} did not become healthy within {timeout}s: {last_error}")


def post_json(url: str, path: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    response = requests.post(url.rstrip("/") + path, json=payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def embed(url: str, inputs: str | list[str], timeout: float) -> list[list[float]]:
    status, body = post_json(url, "/embed", {"inputs": inputs}, timeout)
    if status != 200:
        raise RuntimeError(f"/embed failed status={status} body={body}")
    return body


def openai_embed(url: str, inputs: str | list[str], timeout: float) -> dict[str, Any]:
    status, body = post_json(url, "/v1/embeddings", {"input": inputs, "model": "snowflake"}, timeout)
    if status != 200:
        raise RuntimeError(f"/v1/embeddings failed status={status} body={body}")
    return body


def assert_embedding_shape(vectors: list[list[float]], expected_count: int, expected_dim: int) -> None:
    if len(vectors) != expected_count:
        raise AssertionError(f"expected {expected_count} vectors, got {len(vectors)}")
    for idx, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise AssertionError(f"vector {idx} expected dim={expected_dim}, got {len(vector)}")
        if not finite_vector(vector):
            raise AssertionError(f"vector {idx} is empty or contains non-finite values")
        n = norm(vector)
        if not 0.99 <= n <= 1.01:
            raise AssertionError(f"vector {idx} is not normalized, norm={n}")


def run_case(cpp_url: str, tei_url: str | None, case: Case, dim: int, timeout: float) -> dict[str, Any]:
    texts = as_list(case.inputs)
    cpp_vectors = embed(cpp_url, case.inputs, timeout)
    assert_embedding_shape(cpp_vectors, len(texts), dim)

    single_vectors = [embed(cpp_url, text, timeout)[0] for text in texts]
    batch_cos = [cosine(a, b) for a, b in zip(cpp_vectors, single_vectors)]
    min_batch_cos = min(batch_cos) if batch_cos else 1.0
    if min_batch_cos < 0.999:
        raise AssertionError(f"batch-vs-single min cosine {min_batch_cos:.6f} below 0.999")

    row: dict[str, Any] = {
        "case": case.name,
        "status": "ok",
        "count": len(texts),
        "dim": dim,
        "cpp_batch_vs_single_min_cos": min_batch_cos,
    }

    if tei_url:
        tei_vectors = embed(tei_url, case.inputs, timeout)
        assert_embedding_shape(tei_vectors, len(texts), dim)
        tei_cos = [cosine(a, b) for a, b in zip(cpp_vectors, tei_vectors)]
        min_tei_cos = min(tei_cos) if tei_cos else 1.0
        row["cpp_vs_tei_min_cos"] = min_tei_cos
        if min_tei_cos < case.min_cos:
            raise AssertionError(f"cpp-vs-tei min cosine {min_tei_cos:.6f} below {case.min_cos}")

    return row


def run_openai_shape(cpp_url: str, dim: int, timeout: float) -> dict[str, Any]:
    body = openai_embed(cpp_url, ["hello world", "你好，世界"], timeout)
    if body.get("object") != "list":
        raise AssertionError(f"unexpected object: {body.get('object')}")
    data = body.get("data")
    if not isinstance(data, list) or len(data) != 2:
        raise AssertionError("OpenAI response data must contain two embeddings")
    vectors = [item["embedding"] for item in data]
    assert_embedding_shape(vectors, 2, dim)
    if [item.get("index") for item in data] != [0, 1]:
        raise AssertionError("OpenAI response indexes are not stable")
    return {"case": "openai_batch_shape", "status": "ok", "count": 2, "dim": dim}


def run_error_cases(cpp_url: str, timeout: float) -> list[dict[str, Any]]:
    cases = [
        ("missing_inputs", "/embed", {}, {422}),
        ("empty_inputs", "/embed", {"inputs": []}, {422}),
        ("empty_string", "/embed", {"inputs": [""]}, {400}),
        ("invalid_inputs", "/embed", {"inputs": [123]}, {422}),
        ("bad_encoding_format", "/v1/embeddings", {"input": "hello", "encoding_format": "base64"}, {400}),
    ]
    rows = []
    for name, path, payload, expected in cases:
        status, body = post_json(cpp_url, path, payload, timeout)
        if status not in expected:
            raise AssertionError(f"{name} expected status in {expected}, got {status}, body={body}")
        rows.append({"case": name, "status": "ok", "http_status": status})
    return rows


def run_token_limit_case(cpp_url: str, timeout: float) -> dict[str, Any]:
    long_text = " ".join(["token-limit"] * 1024)
    status, body = post_json(cpp_url, "/embed", {"inputs": [long_text]}, timeout)
    if status != 413:
        raise AssertionError(f"token limit case expected 413, got {status}, body={body}")
    return {"case": "max_batch_tokens", "status": "ok", "http_status": status}


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"server_compare_{stamp}.json"
    md_path = output_dir / f"server_compare_{stamp}.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Server Compare\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| Case | Status | Count | Dim | Batch min cos | TEI min cos | HTTP |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('case', '')} | {row.get('status', '')} | "
                f"{row.get('count', '')} | {row.get('dim', '')} | "
                f"{float(row.get('cpp_batch_vs_single_min_cos', 0.0)):.6f} | "
                f"{float(row.get('cpp_vs_tei_min_cos', 0.0)):.6f} | "
                f"{row.get('http_status', '')} |\n"
            )
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare embeddings.cpp HTTP server behavior against TEI-compatible expectations.")
    parser.add_argument("--cpp-url", default="http://127.0.0.1:8080")
    parser.add_argument("--tei-url", help="Optional TEI URL for cosine comparison.")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--expect-token-limit", action="store_true", help="Expect the cpp server to reject a large request with 413.")
    parser.add_argument("--only-token-limit", action="store_true", help="Only run health and max-batch-token rejection checks.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    rows.append({"case": "cpp_health", "status": "ok", "detail": wait_ready(args.cpp_url, args.ready_timeout)})
    if args.tei_url:
        rows.append({"case": "tei_health", "status": "ok", "detail": wait_ready(args.tei_url, args.ready_timeout)})

    if not args.only_token_limit:
        for case in CASES:
            try:
                rows.append(run_case(args.cpp_url, args.tei_url, case, args.dim, args.timeout))
            except Exception as exc:
                rows.append({"case": case.name, "status": "error", "error": str(exc)})

        for runner in (run_openai_shape,):
            try:
                rows.append(runner(args.cpp_url, args.dim, args.timeout))
            except Exception as exc:
                rows.append({"case": runner.__name__, "status": "error", "error": str(exc)})

        try:
            rows.extend(run_error_cases(args.cpp_url, args.timeout))
        except Exception as exc:
            rows.append({"case": "error_cases", "status": "error", "error": str(exc)})

    if args.expect_token_limit:
        try:
            rows.append(run_token_limit_case(args.cpp_url, args.timeout))
        except Exception as exc:
            rows.append({"case": "max_batch_tokens", "status": "error", "error": str(exc)})

    json_path, md_path = write_outputs(rows, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        print(json.dumps(failed, indent=2, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
