#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec


OUTPUT_DIR = ROOT / "scripts" / "output"
TEXTS = [
    "你好，今天天气怎么样？",
    "What's the weather like today?",
    "Embedding alignment should be stable in batch mode.",
    "今天天气很好，适合出去散步。",
    "A short English sentence for embedding throughput.",
    "这是一个用于批量推理测试的中文句子。",
    "Repeated text should not change batch correctness.",
    "The quick brown fox jumps over the lazy dog.",
]


def make_texts(batch_size: int) -> list[str]:
    return (TEXTS * ((batch_size + len(TEXTS) - 1) // len(TEXTS)))[:batch_size]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def rss_mb() -> float:
    import psutil

    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / 1024 / 1024


def run_local(args: argparse.Namespace) -> dict[str, Any]:
    from embeddings_cpp import load

    spec = get_model_spec(args.model_id)
    model = load(spec.model_id, gguf_path=args.gguf_path)
    texts = make_texts(args.batch_size)

    def encode() -> None:
        model.batch_encode(texts)

    for _ in range(args.warmup):
        encode()

    timings = []
    peak_rss = rss_mb()
    for _ in range(args.iterations):
        start = time.perf_counter()
        encode()
        timings.append(time.perf_counter() - start)
        peak_rss = max(peak_rss, rss_mb())

    return summarize("embeddings_cpp", args, timings, peak_rss)


def run_http(args: argparse.Namespace) -> dict[str, Any]:
    import requests

    texts = make_texts(args.batch_size)
    url = args.url.rstrip("/") + "/embed"

    def encode() -> None:
        response = requests.post(url, json={"inputs": texts}, timeout=args.timeout)
        response.raise_for_status()
        vectors = response.json()
        if len(vectors) != len(texts):
            raise RuntimeError(f"expected {len(texts)} vectors, got {len(vectors)}")

    for _ in range(args.warmup):
        encode()

    timings = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        encode()
        timings.append(time.perf_counter() - start)

    return summarize("http", args, timings, 0.0)


def summarize(runner: str, args: argparse.Namespace, timings: list[float], peak_rss: float) -> dict[str, Any]:
    mean = statistics.fmean(timings)
    return {
        "runner": runner,
        "model_id": args.model_id,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "latency_ms_mean": mean * 1000,
        "latency_ms_p50": percentile(timings, 50) * 1000,
        "latency_ms_p95": percentile(timings, 95) * 1000,
        "texts_per_second": args.batch_size / mean,
        "rss_mb": peak_rss,
    }


def run_child(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--runner",
        args.runner,
        "--model-id",
        args.model_id,
        "--batch-size",
        str(batch_size),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--timeout",
        str(args.timeout),
    ]
    if args.gguf_path:
        cmd.extend(["--gguf-path", args.gguf_path])
    if args.url:
        cmd.extend(["--url", args.url])
    worker_timeout = args.worker_timeout
    if worker_timeout is None:
        worker_timeout = max(30.0, args.timeout * (args.warmup + args.iterations + 1))

    try:
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=worker_timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "runner": args.runner,
            "model_id": args.model_id,
            "batch_size": batch_size,
            "status": "error",
            "stderr": f"worker exceeded deadline of {worker_timeout:.1f}s",
            "stdout": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
        }
    if completed.returncode != 0:
        return {
            "runner": args.runner,
            "model_id": args.model_id,
            "batch_size": batch_size,
            "status": "error",
            "stderr": completed.stderr[-4000:],
            "stdout": completed.stdout[-2000:],
        }
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{") and line.endswith("}"):
            row = json.loads(line)
            row["status"] = "ok"
            return row
    return {"runner": args.runner, "batch_size": batch_size, "status": "error", "stdout": completed.stdout[-2000:]}


def write_report(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{stamp}.json"
    md_path = output_dir / f"benchmark_{stamp}.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Embeddings Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| Runner | Model | Batch | Status | Mean ms | P50 ms | P95 ms | Text/s | RSS MB |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('runner', '')} | {row.get('model_id', '')} | {row.get('batch_size', '')} | {row.get('status', '')} | "
                f"{float(row.get('latency_ms_mean', 0.0)):.2f} | "
                f"{float(row.get('latency_ms_p50', 0.0)):.2f} | "
                f"{float(row.get('latency_ms_p95', 0.0)):.2f} | "
                f"{float(row.get('texts_per_second', 0.0)):.2f} | "
                f"{float(row.get('rss_mb', 0.0)):.1f} |\n"
            )
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark embeddings.cpp local or HTTP server inference.")
    parser.add_argument("--model-id", default="Snowflake/snowflake-arctic-embed-m-v2.0")
    parser.add_argument("--runner", choices=("local", "http"), default="local")
    parser.add_argument("--gguf-path")
    parser.add_argument("--url", default="http://127.0.0.1:80")
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0, help="Per HTTP request timeout in seconds.")
    parser.add_argument(
        "--worker-timeout",
        type=float,
        help="Total worker subprocess deadline in seconds. Defaults to request timeout scaled by warmup+iterations.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        row = run_local(args) if args.runner == "local" else run_http(args)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return 0

    spec = get_model_spec(args.model_id)
    batch_sizes = args.batch_sizes or spec.benchmark.get("batch_sizes", [1, 2, 4, 8])
    rows = [run_child(args, batch_size) for batch_size in batch_sizes]
    json_path, md_path = write_report(rows, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
