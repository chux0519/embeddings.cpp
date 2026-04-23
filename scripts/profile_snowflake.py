#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "scripts" / "output"
DEFAULT_MODEL = ROOT / "models" / "snowflake-arctic-embed-m-v2.0.q8_0.gguf"
DEFAULT_REPO_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
TEXTS = [
    "你好，今天天气怎么样？",
    "What's the weather like today?",
    "Embedding alignment should be stable in batch mode.",
    "今天天气很好，适合出去散步。",
]


def make_texts(batch_size: int) -> list[str]:
    return (TEXTS * ((batch_size + len(TEXTS) - 1) // len(TEXTS)))[:batch_size]


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


def worker(args: argparse.Namespace) -> int:
    import numpy as np

    threads = args.threads[0] if isinstance(args.threads, list) else args.threads
    texts = make_texts(args.batch_size)

    if args.runner == "embeddings_cpp":
        from embeddings_cpp import load

        os.environ["EMBEDDINGS_CPP_THREADS"] = str(threads)
        model = load(args.repo_id, gguf_path=str(args.model))

        def encode() -> None:
            model.batch_encode(texts, True)

    elif args.runner == "python_cpu":
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.repo_id,
            trust_remote_code=True,
            add_pooling_layer=False,
            use_memory_efficient_attention=False,
        )
        model.eval()
        model.to("cpu")

        def encode() -> None:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=8192)
            batch_size, seq_length = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                emb = hidden[:, 0]
                torch.nn.functional.normalize(emb, p=2, dim=1)

    else:
        raise ValueError(f"unsupported runner: {args.runner}")

    for _ in range(args.warmup):
        encode()

    timings: list[float] = []
    peak_rss = rss_mb()
    for _ in range(args.iterations):
        start = time.perf_counter()
        encode()
        timings.append(time.perf_counter() - start)
        peak_rss = max(peak_rss, rss_mb())

    arr = np.asarray(timings, dtype=np.float64)
    print(
        json.dumps(
            {
                "effective_threads": threads,
                "threads": threads,
                "runner": args.runner,
                "batch_size": args.batch_size,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "latency_ms_mean": float(arr.mean() * 1000),
                "latency_ms_p50": float(np.percentile(arr, 50) * 1000),
                "latency_ms_p95": float(np.percentile(arr, 95) * 1000),
                "texts_per_second": float(args.batch_size / arr.mean()),
                "rss_mb": float(peak_rss),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def run_child(args: argparse.Namespace, threads: int, batch_size: int) -> dict:
    env = os.environ.copy()
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["EMBEDDINGS_CPP_THREADS"] = str(threads)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(args.model),
        "--repo-id",
        args.repo_id,
        "--runner",
        args.runner,
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "threads": threads,
            "batch_size": batch_size,
            "status": "error",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
        }
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            row = json.loads(line)
            row["status"] = "ok"
            return row
    return {
        "threads": threads,
        "batch_size": batch_size,
        "status": "error",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-4000:],
    }


def write_outputs(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"snowflake_profile_{timestamp}.json"
    md_path = output_dir / f"snowflake_profile_{timestamp}.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Snowflake Profile\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| Runner | Threads | Batch | Status | Mean ms | P50 ms | P95 ms | Text/s | RSS MB |\n")
        f.write("|---|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('runner', '')} | {row.get('threads', '')} | {row.get('batch_size', '')} | {row.get('status', '')} | "
                f"{float(row.get('latency_ms_mean', 0.0)):.2f} | "
                f"{float(row.get('latency_ms_p50', 0.0)):.2f} | "
                f"{float(row.get('latency_ms_p95', 0.0)):.2f} | "
                f"{float(row.get('texts_per_second', 0.0)):.2f} | "
                f"{float(row.get('rss_mb', 0.0)):.1f} |\n"
            )
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible Snowflake CPU profiling for embeddings.cpp.")
    parser.add_argument("--runner", choices=("embeddings_cpp", "python_cpu"), default="embeddings_cpp")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10, 12, 16])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--batch-size", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)

    rows: list[dict] = []
    for batch_size in args.batch_sizes:
        for threads in args.threads:
            print(f"profiling runner={args.runner} batch={batch_size} threads={threads}", file=sys.stderr, flush=True)
            row = run_child(args, threads, batch_size)
            rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"  {row['texts_per_second']:.2f} text/s, {row['latency_ms_mean']:.2f} ms, {row['rss_mb']:.1f} MB",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"  error: {row.get('stderr', row.get('stdout', ''))[-400:]}", file=sys.stderr, flush=True)

    json_path, md_path = write_outputs(rows, args.output_dir)
    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
