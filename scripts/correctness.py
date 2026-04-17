#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec

def main() -> int:
    parser = argparse.ArgumentParser(description="Run registry-driven embeddings.cpp correctness checks.")
    parser.add_argument("--model-id", default="Snowflake/snowflake-arctic-embed-m-v2.0")
    parser.add_argument("--quantizations", nargs="+", default=None)
    parser.add_argument("--convert-missing", action="store_true")
    parser.add_argument("--with-tei", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    spec = get_model_spec(args.model_id)
    quantizations = args.quantizations or [spec.artifact_file.removesuffix(".gguf").removeprefix(f"{spec.slug}.")]
    min_cos = str(spec.correctness.get("python_min_cos", 0.999))
    batch_min_cos = str(spec.correctness.get("batch_min_cos", 0.999999))

    cmd = [
        sys.executable,
        "scripts/alignment.py",
        "--models",
        spec.model_id,
        "--quantizations",
        *quantizations,
        "--min-cos",
        min_cos,
        "--batch-min-cos",
        batch_min_cos,
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
    ]
    if spec.benchmark.get("default_threads"):
        cmd.extend(["--cpp-threads", str(spec.benchmark["default_threads"])])
    if args.convert_missing:
        cmd.append("--convert-missing")
    if args.with_tei:
        cmd.append("--tei-start")
    if args.benchmark:
        cmd.append("--benchmark")
        cmd.extend(["--benchmark-batches", *[str(x) for x in spec.benchmark.get("batch_sizes", [1, 2, 4, 8])]])

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
