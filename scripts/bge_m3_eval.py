#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_GGUF = ROOT / "models" / "bge-m3.fp16.gguf"
OUTPUT_DIR = ROOT / "scripts" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for BGE-M3 benchmarks. New benchmark work should use "
            "scripts/model_bench.py directly."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gguf-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--convert-missing", action="store_true")
    parser.add_argument("--quantizations", nargs="+", default=["fp16"])
    parser.add_argument("--quantize-missing", action="store_true")
    parser.add_argument("--quantized-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--quantize-bin", type=Path, default=ROOT / "build" / "quantize")
    parser.add_argument("--repack-modes", nargs="+", choices=("default", "on", "off"), default=["default"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--min-cos", type=float)
    parser.add_argument("--batch-min-cos", type=float)
    parser.add_argument("--quantized-batch-min-cos", type=float)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--cpp-threads", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "model_bench.py"),
        "--models",
        args.model_id,
        "--runners",
        "python_cpu",
        "embeddings_cpp",
        "--gguf-path",
        str(args.gguf_path),
        "--models-dir",
        str(args.quantized_dir),
        "--quantize-bin",
        str(args.quantize_bin),
        "--quantizations",
        *args.quantizations,
        "--repack-modes",
        *args.repack_modes,
        "--batch-sizes",
        *[str(size) for size in args.batch_sizes],
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.convert_missing:
        cmd.append("--convert-missing")
    if args.quantize_missing:
        cmd.append("--quantize-missing")
    if args.fail_on_threshold:
        cmd.append("--fail-on-threshold")
    if args.min_cos is not None:
        cmd.extend(["--min-cos", str(args.min_cos)])
    if args.batch_min_cos is not None:
        cmd.extend(["--batch-min-cos", str(args.batch_min_cos)])
    if args.quantized_batch_min_cos is not None:
        cmd.extend(["--quantized-batch-min-cos", str(args.quantized_batch_min_cos)])
    if args.torch_threads is not None:
        cmd.extend(["--torch-threads", str(args.torch_threads)])
    if args.cpp_threads is not None:
        cmd.extend(["--cpp-threads", str(args.cpp_threads)])
    print(
        "scripts/bge_m3_eval.py is a compatibility wrapper; use scripts/model_bench.py for new runs.",
        file=sys.stderr,
    )
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
