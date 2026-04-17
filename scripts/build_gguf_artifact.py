#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec



def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a registry-defined optimized GGUF artifact.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--quantize-bin", type=Path, default=ROOT / "build" / "quantize")
    args = parser.parse_args()

    spec = get_model_spec(args.model_id)
    models_dir = args.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    fp16_path = models_dir / f"{spec.slug}.{spec.source_dtype}.gguf"
    run(["uv", "run", "scripts/convert.py", spec.model_id, str(fp16_path), spec.source_dtype])

    current = fp16_path
    for step in spec.quantization_steps:
        output = models_dir / step["output_file"]
        env = os.environ.copy()
        skip_patterns = step.get("skip_patterns") or []
        if skip_patterns:
            env["EMBEDDINGS_CPP_SKIP_QUANT_PATTERNS"] = ",".join(skip_patterns)
        run([str(args.quantize_bin), str(current), str(output), step["quant"]], env=env)
        current = output

    final_path = models_dir / spec.artifact_file
    print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
