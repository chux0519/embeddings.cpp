#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Hugging Face README from the registry model card template.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = get_model_spec(args.model_id)
    readme = args.template.read_text(encoding="utf-8")
    size_mb = args.gguf.stat().st_size / 1024 / 1024
    sha = sha256_file(args.gguf)

    replacements = {
        "{{MODEL_ID}}": spec.model_id,
        "{{HF_REPO_ID}}": spec.hf_repo_id,
        "{{GGUF_FILE}}": spec.artifact_file,
        "{{GGUF_SIZE}}": f"{size_mb:.2f} MB",
        "{{GGUF_SHA256}}": sha,
        "{{POOLING}}": spec.pooling,
        "{{EMBEDDING_DIM}}": str(spec.embedding_dim),
        "{{RUNTIME_ENV}}": "\n".join(f"{k}={v} \\" for k, v in spec.runtime_env.items()).rstrip(" \\"),
    }
    for key, value in replacements.items():
        readme = readme.replace(key, value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(readme, encoding="utf-8")
    print(sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
