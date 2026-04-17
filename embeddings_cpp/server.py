from __future__ import annotations

import argparse
import os
import time
from typing import Any

from embeddings_cpp import load


def _make_app(model_id: str, gguf_path: str | None, cache_dir: str | None, max_batch_size: int | None):
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError("Install embeddings.cpp with the 'server' extra to run the HTTP server.") from exc

    model = load(model_id, gguf_path=gguf_path, cache_dir=cache_dir)
    app = FastAPI(title="embeddings.cpp server", version="0.1.0")
    started_at = time.time()

    def normalize_inputs(value: str | list[str]) -> list[str]:
        texts = [value] if isinstance(value, str) else value
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise HTTPException(status_code=422, detail="inputs must be a string or a list of strings")
        if max_batch_size is not None and len(texts) > max_batch_size:
            raise HTTPException(status_code=413, detail=f"batch size {len(texts)} exceeds max_batch_size={max_batch_size}")
        return texts

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": model.spec.model_id,
            "artifact_file": model.spec.artifact_file,
            "uptime_seconds": time.time() - started_at,
        }

    @app.post("/embed")
    def embed(req: dict[str, Any] = Body(...)) -> list[list[float]]:
        if "inputs" not in req:
            raise HTTPException(status_code=422, detail="missing required field: inputs")
        return model.batch_encode(normalize_inputs(req["inputs"]), normalize=bool(req.get("normalize", True)))

    @app.post("/v1/embeddings")
    def openai_embeddings(req: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if "input" not in req:
            raise HTTPException(status_code=422, detail="missing required field: input")
        if req.get("encoding_format", "float") != "float":
            raise HTTPException(status_code=400, detail="Only encoding_format='float' is supported")
        texts = normalize_inputs(req["input"])
        vectors = model.batch_encode(texts, normalize=True)
        return {
            "object": "list",
            "model": req.get("model") or model.spec.model_id,
            "data": [
                {"object": "embedding", "index": idx, "embedding": vector}
                for idx, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an embeddings.cpp HTTP server.")
    parser.add_argument("--model-id", default=os.environ.get("EMBEDDINGS_CPP_MODEL_ID", "Snowflake/snowflake-arctic-embed-m-v2.0"))
    parser.add_argument("--gguf-path", default=os.environ.get("EMBEDDINGS_CPP_GGUF_PATH"))
    parser.add_argument("--cache-dir", default=os.environ.get("EMBEDDINGS_CPP_CACHE_DIR"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "80")))
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads is not None:
        os.environ["EMBEDDINGS_CPP_THREADS"] = str(args.threads)

    import uvicorn

    app = _make_app(args.model_id, args.gguf_path, args.cache_dir, args.max_batch_size)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
