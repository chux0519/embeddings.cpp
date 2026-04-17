from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from embeddings_cpp import create_embedding
from embeddings_cpp.registry import ModelSpec, get_model_spec, pooling_name_to_enum


class LoadedEmbedding:
    def __init__(self, model, spec: ModelSpec, gguf_path: str):
        self._model = model
        self.spec = spec
        self.gguf_path = gguf_path
        self.pooling_method = pooling_name_to_enum(spec.pooling)

    def encode(self, text: str, normalize: bool = True, pooling_method=None):
        return self._model.encode(text, normalize, pooling_method or self.pooling_method)

    def batch_encode(self, texts: Sequence[str], normalize: bool = True, pooling_method=None):
        return self._model.batch_encode(list(texts), normalize, pooling_method or self.pooling_method)

    def tokenize(self, text: str, add_special_tokens: bool = True):
        return self._model.tokenize(text, add_special_tokens)

    def batch_tokenize(self, texts: Sequence[str], add_special_tokens: bool = True):
        return self._model.batch_tokenize(list(texts), add_special_tokens)

    @property
    def raw_model(self):
        return self._model


def _download_from_hf(spec: ModelSpec, revision: str | None = None, cache_dir: str | Path | None = None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install embeddings.cpp with the 'hub' extra to load models from Hugging Face.") from exc

    return hf_hub_download(
        repo_id=spec.hf_repo_id,
        filename=spec.artifact_file,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def load(
    model_id: str = "Snowflake/snowflake-arctic-embed-m-v2.0",
    *,
    gguf_path: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    apply_runtime_env: bool = True,
) -> LoadedEmbedding:
    spec = get_model_spec(model_id)
    if apply_runtime_env:
        for key, value in spec.runtime_env.items():
            os.environ.setdefault(key, value)

    resolved_path = str(gguf_path) if gguf_path is not None else _download_from_hf(spec, revision, cache_dir)
    return LoadedEmbedding(create_embedding(resolved_path), spec, resolved_path)
