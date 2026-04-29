from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    slug: str
    hf_repo_id: str | None
    artifact_file: str | None
    source_dtype: str
    pooling: str
    embedding_dim: int
    runtime_env: dict[str, str]
    cmake_flags: list[str]
    quantization_steps: list[dict[str, Any]]
    correctness: dict[str, Any]
    benchmark: dict[str, Any]


def _registry() -> dict[str, Any]:
    with resources.files(__package__).joinpath("registry.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def list_models() -> list[str]:
    registry = _registry()
    return sorted(key for key, value in registry.items() if "alias_for" not in value)


def get_model_spec(model_id: str) -> ModelSpec:
    registry = _registry()
    if model_id not in registry:
        known = ", ".join(list_models())
        raise KeyError(f"Unknown embeddings.cpp model '{model_id}'. Known models: {known}")

    entry = registry[model_id]
    if "alias_for" in entry:
        entry = registry[entry["alias_for"]]

    return ModelSpec(
        model_id=entry["model_id"],
        slug=entry["slug"],
        hf_repo_id=entry.get("hf_repo_id"),
        artifact_file=entry.get("artifact_file"),
        source_dtype=entry.get("source_dtype", "f16"),
        pooling=entry.get("pooling", "mean"),
        embedding_dim=int(entry["embedding_dim"]),
        runtime_env=dict(entry.get("runtime_env", {})),
        cmake_flags=list(entry.get("cmake_flags", [])),
        quantization_steps=list(entry.get("quantization_steps", [])),
        correctness=dict(entry.get("correctness", {})),
        benchmark=dict(entry.get("benchmark", {})),
    )


def pooling_name_to_enum(name: str):
    from embeddings_cpp import PoolingMethod

    normalized = name.lower()
    if normalized == "cls":
        return PoolingMethod.CLS
    if normalized == "mean":
        return PoolingMethod.MEAN
    raise ValueError(f"Unsupported pooling method: {name}")
