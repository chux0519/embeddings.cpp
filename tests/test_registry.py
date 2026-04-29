from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec, list_models
from embeddings_cpp.hub import load
from scripts.correctness import default_quantizations


def test_bge_m3_registry_entry_is_local_gguf_compatible():
    spec = get_model_spec("BAAI/bge-m3")

    assert "BAAI/bge-m3" in list_models()
    assert spec.model_id == "BAAI/bge-m3"
    assert spec.slug == "bge-m3"
    assert spec.hf_repo_id is None
    assert spec.artifact_file is None
    assert spec.source_dtype == "f16"
    assert spec.pooling == "cls"
    assert spec.embedding_dim == 1024
    assert spec.correctness["python_min_cos"] >= 0.999
    assert spec.correctness["batch_min_cos"] >= 0.999999


def test_bge_m3_alias_resolves_to_canonical_spec():
    assert get_model_spec("bge-m3") == get_model_spec("BAAI/bge-m3")


def test_local_only_registry_entry_defaults_to_source_dtype_quantization():
    assert default_quantizations(get_model_spec("BAAI/bge-m3")) == ["fp16"]


def test_published_registry_entry_defaults_to_artifact_quantization():
    assert default_quantizations(get_model_spec("Snowflake/snowflake-arctic-embed-m-v2.0")) == [
        "q4_k_mlp_q8_attn"
    ]


def test_local_only_registry_entry_requires_explicit_gguf_path():
    try:
        load("BAAI/bge-m3")
    except ValueError as exc:
        assert "No published GGUF artifact is configured for BAAI/bge-m3" in str(exc)
        assert "pass gguf_path" in str(exc)
    else:
        raise AssertionError("expected local-only BGE-M3 load to require gguf_path")


if __name__ == "__main__":
    test_bge_m3_registry_entry_is_local_gguf_compatible()
    test_bge_m3_alias_resolves_to_canonical_spec()
    test_local_only_registry_entry_defaults_to_source_dtype_quantization()
    test_published_registry_entry_defaults_to_artifact_quantization()
    test_local_only_registry_entry_requires_explicit_gguf_path()
