from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec, list_models
from scripts.correctness import default_quantizations


def test_bge_m3_registry_entry_publishes_default_q8_artifact():
    spec = get_model_spec("BAAI/bge-m3")

    assert "BAAI/bge-m3" in list_models()
    assert spec.model_id == "BAAI/bge-m3"
    assert spec.slug == "bge-m3"
    assert spec.hf_repo_id == "chux0519/bge-m3-gguf-embeddings-cpp"
    assert spec.artifact_file == "bge-m3.q8_0.gguf"
    assert spec.source_dtype == "f16"
    assert spec.pooling == "cls"
    assert spec.embedding_dim == 1024
    assert spec.runtime_env["EMBEDDINGS_CPP_CPU_REPACK"] == "1"
    assert [step["name"] for step in spec.quantization_steps] == ["q8_0"]
    assert spec.correctness["python_min_cos"] >= 0.999
    assert spec.correctness["batch_min_cos"] >= 0.999999


def test_bge_m3_alias_resolves_to_canonical_spec():
    assert get_model_spec("bge-m3") == get_model_spec("BAAI/bge-m3")


def test_bge_m3_defaults_to_published_artifact_quantization():
    assert default_quantizations(get_model_spec("BAAI/bge-m3")) == ["q8_0"]


def test_published_registry_entry_defaults_to_artifact_quantization():
    assert default_quantizations(get_model_spec("Snowflake/snowflake-arctic-embed-m-v2.0")) == [
        "q4_k_mlp_q8_attn"
    ]


if __name__ == "__main__":
    test_bge_m3_registry_entry_publishes_default_q8_artifact()
    test_bge_m3_alias_resolves_to_canonical_spec()
    test_bge_m3_defaults_to_published_artifact_quantization()
    test_published_registry_entry_defaults_to_artifact_quantization()
