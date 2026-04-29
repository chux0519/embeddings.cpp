from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec
from scripts.model_bench import build_cases, build_correctness_rows, model_config, tolerance_tier


def test_model_config_comes_from_registry():
    spec = get_model_spec("BAAI/bge-m3")

    config = model_config(spec, ROOT / "models", ["q8_0"])

    assert config.model_id == "BAAI/bge-m3"
    assert config.pooling == "cls"
    assert config.embedding_dim == 1024
    assert config.python_min_cos == 0.999
    assert config.batch_min_cos == 0.999999
    assert config.batch_sizes == [1, 4, 8]
    assert config.gguf_paths["q8_0"].endswith("models/bge-m3.q8_0.gguf")


def test_build_cases_uses_shared_randomized_long_texts():
    first = build_cases(20260429)
    repeated = build_cases(20260429)
    different = build_cases(20260430)

    assert first == repeated
    assert first["random_batch_8"] != different["random_batch_8"]
    assert len(first["mixed_batch_4"]) == 4
    assert len(first["random_batch_8"]) == 8
    all_texts = first["mixed_batch_4"] + first["random_batch_8"] + first["length_skew_batch_4"]
    assert any(len(text) > 100 for text in all_texts)


def test_tolerance_tier_labels_cross_model_quality_bands():
    assert tolerance_tier(0.9991) == "strict"
    assert tolerance_tier(0.995) == "practical"
    assert tolerance_tier(0.951) == "relaxed"
    assert tolerance_tier(0.94) == "outside_relaxed"


def test_build_correctness_rows_reports_drift_without_fail_semantics():
    spec = get_model_spec("BAAI/bge-m3")
    config = model_config(spec, ROOT / "models", ["q8_0"])
    python_result = {
        "status": "ok",
        "cases": {
            "single_en": [[1.0, 0.0]],
        },
    }
    cpp_results = [
        {
            "status": "ok",
            "variant": "q8_0",
            "cases": {
                "single_en": {
                    "batch": [[0.995, 0.1]],
                    "single": [[0.995, 0.1]],
                }
            },
        }
    ]

    rows = build_correctness_rows("BAAI/bge-m3", python_result, cpp_results, config)

    assert len(rows) == 2
    assert rows[0]["runner"] == "embeddings_cpp_vs_python_cpu"
    assert rows[0]["status"] == "outside_tolerance"
    assert rows[0]["tier"] == "practical"
    assert rows[1]["runner"] == "embeddings_cpp_batch_vs_single"
    assert rows[1]["status"] == "within_tolerance"
    assert rows[1]["tier"] == "strict"
