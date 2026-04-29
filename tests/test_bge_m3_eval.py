from pathlib import Path
import argparse
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.embedding_eval_common import compare_vectors as compare
from scripts.embedding_eval_common import status_from_metrics as status
from scripts.model_bench import apply_default_thresholds, build_analysis, build_cases, threshold_exit_code


def test_compare_reports_cosine_and_distance_metrics():
    metrics = compare([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]])

    assert metrics["min_cos"] == 0.0
    assert metrics["mean_cos"] == 0.5
    assert metrics["mse"] == 0.5
    assert metrics["max_abs"] == 1.0
    assert status(metrics, 0.9) == "outside_tolerance"


def test_build_cases_uses_seeded_realistic_text_batches():
    first = build_cases(20260429)
    repeated = build_cases(20260429)
    different = build_cases(20260430)

    assert first["random_batch_8"] == repeated["random_batch_8"]
    assert first["random_batch_8"] != different["random_batch_8"]
    assert len(first["random_batch_8"]) == 8
    assert any(len(text) > 100 for text in first["mixed_batch_4"] + first["random_batch_8"] + first["length_skew_batch_4"])


def test_default_thresholds_follow_bge_m3_registry_contract():
    args = argparse.Namespace(
        model_id="BAAI/bge-m3",
        min_cos=None,
        batch_min_cos=None,
        quantized_batch_min_cos=None,
    )

    apply_default_thresholds(args)

    assert args.min_cos == 0.999
    assert args.batch_min_cos == 0.999999
    assert args.quantized_batch_min_cos == 0.999999


def test_explicit_thresholds_are_preserved_for_quantization_sweeps():
    args = argparse.Namespace(
        model_id="BAAI/bge-m3",
        min_cos=0.99,
        batch_min_cos=0.99,
        quantized_batch_min_cos=0.99,
    )

    apply_default_thresholds(args)

    assert args.min_cos == 0.99
    assert args.batch_min_cos == 0.99
    assert args.quantized_batch_min_cos == 0.99


def test_threshold_failures_are_report_only_by_default():
    correctness = [{"status": "outside_tolerance"}]

    assert threshold_exit_code(correctness, fail_on_threshold=False) == 0
    assert threshold_exit_code(correctness, fail_on_threshold=True) == 1


def test_invalid_outputs_always_fail_exit_code():
    correctness = [{"status": "invalid"}]

    assert threshold_exit_code(correctness, fail_on_threshold=False) == 1


def test_build_analysis_compares_isolated_runner_performance():
    correctness = [
        {
            "case": "single_en",
            "variant": "fp16",
            "comparison": "embeddings_cpp_batch_vs_python_cpu",
            "status": "within_tolerance",
            "min_cos": 0.9995,
        },
        {
            "case": "mixed_batch_4",
            "variant": "fp16",
            "comparison": "embeddings_cpp_batch_vs_python_cpu",
            "status": "within_tolerance",
            "min_cos": 0.9991,
        },
    ]
    performance = [
        {
            "runner": "python_cpu",
            "batch_size": 4,
            "latency_ms_mean": 200.0,
            "texts_per_second": 20.0,
            "rss_mb": 1200.0,
            "load_rss_mb": 1000.0,
        },
        {
            "runner": "embeddings_cpp",
            "variant": "fp16",
            "quantization": "fp16",
            "repack": "default",
            "batch_size": 4,
            "latency_ms_mean": 50.0,
            "texts_per_second": 80.0,
            "rss_mb": 500.0,
            "load_rss_mb": 400.0,
        },
    ]

    analysis = build_analysis(correctness, performance)

    assert analysis["worst_correctness"]["embeddings_cpp_batch_vs_python_cpu"] == {
        "case": "mixed_batch_4",
        "variant": "fp16",
        "status": "within_tolerance",
        "tier": "unknown",
        "min_cos": 0.9991,
    }
    comparison = analysis["performance_comparison"][0]
    assert comparison["batch_size"] == 4
    assert comparison["variant"] == "fp16"
    assert comparison["quantization"] == "fp16"
    assert comparison["repack"] == "default"
    assert comparison["faster_runner"] == "embeddings_cpp"
    assert comparison["latency_ratio_python_over_cpp"] == 4.0
    assert comparison["throughput_ratio_cpp_over_python"] == 4.0
    assert comparison["rss_delta_mb_cpp_minus_python"] == -700.0
    assert comparison["load_rss_delta_mb_cpp_minus_python"] == -600.0
    assert comparison["min_cos_vs_python_cpu"] == 0.9991
    assert analysis["within_tolerance"] is True
    assert analysis["faster_batches"] == [4]
    assert analysis["slower_batches"] == []
    assert analysis["lower_peak_rss_batches"] == [4]
    assert analysis["ready_for_quantization"] is True
    assert "start quantization next" in analysis["recommendation"]


def test_build_analysis_recommends_batch_optimization_when_python_cpu_wins_batch():
    correctness = [
        {
            "case": "mixed_batch_4",
            "variant": "fp16",
            "comparison": "embeddings_cpp_batch_vs_python_cpu",
            "status": "within_tolerance",
            "min_cos": 0.9991,
        }
    ]
    performance = [
        {
            "runner": "python_cpu",
            "batch_size": 8,
            "latency_ms_mean": 100.0,
            "texts_per_second": 80.0,
            "rss_mb": 2200.0,
            "load_rss_mb": 1000.0,
        },
        {
            "runner": "embeddings_cpp",
            "variant": "fp16",
            "batch_size": 8,
            "latency_ms_mean": 125.0,
            "texts_per_second": 64.0,
            "rss_mb": 1400.0,
            "load_rss_mb": 1300.0,
        },
    ]

    analysis = build_analysis(correctness, performance)

    assert analysis["within_tolerance"] is True
    assert analysis["faster_batches"] == []
    assert analysis["slower_batches"] == [8]
    assert analysis["lower_peak_rss_batches"] == [8]
    assert analysis["ready_for_quantization"] is False
    assert "Optimize embeddings.cpp batch throughput" in analysis["recommendation"]


def test_build_analysis_tracks_best_quantized_repack_variant_by_batch():
    correctness = [
        {
            "case": "mixed_batch_4",
            "variant": "q8_0+repack_off",
            "comparison": "q8_0+repack_off_batch_vs_python_cpu",
            "status": "within_tolerance",
            "min_cos": 0.999,
        },
        {
            "case": "mixed_batch_4",
            "variant": "q8_0+repack_on",
            "comparison": "q8_0+repack_on_batch_vs_python_cpu",
            "status": "within_tolerance",
            "min_cos": 0.999,
        },
    ]
    performance = [
        {
            "runner": "python_cpu",
            "batch_size": 4,
            "latency_ms_mean": 100.0,
            "texts_per_second": 40.0,
            "rss_mb": 1200.0,
            "load_rss_mb": 1000.0,
        },
        {
            "runner": "embeddings_cpp",
            "variant": "q8_0+repack_off",
            "quantization": "q8_0",
            "repack": "off",
            "batch_size": 4,
            "latency_ms_mean": 120.0,
            "texts_per_second": 33.3,
            "rss_mb": 800.0,
            "load_rss_mb": 700.0,
        },
        {
            "runner": "embeddings_cpp",
            "variant": "q8_0+repack_on",
            "quantization": "q8_0",
            "repack": "on",
            "batch_size": 4,
            "latency_ms_mean": 80.0,
            "texts_per_second": 50.0,
            "rss_mb": 760.0,
            "load_rss_mb": 690.0,
        },
    ]

    analysis = build_analysis(correctness, performance)

    assert analysis["faster_batches"] == [4]
    assert analysis["slower_batches"] == []
    assert analysis["best_by_batch"][0]["variant"] == "q8_0+repack_on"
    assert len(analysis["performance_comparison"]) == 2
    off_summary, on_summary = analysis["kquant_summary"]
    assert off_summary["variant"] == "q8_0+repack_off"
    assert off_summary["quantization"] == "q8_0"
    assert off_summary["repack"] == "off"
    assert off_summary["kquant_standard_status"] == "outside_tolerance"
    assert off_summary["within_tolerance"] == 1
    assert off_summary["correctness_total"] == 1
    assert off_summary["min_cos_vs_python_cpu"] == 0.999
    assert math.isnan(off_summary["min_cos_batch_vs_single"])
    assert off_summary["peak_rss_mb"] == 800.0
    assert off_summary["peak_rss_delta_mb_cpp_minus_python"] == -400.0
    assert off_summary["load_rss_mb"] == 700.0
    assert off_summary["load_rss_delta_mb_cpp_minus_python"] == -300.0
    assert off_summary["batch_4_throughput_ratio"] == 33.3 / 40.0
    assert off_summary["batch_4_latency_ms"] == 120.0

    assert on_summary["variant"] == "q8_0+repack_on"
    assert on_summary["kquant_standard_status"] == "outside_tolerance"
    assert on_summary["peak_rss_mb"] == 760.0
    assert on_summary["batch_4_throughput_ratio"] == 50.0 / 40.0


def test_build_analysis_treats_cosine_drift_as_tolerance_metric():
    correctness = [
        {
            "case": "single_en",
            "variant": "fp16",
            "comparison": "embeddings_cpp_batch_vs_python_cpu",
            "status": "outside_tolerance",
            "min_cos": 0.9,
        }
    ]

    analysis = build_analysis(correctness, [])

    assert analysis["within_tolerance"] is False
    assert analysis["valid_outputs"] is True
    assert analysis["ready_for_quantization"] is False
    assert "below the configured tolerance" in analysis["recommendation"]
    assert "Tolerance outside configured range" in analysis["conclusion"]
