#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "scripts" / "output"
DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_GGUF = ROOT / "models" / "bge-m3.fp16.gguf"
DEFAULT_QUANTIZATIONS = ["fp16"]
REPACK_ENV = "EMBEDDINGS_CPP_CPU_REPACK"
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import get_model_spec
from scripts.embedding_eval_common import (
    RssSampler,
    compare_vectors as compare,
    format_cell,
    make_realistic_texts,
    rss_mb,
    status_from_metrics as status,
    summarize_timings,
)


def apply_default_thresholds(args: argparse.Namespace) -> None:
    try:
        correctness = get_model_spec(args.model_id).correctness
    except KeyError:
        correctness = {}
    if args.min_cos is None:
        args.min_cos = float(correctness.get("python_min_cos", 0.999))
    if args.batch_min_cos is None:
        args.batch_min_cos = float(correctness.get("batch_min_cos", 0.999999))
    if args.quantized_batch_min_cos is None:
        args.quantized_batch_min_cos = args.batch_min_cos


def build_cases(seed: int) -> dict[str, list[str]]:
    return {
        "single_en": ["The quick brown fox jumps over the lazy dog."],
        "single_zh": ["机器学习是人工智能的一个重要分支。"],
        "mixed_batch_4": make_realistic_texts(4, seed + 11),
        "random_batch_8": make_realistic_texts(8, seed + 17),
        "length_skew_batch_4": [
            "short",
            " ".join(["long-context"] * 96),
            make_realistic_texts(1, seed + 23)[0],
            make_realistic_texts(1, seed + 29)[0],
        ],
    }


def load_python_runner(model_id: str, torch_threads: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(torch_threads)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            add_pooling_layer=False,
            use_memory_efficient_attention=False,
        )
    except TypeError:
        try:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True, add_pooling_layer=False)
        except TypeError:
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model.to("cpu")

    def encode(texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=8192)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            embeddings = torch.nn.functional.normalize(hidden[:, 0], p=2, dim=1)
        return embeddings.cpu().tolist()

    return encode


def load_cpp_runner(model_id: str, gguf_path: Path):
    sys.path.insert(0, str(ROOT))
    from embeddings_cpp import load

    loaded = load(model_id, gguf_path=str(gguf_path))
    model = loaded.raw_model
    pooling = loaded.pooling_method

    def encode_batch(texts: list[str]) -> list[list[float]]:
        return model.batch_encode(texts, True, pooling)

    def encode_single(texts: list[str]) -> list[list[float]]:
        return [model.encode(text, True, pooling) for text in texts]

    return encode_batch, encode_single


def benchmark(
    encode: Callable[[list[str]], list[list[float]]],
    batch_size: int,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    texts = make_realistic_texts(batch_size, seed)
    for _ in range(warmup):
        encode(texts)
    timings: list[float] = []
    with RssSampler() as sampler:
        for _ in range(iterations):
            start = time.perf_counter()
            vectors = encode(texts)
            elapsed = time.perf_counter() - start
            if len(vectors) != len(texts):
                raise RuntimeError(f"expected {len(texts)} vectors, got {len(vectors)}")
            timings.append(elapsed)
    return summarize_timings(timings, batch_size, sampler.peak_mb)


def run_worker(args: argparse.Namespace) -> int:
    variant = args.variant_label or args.runner
    quantization = args.quantization or ("python_cpu" if args.runner == "python_cpu" else "fp16")
    repack = args.repack or "default"
    cases = build_cases(args.seed)
    if args.runner == "python_cpu":
        start = time.perf_counter()
        encode = load_python_runner(args.model_id, args.torch_threads)
        load_ms = (time.perf_counter() - start) * 1000
        load_rss = rss_mb()
        case_outputs = {name: {"batch": encode(texts)} for name, texts in cases.items()}
        benchmark_encode = encode
    else:
        gguf_path = Path(args.gguf_path)
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        start = time.perf_counter()
        encode_batch, encode_single = load_cpp_runner(args.model_id, gguf_path)
        load_ms = (time.perf_counter() - start) * 1000
        load_rss = rss_mb()
        case_outputs = {
            name: {
                "batch": encode_batch(texts),
                "single": encode_single(texts),
            }
            for name, texts in cases.items()
        }

    perf_rows = []
    for batch_size in args.batch_sizes:
        api = "encode"
        if args.runner == "embeddings_cpp" and batch_size == 1:
            benchmark_encode = encode_single
        elif args.runner == "embeddings_cpp":
            benchmark_encode = encode_batch
            api = "batch_encode"
        perf_rows.append(
            {
                "runner": args.runner,
                "variant": variant,
                "quantization": quantization,
                "repack": repack,
                "model": args.model_id,
                "batch_size": batch_size,
                "mode": "single" if batch_size == 1 else "batch",
                "api": api,
                **benchmark(benchmark_encode, batch_size, args.warmup, args.iterations, args.seed + batch_size * 97),
            }
        )

    result = {
        "runner": args.runner,
        "variant": variant,
        "quantization": quantization,
        "repack": repack,
        "model": args.model_id,
        "load_ms": load_ms,
        "load_rss_mb": load_rss,
        "cases": case_outputs,
        "performance": perf_rows,
    }
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


def run_subprocess(
    args: argparse.Namespace,
    runner: str,
    output_json: Path,
    *,
    gguf_path: Path | None = None,
    variant_label: str | None = None,
    quantization: str | None = None,
    repack: str = "default",
) -> dict[str, Any]:
    worker_gguf_path = gguf_path or Path(args.gguf_path)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--runner",
        runner,
        "--model-id",
        args.model_id,
        "--gguf-path",
        str(worker_gguf_path),
        "--output-json",
        str(output_json),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--torch-threads",
        str(args.torch_threads),
        "--batch-sizes",
        *[str(size) for size in args.batch_sizes],
        "--seed",
        str(args.seed),
        "--quantized-batch-min-cos",
        str(args.quantized_batch_min_cos),
    ]
    if variant_label:
        cmd.extend(["--variant-label", variant_label])
    if quantization:
        cmd.extend(["--quantization", quantization])
    if repack:
        cmd.extend(["--repack", repack])
    env = os.environ.copy()
    if args.cpp_threads and runner == "embeddings_cpp":
        env["EMBEDDINGS_CPP_THREADS"] = str(args.cpp_threads)
    if runner == "embeddings_cpp":
        if repack == "on":
            env[REPACK_ENV] = "1"
        elif repack == "off":
            env[REPACK_ENV] = "0"
        elif repack == "default":
            env.pop(REPACK_ENV, None)
    completed = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{runner} worker failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return json.loads(output_json.read_text(encoding="utf-8"))


def quantized_path(args: argparse.Namespace, quantization: str) -> Path:
    if quantization in {"fp16", "f16"}:
        return Path(args.gguf_path)
    stem = Path(args.gguf_path).name
    for suffix in (".fp16.gguf", ".f16.gguf", ".gguf"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return Path(args.quantized_dir) / f"{stem}.{quantization}.gguf"


def variant_label(quantization: str, repack: str) -> str:
    if repack == "default":
        return quantization
    return f"{quantization}+repack_{repack}"


def ensure_quantized_model(args: argparse.Namespace, quantization: str) -> Path:
    path = quantized_path(args, quantization)
    if path.exists() or quantization in {"fp16", "f16"}:
        return path
    if not args.quantize_missing:
        raise FileNotFoundError(
            f"Missing quantized GGUF for {quantization}: {path}\n"
            "Pass --quantize-missing to create it with build/quantize, or remove it from --quantizations."
        )
    quantize_bin = Path(args.quantize_bin)
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"Quantize executable not found: {quantize_bin}\n"
            "Build it first, for example: cmake --build build --target quantize"
        )
    source = Path(args.gguf_path)
    if not source.exists():
        raise FileNotFoundError(f"Source fp16 GGUF not found: {source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(quantize_bin), str(source), str(path), quantization], cwd=ROOT, check=True)
    return path


def convert_missing(args: argparse.Namespace) -> None:
    gguf_path = Path(args.gguf_path)
    if gguf_path.exists():
        return
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "convert.py"), args.model_id, str(gguf_path), "f16"],
        cwd=ROOT,
        check=True,
    )


def build_analysis(correctness: list[dict[str, Any]], performance: list[dict[str, Any]]) -> dict[str, Any]:
    valid_outputs = all(row["status"] != "invalid" for row in correctness)
    tolerance_passed = all(row["status"] == "within_tolerance" for row in correctness)
    worst_correctness: dict[str, dict[str, Any]] = {}
    variant_correctness: dict[str, dict[str, dict[str, Any]]] = {}
    for row in correctness:
        current = worst_correctness.get(row["comparison"])
        if current is None or row["min_cos"] < current["min_cos"]:
            worst_correctness[row["comparison"]] = {
                "case": row["case"],
                "variant": row.get("variant", "embeddings_cpp"),
                "status": row["status"],
                "min_cos": row["min_cos"],
            }
        variant = row.get("variant", "embeddings_cpp")
        comparison_kind = "python_cpu" if row["comparison"].endswith("_vs_python_cpu") else "batch_vs_single"
        current = variant_correctness.setdefault(variant, {}).get(comparison_kind)
        if current is None or row["min_cos"] < current["min_cos"]:
            variant_correctness[variant][comparison_kind] = {
                "case": row["case"],
                "status": row["status"],
                "min_cos": row["min_cos"],
            }

    by_batch: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in performance:
        by_batch.setdefault(row["batch_size"], {}).setdefault(row["runner"], []).append(row)

    performance_comparison = []
    for batch_size, rows in sorted(by_batch.items()):
        python_rows = rows.get("python_cpu", [])
        cpp_rows = rows.get("embeddings_cpp", [])
        if not python_rows or not cpp_rows:
            continue
        python_row = python_rows[0]
        for cpp_row in sorted(cpp_rows, key=lambda item: item.get("variant", "")):
            latency_ratio = python_row["latency_ms_mean"] / cpp_row["latency_ms_mean"]
            throughput_ratio = cpp_row["texts_per_second"] / python_row["texts_per_second"]
            rss_delta_mb = cpp_row["rss_mb"] - python_row["rss_mb"]
            load_rss_delta_mb = cpp_row["load_rss_mb"] - python_row["load_rss_mb"]
            variant = cpp_row.get("variant", "embeddings_cpp")
            correctness_for_variant = variant_correctness.get(variant, {})
            python_correctness = correctness_for_variant.get("python_cpu", {})
            batch_correctness = correctness_for_variant.get("batch_vs_single", {})
            performance_comparison.append(
                {
                    "batch_size": batch_size,
                    "variant": variant,
                    "quantization": cpp_row.get("quantization", "unknown"),
                    "repack": cpp_row.get("repack", "default"),
                    "faster_runner": "embeddings_cpp" if latency_ratio > 1 else "python_cpu",
                    "latency_ratio_python_over_cpp": latency_ratio,
                    "throughput_ratio_cpp_over_python": throughput_ratio,
                    "python_latency_ms_mean": python_row["latency_ms_mean"],
                    "cpp_latency_ms_mean": cpp_row["latency_ms_mean"],
                    "python_texts_per_second": python_row["texts_per_second"],
                    "cpp_texts_per_second": cpp_row["texts_per_second"],
                    "python_rss_mb": python_row["rss_mb"],
                    "cpp_rss_mb": cpp_row["rss_mb"],
                    "rss_delta_mb_cpp_minus_python": rss_delta_mb,
                    "python_load_rss_mb": python_row["load_rss_mb"],
                    "cpp_load_rss_mb": cpp_row["load_rss_mb"],
                    "load_rss_delta_mb_cpp_minus_python": load_rss_delta_mb,
                    "min_cos_vs_python_cpu": python_correctness.get("min_cos", float("nan")),
                    "correctness_status_vs_python_cpu": python_correctness.get("status", "missing"),
                    "min_cos_batch_vs_single": batch_correctness.get("min_cos", float("nan")),
                    "correctness_status_batch_vs_single": batch_correctness.get("status", "missing"),
                }
            )

    faster_batches = [
        batch_size
        for batch_size in sorted({row["batch_size"] for row in performance_comparison})
        if any(
            row["batch_size"] == batch_size and row["faster_runner"] == "embeddings_cpp"
            for row in performance_comparison
        )
    ]
    slower_batches = [
        batch_size
        for batch_size in sorted({row["batch_size"] for row in performance_comparison})
        if not any(
            row["batch_size"] == batch_size and row["faster_runner"] == "embeddings_cpp"
            for row in performance_comparison
        )
    ]
    lower_peak_rss = [
        batch_size
        for batch_size in sorted({row["batch_size"] for row in performance_comparison})
        if any(
            row["batch_size"] == batch_size and row["rss_delta_mb_cpp_minus_python"] < 0
            for row in performance_comparison
        )
    ]
    best_by_batch = []
    for batch_size in sorted({row["batch_size"] for row in performance_comparison}):
        candidates = [row for row in performance_comparison if row["batch_size"] == batch_size]
        best_by_batch.append(max(candidates, key=lambda row: row["throughput_ratio_cpp_over_python"]))

    correctness_counts: dict[str, dict[str, int]] = {}
    for row in correctness:
        counts = correctness_counts.setdefault(
            row.get("variant", "embeddings_cpp"),
            {"within_tolerance": 0, "invalid": 0, "total": 0},
        )
        counts["total"] += 1
        if row["status"] == "within_tolerance":
            counts["within_tolerance"] += 1
        elif row["status"] == "invalid":
            counts["invalid"] += 1

    kquant_summary = []
    for variant in sorted({row["variant"] for row in performance_comparison}):
        rows = sorted((row for row in performance_comparison if row["variant"] == variant), key=lambda row: row["batch_size"])
        if not rows:
            continue
        first = rows[0]
        correctness_for_variant = variant_correctness.get(variant, {})
        python_correctness = correctness_for_variant.get("python_cpu", {})
        batch_correctness = correctness_for_variant.get("batch_vs_single", {})
        counts = correctness_counts.get(variant, {"within_tolerance": 0, "invalid": 0, "total": 0})
        standard_status = (
            "within_tolerance"
            if counts["total"] > 0
            and counts["within_tolerance"] == counts["total"]
            and python_correctness.get("status") == "within_tolerance"
            and batch_correctness.get("status") == "within_tolerance"
            else "invalid"
            if counts.get("invalid", 0) > 0
            else "outside_tolerance"
        )
        speed_by_batch = {f"batch_{row['batch_size']}_throughput_ratio": row["throughput_ratio_cpp_over_python"] for row in rows}
        latency_by_batch = {f"batch_{row['batch_size']}_latency_ms": row["cpp_latency_ms_mean"] for row in rows}
        kquant_summary.append(
            {
                "variant": variant,
                "quantization": first["quantization"],
                "repack": first["repack"],
                "kquant_standard_status": standard_status,
                "within_tolerance": counts["within_tolerance"],
                "invalid": counts.get("invalid", 0),
                "correctness_total": counts["total"],
                "min_cos_vs_python_cpu": python_correctness.get("min_cos", float("nan")),
                "correctness_status_vs_python_cpu": python_correctness.get("status", "missing"),
                "min_cos_batch_vs_single": batch_correctness.get("min_cos", float("nan")),
                "correctness_status_batch_vs_single": batch_correctness.get("status", "missing"),
                "peak_rss_mb": max(row["cpp_rss_mb"] for row in rows),
                "peak_rss_delta_mb_cpp_minus_python": max(row["rss_delta_mb_cpp_minus_python"] for row in rows),
                "load_rss_mb": first["cpp_load_rss_mb"],
                "load_rss_delta_mb_cpp_minus_python": first["load_rss_delta_mb_cpp_minus_python"],
                **speed_by_batch,
                **latency_by_batch,
            }
        )

    if not valid_outputs:
        recommendation = "Fix invalid embedding output before performance optimization or quantization."
    elif not tolerance_passed:
        recommendation = (
            "Review the tolerance trade-off before choosing this variant; embeddings were produced, "
            "but at least one cosine metric is below the configured tolerance."
        )
    elif slower_batches:
        recommendation = (
            "Optimize embeddings.cpp batch throughput before making quantization the main next step; "
            f"Python CPU is faster for batch sizes {slower_batches} in this run."
        )
    else:
        recommendation = (
            "embeddings.cpp is faster across the measured batch sizes; start quantization next while "
            "preserving the same correctness thresholds."
        )

    conclusion = (
        f"Embedding output {'valid' if valid_outputs else 'invalid'}. "
        f"Tolerance {'met' if tolerance_passed else 'outside configured range'}. "
        f"embeddings.cpp is faster for batch sizes {faster_batches or []} and Python CPU is faster for "
        f"batch sizes {slower_batches or []}. "
        f"embeddings.cpp uses lower peak RSS for batch sizes {lower_peak_rss or []}."
    )

    return {
        "correctness_passed": tolerance_passed,
        "valid_outputs": valid_outputs,
        "tolerance_passed": tolerance_passed,
        "within_tolerance": tolerance_passed,
        "worst_correctness": worst_correctness,
        "performance_comparison": performance_comparison,
        "best_by_batch": best_by_batch,
        "kquant_summary": kquant_summary,
        "variant_correctness": variant_correctness,
        "faster_batches": faster_batches,
        "slower_batches": slower_batches,
        "lower_peak_rss_batches": lower_peak_rss,
        "ready_for_quantization": valid_outputs and tolerance_passed and not slower_batches,
        "conclusion": conclusion,
        "recommendation": recommendation,
    }


def threshold_exit_code(correctness: list[dict[str, Any]], fail_on_threshold: bool) -> int:
    has_invalid_output = any(row["status"] == "invalid" for row in correctness)
    outside_tolerance = any(row["status"] == "outside_tolerance" for row in correctness)
    return 1 if has_invalid_output or (fail_on_threshold and outside_tolerance) else 0


def write_report(
    correctness: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"bge_m3_eval_{stamp}.json"
    md_path = output_dir / f"bge_m3_eval_{stamp}.md"
    json_path.write_text(
        json.dumps(
            {"correctness": correctness, "performance": performance, "analysis": analysis},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# BGE-M3 Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        if "metadata" in analysis:
            metadata = analysis["metadata"]
            f.write("## Method\n\n")
            f.write(
                "Inputs: deterministic randomized realistic text batches shared with the Snowflake profiling script.\n\n"
            )
            f.write(
                f"Seed: {metadata['seed']}; min cosine vs Python: {metadata['min_cos']}; "
                f"batch-vs-single min cosine: {metadata['batch_min_cos']}; "
                f"quantized batch-vs-single min cosine: {metadata['quantized_batch_min_cos']}.\n\n"
            )
            f.write(
                "These thresholds are report tolerances. The command exits non-zero only when "
                "`--fail-on-threshold` is set.\n\n"
            )
        f.write("## Conclusion\n\n")
        f.write(f"{analysis['conclusion']}\n\n")
        f.write(f"Recommendation: {analysis['recommendation']}\n\n")
        f.write("## Kquant / Repack Summary\n\n")
        f.write(
            "| Variant | Tolerance | Correctness | Min Cos vs Python | Batch vs Single Min Cos | "
            "B1 C++/Python TPS | B4 C++/Python TPS | B8 C++/Python TPS | Peak RSS MB | RSS Delta MB | Load RSS MB |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in analysis["kquant_summary"]:
            correctness_ratio = f"{row['within_tolerance']}/{row['correctness_total']}"
            f.write(
                f"| {row['variant']} | {row['kquant_standard_status']} | {correctness_ratio} | "
                f"{format_cell(row['min_cos_vs_python_cpu'])} | "
                f"{format_cell(row['min_cos_batch_vs_single'])} | "
                f"{format_cell(row.get('batch_1_throughput_ratio', float('nan')))} | "
                f"{format_cell(row.get('batch_4_throughput_ratio', float('nan')))} | "
                f"{format_cell(row.get('batch_8_throughput_ratio', float('nan')))} | "
                f"{format_cell(row['peak_rss_mb'])} | "
                f"{format_cell(row['peak_rss_delta_mb_cpp_minus_python'])} | "
                f"{format_cell(row['load_rss_mb'])} |\n"
            )
        f.write("\n")
        f.write("## Correctness\n\n")
        f.write("| Variant | Case | Comparison | Tolerance | Min Cos | Mean Cos | MSE | Max Abs |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|\n")
        for row in correctness:
            f.write(
                f"| {row.get('variant', 'embeddings_cpp')} | {row['case']} | {row['comparison']} | "
                f"{row['status']} | {format_cell(row['min_cos'])} | "
                f"{format_cell(row['mean_cos'])} | {format_cell(row['mse'])} | {format_cell(row['max_abs'])} |\n"
            )
        f.write("\n## Correctness Summary\n\n")
        f.write("| Comparison | Variant | Worst Case | Tolerance | Min Cos |\n")
        f.write("|---|---|---|---|---:|\n")
        for comparison, row in analysis["worst_correctness"].items():
            f.write(
                f"| {comparison} | {row.get('variant', 'embeddings_cpp')} | {row['case']} | "
                f"{row['status']} | {format_cell(row['min_cos'])} |\n"
            )
        f.write("\n## Performance\n\n")
        f.write("| Runner | Variant | Quantization | Repack | Mode | API | Batch | Mean ms | P50 ms | P95 ms | Text/s | RSS MB | Load ms | Load RSS MB |\n")
        f.write("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in performance:
            f.write(
                f"| {row['runner']} | {row.get('variant', row['runner'])} | "
                f"{row.get('quantization', '')} | {row.get('repack', '')} | "
                f"{row['mode']} | {row['api']} | {row['batch_size']} | "
                f"{format_cell(row['latency_ms_mean'])} | {format_cell(row['latency_ms_p50'])} | "
                f"{format_cell(row['latency_ms_p95'])} | {format_cell(row['texts_per_second'])} | "
                f"{format_cell(row['rss_mb'])} | {format_cell(row['load_ms'])} | "
                f"{format_cell(row['load_rss_mb'])} |\n"
            )
        f.write("\n## Optimization Sweep\n\n")
        f.write("| Batch | Variant | Quantization | Repack | Faster Runner | Python/C++ Latency | C++/Python Throughput | C++ RSS Delta MB | C++ Load RSS Delta MB | Min Cos vs Python | Batch vs Single Min Cos |\n")
        f.write("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in analysis["performance_comparison"]:
            f.write(
                f"| {row['batch_size']} | {row['variant']} | {row['quantization']} | {row['repack']} | "
                f"{row['faster_runner']} | "
                f"{format_cell(row['latency_ratio_python_over_cpp'])} | "
                f"{format_cell(row['throughput_ratio_cpp_over_python'])} | "
                f"{format_cell(row['rss_delta_mb_cpp_minus_python'])} | "
                f"{format_cell(row['load_rss_delta_mb_cpp_minus_python'])} | "
                f"{format_cell(row['min_cos_vs_python_cpu'])} | "
                f"{format_cell(row['min_cos_batch_vs_single'])} |\n"
            )
        f.write("\n## Best Variant By Batch\n\n")
        f.write("| Batch | Variant | Quantization | Repack | C++/Python Throughput | C++ RSS Delta MB | Min Cos vs Python |\n")
        f.write("|---:|---|---|---|---:|---:|---:|\n")
        for row in analysis["best_by_batch"]:
            f.write(
                f"| {row['batch_size']} | {row['variant']} | {row['quantization']} | {row['repack']} | "
                f"{format_cell(row['throughput_ratio_cpp_over_python'])} | "
                f"{format_cell(row['rss_delta_mb_cpp_minus_python'])} | "
                f"{format_cell(row['min_cos_vs_python_cpu'])} |\n"
            )
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BGE-M3 correctness and isolated CPU performance for Python transformers and embeddings.cpp."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gguf-path", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--convert-missing", action="store_true", help="Convert a missing fp16 GGUF before running.")
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=DEFAULT_QUANTIZATIONS,
        help="GGUF variants to evaluate. fp16 uses --gguf-path; other values use --quantized-dir/<stem>.<quant>.gguf.",
    )
    parser.add_argument(
        "--quantize-missing",
        action="store_true",
        help="Create missing quantized GGUF variants with build/quantize before evaluating them.",
    )
    parser.add_argument("--quantized-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--quantize-bin", type=Path, default=ROOT / "build" / "quantize")
    parser.add_argument(
        "--repack-modes",
        nargs="+",
        choices=("default", "on", "off"),
        default=["default"],
        help="Evaluate embeddings.cpp with default, forced-on, or forced-off CPU repack.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--min-cos",
        type=float,
        default=None,
        help="Python baseline min cosine threshold. Defaults to the model registry correctness contract.",
    )
    parser.add_argument(
        "--batch-min-cos",
        type=float,
        default=None,
        help="Batch-vs-single min cosine threshold. Defaults to the model registry correctness contract.",
    )
    parser.add_argument(
        "--quantized-batch-min-cos",
        type=float,
        default=None,
        help="Batch-vs-single min cosine threshold for quantized GGUF variants. Defaults to --batch-min-cos.",
    )
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with status 1 when any cosine metric is below the configured tolerance.",
    )
    parser.add_argument("--torch-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--cpp-threads", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runner", choices=("python_cpu", "embeddings_cpp"), default="python_cpu", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant-label", help=argparse.SUPPRESS)
    parser.add_argument("--quantization", help=argparse.SUPPRESS)
    parser.add_argument("--repack", choices=("default", "on", "off"), default="default", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_default_thresholds(args)
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if any(size < 1 for size in args.batch_sizes):
        raise ValueError("--batch-sizes values must be positive")
    quantizations = []
    for quantization in args.quantizations:
        normalized = "fp16" if quantization == "f16" else quantization
        if normalized not in quantizations:
            quantizations.append(normalized)

    if args.worker:
        if args.output_json is None:
            raise ValueError("--output-json is required in worker mode")
        return run_worker(args)

    if args.convert_missing:
        convert_missing(args)
    elif not Path(args.gguf_path).exists():
        print(
            f"Missing GGUF: {args.gguf_path}\n"
            "Run with --convert-missing or pass --gguf-path to an existing BGE-M3 GGUF.",
            file=sys.stderr,
        )
        return 2
    for quantization in quantizations:
        ensure_quantized_model(args, quantization)

    with tempfile.TemporaryDirectory(prefix="bge-m3-eval-") as temp_dir:
        temp_path = Path(temp_dir)
        python_result = run_subprocess(args, "python_cpu", temp_path / "python.json")
        cpp_results = []
        for quantization in quantizations:
            gguf_path = quantized_path(args, quantization)
            for repack in args.repack_modes:
                label = variant_label(quantization, repack)
                cpp_results.append(
                    run_subprocess(
                        args,
                        "embeddings_cpp",
                        temp_path / f"cpp_{label}.json",
                        gguf_path=gguf_path,
                        variant_label=label,
                        quantization=quantization,
                        repack=repack,
                    )
                )

    correctness = []
    for cpp_result in cpp_results:
        variant = cpp_result["variant"]
        for case_name, python_case in python_result["cases"].items():
            cpp_case = cpp_result["cases"][case_name]
            metrics = compare(cpp_case["batch"], python_case["batch"])
            correctness.append(
                {
                    "variant": variant,
                    "quantization": cpp_result["quantization"],
                    "repack": cpp_result["repack"],
                    "case": case_name,
                    "comparison": f"{variant}_batch_vs_python_cpu",
                    "status": status(metrics, args.min_cos),
                    **metrics,
                }
            )
            metrics = compare(cpp_case["batch"], cpp_case["single"])
            batch_threshold = (
                args.batch_min_cos
                if cpp_result["quantization"] in {"fp16", "f16"}
                else min(args.batch_min_cos, args.quantized_batch_min_cos)
            )
            correctness.append(
                {
                    "variant": variant,
                    "quantization": cpp_result["quantization"],
                    "repack": cpp_result["repack"],
                    "case": case_name,
                    "comparison": f"{variant}_batch_vs_single",
                    "status": status(metrics, batch_threshold),
                    **metrics,
                }
            )

    performance = []
    for result in (python_result, *cpp_results):
        for row in result["performance"]:
            performance.append(
                {
                    **row,
                    "load_ms": result["load_ms"],
                    "load_rss_mb": result["load_rss_mb"],
                }
            )

    analysis = build_analysis(correctness, performance)
    analysis["metadata"] = {
        "seed": args.seed,
        "min_cos": args.min_cos,
        "batch_min_cos": args.batch_min_cos,
        "quantized_batch_min_cos": args.quantized_batch_min_cos,
        "fail_on_threshold": args.fail_on_threshold,
    }
    json_path, md_path = write_report(correctness, performance, analysis, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(analysis["conclusion"])
    print(f"Recommendation: {analysis['recommendation']}")
    return threshold_exit_code(correctness, args.fail_on_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
