#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "scripts" / "output"
MODELS_DIR = ROOT / "models"
DEFAULT_RUNNERS = ["python_cpu", "embeddings_cpp"]
REPACK_ENV = "EMBEDDINGS_CPP_CPU_REPACK"
sys.path.insert(0, str(ROOT))

from embeddings_cpp.registry import ModelSpec, get_model_spec
from scripts.embedding_eval_common import (
    RssSampler,
    compare_vectors,
    format_cell,
    make_realistic_texts,
    rss_mb,
    status_from_metrics,
    summarize_timings,
)


@dataclass(frozen=True)
class ModelBenchConfig:
    model_id: str
    slug: str
    pooling: str
    embedding_dim: int
    python_min_cos: float
    batch_min_cos: float
    quantized_batch_min_cos: float
    batch_sizes: list[int]
    max_batch_tokens: int | None
    gguf_paths: dict[str, str]


def model_config(spec: ModelSpec, models_dir: Path, quantizations: list[str]) -> ModelBenchConfig:
    stem = spec.slug
    gguf_paths = {}
    for quantization in quantizations:
        if quantization == "artifact" and spec.artifact_file:
            gguf_paths[quantization] = str(models_dir / spec.artifact_file)
        else:
            gguf_paths[quantization] = str(models_dir / f"{stem}.{quantization}.gguf")
    correctness = spec.correctness
    benchmark = spec.benchmark
    return ModelBenchConfig(
        model_id=spec.model_id,
        slug=spec.slug,
        pooling=spec.pooling,
        embedding_dim=spec.embedding_dim,
        python_min_cos=float(correctness.get("python_min_cos", 0.999)),
        batch_min_cos=float(correctness.get("batch_min_cos", 0.999999)),
        quantized_batch_min_cos=float(correctness.get("batch_min_cos", 0.999999)),
        batch_sizes=[int(v) for v in benchmark.get("batch_sizes", [1, 4, 8])],
        max_batch_tokens=benchmark.get("max_batch_tokens"),
        gguf_paths=gguf_paths,
    )


def apply_default_thresholds(args: argparse.Namespace) -> None:
    correctness = get_model_spec(args.model_id).correctness
    if args.min_cos is None:
        args.min_cos = float(correctness.get("python_min_cos", 0.999))
    if args.batch_min_cos is None:
        args.batch_min_cos = float(correctness.get("batch_min_cos", 0.999999))
    if args.quantized_batch_min_cos is None:
        args.quantized_batch_min_cos = args.batch_min_cos


def gguf_paths_from_override(gguf_path: Path, quantizations: list[str]) -> dict[str, str]:
    stem = gguf_path.name
    for suffix in (".fp16.gguf", ".f16.gguf", ".gguf"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    paths = {}
    for quantization in quantizations:
        if quantization in {"fp16", "f16"}:
            paths["fp16"] = str(gguf_path)
        else:
            paths[quantization] = str(gguf_path.parent / f"{stem}.{quantization}.gguf")
    return paths


def convert_missing(spec: ModelSpec, gguf_path: Path) -> None:
    if gguf_path.exists():
        return
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "convert.py"), spec.model_id, str(gguf_path), "f16"],
        cwd=ROOT,
        check=True,
    )


def ensure_quantized_model(args: argparse.Namespace, source_path: Path, quantization: str, target_path: Path) -> None:
    if target_path.exists() or quantization in {"fp16", "f16"}:
        return
    if not args.quantize_missing:
        raise FileNotFoundError(
            f"Missing quantized GGUF for {quantization}: {target_path}\n"
            "Pass --quantize-missing to create it with build/quantize, or remove it from --quantizations."
        )
    quantize_bin = Path(args.quantize_bin)
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"Quantize executable not found: {quantize_bin}\n"
            "Build it first, for example: cmake --build build --target quantize"
        )
    if not source_path.exists():
        raise FileNotFoundError(f"Source fp16 GGUF not found: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(quantize_bin), str(source_path), str(target_path), quantization], cwd=ROOT, check=True)


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


def load_python_runner(spec: ModelSpec, torch_threads: int) -> Callable[[list[str]], list[list[float]]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(torch_threads)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            spec.model_id,
            trust_remote_code=True,
            add_pooling_layer=False,
            use_memory_efficient_attention=False,
        )
    except TypeError:
        try:
            model = AutoModel.from_pretrained(spec.model_id, trust_remote_code=True, add_pooling_layer=False)
        except TypeError:
            model = AutoModel.from_pretrained(spec.model_id, trust_remote_code=True)
    model.eval()
    model.to("cpu")

    def encode(texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=8192)
        if spec.model_id.startswith("Snowflake/"):
            batch_size, seq_length = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            if spec.pooling == "cls":
                embeddings = hidden[:, 0]
            elif spec.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).bool()
                hidden = hidden.masked_fill(~mask, 0)
                mask_f = mask.float()
                embeddings = torch.sum(hidden, dim=1) / torch.clamp(mask_f.sum(dim=1), min=1e-9)
            else:
                raise ValueError(f"unsupported pooling: {spec.pooling}")
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    return encode


def load_cpp_runner(spec: ModelSpec, gguf_path: Path):
    from embeddings_cpp import load

    loaded = load(spec.model_id, gguf_path=str(gguf_path))
    model = loaded.raw_model
    pooling = loaded.pooling_method

    def encode_batch(texts: list[str]) -> list[list[float]]:
        return model.batch_encode(texts, True, pooling)

    def encode_single(texts: list[str]) -> list[list[float]]:
        return [model.encode(text, True, pooling) for text in texts]

    return encode_batch, encode_single


def benchmark_runner(
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
            started = time.perf_counter()
            vectors = encode(texts)
            timings.append(time.perf_counter() - started)
            if len(vectors) != len(texts):
                raise RuntimeError(f"expected {len(texts)} vectors, got {len(vectors)}")
    return summarize_timings(timings, batch_size, sampler.peak_mb)


def run_python_worker(args: argparse.Namespace, spec: ModelSpec) -> dict[str, Any]:
    started = time.perf_counter()
    encode = load_python_runner(spec, args.torch_threads)
    load_ms = (time.perf_counter() - started) * 1000
    load_rss = rss_mb()
    cases = build_cases(args.seed)
    return {
        "runner": "python_cpu",
        "model_id": spec.model_id,
        "load_ms": load_ms,
        "load_rss_mb": load_rss,
        "cases": {name: encode(texts) for name, texts in cases.items()},
        "performance": [
            {
                "runner": "python_cpu",
                "model_id": spec.model_id,
                "variant": "python_cpu",
                "quantization": "python_cpu",
                "repack": "default",
                "batch_size": batch_size,
                "mode": "single" if batch_size == 1 else "batch",
                "api": "encode",
                **benchmark_runner(encode, batch_size, args.warmup, args.iterations, args.seed + batch_size * 97),
            }
            for batch_size in args.batch_sizes
        ],
    }


def run_cpp_worker(args: argparse.Namespace, spec: ModelSpec) -> dict[str, Any]:
    gguf_path = Path(args.gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    started = time.perf_counter()
    encode_batch, encode_single = load_cpp_runner(spec, gguf_path)
    load_ms = (time.perf_counter() - started) * 1000
    load_rss = rss_mb()
    cases = build_cases(args.seed)
    case_outputs = {
        name: {
            "batch": encode_batch(texts),
            "single": encode_single(texts),
        }
        for name, texts in cases.items()
    }
    perf_rows = []
    for batch_size in args.batch_sizes:
        encode = encode_single if batch_size == 1 else encode_batch
        perf_rows.append(
            {
                "runner": "embeddings_cpp",
                "model_id": spec.model_id,
                "variant": args.variant,
                "quantization": args.quantization,
                "repack": args.repack,
                "gguf_path": str(gguf_path),
                "batch_size": batch_size,
                "mode": "single" if batch_size == 1 else "batch",
                "api": "encode" if batch_size == 1 else "batch_encode",
                **benchmark_runner(encode, batch_size, args.warmup, args.iterations, args.seed + batch_size * 97),
            }
        )
    return {
        "runner": "embeddings_cpp",
        "model_id": spec.model_id,
        "variant": args.variant,
        "quantization": args.quantization,
        "repack": args.repack,
        "gguf_path": str(gguf_path),
        "load_ms": load_ms,
        "load_rss_mb": load_rss,
        "cases": case_outputs,
        "performance": perf_rows,
    }


def run_tei_engine_ort_worker(args: argparse.Namespace, spec: ModelSpec) -> dict[str, Any]:
    from scripts.profile_snowflake import benchmark_tei_engine

    perf_rows = []
    tei_args = argparse.Namespace(
        repo_id=spec.model_id,
        tei_repo_dir=args.tei_repo_dir,
        tei_cache_dir=args.tei_cache_dir,
        tei_backend="ort",
        scope=args.scope,
        warmup=args.warmup,
        iterations=args.iterations,
        timeout=args.timeout,
    )
    for batch_size in args.batch_sizes:
        texts = make_realistic_texts(batch_size, args.seed + batch_size * 97)
        row = benchmark_tei_engine(tei_args, args.tei_threads, texts)
        row.update(
            {
                "runner": "tei_engine_ort",
                "model_id": spec.model_id,
                "variant": "tei_engine_ort",
                "quantization": "n/a",
                "repack": "n/a",
                "batch_size": batch_size,
                "mode": "engine",
                "api": args.scope,
            }
        )
        perf_rows.append(row)
    return {
        "runner": "tei_engine_ort",
        "model_id": spec.model_id,
        "variant": "tei_engine_ort",
        "cases": {},
        "performance": perf_rows,
    }


def worker_main(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model_id)
    if args.runner == "python_cpu":
        result = run_python_worker(args, spec)
    elif args.runner == "embeddings_cpp":
        result = run_cpp_worker(args, spec)
    elif args.runner == "tei_engine_ort":
        result = run_tei_engine_ort_worker(args, spec)
    else:
        raise ValueError(f"unknown runner: {args.runner}")
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


def run_child(
    args: argparse.Namespace,
    *,
    model_id: str,
    runner: str,
    output_json: Path,
    batch_sizes: list[int],
    gguf_path: Path | None = None,
    quantization: str | None = None,
    variant: str | None = None,
    repack: str = "default",
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model-id",
        model_id,
        "--runner",
        runner,
        "--output-json",
        str(output_json),
        "--batch-sizes",
        *[str(v) for v in batch_sizes],
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--torch-threads",
        str(args.torch_threads),
        "--tei-threads",
        str(args.tei_threads),
        "--tei-repo-dir",
        str(args.tei_repo_dir),
        "--tei-cache-dir",
        str(args.tei_cache_dir),
        "--scope",
        args.scope,
        "--timeout",
        str(args.timeout),
    ]
    if gguf_path is not None:
        cmd.extend(["--gguf-path", str(gguf_path)])
    if quantization is not None:
        cmd.extend(["--quantization", quantization, "--variant", variant or quantization])
    cmd.extend(["--repack", repack])

    env = os.environ.copy()
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    if args.cpp_threads:
        env["EMBEDDINGS_CPP_THREADS"] = str(args.cpp_threads)
    if runner == "embeddings_cpp":
        if repack == "on":
            env[REPACK_ENV] = "1"
        elif repack == "off":
            env[REPACK_ENV] = "0"
        else:
            env.pop(REPACK_ENV, None)

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "runner": runner,
            "model_id": model_id,
            "status": "error",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
            "cases": {},
            "performance": [],
        }
    try:
        result = json.loads(output_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "runner": runner,
            "model_id": model_id,
            "status": "error",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": f"failed to read worker json {output_json}: {exc}",
            "cases": {},
            "performance": [],
        }
    result["status"] = "ok"
    return result


def tolerance_tier(min_cos: float) -> str:
    if math.isnan(min_cos):
        return "invalid"
    if min_cos >= 0.999:
        return "strict"
    if min_cos >= 0.99:
        return "practical"
    if min_cos >= 0.95:
        return "relaxed"
    return "outside_relaxed"


def variant_label(quantization: str, repack: str) -> str:
    if repack == "default":
        return quantization
    return f"{quantization}+repack_{repack}"


def build_correctness_rows(
    model_id: str,
    python_result: dict[str, Any] | None,
    cpp_results: list[dict[str, Any]],
    config: ModelBenchConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if python_result is None or python_result.get("status") != "ok":
        return rows
    python_cases = python_result["cases"]
    for result in cpp_results:
        if result.get("status") != "ok":
            continue
        variant = result.get("variant", "embeddings_cpp")
        for case_name, cpp_case in result.get("cases", {}).items():
            if case_name not in python_cases:
                continue
            metrics = compare_vectors(cpp_case["batch"], python_cases[case_name])
            rows.append(
                {
                    "model_id": model_id,
                    "case": case_name,
                    "runner": "embeddings_cpp_vs_python_cpu",
                    "comparison": f"{variant}_batch_vs_python_cpu",
                    "variant": variant,
                    "quantization": result.get("quantization", "unknown"),
                    "repack": result.get("repack", "default"),
                    "status": status_from_metrics(metrics, config.python_min_cos),
                    "tier": tolerance_tier(metrics["min_cos"]),
                    "threshold": config.python_min_cos,
                    **metrics,
                }
            )
            metrics = compare_vectors(cpp_case["batch"], cpp_case["single"])
            batch_threshold = (
                config.batch_min_cos
                if result.get("quantization") in {"fp16", "f16"}
                else min(config.batch_min_cos, config.quantized_batch_min_cos)
            )
            rows.append(
                {
                    "model_id": model_id,
                    "case": case_name,
                    "runner": "embeddings_cpp_batch_vs_single",
                    "comparison": f"{variant}_batch_vs_single",
                    "variant": variant,
                    "quantization": result.get("quantization", "unknown"),
                    "repack": result.get("repack", "default"),
                    "status": status_from_metrics(metrics, batch_threshold),
                    "tier": tolerance_tier(metrics["min_cos"]),
                    "threshold": batch_threshold,
                    **metrics,
                }
            )
    return rows


def collect_performance(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in result.get("performance", []):
            enriched = dict(row)
            enriched["status"] = result.get("status", "unknown")
            if "load_ms" in result:
                enriched["load_ms"] = result["load_ms"]
            if "load_rss_mb" in result:
                enriched["load_rss_mb"] = result["load_rss_mb"]
            rows.append(enriched)
        if result.get("status") != "ok" and not result.get("performance"):
            rows.append(
                {
                    "runner": result.get("runner"),
                    "model_id": result.get("model_id"),
                    "variant": result.get("variant", result.get("runner")),
                    "batch_size": "",
                    "status": "error",
                    "error": result.get("stderr", ""),
                }
            )
    return rows


def build_analysis(correctness: list[dict[str, Any]], performance: list[dict[str, Any]]) -> dict[str, Any]:
    valid_outputs = all(row["status"] != "invalid" for row in correctness)
    tolerance_passed = all(row["status"] == "within_tolerance" for row in correctness)
    worst_correctness: dict[str, dict[str, Any]] = {}
    variant_correctness: dict[str, dict[str, dict[str, Any]]] = {}
    for row in correctness:
        comparison = row.get("comparison", row.get("runner", "unknown"))
        current = worst_correctness.get(comparison)
        if current is None or row["min_cos"] < current["min_cos"]:
            worst_correctness[comparison] = {
                "case": row["case"],
                "variant": row.get("variant", "embeddings_cpp"),
                "status": row["status"],
                "tier": row.get("tier", "unknown"),
                "min_cos": row["min_cos"],
            }
        variant = row.get("variant", "embeddings_cpp")
        runner_name = row.get("runner", "")
        comparison_kind = (
            "python_cpu"
            if runner_name.endswith("_vs_python_cpu") or comparison.endswith("_vs_python_cpu")
            else "batch_vs_single"
        )
        current = variant_correctness.setdefault(variant, {}).get(comparison_kind)
        if current is None or row["min_cos"] < current["min_cos"]:
            variant_correctness[variant][comparison_kind] = {
                "case": row["case"],
                "status": row["status"],
                "tier": row.get("tier", "unknown"),
                "min_cos": row["min_cos"],
            }

    by_batch: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in performance:
        if row.get("status", "ok") != "ok" or not isinstance(row.get("batch_size"), int):
            continue
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
            load_rss_delta_mb = cpp_row.get("load_rss_mb", float("nan")) - python_row.get("load_rss_mb", float("nan"))
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
                    "python_load_rss_mb": python_row.get("load_rss_mb", float("nan")),
                    "cpp_load_rss_mb": cpp_row.get("load_rss_mb", float("nan")),
                    "load_rss_delta_mb_cpp_minus_python": load_rss_delta_mb,
                    "min_cos_vs_python_cpu": python_correctness.get("min_cos", float("nan")),
                    "correctness_status_vs_python_cpu": python_correctness.get("status", "missing"),
                    "min_cos_batch_vs_single": batch_correctness.get("min_cos", float("nan")),
                    "correctness_status_batch_vs_single": batch_correctness.get("status", "missing"),
                }
            )

    measured_batches = sorted({row["batch_size"] for row in performance_comparison})
    faster_batches = [
        batch_size
        for batch_size in measured_batches
        if any(row["batch_size"] == batch_size and row["faster_runner"] == "embeddings_cpp" for row in performance_comparison)
    ]
    slower_batches = [
        batch_size
        for batch_size in measured_batches
        if not any(row["batch_size"] == batch_size and row["faster_runner"] == "embeddings_cpp" for row in performance_comparison)
    ]
    lower_peak_rss = [
        batch_size
        for batch_size in measured_batches
        if any(row["batch_size"] == batch_size and row["rss_delta_mb_cpp_minus_python"] < 0 for row in performance_comparison)
    ]
    best_by_batch = []
    for batch_size in measured_batches:
        candidates = [row for row in performance_comparison if row["batch_size"] == batch_size]
        best_by_batch.append(max(candidates, key=lambda row: row["throughput_ratio_cpp_over_python"]))

    correctness_counts: dict[str, dict[str, int]] = {}
    for row in correctness:
        counts = correctness_counts.setdefault(row.get("variant", "embeddings_cpp"), {"within_tolerance": 0, "invalid": 0, "total": 0})
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
        recommendation = "embeddings.cpp is faster across the measured batch sizes; start quantization next while preserving the same correctness thresholds."

    conclusion = (
        f"Embedding output {'valid' if valid_outputs else 'invalid'}. "
        f"Tolerance {'met' if tolerance_passed else 'outside configured range'}. "
        f"embeddings.cpp is faster for batch sizes {faster_batches or []} and Python CPU is faster for "
        f"batch sizes {slower_batches or []}. "
        f"embeddings.cpp uses lower peak RSS for batch sizes {lower_peak_rss or []}."
    )
    return {
        "valid_outputs": valid_outputs,
        "tolerance_passed": tolerance_passed,
        "within_tolerance": tolerance_passed,
        "worst_correctness": worst_correctness,
        "variant_correctness": variant_correctness,
        "performance_comparison": performance_comparison,
        "best_by_batch": best_by_batch,
        "kquant_summary": kquant_summary,
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


def write_outputs(
    metadata: dict[str, Any],
    configs: list[ModelBenchConfig],
    correctness: list[dict[str, Any]],
    performance: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"model_bench_{stamp}.json"
    md_path = output_dir / f"model_bench_{stamp}.md"
    payload = {
        "metadata": metadata,
        "configs": [asdict(config) for config in configs],
        "correctness": correctness,
        "performance": performance,
        "analysis": analysis,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Model Benchmark Report\n\n")
        f.write(f"Generated: {metadata['generated_at']}\n\n")
        f.write("Protocol: `benchmarks/STANDARD.md`. Cosine drift is a tolerance metric; invalid output is failure.\n\n")
        f.write("## Conclusion\n\n")
        f.write(f"{analysis['conclusion']}\n\n")
        f.write(f"Recommendation: {analysis['recommendation']}\n\n")
        f.write("## Model Configs\n\n")
        f.write("| Model | Pooling | Dim | Python Tol | Batch Tol | Batch Sizes | GGUF Variants |\n")
        f.write("|---|---|---:|---:|---:|---|---|\n")
        for config in configs:
            f.write(
                f"| {config.model_id} | {config.pooling} | {config.embedding_dim} | "
                f"{config.python_min_cos} | {config.batch_min_cos} | "
                f"{', '.join(str(v) for v in config.batch_sizes)} | "
                f"{', '.join(config.gguf_paths)} |\n"
            )
        f.write("\n## Correctness\n\n")
        f.write("| Model | Case | Runner | Variant | Status | Tier | Threshold | Min Cos | Mean Cos | MSE | Max Abs |\n")
        f.write("|---|---|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in correctness:
            f.write(
                f"| {row['model_id']} | {row['case']} | {row['runner']} | {row['variant']} | "
                f"{row['status']} | {row['tier']} | {format_cell(row['threshold'])} | "
                f"{format_cell(row['min_cos'])} | {format_cell(row['mean_cos'])} | "
                f"{format_cell(row['mse'])} | {format_cell(row['max_abs'])} |\n"
            )
        f.write("\n## Kquant / Repack Summary\n\n")
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
        f.write("\n## Performance\n\n")
        f.write("| Model | Runner | Variant | Batch | Status | Mean ms | P50 ms | P95 ms | Text/s | RSS MB | Load RSS MB |\n")
        f.write("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in performance:
            f.write(
                f"| {row.get('model_id', '')} | {row.get('runner', '')} | {row.get('variant', '')} | "
                f"{row.get('batch_size', '')} | {row.get('status', '')} | "
                f"{format_cell(row.get('latency_ms_mean', float('nan')))} | "
                f"{format_cell(row.get('latency_ms_p50', float('nan')))} | "
                f"{format_cell(row.get('latency_ms_p95', float('nan')))} | "
                f"{format_cell(row.get('texts_per_second', float('nan')))} | "
                f"{format_cell(row.get('rss_mb', float('nan')))} | "
                f"{format_cell(row.get('load_rss_mb', float('nan')))} |\n"
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
        description="Unified model benchmark for Python CPU, TEI engine ORT, and embeddings.cpp."
    )
    parser.add_argument("--models", nargs="+", default=["BAAI/bge-m3"], help="Registered model ids or aliases.")
    parser.add_argument(
        "--runners",
        nargs="+",
        choices=("python_cpu", "tei_engine_ort", "embeddings_cpp"),
        default=DEFAULT_RUNNERS,
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--gguf-path",
        type=Path,
        help="Optional fp16 GGUF source path for a single model. Quantized paths are derived from this stem.",
    )
    parser.add_argument("--quantizations", nargs="+", default=["fp16"])
    parser.add_argument(
        "--quantize-missing",
        action="store_true",
        help="Create missing quantized GGUF variants with build/quantize before evaluating them.",
    )
    parser.add_argument("--quantize-bin", type=Path, default=ROOT / "build" / "quantize")
    parser.add_argument("--convert-missing", action="store_true", help="Convert missing fp16 GGUF variants before running.")
    parser.add_argument(
        "--repack-modes",
        nargs="+",
        choices=("default", "on", "off"),
        default=["default"],
        help="Evaluate embeddings.cpp with default, forced-on, or forced-off CPU repack.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--min-cos", type=float, help="Override Python baseline min cosine reporting tolerance.")
    parser.add_argument("--batch-min-cos", type=float, help="Override batch-vs-single min cosine reporting tolerance.")
    parser.add_argument(
        "--quantized-batch-min-cos",
        type=float,
        help="Override quantized batch-vs-single min cosine reporting tolerance. Defaults to --batch-min-cos.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with status 1 when any cosine metric is outside the configured tolerance.",
    )
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--torch-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--cpp-threads", type=int)
    parser.add_argument("--tei-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--tei-repo-dir", type=Path, default=ROOT.parent / "text-embeddings-inference")
    parser.add_argument("--tei-cache-dir", type=Path, default=ROOT / ".cache" / "tei")
    parser.add_argument("--scope", choices=("end_to_end", "engine_only"), default="end_to_end")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--print-config", action="store_true", help="Print resolved structured config and exit.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runner", choices=("python_cpu", "tei_engine_ort", "embeddings_cpp"), help=argparse.SUPPRESS)
    parser.add_argument("--model-id", help=argparse.SUPPRESS)
    parser.add_argument("--quantization", default="fp16", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="fp16", help=argparse.SUPPRESS)
    parser.add_argument("--repack", choices=("default", "on", "off"), default="default", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(args)
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.batch_sizes and any(size < 1 for size in args.batch_sizes):
        raise ValueError("--batch-sizes values must be positive")

    configs: list[ModelBenchConfig] = []
    specs = [get_model_spec(model_id) for model_id in args.models]
    if args.gguf_path and len(specs) != 1:
        raise ValueError("--gguf-path can only be used with a single --models entry")
    quantizations = []
    for quantization in args.quantizations:
        normalized = "fp16" if quantization == "f16" else quantization
        if normalized not in quantizations:
            quantizations.append(normalized)
    for spec in specs:
        config = model_config(spec, args.models_dir, quantizations)
        if args.gguf_path:
            config = ModelBenchConfig(
                **{
                    **asdict(config),
                    "gguf_paths": gguf_paths_from_override(args.gguf_path, quantizations),
                }
            )
        overrides = {}
        if args.batch_sizes:
            overrides["batch_sizes"] = args.batch_sizes
        if args.min_cos is not None:
            overrides["python_min_cos"] = args.min_cos
        if args.batch_min_cos is not None:
            overrides["batch_min_cos"] = args.batch_min_cos
        if args.quantized_batch_min_cos is not None:
            overrides["quantized_batch_min_cos"] = args.quantized_batch_min_cos
        elif args.batch_min_cos is not None:
            overrides["quantized_batch_min_cos"] = args.batch_min_cos
        if overrides:
            config = ModelBenchConfig(
                **{
                    **asdict(config),
                    **overrides,
                }
            )
        configs.append(config)

    if args.print_config:
        print(json.dumps([asdict(config) for config in configs], indent=2, ensure_ascii=False))
        return 0

    all_results: list[dict[str, Any]] = []
    all_correctness: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec, config in zip(specs, configs):
        print(f"benchmarking {spec.model_id}", file=sys.stderr, flush=True)
        fp16_path = Path(config.gguf_paths.get("fp16", args.models_dir / f"{spec.slug}.fp16.gguf"))
        if args.convert_missing:
            convert_missing(spec, fp16_path)
        for quantization, gguf_path_str in config.gguf_paths.items():
            ensure_quantized_model(args, fp16_path, quantization, Path(gguf_path_str))
        python_result = None
        cpp_results: list[dict[str, Any]] = []
        if "python_cpu" in args.runners:
            output_json = args.output_dir / f".model_bench_{spec.slug}_python_cpu.json"
            python_result = run_child(
                args,
                model_id=spec.model_id,
                runner="python_cpu",
                output_json=output_json,
                batch_sizes=config.batch_sizes,
            )
            all_results.append(python_result)
        if "embeddings_cpp" in args.runners:
            for quantization, gguf_path_str in config.gguf_paths.items():
                for repack in args.repack_modes:
                    variant = variant_label(quantization, repack)
                    safe_variant = variant.replace("+", "_").replace("/", "_")
                    output_json = args.output_dir / f".model_bench_{spec.slug}_embeddings_cpp_{safe_variant}.json"
                    result = run_child(
                        args,
                        model_id=spec.model_id,
                        runner="embeddings_cpp",
                        output_json=output_json,
                        batch_sizes=config.batch_sizes,
                        gguf_path=Path(gguf_path_str),
                        quantization=quantization,
                        variant=variant,
                        repack=repack,
                    )
                    cpp_results.append(result)
                    all_results.append(result)
        if "tei_engine_ort" in args.runners:
            output_json = args.output_dir / f".model_bench_{spec.slug}_tei_engine_ort.json"
            result = run_child(
                args,
                model_id=spec.model_id,
                runner="tei_engine_ort",
                output_json=output_json,
                batch_sizes=config.batch_sizes,
            )
            all_results.append(result)
        all_correctness.extend(build_correctness_rows(spec.model_id, python_result, cpp_results, config))

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "runners": args.runners,
        "fail_on_threshold": args.fail_on_threshold,
    }
    performance = collect_performance(all_results)
    analysis = build_analysis(all_correctness, performance)
    json_path, md_path = write_outputs(metadata, configs, all_correctness, performance, analysis, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    runner_errors = any(result.get("status") == "error" for result in all_results)
    threshold_exit = threshold_exit_code(all_correctness, args.fail_on_threshold)
    return 1 if runner_errors else threshold_exit


if __name__ == "__main__":
    raise SystemExit(main())
