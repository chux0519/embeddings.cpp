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
        batch_sizes=[int(v) for v in benchmark.get("batch_sizes", [1, 4, 8])],
        max_batch_tokens=benchmark.get("max_batch_tokens"),
        gguf_paths=gguf_paths,
    )


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
                "batch_size": batch_size,
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
                "gguf_path": str(gguf_path),
                "batch_size": batch_size,
                "api": "encode" if batch_size == 1 else "batch_encode",
                **benchmark_runner(encode, batch_size, args.warmup, args.iterations, args.seed + batch_size * 97),
            }
        )
    return {
        "runner": "embeddings_cpp",
        "model_id": spec.model_id,
        "variant": args.variant,
        "quantization": args.quantization,
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
                "batch_size": batch_size,
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
        cmd.extend(["--quantization", quantization, "--variant", quantization])

    env = os.environ.copy()
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    if args.cpp_threads:
        env["EMBEDDINGS_CPP_THREADS"] = str(args.cpp_threads)

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
                    "variant": variant,
                    "status": status_from_metrics(metrics, config.python_min_cos),
                    "tier": tolerance_tier(metrics["min_cos"]),
                    "threshold": config.python_min_cos,
                    **metrics,
                }
            )
            metrics = compare_vectors(cpp_case["batch"], cpp_case["single"])
            rows.append(
                {
                    "model_id": model_id,
                    "case": case_name,
                    "runner": "embeddings_cpp_batch_vs_single",
                    "variant": variant,
                    "status": status_from_metrics(metrics, config.batch_min_cos),
                    "tier": tolerance_tier(metrics["min_cos"]),
                    "threshold": config.batch_min_cos,
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


def write_outputs(
    metadata: dict[str, Any],
    configs: list[ModelBenchConfig],
    correctness: list[dict[str, Any]],
    performance: list[dict[str, Any]],
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
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Model Benchmark Report\n\n")
        f.write(f"Generated: {metadata['generated_at']}\n\n")
        f.write("Protocol: `benchmarks/STANDARD.md`. Cosine drift is a tolerance metric; invalid output is failure.\n\n")
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
    parser.add_argument("--quantizations", nargs="+", default=["fp16"])
    parser.add_argument("--batch-sizes", nargs="+", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
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
    parser.add_argument("--gguf-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--quantization", default="fp16", help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="fp16", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(args)

    configs: list[ModelBenchConfig] = []
    specs = [get_model_spec(model_id) for model_id in args.models]
    for spec in specs:
        config = model_config(spec, args.models_dir, args.quantizations)
        if args.batch_sizes:
            config = ModelBenchConfig(
                **{
                    **asdict(config),
                    "batch_sizes": args.batch_sizes,
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
                output_json = args.output_dir / f".model_bench_{spec.slug}_embeddings_cpp_{quantization}.json"
                result = run_child(
                    args,
                    model_id=spec.model_id,
                    runner="embeddings_cpp",
                    output_json=output_json,
                    batch_sizes=config.batch_sizes,
                    gguf_path=Path(gguf_path_str),
                    quantization=quantization,
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
    }
    performance = collect_performance(all_results)
    json_path, md_path = write_outputs(metadata, configs, all_correctness, performance, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    invalid = any(row["status"] == "invalid" for row in all_correctness)
    runner_errors = any(result.get("status") == "error" for result in all_results)
    return 1 if invalid or runner_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
