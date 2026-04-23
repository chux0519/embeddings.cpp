#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
README = ROOT / "README.md"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = SCRIPTS_DIR / "output"
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    pooling: str
    tei_supported: bool = False
    max_length: int = 8192
    min_cos: float = 0.999

    @property
    def file_stem(self) -> str:
        return self.repo_id.split("/")[-1]


MODEL_SPECS: dict[str, ModelSpec] = {
    "BAAI/bge-m3": ModelSpec("BAAI/bge-m3", "cls"),
    "BAAI/bge-base-zh-v1.5": ModelSpec("BAAI/bge-base-zh-v1.5", "cls"),
    "shibing624/text2vec-base-multilingual": ModelSpec("shibing624/text2vec-base-multilingual", "mean"),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ModelSpec(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "mean", max_length=128
    ),
    "Snowflake/snowflake-arctic-embed-m-v2.0": ModelSpec(
        "Snowflake/snowflake-arctic-embed-m-v2.0", "cls", tei_supported=True, min_cos=0.96
    ),
}

QUANTIZATIONS = (
    "fp16",
    "q8_0",
    "q6_k",
    "q5_k",
    "q5_0",
    "q4_k",
    "q4_0",
    "q3_k",
    "q2_k",
    "q4_k_mlp_q8_attn",
)

CORRECTNESS_CASES: dict[str, list[str]] = {
    "single_en": ["The quick brown fox jumps over the lazy dog."],
    "single_zh": ["机器学习是人工智能的一个重要分支。"],
    "snowflake_batch_regression": ["A cute cat....", "A cute cat."],
    "mixed_batch_4": [
        "你好，今天天气怎么样？",
        "What's the weather like today?",
        "Embedding alignment should be stable in batch mode.",
        "今天天气很好，适合出去散步。",
    ],
    "duplicates_batch_8": [
        "same sentence",
        "same sentence",
        "same sentence.",
        "same sentence!",
        "A cute cat....",
        "A cute cat.",
        "短文本",
        "短文本",
    ],
    "length_skew_batch_4": [
        "short",
        " ".join(["long-context"] * 64),
        "符号、数字 12345 mixed tokens.",
        "newline separated text\nstill one embedding input",
    ],
}


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_readme_models() -> list[str]:
    if not README.exists():
        return list(MODEL_SPECS)
    text = README.read_text(encoding="utf-8")
    models = [repo_id for repo_id in MODEL_SPECS if repo_id in text]
    return models or list(MODEL_SPECS)


def load_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.models:
        repo_ids = args.models
    elif args.models_file:
        repo_ids = [
            line.strip()
            for line in Path(args.models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        repo_ids = parse_readme_models()

    specs: list[ModelSpec] = []
    for repo_id in repo_ids:
        specs.append(MODEL_SPECS.get(repo_id, ModelSpec(repo_id, args.default_pooling)))
    return specs


def cpp_pooling(pooling: str):
    from embeddings_cpp import PoolingMethod

    return PoolingMethod.CLS if pooling == "cls" else PoolingMethod.MEAN


def normalize_rows(values: np.ndarray) -> np.ndarray:
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-12)


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    import numpy as np

    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if candidate.shape != reference.shape:
        return {
            "mse": float("nan"),
            "max_abs": float("nan"),
            "mean_cos": float("nan"),
            "min_cos": float("nan"),
        }
    diff = candidate - reference
    cand_n = normalize_rows(candidate)
    ref_n = normalize_rows(reference)
    cos = np.sum(cand_n * ref_n, axis=1)
    return {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_cos": float(np.mean(cos)),
        "min_cos": float(np.min(cos)),
    }


def status_from_metrics(metrics: dict[str, float], min_cos_threshold: float) -> str:
    import numpy as np

    min_cos = metrics.get("min_cos", float("nan"))
    if np.isnan(min_cos):
        return "failed-shape"
    return "pass" if min_cos >= min_cos_threshold else "fail"


def rss_mb(pid: int | None = None) -> float:
    import psutil

    proc = psutil.Process(pid or os.getpid())
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / 1024 / 1024


class PythonRunner:
    name = "python_cpu"

    def __init__(self, spec: ModelSpec):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        try:
            self.model = AutoModel.from_pretrained(
                spec.repo_id,
                trust_remote_code=True,
                add_pooling_layer=False,
                use_memory_efficient_attention=False,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(spec.repo_id, trust_remote_code=True)
        self.model.eval()
        self.model.to("cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.spec.max_length,
        )
        if self.spec.repo_id.startswith("Snowflake/"):
            batch_size, seq_length = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            if self.spec.pooling == "cls":
                emb = hidden[:, 0]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).bool()
                hidden = hidden.masked_fill(~mask, 0)
                mask_f = mask.float()
                emb = torch.sum(hidden, dim=1) / torch.clamp(mask_f.sum(dim=1), min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()


class CppRunner:
    name = "embeddings_cpp"

    def __init__(self, spec: ModelSpec, gguf_path: Path):
        from embeddings_cpp import load

        self.spec = spec
        self.gguf_path = gguf_path
        self.model = load(spec.repo_id, gguf_path=str(gguf_path))
        self.pooling = self.model.pooling_method

    def encode(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        return np.asarray(self.model.batch_encode(texts, True, self.pooling), dtype=np.float32)

    def encode_one_by_one(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        return np.asarray([self.model.encode(text, True, self.pooling) for text in texts], dtype=np.float32)


class TeiRunner:
    name = "tei"

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def encode(self, texts: list[str]) -> np.ndarray:
        import numpy as np
        import requests

        response = requests.post(
            f"{self.url}/embed",
            json={"inputs": texts},
            timeout=300,
        )
        response.raise_for_status()
        values = response.json()
        if values and isinstance(values[0], (int, float)):
            values = [values]
        return normalize_rows(np.asarray(values, dtype=np.float32))


def gguf_path(models_dir: Path, spec: ModelSpec, quantization: str) -> Path:
    return models_dir / f"{spec.file_stem}.{quantization}.gguf"


def available_quantizations(models_dir: Path, spec: ModelSpec, requested: Iterable[str]) -> list[tuple[str, Path]]:
    found = []
    for quant in requested:
        path = gguf_path(models_dir, spec, quant)
        if path.exists():
            found.append((quant, path))
    return found


def convert_missing(spec: ModelSpec, models_dir: Path, quantization: str) -> Path:
    path = gguf_path(models_dir, spec, quantization)
    if path.exists():
        return path
    if quantization != "fp16":
        raise RuntimeError(f"Cannot auto-convert missing non-fp16 GGUF: {path}")
    models_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPTS_DIR / "convert.py"), spec.repo_id, str(path), "f16"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return path


def start_tei_container(args: argparse.Namespace, spec: ModelSpec) -> tuple[str, str]:
    port = int(args.tei_port)
    cache_dir = Path(args.tei_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"embeddings-cpp-tei-{re.sub(r'[^A-Za-z0-9_.-]+', '-', spec.file_stem).lower()}-{os.getpid()}"
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-p",
        f"{port}:80",
        "-v",
        f"{cache_dir}:/data",
    ]
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if os.environ.get(key):
            value = os.environ[key].replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
            cmd.extend(["-e", f"{key}={value}"])
    cmd.extend([
        args.tei_image,
        "--model-id",
        spec.repo_id,
        "--dtype",
        args.tei_dtype,
        "--pooling",
        spec.pooling,
        "--max-client-batch-size",
        str(args.max_batch_size),
        "--max-batch-tokens",
        str(args.tei_max_batch_tokens),
    ])
    container_id = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + args.tei_start_timeout
    while time.time() < deadline:
        try:
            import requests

            if requests.get(f"{url}/health", timeout=5).status_code < 500:
                return url, container_id
        except requests.RequestException:
            pass
        time.sleep(2)
    logs = subprocess.run(["docker", "logs", container_id], text=True, capture_output=True, check=False)
    raise RuntimeError(f"TEI did not become healthy for {spec.repo_id}\n{logs.stdout[-4000:]}\n{logs.stderr[-4000:]}")


def stop_container(container_id: str | None) -> None:
    if container_id:
        subprocess.run(["docker", "stop", container_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def docker_rss_mb(container_id: str | None) -> float | None:
    if not container_id:
        return None
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id],
        text=True,
        capture_output=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    used = out.stdout.split("/")[0].strip().upper().replace("IB", "B")
    match = re.match(r"([0-9.]+)\s*([KMGT]?B)", used)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    scale = {"B": 1 / 1024 / 1024, "KB": 1 / 1024, "MB": 1, "GB": 1024, "TB": 1024 * 1024}[unit]
    return value * scale


def benchmark_runner(runner, texts: list[str], warmup: int, iterations: int) -> dict[str, float]:
    import numpy as np

    for _ in range(warmup):
        runner.encode(texts)
    timings = []
    peak = rss_mb()
    for _ in range(iterations):
        before = time.perf_counter()
        runner.encode(texts)
        timings.append(time.perf_counter() - before)
        peak = max(peak, rss_mb())
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "latency_ms_mean": float(arr.mean() * 1000),
        "latency_ms_p50": float(np.percentile(arr, 50) * 1000),
        "latency_ms_p95": float(np.percentile(arr, 95) * 1000),
        "texts_per_second": float(len(texts) / arr.mean()),
        "rss_mb": float(peak),
    }


def write_outputs(rows: list[dict], perf_rows: list[dict], args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"alignment_{timestamp}.json"
    csv_path = output_dir / f"alignment_{timestamp}.csv"
    md_path = output_dir / f"alignment_{timestamp}.md"
    json_path.write_text(json.dumps({"correctness": rows, "performance": perf_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Embeddings Alignment Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("## Correctness\n\n")
        f.write("| Model | Case | Runner | Quant | Status | Min Cos | Mean Cos | MSE | Max Abs |\n")
        f.write("|---|---|---|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['case']} | {row['runner']} | {row['quantization']} | "
                f"{row['status']} | {row['min_cos']} | {row['mean_cos']} | {row['mse']} | {row['max_abs']} |\n"
            )
        f.write("\n## Performance\n\n")
        f.write("| Model | Runner | Quant | Batch | Mean ms | P95 ms | Text/s | RSS MB |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in perf_rows:
            f.write(
                f"| {row['model']} | {row['runner']} | {row['quantization']} | {row['batch_size']} | "
                f"{row['latency_ms_mean']} | {row['latency_ms_p95']} | {row['texts_per_second']} | {row['rss_mb']} |\n"
            )
    return json_path, csv_path, md_path


def rounded_row(base: dict) -> dict:
    out = dict(base)
    for key in ("mse", "max_abs", "mean_cos", "min_cos", "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "texts_per_second", "rss_mb"):
        if key in out and isinstance(out[key], float):
            import numpy as np

            out[key] = "nan" if np.isnan(out[key]) else f"{out[key]:.6g}"
    return out


def run_model(args: argparse.Namespace, spec: ModelSpec) -> tuple[list[dict], list[dict]]:
    eprint(f"\n== {spec.repo_id} ==")
    rows: list[dict] = []
    perf_rows: list[dict] = []
    tei_container: str | None = None

    if args.convert_missing:
        convert_missing(spec, Path(args.models_dir), "fp16")

    requested_quants = args.quantizations or ["fp16"]
    quants = available_quantizations(Path(args.models_dir), spec, requested_quants)
    if not quants:
        rows.append(
            rounded_row(
                {
                    "model": spec.repo_id,
                    "case": "model_load",
                    "runner": "embeddings_cpp",
                    "quantization": ",".join(requested_quants),
                    "status": "missing-gguf",
                    "mse": float("nan"),
                    "max_abs": float("nan"),
                    "mean_cos": float("nan"),
                    "min_cos": float("nan"),
                    "detail": f"No GGUF found in {args.models_dir}",
                }
            )
        )

    python_runner = None
    if not args.skip_python:
        eprint("Loading Python CPU reference")
        import torch

        torch.set_num_threads(args.torch_threads)
        python_runner = PythonRunner(spec)

    tei_runner = None
    if args.tei_url:
        tei_runner = TeiRunner(args.tei_url)
    elif args.tei_start and spec.tei_supported:
        eprint("Starting TEI container")
        url, tei_container = start_tei_container(args, spec)
        tei_runner = TeiRunner(url)
    elif args.tei_start:
        rows.append(
            rounded_row(
                {
                    "model": spec.repo_id,
                    "case": "tei_support",
                    "runner": "tei",
                    "quantization": "n/a",
                    "status": "skipped-unsupported",
                    "mse": float("nan"),
                    "max_abs": float("nan"),
                    "mean_cos": float("nan"),
                    "min_cos": float("nan"),
                    "detail": "Not marked as supported by this repo's TEI matrix",
                }
            )
        )

    try:
        references: dict[str, np.ndarray] = {}
        for case_name, texts in CORRECTNESS_CASES.items():
            if python_runner is None:
                continue
            references[case_name] = python_runner.encode(texts)

        for quant, path in quants:
            eprint(f"Loading C++ GGUF {path.name}")
            cpp_runner = CppRunner(spec, path)
            for case_name, texts in CORRECTNESS_CASES.items():
                try:
                    cpp_batch = cpp_runner.encode(texts)
                    if python_runner is not None:
                        metrics = compare(cpp_batch, references[case_name])
                        min_cos = min(args.min_cos, spec.min_cos)
                        rows.append(
                            rounded_row(
                                {
                                    "model": spec.repo_id,
                                    "case": case_name,
                                    "runner": "embeddings_cpp_vs_python",
                                    "quantization": quant,
                                    "status": status_from_metrics(metrics, min_cos),
                                    **metrics,
                                    "detail": "",
                                }
                            )
                        )
                    cpp_single = cpp_runner.encode_one_by_one(texts)
                    metrics = compare(cpp_batch, cpp_single)
                    batch_min_cos = args.batch_min_cos if quant == "fp16" else min(args.batch_min_cos, 0.999)
                    rows.append(
                        rounded_row(
                            {
                                "model": spec.repo_id,
                                "case": f"{case_name}:batch_vs_single",
                                "runner": "embeddings_cpp_internal",
                                "quantization": quant,
                                "status": status_from_metrics(metrics, batch_min_cos),
                                **metrics,
                                "detail": "",
                            }
                        )
                    )
                except Exception as exc:
                    rows.append(
                        rounded_row(
                            {
                                "model": spec.repo_id,
                                "case": case_name,
                                "runner": "embeddings_cpp",
                                "quantization": quant,
                                "status": "error",
                                "mse": float("nan"),
                                "max_abs": float("nan"),
                                "mean_cos": float("nan"),
                                "min_cos": float("nan"),
                                "detail": repr(exc),
                            }
                        )
                    )

            if args.benchmark:
                for batch_size in args.benchmark_batches:
                    texts = (CORRECTNESS_CASES["mixed_batch_4"] * ((batch_size + 3) // 4))[:batch_size]
                    perf = benchmark_runner(cpp_runner, texts, args.warmup, args.iterations)
                    perf_rows.append(
                        rounded_row(
                            {
                                "model": spec.repo_id,
                                "runner": "embeddings_cpp",
                                "quantization": quant,
                                "batch_size": batch_size,
                                **perf,
                            }
                        )
                    )

        if python_runner is not None and args.benchmark:
            for batch_size in args.benchmark_batches:
                texts = (CORRECTNESS_CASES["mixed_batch_4"] * ((batch_size + 3) // 4))[:batch_size]
                perf = benchmark_runner(python_runner, texts, args.warmup, args.iterations)
                perf_rows.append(
                    rounded_row(
                        {
                            "model": spec.repo_id,
                            "runner": "python_cpu",
                            "quantization": "fp32",
                            "batch_size": batch_size,
                            **perf,
                        }
                    )
                )

        if tei_runner is not None:
            for case_name, texts in CORRECTNESS_CASES.items():
                try:
                    tei_values = tei_runner.encode(texts)
                    if python_runner is not None:
                        metrics = compare(tei_values, references[case_name])
                        min_cos = min(args.min_cos, spec.min_cos)
                        rows.append(
                            rounded_row(
                                {
                                    "model": spec.repo_id,
                                    "case": case_name,
                                    "runner": "tei_vs_python",
                                    "quantization": "n/a",
                                    "status": status_from_metrics(metrics, min_cos),
                                    **metrics,
                                    "detail": "",
                                }
                            )
                        )
                except Exception as exc:
                    rows.append(
                        rounded_row(
                            {
                                "model": spec.repo_id,
                                "case": case_name,
                                "runner": "tei",
                                "quantization": "n/a",
                                "status": "error",
                                "mse": float("nan"),
                                "max_abs": float("nan"),
                                "mean_cos": float("nan"),
                                "min_cos": float("nan"),
                                "detail": repr(exc),
                            }
                        )
                    )
            if args.benchmark:
                for batch_size in args.benchmark_batches:
                    texts = (CORRECTNESS_CASES["mixed_batch_4"] * ((batch_size + 3) // 4))[:batch_size]
                    perf = benchmark_runner(tei_runner, texts, args.warmup, args.iterations)
                    if tei_container:
                        docker_mem = docker_rss_mb(tei_container)
                        if docker_mem is not None:
                            perf["rss_mb"] = docker_mem
                    perf_rows.append(
                        rounded_row(
                            {
                                "model": spec.repo_id,
                                "runner": "tei",
                                "quantization": "n/a",
                                "batch_size": batch_size,
                                **perf,
                            }
                        )
                    )
    finally:
        stop_container(tei_container)

    return rows, perf_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare embeddings.cpp against Python CPU and optionally TEI.")
    parser.add_argument("--models", nargs="+", help="HF repo IDs to test. Defaults to model IDs found in README.")
    parser.add_argument("--models-file", help="Optional file with one HF repo ID per line.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--quantizations", nargs="+", choices=QUANTIZATIONS, default=["fp16"])
    parser.add_argument("--convert-missing", action="store_true", help="Convert missing fp16 GGUF files before testing.")
    parser.add_argument("--default-pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--batch-min-cos", type=float, default=0.999999)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--tei-url", help="Existing TEI base URL, for example http://127.0.0.1:8080")
    parser.add_argument("--tei-start", action="store_true", help="Start a TEI CPU Docker container for marked-supported models.")
    parser.add_argument("--tei-image", default="ghcr.io/huggingface/text-embeddings-inference:cpu-1.9")
    parser.add_argument("--tei-port", type=int, default=8080)
    parser.add_argument("--tei-cache-dir", default=str(ROOT / ".cache" / "tei"))
    parser.add_argument("--tei-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--tei-max-batch-tokens", type=int, default=16384)
    parser.add_argument("--tei-start-timeout", type=int, default=900)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cpp-threads", type=int, help="Set EMBEDDINGS_CPP_THREADS for this run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cpp_threads:
        os.environ["EMBEDDINGS_CPP_THREADS"] = str(args.cpp_threads)
    specs = load_models(args)
    all_rows: list[dict] = []
    all_perf_rows: list[dict] = []
    eprint(f"Testing {len(specs)} model(s): {', '.join(spec.repo_id for spec in specs)}")
    for spec in specs:
        rows, perf_rows = run_model(args, spec)
        all_rows.extend(rows)
        all_perf_rows.extend(perf_rows)
    json_path, csv_path, md_path = write_outputs(all_rows, all_perf_rows, args)
    eprint(f"\nWrote {json_path}")
    eprint(f"Wrote {csv_path}")
    eprint(f"Wrote {md_path}")
    failed = [row for row in all_rows if row.get("status", "").startswith(("fail", "error", "missing"))]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
