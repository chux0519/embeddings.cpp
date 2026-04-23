#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "scripts" / "output"
DEFAULT_MODEL = ROOT / "models" / "snowflake-arctic-embed-m-v2.0.q8_0.gguf"
DEFAULT_REPO_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
DEFAULT_TEI_REPO = ROOT.parent / "text-embeddings-inference"
TEXTS = [
    "你好，今天天气怎么样？",
    "What's the weather like today?",
    "Embedding alignment should be stable in batch mode.",
    "今天天气很好，适合出去散步。",
]


def make_texts(batch_size: int) -> list[str]:
    return (TEXTS * ((batch_size + len(TEXTS) - 1) // len(TEXTS)))[:batch_size]


def rss_mb() -> float:
    import psutil

    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / 1024 / 1024


def resolve_model_root(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    return Path(
        snapshot_download(
            repo_id,
            local_files_only=offline,
            allow_patterns=[
                "config.json",
                "modules.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "sentence_*",
                "config_sentence_transformers.json",
                "*.safetensors",
                "*.bin",
                "*.json",
            ],
        )
    )


def resolve_tei_model_root(repo_id: str, tei_cache_dir: Path) -> Path:
    model_key = repo_id.replace("/", "--")
    snapshots_dir = tei_cache_dir / f"models--{model_key}" / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
        if snapshots:
            return snapshots[-1]

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "onnx/model.onnx",
                "onnx/model.onnx_data",
                "model.onnx",
                "model.onnx_data",
                "modules.json",
                "config_sentence_transformers.json",
                "sentence_*",
            ],
        )
    )


def tei_engine_harness_source() -> str:
    return r"""use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use text_embeddings_backend::{Backend as TeiBackend, DType, ModelType, Pool};
use text_embeddings_backend_core::{Batch, Embedding};
use tokenizers::{Encoding, Tokenizer};

#[derive(Deserialize)]
struct InputPayload {
    model_root: String,
    texts: Vec<String>,
    warmup: usize,
    iterations: usize,
    batch_size: usize,
    threads: usize,
}

#[derive(Serialize)]
struct OutputPayload {
    effective_threads: usize,
    threads: usize,
    runner: &'static str,
    batch_size: usize,
    iterations: usize,
    warmup: usize,
    latency_ms_mean: f64,
    latency_ms_p50: f64,
    latency_ms_p95: f64,
    texts_per_second: f64,
    rss_mb: f64,
}

fn load_tokenizer(model_root: &std::path::Path) -> Result<Tokenizer> {
    let tokenizer_path = model_root.join("tokenizer.json");
    let mut tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

fn batch(encodings: Vec<Encoding>) -> Batch {
    let mut input_ids = Vec::new();
    let mut token_type_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths = Vec::with_capacity(encodings.len() + 1);
    cumulative_seq_lengths.push(0);
    let mut max_length = 0;
    let mut cumulative_length = 0;

    for encoding in &encodings {
        let encoding_length = encoding.len() as u32;
        input_ids.extend(encoding.get_ids().to_vec());
        token_type_ids.extend(encoding.get_type_ids().to_vec());
        position_ids.extend(0..encoding_length);
        cumulative_length += encoding_length;
        cumulative_seq_lengths.push(cumulative_length);
        max_length = max(max_length, encoding_length);
    }

    Batch {
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths,
        max_length,
        pooled_indices: (0..encodings.len() as u32).collect(),
        raw_indices: vec![],
    }
}

fn rss_mb() -> f64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn percentile_ms(mut values_ms: Vec<f64>, percentile: f64) -> f64 {
    values_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((values_ms.len() - 1) as f64 * percentile).round() as usize;
    values_ms[idx]
}

async fn encode_once(backend: &TeiBackend, tokenizer: &Tokenizer, texts: &[String]) -> Result<()> {
    let encodings = texts
        .iter()
        .map(|text| {
            tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let (embeddings, _) = backend.embed(batch(encodings)).await?;
    for (_, embedding) in embeddings {
        match embedding {
            Embedding::Pooled(v) => {
                let _ = v.len();
            }
            Embedding::All(v) => {
                let _ = v.len();
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let input_path = std::env::args()
        .nth(1)
        .context("missing input json path")?;
    let input: InputPayload = serde_json::from_str(&fs::read_to_string(&input_path)?)?;
    let model_root = PathBuf::from(input.model_root);
    let tokenizer = load_tokenizer(&model_root)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let backend = runtime.block_on(TeiBackend::new(
        model_root,
        None,
        DType::Float32,
        ModelType::Embedding(Pool::Cls),
        None,
        String::new(),
        None,
        "tei-engine-bench".to_string(),
    ))?;

    for _ in 0..input.warmup {
        runtime.block_on(encode_once(&backend, &tokenizer, &input.texts))?;
    }

    let mut timings_ms = Vec::with_capacity(input.iterations);
    let mut peak_rss = rss_mb();
    for _ in 0..input.iterations {
        let start = Instant::now();
        runtime.block_on(encode_once(&backend, &tokenizer, &input.texts))?;
        timings_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        peak_rss = peak_rss.max(rss_mb());
    }

    let mean_ms = timings_ms.iter().sum::<f64>() / timings_ms.len() as f64;
    let output = OutputPayload {
        effective_threads: input.threads,
        threads: input.threads,
        runner: "tei_engine",
        batch_size: input.batch_size,
        iterations: input.iterations,
        warmup: input.warmup,
        latency_ms_mean: mean_ms,
        latency_ms_p50: percentile_ms(timings_ms.clone(), 0.50),
        latency_ms_p95: percentile_ms(timings_ms.clone(), 0.95),
        texts_per_second: input.batch_size as f64 / (mean_ms / 1000.0),
        rss_mb: peak_rss,
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
"""


def tei_backend_features(backend: str) -> list[str]:
    if backend == "ort":
        return ["ort"]
    if backend == "candle":
        return ["candle"]
    raise ValueError(f"unsupported TEI backend: {backend}")


def benchmark_tei_engine(args: argparse.Namespace, threads: int, texts: list[str]) -> dict:
    tei_repo_dir = Path(args.tei_repo_dir).resolve()
    if not tei_repo_dir.exists():
        raise FileNotFoundError(f"TEI repo not found: {tei_repo_dir}")

    if shutil.which("cargo") is None:
        raise RuntimeError("cargo is required for tei_engine benchmark")

    if args.tei_backend == "ort":
        model_root = resolve_tei_model_root(args.repo_id, Path(args.tei_cache_dir).resolve())
    elif args.tei_backend == "candle":
        model_root = resolve_model_root(args.repo_id)
    else:
        raise ValueError(f"unsupported TEI backend: {args.tei_backend}")
    with tempfile.TemporaryDirectory(prefix="tei-engine-bench-") as tmp:
        tmpdir = Path(tmp)
        src_dir = tmpdir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        features = ", ".join(f'"{feature}"' for feature in tei_backend_features(args.tei_backend))
        cargo_toml = f"""
[package]
name = "tei-engine-bench"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
tokenizers = {{ version = "0.21.0", default-features = false, features = ["onig", "esaxx_fast"] }}
tokio = {{ version = "1.25", features = ["rt", "sync"] }}
text-embeddings-backend = {{ path = "{(tei_repo_dir / 'backends').as_posix()}", features = [{features}] }}
text-embeddings-backend-core = {{ path = "{(tei_repo_dir / 'backends' / 'core').as_posix()}" }}

[patch.crates-io]
cudarc = {{ git = "https://github.com/Narsil/cudarc", rev = "8b4f18b4bcd5e4b1a9daf40abc3a2e27f83f06e9" }}
candle = {{ git = "https://github.com/huggingface/candle", rev = "6381023982251959a2c9bab7378b3013304e192b", package = "candle-core" }}
candle-nn = {{ git = "https://github.com/huggingface/candle", rev = "6381023982251959a2c9bab7378b3013304e192b", package = "candle-nn" }}
candle-transformers = {{ git = "https://github.com/huggingface/candle", rev = "6381023982251959a2c9bab7378b3013304e192b", package = "candle-transformers" }}
candle-flash-attn = {{ git = "https://github.com/huggingface/candle", rev = "6381023982251959a2c9bab7378b3013304e192b", package = "candle-flash-attn" }}
"""
        (tmpdir / "Cargo.toml").write_text(cargo_toml.strip() + "\n", encoding="utf-8")
        (src_dir / "main.rs").write_text(tei_engine_harness_source(), encoding="utf-8")
        input_path = tmpdir / "input.json"
        input_path.write_text(
            json.dumps(
                {
                    "model_root": str(model_root),
                    "texts": texts,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "batch_size": len(texts),
                    "threads": threads,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env.setdefault("CARGO_TARGET_DIR", str(ROOT / ".cache" / "tei_engine_target"))
        cmd = ["cargo", "run", "--release", "--quiet", "--", str(input_path)]
        completed = subprocess.run(
            cmd,
            cwd=tmpdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])

    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            row = json.loads(line)
            row["runner"] = f"tei_engine_{args.tei_backend}"
            row["tei_backend"] = args.tei_backend
            return row
    raise RuntimeError(f"tei_engine produced no json output:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")


def worker(args: argparse.Namespace) -> int:
    import numpy as np

    threads = args.threads[0] if isinstance(args.threads, list) else args.threads
    texts = make_texts(args.batch_size)

    if args.runner == "embeddings_cpp":
        from embeddings_cpp import load

        os.environ["EMBEDDINGS_CPP_THREADS"] = str(threads)
        model = load(args.repo_id, gguf_path=str(args.model))

        def encode() -> None:
            model.batch_encode(texts, True)

    elif args.runner == "python_cpu":
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.repo_id,
            trust_remote_code=True,
            add_pooling_layer=False,
            use_memory_efficient_attention=False,
        )
        model.eval()
        model.to("cpu")

        def encode() -> None:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=8192)
            batch_size, seq_length = inputs["input_ids"].shape
            inputs["position_ids"] = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            with torch.no_grad():
                outputs = model(**inputs)
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                emb = hidden[:, 0]
                torch.nn.functional.normalize(emb, p=2, dim=1)

    elif args.runner == "tei_engine":
        row = benchmark_tei_engine(args, threads, texts)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return 0

    else:
        raise ValueError(f"unsupported runner: {args.runner}")

    for _ in range(args.warmup):
        encode()

    timings: list[float] = []
    peak_rss = rss_mb()
    for _ in range(args.iterations):
        start = time.perf_counter()
        encode()
        timings.append(time.perf_counter() - start)
        peak_rss = max(peak_rss, rss_mb())

    arr = np.asarray(timings, dtype=np.float64)
    print(
        json.dumps(
            {
                "effective_threads": threads,
                "threads": threads,
                "runner": args.runner,
                "batch_size": args.batch_size,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "latency_ms_mean": float(arr.mean() * 1000),
                "latency_ms_p50": float(np.percentile(arr, 50) * 1000),
                "latency_ms_p95": float(np.percentile(arr, 95) * 1000),
                "texts_per_second": float(args.batch_size / arr.mean()),
                "rss_mb": float(peak_rss),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def run_child(args: argparse.Namespace, threads: int, batch_size: int) -> dict:
    env = os.environ.copy()
    env.setdefault("NO_PROXY", "127.0.0.1,localhost")
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["EMBEDDINGS_CPP_THREADS"] = str(threads)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(args.model),
        "--repo-id",
        args.repo_id,
        "--runner",
        args.runner,
        "--threads",
        str(threads),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timeout",
        str(args.timeout),
    ]
    if args.runner == "tei_engine":
        cmd.extend(["--tei-backend", args.tei_backend])
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
            "threads": threads,
            "batch_size": batch_size,
            "status": "error",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-4000:],
        }
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            row = json.loads(line)
            row["status"] = "ok"
            return row
    return {
        "threads": threads,
        "batch_size": batch_size,
        "status": "error",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-4000:],
    }


def write_outputs(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"snowflake_profile_{timestamp}.json"
    md_path = output_dir / f"snowflake_profile_{timestamp}.md"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Snowflake Profile\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| Runner | Threads | Batch | Status | Mean ms | P50 ms | P95 ms | Text/s | RSS MB |\n")
        f.write("|---|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('runner', '')} | {row.get('threads', '')} | {row.get('batch_size', '')} | {row.get('status', '')} | "
                f"{float(row.get('latency_ms_mean', 0.0)):.2f} | "
                f"{float(row.get('latency_ms_p50', 0.0)):.2f} | "
                f"{float(row.get('latency_ms_p95', 0.0)):.2f} | "
                f"{float(row.get('texts_per_second', 0.0)):.2f} | "
                f"{float(row.get('rss_mb', 0.0)):.1f} |\n"
            )
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible Snowflake CPU profiling for embeddings.cpp.")
    parser.add_argument("--runner", choices=("embeddings_cpp", "python_cpu", "tei_engine"), default="embeddings_cpp")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--tei-repo-dir", type=Path, default=DEFAULT_TEI_REPO)
    parser.add_argument("--tei-cache-dir", type=Path, default=ROOT / ".cache" / "tei")
    parser.add_argument("--tei-backend", choices=("ort", "candle"), default="ort")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10, 12, 16])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--batch-size", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)

    rows: list[dict] = []
    for batch_size in args.batch_sizes:
        for threads in args.threads:
            print(f"profiling runner={args.runner} batch={batch_size} threads={threads}", file=sys.stderr, flush=True)
            row = run_child(args, threads, batch_size)
            rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"  {row['texts_per_second']:.2f} text/s, {row['latency_ms_mean']:.2f} ms, {row['rss_mb']:.1f} MB",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"  error: {row.get('stderr', row.get('stdout', ''))[-400:]}", file=sys.stderr, flush=True)

    json_path, md_path = write_outputs(rows, args.output_dir)
    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Wrote {md_path}", file=sys.stderr)
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
