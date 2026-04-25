#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
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
REALISTIC_TEXT_POOL = [
    "你好，今天天气怎么样？",
    "What's the weather like today?",
    "Embedding alignment should be stable in batch mode.",
    "今天天气很好，适合出去散步。",
    "How do I reset my password if I lost access to my phone?",
    "订单已经支付，但是页面一直显示待付款。",
    "The invoice total does not match the amount charged to my card.",
    "帮我找一下最近三个月关于向量数据库性能优化的笔记。",
    "Summarize the security implications of exposing an internal metrics endpoint.",
    "用户反馈搜索结果相关性下降，尤其是中文短查询。",
    "What are the differences between cosine similarity and dot product for normalized embeddings?",
    "这个接口在批量请求时偶尔超时，需要排查日志。",
    "A customer wants to migrate from a hosted embedding API to an on-premise service.",
    "请把 Kubernetes deployment 里的 CPU request 和 limit 调整到更保守的配置。",
    "The query contains a product SKU, a city name, and a short complaint about delivery.",
    "会议纪要：模型服务内存占用过高，优先评估量化方案。",
    "Compare Snowflake Arctic Embed with BGE-M3 for multilingual retrieval.",
    "代码审查时发现 tokenizer 的 max length 没有统一配置。",
    "The legal team asked whether user generated text is stored after embedding.",
    "搜索“退款多久到账”时，应该优先返回财务和客服知识库。",
    "A support ticket says the CPU container was killed after a spike in concurrent requests.",
    "请根据这段报错定位可能的依赖冲突：undefined symbol: cblas_sgemm.",
    "We need a benchmark that reports p50, p95, throughput, and resident set size.",
    "产品文档里提到的模型都应该有正确性回归测试。",
    "The input can be a single sentence, a paragraph, or a mixed-language query.",
    "用户输入包含换行符\n第二行仍然属于同一个 embedding 请求。",
    "How should I tune max batch tokens for a CPU-only text embedding deployment?",
    "公司内部知识库需要支持英文、中文和少量代码片段检索。",
    "请查找关于 OAuth 回调地址配置错误的排障步骤。",
    "The service returns HTTP 413 when the request exceeds the configured token budget.",
    "我们希望替换 text-embeddings-inference，因为当前内存占用太高。",
    "Short query",
    "短文本",
    "A cute cat.",
    "A cute cat....",
    "same sentence",
    "same sentence.",
    "符号、数字 12345 mixed tokens.",
    "newline separated text\nstill one embedding input",
    "The deployment uses two CPU cores and sixteen gigabytes of memory in production.",
    "这是一条较长的用户问题，包含背景、目标和约束：我们需要在不增加机器规格的情况下提升检索吞吐，并且保持召回质量稳定。",
    "When comparing engines, avoid mixing fp32 baselines with int4 production tradeoffs in the same headline table.",
    "请生成一份说明，解释为什么同一个模型在 ORT 后端和 Candle 后端的 CPU 表现差异很大。",
    "The benchmark should use realistic randomized inputs instead of repeating the same four sentences.",
    "A developer asks whether batching empty strings should return an error or a zero vector.",
    "请检查 README 里的 Docker 示例是否包含持久化模型缓存目录。",
    "The current implementation compresses valid tokens, runs attention per sequence, then concatenates outputs.",
    "我们需要知道 batch size 从 1 到 16 时吞吐是否稳定增长。",
    "Find documents about memory bandwidth, quantized matrix multiplication, and CPU cache behavior.",
    "这条请求模拟真实生产流量：短查询、长查询、中文、英文和重复用户意图混在一起。",
    "What is the expected cosine drift when using q4_k_mlp_q8_attn compared with fp32?",
    "请根据用户输入判断最相关的知识库文章，而不是生成答案。",
    "The router is not the bottleneck; the backend engine and graph optimization dominate latency.",
    "需要把 benchmark 结果写进 README，并标注测试平台和线程数。",
    "A search query about billing: invoice failed, payment succeeded, receipt missing.",
    "请帮我对齐 Python CPU、TEI ORT 和 embeddings.cpp 的输出向量。",
    "The model should handle multilingual retrieval queries with punctuation, numbers, and whitespace.",
    "用户在生产服务里通过 HTTP 调用 embedding endpoint，每次请求可能包含多条文本。",
    "We should measure engine-only latency separately from HTTP service latency.",
    "这是一段模拟日志：request_id=abc latency_ms=92 batch=8 tokens=134 status=200.",
    "Explain why resident set size matters when replacing a text embedding service in Kubernetes.",
    "The customer says recall got worse after quantization, but memory usage improved significantly.",
    "请把这段自然语言问题转换成可以检索内部文档的 embedding。",
    "A medium length English sentence with several clauses, commas, and domain-specific terms about embeddings.",
    "模型加载之后的常驻内存，比单次请求峰值更适合做容量规划。",
    "The input text may include code like `torch.set_num_threads(12)` and command flags.",
    "请问 snowflake-arctic-embed-m-v2.0 的 batch 行为是否和单条编码一致？",
    "We need deterministic benchmarks, but deterministic should not mean identical repeated samples.",
    "用户希望用本地 GGUF 文件启动服务，不依赖外部网络下载。",
    "This paragraph is intentionally longer to simulate a knowledge base passage. It mentions deployment constraints, CPU-only inference, quantization strategy, latency percentiles, memory footprint, and correctness thresholds for embedding vectors.",
    "另一个较长中文段落，用于模拟知识库中的FAQ答案。内容包括服务部署、模型缓存、Docker镜像、Hugging Face模型地址、以及如何通过环境变量控制线程数量。",
]


def make_texts(batch_size: int, seed: int) -> list[str]:
    rng = random.Random(seed + batch_size * 1009)
    if batch_size <= len(REALISTIC_TEXT_POOL):
        return rng.sample(REALISTIC_TEXT_POOL, batch_size)
    return [rng.choice(REALISTIC_TEXT_POOL) for _ in range(batch_size)]


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
    scope: String,
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

async fn encode_cached(backend: &TeiBackend, encodings: &[Encoding]) -> Result<()> {
    let (embeddings, _) = backend.embed(batch(encodings.to_vec())).await?;
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
    let cached_encodings = if input.scope == "engine_only" {
        Some(
            input
                .texts
                .iter()
                .map(|text| {
                    tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    for _ in 0..input.warmup {
        if let Some(encodings) = &cached_encodings {
            runtime.block_on(encode_cached(&backend, encodings))?;
        } else {
            runtime.block_on(encode_once(&backend, &tokenizer, &input.texts))?;
        }
    }

    let mut timings_ms = Vec::with_capacity(input.iterations);
    let mut peak_rss = rss_mb();
    for _ in 0..input.iterations {
        let start = Instant::now();
        if let Some(encodings) = &cached_encodings {
            runtime.block_on(encode_cached(&backend, encodings))?;
        } else {
            runtime.block_on(encode_once(&backend, &tokenizer, &input.texts))?;
        }
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
                    "scope": args.scope,
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
            row["scope"] = args.scope
            return row
    raise RuntimeError(f"tei_engine produced no json output:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")


def worker(args: argparse.Namespace) -> int:
    import numpy as np

    threads = args.threads[0] if isinstance(args.threads, list) else args.threads
    texts = make_texts(args.batch_size, args.seed)

    if args.runner == "embeddings_cpp":
        if args.scope != "end_to_end":
            raise ValueError("embeddings_cpp currently supports only --scope end_to_end")
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

        if args.scope == "engine_only":
            cached_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=8192)
            batch_size, seq_length = cached_inputs["input_ids"].shape
            cached_inputs["position_ids"] = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

            def encode() -> None:
                with torch.no_grad():
                    outputs = model(**cached_inputs)
                    hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                    emb = hidden[:, 0]
                    torch.nn.functional.normalize(emb, p=2, dim=1)
        else:
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
                "scope": args.scope,
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
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.timeout),
        "--scope",
        str(args.scope),
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
        f.write("This report is intended for serial benchmark runs. Do not run multiple benchmark processes on the same host when collecting headline numbers.\n\n")
        f.write("| Runner | Scope | Threads | Batch | Status | Mean ms | P50 ms | P95 ms | Text/s | RSS MB |\n")
        f.write("|---|---|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('runner', '')} | {row.get('scope', 'end_to_end')} | {row.get('threads', '')} | {row.get('batch_size', '')} | {row.get('status', '')} | "
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
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--scope", choices=("end_to_end", "engine_only"), default="end_to_end")
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
                    f"  p50 {row['latency_ms_p50']:.2f} ms, p95 {row['latency_ms_p95']:.2f} ms, "
                    f"{row['texts_per_second']:.2f} text/s, {row['rss_mb']:.1f} MB",
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
