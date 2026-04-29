from __future__ import annotations

import math
import os
import random
import statistics
import threading
from typing import Any


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


def make_realistic_texts(batch_size: int, seed: int) -> list[str]:
    rng = random.Random(seed + batch_size * 1009)
    if batch_size <= len(REALISTIC_TEXT_POOL):
        return rng.sample(REALISTIC_TEXT_POOL, batch_size)
    return [rng.choice(REALISTIC_TEXT_POOL) for _ in range(batch_size)]


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


class RssSampler:
    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.peak_mb = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "RssSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb())

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            self._stop.wait(self.interval_s)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize_timings(timings: list[float], batch_size: int, peak_rss: float) -> dict[str, float]:
    mean = statistics.fmean(timings)
    return {
        "latency_ms_mean": mean * 1000,
        "latency_ms_p50": percentile(timings, 50) * 1000,
        "latency_ms_p95": percentile(timings, 95) * 1000,
        "texts_per_second": batch_size / mean,
        "rss_mb": peak_rss,
    }


def normalize_row(row: list[float]) -> list[float]:
    denom = math.sqrt(sum(value * value for value in row))
    if denom <= 1e-12:
        return [0.0 for _ in row]
    return [value / denom for value in row]


def compare_vectors(candidate: list[list[float]], reference: list[list[float]]) -> dict[str, float]:
    if len(candidate) != len(reference) or any(len(a) != len(b) for a, b in zip(candidate, reference)):
        return {"mse": float("nan"), "max_abs": float("nan"), "mean_cos": float("nan"), "min_cos": float("nan")}

    squared = []
    abs_values = []
    cosines = []
    for cand, ref in zip(candidate, reference):
        squared.extend((a - b) * (a - b) for a, b in zip(cand, ref))
        abs_values.extend(abs(a - b) for a, b in zip(cand, ref))
        cand_n = normalize_row(cand)
        ref_n = normalize_row(ref)
        cosines.append(sum(a * b for a, b in zip(cand_n, ref_n)))
    return {
        "mse": statistics.fmean(squared) if squared else float("nan"),
        "max_abs": max(abs_values) if abs_values else float("nan"),
        "mean_cos": statistics.fmean(cosines) if cosines else float("nan"),
        "min_cos": min(cosines) if cosines else float("nan"),
    }


def status_from_metrics(metrics: dict[str, float], threshold: float) -> str:
    min_cos = metrics["min_cos"]
    return "fail" if math.isnan(min_cos) or min_cos < threshold else "pass"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6g}"
    return str(value)
