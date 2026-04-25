#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "factory.h"

using namespace embeddings;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model_path;
  std::string texts_path;
  int warmup = 1;
  int iterations = 3;
  bool normalize = true;
  PoolingMethod pooling = PoolingMethod::MEAN;
};

struct Stats {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double texts_per_sec = 0.0;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile_ms(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double idx = q * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) {
    return values[lo];
  }
  const double frac = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::vector<std::string> split_tsv_texts(const std::string &path) {
  std::ifstream fin(path);
  if (!fin) {
    throw std::runtime_error("failed to open texts file: " + path);
  }
  std::vector<std::string> texts;
  std::string line;
  while (std::getline(fin, line)) {
    if (!line.empty()) {
      texts.push_back(line);
    }
  }
  if (texts.empty()) {
    throw std::runtime_error("texts file is empty: " + path);
  }
  return texts;
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--model") {
      opts.model_path = need_value(arg);
    } else if (arg == "--texts") {
      opts.texts_path = need_value(arg);
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(need_value(arg));
    } else if (arg == "--iterations") {
      opts.iterations = std::stoi(need_value(arg));
    } else if (arg == "--no-normalize") {
      opts.normalize = false;
    } else if (arg == "--pooling") {
      const std::string value = need_value(arg);
      if (value == "mean") {
        opts.pooling = PoolingMethod::MEAN;
      } else if (value == "cls") {
        opts.pooling = PoolingMethod::CLS;
      } else {
        throw std::runtime_error("unsupported pooling: " + value);
      }
    } else if (arg == "--help") {
      std::cout
          << "Usage: embedding_wasm_bench --model MODEL.gguf --texts texts.txt "
             "[--warmup N] [--iterations N] [--pooling mean|cls] [--no-normalize]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opts.model_path.empty() || opts.texts_path.empty()) {
    throw std::runtime_error("--model and --texts are required");
  }
  if (opts.warmup < 0 || opts.iterations <= 0) {
    throw std::runtime_error("warmup must be >= 0 and iterations must be > 0");
  }
  return opts;
}

Stats run_bench(Embedding &embedding, const std::vector<std::string> &texts,
                const Options &opts) {
  for (int i = 0; i < opts.warmup; ++i) {
    embedding.batch_encode(texts, opts.normalize, opts.pooling);
  }

  std::vector<double> timings_ms;
  timings_ms.reserve(opts.iterations);
  for (int i = 0; i < opts.iterations; ++i) {
    const auto start = Clock::now();
    auto vectors = embedding.batch_encode(texts, opts.normalize, opts.pooling);
    const auto end = Clock::now();
    if (vectors.size() != texts.size()) {
      throw std::runtime_error("unexpected batch size mismatch");
    }
    timings_ms.push_back(elapsed_ms(start, end));
  }

  const double total_ms =
      std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
  Stats stats;
  stats.mean_ms = total_ms / static_cast<double>(timings_ms.size());
  stats.p50_ms = percentile_ms(timings_ms, 0.50);
  stats.p95_ms = percentile_ms(timings_ms, 0.95);
  stats.texts_per_sec = (1000.0 * texts.size()) / stats.mean_ms;
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options opts = parse_args(argc, argv);
    const auto texts = split_tsv_texts(opts.texts_path);
    auto embedding = create_embedding(opts.model_path);

    const auto bench_start = Clock::now();
    Stats stats = run_bench(*embedding, texts, opts);
    const auto bench_end = Clock::now();

    std::cout << "{\n";
    std::cout << "  \"model\": \"" << opts.model_path << "\",\n";
    std::cout << "  \"texts_path\": \"" << opts.texts_path << "\",\n";
    std::cout << "  \"batch\": " << texts.size() << ",\n";
    std::cout << "  \"warmup\": " << opts.warmup << ",\n";
    std::cout << "  \"iterations\": " << opts.iterations << ",\n";
    std::cout << "  \"mean_ms\": " << stats.mean_ms << ",\n";
    std::cout << "  \"p50_ms\": " << stats.p50_ms << ",\n";
    std::cout << "  \"p95_ms\": " << stats.p95_ms << ",\n";
    std::cout << "  \"texts_per_sec\": " << stats.texts_per_sec << ",\n";
    std::cout << "  \"wall_ms\": " << elapsed_ms(bench_start, bench_end) << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "embedding_wasm_bench error: " << e.what() << "\n";
    return 1;
  }
}
