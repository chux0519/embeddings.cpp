#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gte.h"

using namespace embeddings;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model_path;
  std::string batch_path;
  std::string backend;
  int warmup = 1;
  int iterations = 3;
  int threads = 0;
  bool normalize = true;
  PoolingMethod pooling = PoolingMethod::CLS;
};

struct Stats {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double items_per_sec = 0.0;
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

std::vector<int32_t> parse_csv_i32(const std::string &value) {
  std::vector<int32_t> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(static_cast<int32_t>(std::stoi(item)));
    }
  }
  return out;
}

std::vector<TokenizedInput> load_batch(const std::string &path) {
  std::ifstream fin(path);
  if (!fin) {
    throw std::runtime_error("failed to open token batch file: " + path);
  }

  std::vector<TokenizedInput> batch;
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) {
      continue;
    }
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      throw std::runtime_error("invalid line without tab separator");
    }
    TokenizedInput enc;
    enc.ids = parse_csv_i32(line.substr(0, tab));
    enc.attention_mask = parse_csv_i32(line.substr(tab + 1));
    if (enc.ids.size() != enc.attention_mask.size()) {
      throw std::runtime_error("ids/mask length mismatch");
    }
    enc.no_pad_len = 0;
    for (size_t i = 0; i < enc.attention_mask.size(); ++i) {
      if (enc.attention_mask[i] != 0) {
        enc.no_pad_len = i + 1;
      }
    }
    batch.push_back(std::move(enc));
  }
  if (batch.empty()) {
    throw std::runtime_error("token batch file is empty");
  }
  return batch;
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
    } else if (arg == "--batch") {
      opts.batch_path = need_value(arg);
    } else if (arg == "--backend") {
      opts.backend = need_value(arg);
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(need_value(arg));
    } else if (arg == "--iterations") {
      opts.iterations = std::stoi(need_value(arg));
    } else if (arg == "--threads") {
      opts.threads = std::stoi(need_value(arg));
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
          << "Usage: embedding_wasm_model_bench --model MODEL.gguf --batch batch.txt "
             "[--backend cpu|webgpu] [--warmup N] [--iterations N] [--threads N] [--pooling mean|cls] [--no-normalize]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opts.model_path.empty() || opts.batch_path.empty()) {
    throw std::runtime_error("--model and --batch are required");
  }
  return opts;
}

Stats run_bench(BaseModel &model, const std::vector<TokenizedInput> &batch,
                const Options &opts) {
  for (int i = 0; i < opts.warmup; ++i) {
    model.BatchForward(batch, opts.normalize, opts.pooling);
  }

  std::vector<double> timings_ms;
  timings_ms.reserve(opts.iterations);
  for (int i = 0; i < opts.iterations; ++i) {
    const auto start = Clock::now();
    auto vectors = model.BatchForward(batch, opts.normalize, opts.pooling);
    const auto end = Clock::now();
    if (vectors.size() != batch.size()) {
      throw std::runtime_error("unexpected batch size mismatch");
    }
    timings_ms.push_back(elapsed_ms(start, end));
  }

  Stats stats;
  const double total_ms =
      std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
  stats.mean_ms = total_ms / static_cast<double>(timings_ms.size());
  stats.p50_ms = percentile_ms(timings_ms, 0.50);
  stats.p95_ms = percentile_ms(timings_ms, 0.95);
  stats.items_per_sec = (1000.0 * batch.size()) / stats.mean_ms;
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options opts = parse_args(argc, argv);
    const auto batch = load_batch(opts.batch_path);
    if (!opts.backend.empty()) {
      setenv("EMBEDDINGS_CPP_BACKEND", opts.backend.c_str(), 1);
    }
    if (opts.threads > 0) {
      std::string value = std::to_string(opts.threads);
      setenv("EMBEDDINGS_CPP_THREADS", value.c_str(), 1);
    }

    GteBertModel model(opts.model_path);
    model.Load();

    const auto bench_start = Clock::now();
    Stats stats = run_bench(model, batch, opts);
    const auto bench_end = Clock::now();

    std::cout << "{\n";
    std::cout << "  \"model\": \"" << opts.model_path << "\",\n";
    std::cout << "  \"batch_path\": \"" << opts.batch_path << "\",\n";
    std::cout << "  \"backend\": \"" << opts.backend << "\",\n";
    std::cout << "  \"batch\": " << batch.size() << ",\n";
    std::cout << "  \"warmup\": " << opts.warmup << ",\n";
    std::cout << "  \"iterations\": " << opts.iterations << ",\n";
    std::cout << "  \"threads\": " << opts.threads << ",\n";
    std::cout << "  \"mean_ms\": " << stats.mean_ms << ",\n";
    std::cout << "  \"p50_ms\": " << stats.p50_ms << ",\n";
    std::cout << "  \"p95_ms\": " << stats.p95_ms << ",\n";
    std::cout << "  \"items_per_sec\": " << stats.items_per_sec << ",\n";
    std::cout << "  \"wall_ms\": " << elapsed_ms(bench_start, bench_end) << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "embedding_wasm_model_bench error: " << e.what() << "\n";
    return 1;
  }
}
