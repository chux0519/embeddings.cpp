#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gte.h"

using namespace embeddings;

namespace {

struct Options {
  std::string model_path;
  std::string batch_path;
  std::string backend;
  int threads = 0;
  bool normalize = true;
  PoolingMethod pooling = PoolingMethod::CLS;
};

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
          << "Usage: embedding_wasm_model_encode --model MODEL.gguf --batch batch.txt "
             "[--backend cpu|webgpu] [--threads N] [--pooling mean|cls] [--no-normalize]\n";
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

void print_json_string(const std::string &value) {
  std::cout << '"';
  for (char c : value) {
    switch (c) {
      case '\\':
        std::cout << "\\\\";
        break;
      case '"':
        std::cout << "\\\"";
        break;
      case '\n':
        std::cout << "\\n";
        break;
      default:
        std::cout << c;
        break;
    }
  }
  std::cout << '"';
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
    auto vectors = model.BatchForward(batch, opts.normalize, opts.pooling);

    std::cout << "{\n";
    std::cout << "  \"model\": ";
    print_json_string(opts.model_path);
    std::cout << ",\n";
    std::cout << "  \"batch_path\": ";
    print_json_string(opts.batch_path);
    std::cout << ",\n";
    std::cout << "  \"backend\": ";
    print_json_string(opts.backend);
    std::cout << ",\n";
    std::cout << "  \"threads\": " << opts.threads << ",\n";
    std::cout << "  \"normalize\": " << (opts.normalize ? "true" : "false") << ",\n";
    std::cout << "  \"vectors\": [\n";
    for (size_t i = 0; i < vectors.size(); ++i) {
      std::cout << "    [";
      for (size_t j = 0; j < vectors[i].size(); ++j) {
        if (j != 0) {
          std::cout << ", ";
        }
        std::cout << vectors[i][j];
      }
      std::cout << "]";
      if (i + 1 != vectors.size()) {
        std::cout << ",";
      }
      std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "embedding_wasm_model_encode error: " << e.what() << "\n";
    return 1;
  }
}
