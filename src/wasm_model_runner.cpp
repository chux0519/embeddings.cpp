#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <emscripten/emscripten.h>

#include "gte.h"

using namespace embeddings;

namespace {

struct RunnerState {
  std::unique_ptr<GteBertModel> model;
  bool normalize = true;
  PoolingMethod pooling = PoolingMethod::CLS;
  std::string last_json;
  std::string last_error;
};

RunnerState g_state;

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

std::vector<TokenizedInput> load_batch_stream(std::istream &input) {
  std::vector<TokenizedInput> batch;
  std::string line;
  while (std::getline(input, line)) {
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
    throw std::runtime_error("token batch is empty");
  }
  return batch;
}

std::vector<TokenizedInput> load_batch_text(const std::string &text) {
  std::istringstream input(text);
  return load_batch_stream(input);
}

std::vector<TokenizedInput> load_batch_path(const std::string &path) {
  std::ifstream fin(path);
  if (!fin) {
    throw std::runtime_error("failed to open token batch file: " + path);
  }
  return load_batch_stream(fin);
}

std::string vectors_to_json(const std::vector<std::vector<float>> &vectors) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"vectors\": [\n";
  for (size_t i = 0; i < vectors.size(); ++i) {
    out << "    [";
    for (size_t j = 0; j < vectors[i].size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << vectors[i][j];
    }
    out << "]";
    if (i + 1 != vectors.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

void clear_last() {
  g_state.last_json.clear();
  g_state.last_error.clear();
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int runner_init(const char *model_path, const char *backend,
                                     int threads, int normalize,
                                     int pooling_cls) {
  try {
    clear_last();
    if (!model_path || !*model_path) {
      throw std::runtime_error("model_path is required");
    }
    if (backend && *backend) {
      setenv("EMBEDDINGS_CPP_BACKEND", backend, 1);
    }
    if (threads > 0) {
      std::string value = std::to_string(threads);
      setenv("EMBEDDINGS_CPP_THREADS", value.c_str(), 1);
    }
    g_state.normalize = normalize != 0;
    g_state.pooling = pooling_cls != 0 ? PoolingMethod::CLS : PoolingMethod::MEAN;
    g_state.model = std::make_unique<GteBertModel>(std::string(model_path));
    g_state.model->Load();
    return 0;
  } catch (const std::exception &e) {
    g_state.last_error = e.what();
    g_state.model.reset();
    return 1;
  }
}

EMSCRIPTEN_KEEPALIVE int runner_encode(const char *batch_path) {
  try {
    clear_last();
    if (!g_state.model) {
      throw std::runtime_error("model is not initialized");
    }
    if (!batch_path || !*batch_path) {
      throw std::runtime_error("batch_path is required");
    }
    const auto batch = load_batch_path(std::string(batch_path));
    const auto vectors =
        g_state.model->BatchForward(batch, g_state.normalize, g_state.pooling);
    g_state.last_json = vectors_to_json(vectors);
    return 0;
  } catch (const std::exception &e) {
    g_state.last_error = e.what();
    return 1;
  }
}

EMSCRIPTEN_KEEPALIVE int runner_encode_inline(const char *batch_text) {
  try {
    clear_last();
    if (!g_state.model) {
      throw std::runtime_error("model is not initialized");
    }
    if (!batch_text || !*batch_text) {
      throw std::runtime_error("batch_text is required");
    }
    const auto batch = load_batch_text(std::string(batch_text));
    const auto vectors =
        g_state.model->BatchForward(batch, g_state.normalize, g_state.pooling);
    g_state.last_json = vectors_to_json(vectors);
    return 0;
  } catch (const std::exception &e) {
    g_state.last_error = e.what();
    return 1;
  }
}

EMSCRIPTEN_KEEPALIVE const char *runner_last_json() {
  return g_state.last_json.c_str();
}

EMSCRIPTEN_KEEPALIVE const char *runner_last_error() {
  return g_state.last_error.c_str();
}

EMSCRIPTEN_KEEPALIVE void runner_reset() {
  clear_last();
  g_state.model.reset();
}

}
