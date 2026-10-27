// embeddings_server: a small native HTTP server for embeddings.cpp models.
//
// OpenAI-compatible POST /v1/embeddings (plus TEI-style POST /embed), so any
// OpenAI or TEI client can use it. One model instance serves all requests;
// inference is serialized (ggml already parallelizes inside a batch).
#include <httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "factory.h"

using json = nlohmann::json;
using embeddings::PoolingMethod;

namespace {

constexpr const char* kVersion = "0.1.0";

struct Options {
  std::string model;
  std::string model_name;
  std::string host = "127.0.0.1";
  int port = 8090;
  std::string pooling = "cls";
  int max_tokens = 0;
  int max_batch_size = 64;
  int max_batch_tokens = 16384;
  int max_body_mb = 8;
  int http_threads = 8;
  int threads = 0;
  std::string api_key_env = "EMBEDDINGS_API_KEY";
};

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model FILE.gguf [options]\n"
               "  --host ADDR             listen address (default 127.0.0.1)\n"
               "  --port N                listen port (default 8090)\n"
               "  --model-name NAME       model id reported to clients (default: GGUF name)\n"
               "  --pooling cls|mean      pooling (default cls; Snowflake/BGE use cls)\n"
               "  --max-tokens N          truncate inputs to N tokens (default: tokenizer setting)\n"
               "  --max-batch-size N      inputs per request (default 64)\n"
               "  --max-batch-tokens N    tokens per request after truncation (default 16384)\n"
               "  --max-body-mb N         request body limit (default 8)\n"
               "  --threads N             inference threads (default: all cores)\n"
               "  --http-threads N        connection threads (default 8)\n"
               "  --api-key-env NAME      require 'Authorization: Bearer $NAME' when that\n"
               "                          variable is set (default EMBEDDINGS_API_KEY)\n",
               argv0);
}

bool parse_args(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model" || a == "-m") o.model = next("--model");
    else if (a == "--model-name") o.model_name = next("--model-name");
    else if (a == "--host") o.host = next("--host");
    else if (a == "--port") o.port = std::atoi(next("--port"));
    else if (a == "--pooling") o.pooling = next("--pooling");
    else if (a == "--max-tokens") o.max_tokens = std::atoi(next("--max-tokens"));
    else if (a == "--max-batch-size") o.max_batch_size = std::atoi(next("--max-batch-size"));
    else if (a == "--max-batch-tokens") o.max_batch_tokens = std::atoi(next("--max-batch-tokens"));
    else if (a == "--max-body-mb") o.max_body_mb = std::atoi(next("--max-body-mb"));
    else if (a == "--threads") o.threads = std::atoi(next("--threads"));
    else if (a == "--http-threads") o.http_threads = std::atoi(next("--http-threads"));
    else if (a == "--api-key-env") o.api_key_env = next("--api-key-env");
    else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      usage(argv[0]);
      return false;
    }
  }
  if (o.model.empty()) {
    usage(argv[0]);
    return false;
  }
  if (o.pooling != "cls" && o.pooling != "mean") {
    std::fprintf(stderr, "--pooling must be cls or mean\n");
    return false;
  }
  return true;
}

// HTTP error with an OpenAI-style body.
struct HttpError {
  int status;
  std::string message;
  std::string type;
};

void send_error(httplib::Response& res, const HttpError& e) {
  res.status = e.status;
  res.set_content(json{{"error", {{"message", e.message}, {"type", e.type}, {"code", e.status}}}}.dump(),
                  "application/json");
}

std::vector<std::string> parse_inputs(const json& v, const char* field) {
  std::vector<std::string> out;
  if (v.is_string()) {
    out.push_back(v.get<std::string>());
  } else if (v.is_array()) {
    for (const auto& item : v) {
      if (!item.is_string()) {
        throw HttpError{400, std::string(field) + " must be a string or an array of strings", "invalid_request_error"};
      }
      out.push_back(item.get<std::string>());
    }
  } else {
    throw HttpError{400, std::string(field) + " is required (string or array of strings)", "invalid_request_error"};
  }
  if (out.empty()) {
    throw HttpError{400, std::string(field) + " must not be empty", "invalid_request_error"};
  }
  return out;
}

std::string base64_floats(const std::vector<float>& v) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto* p = reinterpret_cast<const unsigned char*>(v.data());  // little-endian hosts
  const size_t n = v.size() * sizeof(float);
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  for (size_t i = 0; i < n; i += 3) {
    uint32_t b = p[i] << 16;
    if (i + 1 < n) b |= p[i + 1] << 8;
    if (i + 2 < n) b |= p[i + 2];
    out.push_back(tbl[(b >> 18) & 63]);
    out.push_back(tbl[(b >> 12) & 63]);
    out.push_back(i + 1 < n ? tbl[(b >> 6) & 63] : '=');
    out.push_back(i + 2 < n ? tbl[b & 63] : '=');
  }
  return out;
}

// Matryoshka truncation: keep the first dims components and re-normalize.
void shorten(std::vector<float>& v, int dims) {
  if (dims <= 0 || dims >= static_cast<int>(v.size())) return;
  v.resize(dims);
  double norm = 0;
  for (float x : v) norm += double(x) * x;
  norm = std::sqrt(norm);
  if (norm > 0) {
    for (float& x : v) x = static_cast<float>(x / norm);
  }
}

class Service {
 public:
  Service(const Options& o) : opts_(o) {
    embeddings::EmbeddingOptions eo;
    eo.max_tokens = o.max_tokens;
    model_ = embeddings::create_embedding(o.model, eo);
    info_ = model_->info();
    name_ = !o.model_name.empty() ? o.model_name
            : !info_.name.empty() ? info_.name
                                  : std::filesystem::path(o.model).stem().string();
    pooling_ = o.pooling == "mean" ? PoolingMethod::MEAN : PoolingMethod::CLS;
  }

  json info() const {
    return {{"model", name_},
            {"architecture", info_.architecture},
            {"ftype", info_.ftype},
            {"weight_types", info_.weight_types},
            {"dimensions", info_.dimensions},
            {"pooling", opts_.pooling},
            {"max_tokens", info_.max_tokens},
            {"max_batch_size", opts_.max_batch_size},
            {"max_batch_tokens", opts_.max_batch_tokens},
            {"version", kVersion}};
  }

  const std::string& name() const { return name_; }

  struct Result {
    std::vector<std::vector<float>> vectors;
    std::vector<bool> truncated;
    size_t tokens = 0;
  };

  Result embed(const std::vector<std::string>& inputs, bool normalize) {
    if (static_cast<int>(inputs.size()) > opts_.max_batch_size) {
      throw HttpError{413, "too many inputs: " + std::to_string(inputs.size()) + " > max_batch_size " +
                               std::to_string(opts_.max_batch_size), "invalid_request_error"};
    }
    std::lock_guard<std::mutex> lock(mu_);
    Result r;
    auto counts = model_->count_tokens(inputs);
    const size_t limit = info_.max_tokens > 0 ? static_cast<size_t>(info_.max_tokens) : SIZE_MAX;
    for (size_t c : counts) {
      r.truncated.push_back(c > limit);
      r.tokens += std::min(c, limit);
    }
    if (static_cast<int>(r.tokens) > opts_.max_batch_tokens) {
      throw HttpError{413, "too many tokens in request: " + std::to_string(r.tokens) + " > max_batch_tokens " +
                               std::to_string(opts_.max_batch_tokens), "invalid_request_error"};
    }
    r.vectors = model_->batch_encode(inputs, normalize, pooling_);
    return r;
  }

 private:
  Options opts_;
  std::unique_ptr<embeddings::Embedding> model_;
  embeddings::EmbeddingInfo info_;
  std::string name_;
  PoolingMethod pooling_;
  std::mutex mu_;
};

json parse_body(const httplib::Request& req) {
  try {
    return json::parse(req.body);
  } catch (const json::exception&) {
    throw HttpError{400, "request body must be JSON", "invalid_request_error"};
  }
}

httplib::Server* g_server = nullptr;

void on_signal(int) {
  if (g_server) g_server->stop();
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parse_args(argc, argv, opts)) return 2;
  if (opts.threads > 0) {
#ifdef _WIN32
    _putenv_s("EMBEDDINGS_CPP_THREADS", std::to_string(opts.threads).c_str());
#else
    setenv("EMBEDDINGS_CPP_THREADS", std::to_string(opts.threads).c_str(), 1);
#endif
  }

  std::unique_ptr<Service> svc;
  try {
    svc = std::make_unique<Service>(opts);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "failed to load model %s: %s\n", opts.model.c_str(), e.what());
    return 1;
  }
  const char* key_env = std::getenv(opts.api_key_env.c_str());
  const std::string api_key = key_env ? key_env : "";

  httplib::Server server;
  server.new_task_queue = [&] { return new httplib::ThreadPool(std::max(1, opts.http_threads)); };
  server.set_payload_max_length(static_cast<size_t>(opts.max_body_mb) * 1024 * 1024);

  // Authentication (when configured) for everything except /health.
  server.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
    if (api_key.empty() || req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
    if (req.get_header_value("Authorization") != "Bearer " + api_key) {
      send_error(res, {401, "invalid or missing API key", "authentication_error"});
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  // Run a handler, mapping errors to JSON responses.
  auto guarded = [](auto fn) {
    return [fn](const httplib::Request& req, httplib::Response& res) {
      try {
        fn(req, res);
      } catch (const HttpError& e) {
        send_error(res, e);
      } catch (const std::exception& e) {
        send_error(res, {500, std::string("internal error: ") + e.what(), "server_error"});
      }
    };
  };

  server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok"})", "application/json");
  });
  server.Get("/ready", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ready"})", "application/json");  // the model loads before listening
  });
  server.Get("/info", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(svc->info().dump(), "application/json");
  });
  server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    json m = {{"object", "list"},
              {"data", json::array({{{"id", svc->name()}, {"object", "model"}, {"owned_by", "embeddings.cpp"}}})}};
    res.set_content(m.dump(), "application/json");
  });

  // OpenAI-compatible embeddings.
  server.Post("/v1/embeddings", guarded([&](const httplib::Request& req, httplib::Response& res) {
    json body = parse_body(req);
    auto inputs = parse_inputs(body.value("input", json()), "input");
    const std::string format = body.value("encoding_format", "float");
    if (format != "float" && format != "base64") {
      throw HttpError{400, "encoding_format must be float or base64", "invalid_request_error"};
    }
    int dims = body.value("dimensions", 0);
    if (dims < 0) throw HttpError{400, "dimensions must be positive", "invalid_request_error"};
    auto r = svc->embed(inputs, true);
    json data = json::array();
    for (size_t i = 0; i < r.vectors.size(); ++i) {
      shorten(r.vectors[i], dims);
      json item = {{"object", "embedding"}, {"index", i}};
      if (format == "base64") {
        item["embedding"] = base64_floats(r.vectors[i]);
      } else {
        item["embedding"] = r.vectors[i];
      }
      if (r.truncated[i]) item["truncated"] = true;  // extension; OpenAI clients ignore it
      data.push_back(std::move(item));
    }
    json out = {{"object", "list"},
                {"data", std::move(data)},
                {"model", svc->name()},
                {"usage", {{"prompt_tokens", r.tokens}, {"total_tokens", r.tokens}}}};
    res.set_content(out.dump(), "application/json");
  }));

  // TEI-compatible embeddings.
  server.Post("/embed", guarded([&](const httplib::Request& req, httplib::Response& res) {
    json body = parse_body(req);
    auto inputs = parse_inputs(body.value("inputs", json()), "inputs");
    auto r = svc->embed(inputs, body.value("normalize", true));
    if (!body.value("truncate", true)) {
      for (size_t i = 0; i < r.truncated.size(); ++i) {
        if (r.truncated[i]) {
          throw HttpError{413, "input " + std::to_string(i) + " exceeds max_tokens and truncate is false",
                          "invalid_request_error"};
        }
      }
    }
    res.set_content(json(r.vectors).dump(), "application/json");
  }));

  server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    std::fprintf(stderr, "%s %s %d %zuB\n", req.method.c_str(), req.path.c_str(), res.status, res.body.size());
  });

  g_server = &server;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  auto info = svc->info();
  std::fprintf(stderr, "embeddings_server %s: model %s (%s, %s, %d dims, max_tokens %d, pooling %s) on http://%s:%d%s\n",
               kVersion, svc->name().c_str(), info["architecture"].get<std::string>().c_str(),
               info["ftype"].get<std::string>().c_str(), info["dimensions"].get<int>(), info["max_tokens"].get<int>(),
               opts.pooling.c_str(), opts.host.c_str(), opts.port, api_key.empty() ? "" : " (API key required)");
  if (!server.listen(opts.host, opts.port)) {
    std::fprintf(stderr, "failed to listen on %s:%d\n", opts.host.c_str(), opts.port);
    return 1;
  }
  return 0;
}
