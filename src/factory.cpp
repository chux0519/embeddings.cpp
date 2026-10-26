#include "factory.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include "bert.h"
#include "gte.h"
#include "jina_bert.h"
#include "utils.h"  // for get_str, etc.

namespace embeddings {
namespace {

bool should_profile() {
  const char *env = std::getenv("EMBEDDINGS_CPP_PROFILE");
  return env && std::atoi(env) != 0;
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

std::unique_ptr<BaseModel> create_model_from_gguf(
    const std::string& gguf_path) {
  struct ggml_context* ctx_ggml = nullptr;
  struct gguf_init_params gguf_params = {true, &ctx_ggml};
  struct gguf_context* ctx_gguf =
      gguf_init_from_file(gguf_path.c_str(), gguf_params);
  if (!ctx_gguf) {
    throw std::runtime_error("Failed to open GGUF file: " + gguf_path);
  }

  std::string arch = get_str(ctx_gguf, KEY_ARCHITECTURE);
  gguf_free(ctx_gguf);
  ggml_free(ctx_ggml);

  if (arch == ARCH_BERT) {  // Or whatever string is in your BERT GGUF
    return std::make_unique<BertModel>(gguf_path);
  } else if (arch == ARCH_GTE) {  // Fictional architecture name for GTE
    return std::make_unique<GteBertModel>(gguf_path);
  } else if (arch == ARCH_JINABERT) {  // Fictional architecture name for Jina
    return std::make_unique<JinaBertModel>(gguf_path);
  } else if (arch == ARCH_XLMROBERTA) {
    return std::make_unique<BertModel>(
        gguf_path);  // XLM-R is a variant of BERT
  }

  throw std::runtime_error("Unknown or unsupported model architecture: " +
                           arch);
}

class EmbeddingImpl : public Embedding {
 public:
  explicit EmbeddingImpl(const std::string& gguf_path, const EmbeddingOptions& options = {}) {
    struct ggml_context* ctx_ggml = nullptr;
    struct gguf_init_params params = {true, &ctx_ggml};
    struct gguf_context* ctx_gguf =
        gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
      throw std::runtime_error("EmbeddingImpl: Failed to load GGUF from " +
                               gguf_path);
    }
    std::string tokenizer_json = get_str(ctx_gguf, "tokenizer.ggml.file");
    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);

    if (tokenizer_json.empty()) {
      throw std::runtime_error(
          "EmbeddingImpl: tokenizer.ggml.file not found in GGUF: " + gguf_path);
    }

    tok = std::make_unique<Tokenizer>(tokenizer_json, true /* is_blob */);
    if (options.max_tokens > 0 &&
        !tok->GetFastTokenizer()->SetTruncation(static_cast<size_t>(options.max_tokens))) {
      throw std::runtime_error("EmbeddingImpl: cannot set max_tokens");
    }
    max_tokens_ = static_cast<int>(tok->GetFastTokenizer()->GetTruncation());

    model = create_model_from_gguf(gguf_path);

    model->Load();
  }

  EmbeddingInfo info() const override {
    return {model->architecture(), model->name(), get_ftype(model->ftype()), model->hidden_size(), max_tokens_};
  }

  // Not thread-safe (temporarily disables truncation); neither is encoding.
  std::vector<size_t> count_tokens(const std::vector<std::string>& batch) override {
    auto* fast = tok->GetFastTokenizer();
    if (max_tokens_ > 0) fast->SetTruncation(0);
    auto encodings = tok->EncodeBatch(batch, true);
    if (max_tokens_ > 0) fast->SetTruncation(static_cast<size_t>(max_tokens_));
    std::vector<size_t> out;
    out.reserve(encodings.size());
    for (const auto& e : encodings) {
      size_t n = 0;
      for (auto m : e.attention_mask) n += m != 0;
      out.push_back(n);
    }
    return out;
  }

  std::vector<float> encode(
      const std::string& text, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) override {
    std::vector<std::string> batch = {text};
    return batch_encode(batch, normalize, pooling_method)[0];
  }

  TokenizedInput tokenize(const std::string& text, bool add_special_tokens = true) {
    return tok->Encode(text, add_special_tokens);
  }

  std::vector<TokenizedInput> batch_tokenize(const std::vector<std::string>& batch,
                                       bool add_special_tokens = true) {
    return tok->EncodeBatch(batch, add_special_tokens);
  }

  std::vector<std::vector<float>> batch_encode(
      const std::vector<std::string>& batch, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) override {
    const auto t0 = Clock::now();
    auto encodings = tok->EncodeBatch(batch, true);
    const auto t1 = Clock::now();
    auto result = model->BatchForward(encodings, normalize, pooling_method);
    const auto t2 = Clock::now();
    if (should_profile()) {
      fprintf(stderr,
              "profile,tokenize_ms=%.3f,model_forward_ms=%.3f,batch=%zu\n",
              elapsed_ms(t0, t1), elapsed_ms(t1, t2), batch.size());
    }
    return result;
  }

 private:
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<BaseModel> model;
  int max_tokens_ = 0;
};

std::unique_ptr<Embedding> create_embedding(const std::string& gguf_path) {
  return std::make_unique<EmbeddingImpl>(gguf_path);
}

std::unique_ptr<Embedding> create_embedding(const std::string& gguf_path,
                                            const EmbeddingOptions& options) {
  return std::make_unique<EmbeddingImpl>(gguf_path, options);
}

}  // namespace embeddings
