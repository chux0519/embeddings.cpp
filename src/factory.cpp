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
  explicit EmbeddingImpl(const std::string& gguf_path) {
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

    model = create_model_from_gguf(gguf_path);

    model->Load();
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
};

std::unique_ptr<Embedding> create_embedding(const std::string& gguf_path) {
  return std::make_unique<EmbeddingImpl>(gguf_path);
}

}  // namespace embeddings
