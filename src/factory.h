#pragma once
#include <memory>
#include <string>

#include "base_model.h"

namespace embeddings {

// Factory function to create a model instance from a GGUF file.
// Returns a std::unique_ptr for automatic memory management.
std::unique_ptr<BaseModel> create_model_from_gguf(const std::string& gguf_path);

// Options for create_embedding.
struct EmbeddingOptions {
  // Maximum tokens per input (including special tokens); longer inputs are
  // truncated. 0 keeps the tokenizer's own setting (often 512).
  int max_tokens = 0;
};

// Metadata about a loaded embedding model.
struct EmbeddingInfo {
  std::string architecture;
  std::string name;
  std::string ftype;   // e.g. "q4_k", "f16"
  int dimensions = 0;  // embedding size
  int max_tokens = 0;  // truncation length in effect (0 = none)
};

class Embedding {
 public:
  virtual ~Embedding() = default;
  // Not pure: Python subclasses (pybind trampolines) need not implement them.
  virtual EmbeddingInfo info() const { return {}; }
  // Token counts of the inputs without truncation (special tokens
  // included), so callers can tell whether an input was truncated.
  virtual std::vector<size_t> count_tokens(const std::vector<std::string>&) { return {}; }

  virtual TokenizedInput tokenize(const std::string&,
                            bool add_special_tokens = true) = 0;

  virtual std::vector<TokenizedInput> batch_tokenize(
      const std::vector<std::string>&, bool add_special_tokens = true) = 0;

  virtual std::vector<float> encode(
      const std::string& text, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) = 0;

  virtual std::vector<std::vector<float>> batch_encode(
      const std::vector<std::string>& batch, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) = 0;
};

std::unique_ptr<Embedding> create_embedding(const std::string& gguf_path);
std::unique_ptr<Embedding> create_embedding(const std::string& gguf_path,
                                            const EmbeddingOptions& options);

}  // namespace embeddings