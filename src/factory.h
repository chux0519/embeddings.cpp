#pragma once
#include <memory>
#include <string>

#include "base_model.h"

namespace embeddings {

// Factory function to create a model instance from a GGUF file.
// Returns a std::unique_ptr for automatic memory management.
std::unique_ptr<BaseModel> create_model_from_gguf(const std::string& gguf_path);

class Embedding {
 public:
  virtual ~Embedding() = default;

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

}  // namespace embeddings