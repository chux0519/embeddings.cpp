#pragma once
#include "base_model.h"
#include <string>
#include <memory>

namespace embeddings {

// Factory function to create a model instance from a GGUF file.
// Returns a std::unique_ptr for automatic memory management.
std::unique_ptr<BaseModel> create_model_from_gguf(const std::string& gguf_path);

class Embedding {
public:
    virtual ~Embedding() = default;

    // 定义了所有 embedding 对象都必须提供的功能。
    virtual std::vector<float> encode(
        const std::string& text,
        bool normalize = true,
        PoolingMethod pooling_method = PoolingMethod::MEAN) = 0;

    virtual std::vector<std::vector<float>> batch_encode(
        const std::vector<std::string>& batch,
        bool normalize = true,
        PoolingMethod pooling_method = PoolingMethod::MEAN) = 0;
};

std::unique_ptr<Embedding> create_embedding(const std::string& gguf_path);

} // namespace embeddings