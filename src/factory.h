#pragma once
#include "base_model.h"
#include <string>
#include <memory>

namespace embeddings {

// Factory function to create a model instance from a GGUF file.
// Returns a std::unique_ptr for automatic memory management.
std::unique_ptr<BaseModel> create_model_from_gguf(const std::string& gguf_path);

} // namespace embeddings