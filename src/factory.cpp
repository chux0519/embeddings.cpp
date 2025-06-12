#include "factory.h"
#include "bert.h"
#include "gte.h"
#include "jina_bert.h"
#include "utils.h" // for get_str, etc.
#include <stdexcept>

namespace embeddings {

std::unique_ptr<BaseModel> create_model_from_gguf(const std::string& gguf_path) {
    struct ggml_context* ctx_ggml = nullptr;
    struct gguf_init_params gguf_params = {true, &ctx_ggml};
    struct gguf_context* ctx_gguf = gguf_init_from_file(gguf_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        throw std::runtime_error("Failed to open GGUF file: " + gguf_path);
    }

    std::string arch = get_str(ctx_gguf, KEY_ARCHITECTURE);
    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);

    if (arch == ARCH_BERT) { // Or whatever string is in your BERT GGUF
        return std::make_unique<BertModel>(gguf_path);
    } else if (arch == ARCH_GTE) { // Fictional architecture name for GTE
        return std::make_unique<GteBertModel>(gguf_path);
    } else if (arch == ARCH_JINABERT) { // Fictional architecture name for Jina
        return std::make_unique<JinaBertModel>(gguf_path);
    } else if (arch == ARCH_XLMROBERTA) {
         return std::make_unique<BertModel>(gguf_path); // XLM-R is a variant of BERT
    }
    
    throw std::runtime_error("Unknown or unsupported model architecture: " + arch);
}

} // namespace embeddings