#ifndef EMBEDDINGS_GTE_H
#define EMBEDDINGS_GTE_H

#include <string>
#include <vector>

#include "base_model.h"
#include "ggml.h"
#include "tokenizer.h"
#include "utils.h"

namespace embeddings {

struct GteBertConfig : public BaseConfig {
  int32_t max_position_embeddings = 0;
  int32_t intermediate_size = 0;
  float rope_theta = 0.0f;
};

struct GteBertLayer {
  // attention
  ggml_tensor *qkv_proj_w = nullptr;
  ggml_tensor *qkv_proj_b = nullptr;
  ggml_tensor *o_proj_w = nullptr;
  ggml_tensor *o_proj_b = nullptr;
  ggml_tensor *attn_ln_w = nullptr;
  ggml_tensor *attn_ln_b = nullptr;

  // ff
  ggml_tensor *up_gate_proj_w = nullptr;
  ggml_tensor *down_proj_w = nullptr;
  ggml_tensor *down_proj_b = nullptr;
  ggml_tensor *mlp_ln_w = nullptr;
  ggml_tensor *mlp_ln_b = nullptr;
};

struct GteBertEmbeddings {
  ggml_tensor *word_embeddings = nullptr;
  ggml_tensor *token_type_embeddings = nullptr;
  ggml_tensor *LayerNorm_w = nullptr;
  ggml_tensor *LayerNorm_b = nullptr;
};

struct GteBertModel : public BaseModel {
  GteBertModel(const std::string &gguf_model);

 protected:
  struct ggml_cgraph *BuildGraph(
      const std::vector<TokenizedInput> &batch, bool normalize,
      PoolingMethod pooling_method = PoolingMethod::MEAN) override;
  void LoadHyperparameters(struct gguf_context *ctx_gguf) override;
  void LoadTensors() override;

 public:
  GteBertEmbeddings embeddings;
  std::vector<GteBertLayer> layers;
};

}  // namespace embeddings

#endif  // EMBEDDINGS_GTE_H
