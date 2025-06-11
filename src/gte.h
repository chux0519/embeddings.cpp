#ifndef EMBEDDINGS_GTE_H
#define EMBEDDINGS_GTE_H

#include <string>
#include <vector>

#include "ggml.h"
#include "tokenizer.h"
#include "utils.h"
#include "base_model.h"

namespace embeddings {

struct GteBertConfig : public BaseConfig {
  int intermediate_size = 0;
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
  GteBertModel() = default;
  GteBertModel(const std::string &gguf_model);

  std::vector<float> Forward(const Encoding &enc, bool normalize,
                                      int pooling_method) override;
  std::vector<std::vector<float>> BatchForward(
      const std::vector<Encoding> &batch, bool normalize, int pooling_method) override;

 protected:
  struct ggml_cgraph *BuildGraph(const std::vector<Encoding> &batch,
                                          bool normalize, int pooling_method) override;

 public:
  GteBertConfig hparams;
  GteBertEmbeddings embeddings;
  std::vector<GteBertLayer> layers;
};

struct GteEmbedding : public BaseEmbedding<GteBertModel> {
  GteEmbedding() = default;
  GteEmbedding(const std::string &hf_token_json, const std::string &gguf_model)
      : BaseEmbedding<GteBertModel>(hf_token_json, gguf_model) {}
};

}  // namespace embeddings

#endif  // EMBEDDINGS_GTE_H
