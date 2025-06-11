#ifndef EMBEDDINGS_GTE_H
#define EMBEDDINGS_GTE_H

#include <string>
#include <vector>

#include "ggml.h"
#include "tokenizer.h"
#include "utils.h"
#include "bert.h" // 包含 BackendContext

namespace embeddings {

struct GteBertConfig {
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_attention_heads = 0;
  int num_hidden_layers = 0;
  float layer_norm_eps = 0.0f;
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

struct GteBertModel {
  GteBertModel() = default;
  GteBertModel(const std::string &gguf_model);

  std::vector<float> Forward(const Encoding &enc, bool normalize,
                                      int pooling_method);
  std::vector<std::vector<float>> BatchForward(
      const std::vector<Encoding> &batch, bool normalize, int pooling_method);

 private:
  void Clear();
  struct ggml_cgraph *BuildGraph(const std::vector<Encoding> &batch,
                                          bool normalize, int pooling_method);

 public:
  GteBertConfig hparams;
  GteBertEmbeddings embeddings;
  std::vector<GteBertLayer> layers;
  std::string arch;

 private:
  BackendContext ctx;
  ggml_tensor *get_tensor(ggml_context *ctx, const std::string &name) {
    ggml_tensor *tensor = ggml_get_tensor(ctx, name.c_str());
    if (!tensor) {
      fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name.c_str());
      throw "";
    }
    return tensor;
  }
};

struct GteEmbedding {
  GteEmbedding() = default;
  GteEmbedding(const std::string &hf_token_json, const std::string &gguf_model);

  std::vector<float> Encode(const std::string &text, bool normalize,
                                     int pooling_method);
  std::vector<std::vector<float>> BatchEncode(
      const std::vector<std::string> &batch, bool normalize, int pooling_method);

 private:
  Tokenizer *tok = nullptr;
  GteBertModel *model = nullptr;
};

}  // namespace embeddings

#endif  // EMBEDDINGS_GTE_H
