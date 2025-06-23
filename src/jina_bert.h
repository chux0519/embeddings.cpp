#pragma once

#include "base_model.h"

namespace embeddings {

struct JinaBertConfig : public BaseConfig {
  int32_t intermediate_size = 0;
  int32_t type_vocab_size = 0;
  int32_t pad_token_id = 0;
};

struct JinaBertEmbedding {
  struct ggml_tensor *word_embeddings;
  struct ggml_tensor *token_type_embeddings;
  struct ggml_tensor *ln_e_w;
  struct ggml_tensor *ln_e_b;
};

struct JinaEncoderBlock {
  // attention
  struct ggml_tensor *Wqkv_w;
  struct ggml_tensor *Wqkv_b;

  struct ggml_tensor *o_w;
  struct ggml_tensor *o_b;

  struct ggml_tensor *norm1_w;
  struct ggml_tensor *norm1_b;

  // glumlp
  struct ggml_tensor *mlp_gated_layers_w;

  struct ggml_tensor *mlp_out_w;
  struct ggml_tensor *mlp_out_b;

  struct ggml_tensor *norm2_w;
  struct ggml_tensor *norm2_b;
};

class JinaBertModel : public BaseModel {
 public:
  JinaBertModel(const std::string &);

 protected:
  struct ggml_cgraph *BuildGraph(
      const std::vector<Encoding> &batch, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) override;
  void LoadHyperparameters(struct gguf_context *ctx_gguf) override;
  void LoadTensors() override;

 private:
  JinaBertEmbedding embeddings;
  std::vector<JinaEncoderBlock> layers;
};

}  // namespace embeddings