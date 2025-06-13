#pragma once

#include "base_model.h"

namespace embeddings {

struct BertConfig : public BaseConfig {
  int32_t max_position_embeddings = 0;
  int32_t intermediate_size = 0;
};

struct BertEmbedding {
  struct ggml_tensor *word_embeddings;
  struct ggml_tensor *token_type_embeddings;
  struct ggml_tensor *position_embeddings;
  struct ggml_tensor *ln_e_w;
  struct ggml_tensor *ln_e_b;
};

struct EncoderBlock {
  // attention
  struct ggml_tensor *q_w;
  struct ggml_tensor *q_b;
  struct ggml_tensor *k_w;
  struct ggml_tensor *k_b;
  struct ggml_tensor *v_w;
  struct ggml_tensor *v_b;

  struct ggml_tensor *o_w;
  struct ggml_tensor *o_b;

  struct ggml_tensor *ln_att_w;
  struct ggml_tensor *ln_att_b;

  // ff
  struct ggml_tensor *ff_i_w;
  struct ggml_tensor *ff_i_b;

  struct ggml_tensor *ff_o_w;
  struct ggml_tensor *ff_o_b;

  struct ggml_tensor *ln_out_w;
  struct ggml_tensor *ln_out_b;
};

class BertModel : public BaseModel {
 public:
  BertModel(const std::string &gguf_model);

 protected:
  struct ggml_cgraph *BuildGraph(
      const std::vector<Encoding> &batch, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) override;
  void LoadHyperparameters(struct gguf_context *ctx_gguf) override;
  void LoadTensors() override;

 private:
  BertEmbedding embeddings;
  std::vector<EncoderBlock> layers;
};

class Embedding : public BaseEmbedding<BertModel> {
 public:
  Embedding(const std::string &gguf_model)
      : BaseEmbedding<BertModel>(gguf_model) {}
};
}  // namespace embeddings