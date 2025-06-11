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
  std::vector<float> Forward(const Encoding &, bool normalize = true,
                             int pooling_method = 0) override;
  std::vector<std::vector<float>> BatchForward(const std::vector<Encoding> &,
                                               bool normalize = true,
                                               int pooling_method = 0) override;

 protected:
  struct ggml_cgraph *BuildGraph(const std::vector<Encoding> &batch,
                                 bool normalize = true, int pooling_method = 0) override;

 private:
  JinaBertConfig hparams;
  JinaBertEmbedding embeddings;
  std::vector<JinaEncoderBlock> layers;
};

class JinaEmbedding : public BaseEmbedding<JinaBertModel> {
 public:
  JinaEmbedding(const std::string &hf_token_json, const std::string &gguf_model)
      : BaseEmbedding<JinaBertModel>(hf_token_json, gguf_model) {}
};

}  // namespace embeddings