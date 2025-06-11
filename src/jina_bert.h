#pragma once

#include "bert.h"

namespace embeddings {

struct JinaBertConfig {
  int32_t vocab_size;
  int32_t hidden_size;
  int32_t num_hidden_layers;
  int32_t num_attention_heads;
  int32_t intermediate_size;
  int32_t type_vocab_size;
  int32_t pad_token_id;

  float_t layer_norm_eps;
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

class JinaBertModel {
 public:
  JinaBertModel(const std::string &);
  std::vector<float> Forward(const Encoding &, bool normalize = true,
                             int pooling_method = 0);
  std::vector<std::vector<float>> BatchForward(const std::vector<Encoding> &,
                                               bool normalize = true,
                                               int pooling_method = 0);

 private:
  struct ggml_cgraph *BuildGraph(const std::vector<Encoding> &batch,
                                 bool normalize = true, int pooling_method = 0);
  void Clear();

  std::string arch;
  JinaBertConfig hparams;
  BackendContext ctx;

  JinaBertEmbedding embeddings;
  std::vector<JinaEncoderBlock> layers;
};

class JinaEmbedding {
 public:
  JinaEmbedding(const std::string &hf_token_json,
                const std::string &gguf_model);
  std::vector<float> Encode(const std::string &, bool normalize = true,
                            int pooling_method = 0);
  std::vector<std::vector<float>> BatchEncode(const std::vector<std::string> &,
                                              bool normalize = true,
                                              int pooling_method = 0);

 private:
  Tokenizer *tok;
  JinaBertModel *model;
};

}  // namespace embeddings