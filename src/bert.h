#pragma once

#include "base_model.h"

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_ARCHITECTURE "general.architecture"
#define ARCH_XLMROBERTA "XLMRobertaModel"

#define POOLING_METHOD_MEAN 0
#define POOLING_METHOD_CLS 1

namespace embeddings {

struct BertConfig : public BaseConfig {
  int32_t max_position_embedding = 0;
  int32_t intermediate_size = 0;
};

struct BertEmbedding {
  struct ggml_tensor *word_embeddings;
  struct ggml_tensor *token_type_embeddings;
  struct ggml_tensor *position_embeddings;
  struct ggml_tensor *ln_e_w;
  struct ggml_tensor *ln_e_b;
  struct ggml_tensor *pooler_e_w;
  struct ggml_tensor *pooler_e_b;
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
  BertModel(const std::string &);
  std::vector<float> Forward(const Encoding &, bool normalize = true,
                             int pooling_method = 0) override;
  std::vector<std::vector<float>> BatchForward(const std::vector<Encoding> &,
                                               bool normalize = true,
                                               int pooling_method = 0) override;

 protected:
  struct ggml_cgraph *BuildGraph(const std::vector<Encoding> &batch,
                                 bool normalize = true, int pooling_method = 0) override;

 private:
  BertConfig hparams;
  BertEmbedding embeddings;
  std::vector<EncoderBlock> layers;
};

class Embedding : public BaseEmbedding<BertModel> {
 public:
  Embedding(const std::string &hf_token_json, const std::string &gguf_model)
      : BaseEmbedding<BertModel>(hf_token_json, gguf_model) {}
};
}  // namespace embeddings