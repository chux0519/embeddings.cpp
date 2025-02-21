#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tokenizer.h"

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_ARCHITECTURE "general.architecture"
#define ARCH_XLMROBERTA "XLMRobertaModel"

#define POOLING_METHOD_MEAN 0
#define POOLING_METHOD_CLS 1

namespace embeddings {
struct BertConfig {
  int32_t vocab_size;
  int32_t max_position_embedding;
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t num_attention_heads;
  int32_t num_hidden_layers;
  float_t layer_norm_eps;
};

class BackendContext {
 public:
  // ggml context for weights
  struct ggml_context *ctx_data = NULL;

  // memory buffers to evaluate the model
  ggml_backend_t backend = NULL;
  ggml_backend_buffer_t weights_buffer = NULL;

  // load tokens into here, to compute
  struct ggml_context *compute_ctx = NULL;
  ggml_backend_buffer_t compute_buffer = NULL;

  // the compute graph for each forward process
  struct ggml_context *compute_graph_ctx = NULL;
  ggml_gallocr_t compute_allocr = NULL;
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

class BertModel {
 public:
  BertModel(const std::string &);
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
  BertConfig hparams;
  BackendContext ctx;

  BertEmbedding embeddings;
  std::vector<EncoderBlock> layers;
};

class Embedding {
 public:
  Embedding(const std::string &hf_token_json, const std::string &gguf_model);
  std::vector<float> Encode(const std::string &, bool normalize = true,
                            int pooling_method = 0);
  std::vector<std::vector<float>> BatchEncode(const std::vector<std::string> &,
                                              bool normalize = true,
                                              int pooling_method = 0);

 private:
  Tokenizer *tok;
  BertModel *model;
};
}  // namespace embeddings