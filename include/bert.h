#pragma once
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "tokenizer.h"

#include <cmath>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"
#define KEY_ARCHITECTURE "general.architecture"

#define ARCH_XLMROBERTA "XLMRobertaModel"

#define POOLING_METHOD_MEAN 0
#define POOLING_METHOD_CLS 1

namespace embeddings {
class EncoderBlock {
public:
  void Forward();

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

struct BertEncoderConfig {
  int32_t n_vocab;
  int32_t n_max_tokens;
  int32_t n_embd;
  int32_t n_intermediate;
  int32_t n_head;
  int32_t n_layer;
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

class BertEncoder {
public:
  BertEncoder(const std::string &);
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
  BertEncoderConfig hparams;
  BackendContext ctx;
  struct ggml_tensor *word_embeddings;
  struct ggml_tensor *token_type_embeddings;
  struct ggml_tensor *position_embeddings;
  struct ggml_tensor *ln_e_w;
  struct ggml_tensor *ln_e_b;
  struct ggml_tensor *pooler_e_w;
  struct ggml_tensor *pooler_e_b;

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
  BertEncoder *model;
};
} // namespace embeddings