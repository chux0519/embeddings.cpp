#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

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

// 前向声明
struct Encoding;

// 基础配置接口
struct BaseConfig {
  int32_t vocab_size = 0;
  int32_t hidden_size = 0;
  int32_t num_hidden_layers = 0;
  int32_t num_attention_heads = 0;
  float layer_norm_eps = 1e-12f;
  
  virtual ~BaseConfig() = default;
};

// 后端上下文 - 从 bert.h 移动到这里
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

// 基础模型抽象类
class BaseModel {
 public:
  BaseModel() = default;
  virtual ~BaseModel() = default;

  // 纯虚函数 - 子类必须实现
  virtual std::vector<float> Forward(const Encoding &enc, bool normalize = true, 
                                   int pooling_method = 0) = 0;
  virtual std::vector<std::vector<float>> BatchForward(
      const std::vector<Encoding> &batch, bool normalize = true, 
      int pooling_method = 0) = 0;

 protected:
  // 受保护的虚函数 - 子类可以重写
  virtual struct ggml_cgraph *BuildGraph(const std::vector<Encoding> &batch,
                                       bool normalize = true, 
                                       int pooling_method = 0) = 0;
  virtual void Clear();
  virtual void InitializeBackend();
  virtual void LoadModel(const std::string &gguf_model);
  // 通用的辅助函数
  ggml_tensor *get_tensor(ggml_context *ctx, const std::string &name);
  struct ggml_cgraph* CommonBatchForwardSetup(const std::vector<Encoding> &batch, 
                             bool normalize, int pooling_method);
  std::vector<std::vector<float>> ExtractResults(struct ggml_cgraph *graph, 
                                                int batch_size, int hidden_size);

  // 成员变量
  std::string arch;
  BackendContext ctx;
};

// 基础嵌入包装类
template<typename ModelType>
class BaseEmbedding {
 public:
  BaseEmbedding(const std::string &hf_token_json, const std::string &gguf_model) {
    tok = new Tokenizer(hf_token_json);
    model = new ModelType(gguf_model);
  }

  virtual ~BaseEmbedding() {
    delete tok;
    delete model;
  }

  std::vector<float> Encode(const std::string &text, bool normalize = true,
                          int pooling_method = 0) {
    std::vector<std::string> batch = {text};
    return BatchEncode(batch, normalize, pooling_method)[0];
  }

  std::vector<std::vector<float>> BatchEncode(const std::vector<std::string> &batch,
                                            bool normalize = true,
                                            int pooling_method = 0) {
    auto encodings = tok->EncodeBatch(batch);
    auto embeddings = model->BatchForward(encodings, normalize, pooling_method);
    return embeddings;
  }

 protected:
  Tokenizer *tok = nullptr;
  ModelType *model = nullptr;
};

}  // namespace embeddings
