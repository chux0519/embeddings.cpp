#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "tokenizer.h"
#include "utils.h"

namespace embeddings {
// Forward declaration
struct Encoding;

// Base configuration interface
struct BaseConfig {
  int32_t vocab_size = 0;
  int32_t hidden_size = 0;
  int32_t num_hidden_layers = 0;
  int32_t num_attention_heads = 0;
  float layer_norm_eps = 1e-12f;

  virtual ~BaseConfig() = default;
};

// Backend context
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

enum class PoolingMethod { MEAN = 0, CLS = 1 };

// Base model abstract class
class BaseModel {
 public:
  // NEW: Constructor will trigger model loading
  BaseModel(const std::string &gguf_model);
  virtual ~BaseModel();

  void Load();
  // NEW: Forward and BatchForward are no longer pure virtual functions, their
  // implementations are generic
  std::vector<float> Forward(
      const Encoding &enc, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN);
  std::vector<std::vector<float>> BatchForward(
      const std::vector<Encoding> &batch, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN);

 protected:
  // Pure virtual function - subclasses must implement
  virtual struct ggml_cgraph *BuildGraph(
      const std::vector<Encoding> &batch, bool normalize = true,
      PoolingMethod pooling_method = PoolingMethod::MEAN) = 0;

  // NEW: New pure virtual functions, subclasses must implement to load their
  // specific parts
  virtual void LoadHyperparameters(struct gguf_context *ctx_gguf) = 0;
  virtual void LoadTensors() = 0;

  // Protected functions, implement common logic
  void InitializeBackend();
  void Clear();

  struct ggml_cgraph *CommonBatchForwardSetup(
      const std::vector<Encoding> &batch, bool normalize,
      PoolingMethod pooling_method);
  std::vector<std::vector<float>> ExtractResults(struct ggml_cgraph *graph,
                                                 int batch_size,
                                                 int hidden_size);

  // Member variables
  std::string model_path;
  std::string arch;
  BaseConfig *hparams =
      nullptr;  // NEW: Use base class pointer to store hparams
  BackendContext ctx;

 private:
  void LoadModelImpl(const std::string &gguf_model);
};

}  // namespace embeddings
