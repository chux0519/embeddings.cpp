#include "base_model.h"

#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <fstream>

#include "utils.h"

namespace embeddings {

BaseModel::BaseModel(const std::string &gguf_model) : model_path(gguf_model) {}

BaseModel::~BaseModel() {
  if (hparams) {
    delete hparams;
    hparams = nullptr;
  }
}

void BaseModel::Load() { LoadModelImpl(this->model_path); }

std::vector<float> BaseModel::Forward(const Encoding &enc, bool normalize,
                                      int pooling_method) {
  std::vector<Encoding> batch = {enc};
  return BatchForward(batch, normalize, pooling_method)[0];
}

std::vector<std::vector<float>> BaseModel::BatchForward(
    const std::vector<Encoding> &batch, bool normalize, int pooling_method) {
  if (batch.empty()) {
    return {};
  }
  auto graph = CommonBatchForwardSetup(batch, normalize, pooling_method);
  return ExtractResults(graph, batch.size(), hparams->hidden_size);
}

void BaseModel::LoadModelImpl(const std::string &gguf_model) {
  struct ggml_context *ctx_ggml = NULL;
  struct gguf_init_params gguf_params = {true, &ctx_ggml};

  struct gguf_context *ctx_gguf =
      gguf_init_from_file(gguf_model.c_str(), gguf_params);
  if (!ctx_gguf) {
    fprintf(stderr, "%s: failed to load model from %s.\n", __func__,
            gguf_model.c_str());
    throw std::runtime_error("failed to load model file");
  }

  // 1. Read general metadata
  fprintf(stderr, "\n%s: GGUF meta-data\n", __func__);
  arch = get_str(ctx_gguf, KEY_ARCHITECTURE);
  fprintf(stderr, "%s: model name:   %s\n", __func__,
          get_str(ctx_gguf, KEY_NAME).c_str());
  fprintf(stderr, "%s: architecture: %s\n", __func__, arch.c_str());
  fprintf(stderr, "%s: ftype:        %s\n", __func__,
          get_ftype(get_u32(ctx_gguf, KEY_FTYPE)).c_str());

  // 2. Call virtual functions to let subclasses load specific hyperparameters
  LoadHyperparameters(ctx_gguf);

  // 3. Initialize backend
  InitializeBackend();

  // 4. General tensor loading process
  {
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    size_t buffer_size = 0;
    for (int i = 0; i < n_tensors; ++i) {
      buffer_size += ggml_nbytes(
          ggml_get_tensor(ctx_ggml, gguf_get_tensor_name(ctx_gguf, i)));
    }
    fprintf(stderr, "%s: model size = %.2f MB / num tensors = %d\n", __func__,
            buffer_size / (1024.0 * 1024.0), n_tensors);

    struct ggml_init_params params = {(n_tensors + 1) * ggml_tensor_overhead(),
                                      NULL, true};
    ctx.ctx_data = ggml_init(params);
    if (!ctx.ctx_data) {
      throw std::runtime_error("ggml_init() failed for model data context");
    }

    // Copy tensor metadata
    for (int i = 0; i < n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      struct ggml_tensor *ten = ggml_get_tensor(ctx_ggml, name);
      struct ggml_tensor *cur = ggml_dup_tensor(ctx.ctx_data, ten);
      ggml_set_name(cur, name);
    }

    // Allocate and read weights
    ctx.weights_buffer = ggml_backend_alloc_buffer(ctx.backend, buffer_size);
    auto alloc = ggml_tallocr_new(ctx.weights_buffer);
    std::ifstream fin(gguf_model, std::ios::binary);
    std::vector<uint8_t> read_buf;

    for (int i = 0; i < n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      struct ggml_tensor *cur = ggml_get_tensor(ctx.ctx_data, name);
      ggml_tallocr_alloc(&alloc, cur);

      const size_t offset =
          gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
      fin.seekg(offset, std::ios::beg);

      // read in data and copy to device if needed
      int num_bytes = ggml_nbytes(cur);
      if (ggml_backend_buffer_is_host(ctx.weights_buffer)) {
        // for the CPU and Metal backend, we can read directly into the tensor
        fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
      } else {
        // read into a temporary buffer first, then copy to device memory
        read_buf.resize(num_bytes);
        fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
        ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
      }
    }
  }

  // 5. Call virtual functions to let subclasses set their own pointers based on loaded tensors
  LoadTensors();

  // 6. Cleanup
  gguf_free(ctx_gguf);
  ggml_free(ctx_ggml);
}

void BaseModel::Clear() {
  if (ctx.compute_graph_ctx) {
    ggml_free(ctx.compute_graph_ctx);
    ctx.compute_graph_ctx = NULL;
  }
  if (ctx.compute_allocr) {
    ggml_gallocr_free(ctx.compute_allocr);
    ctx.compute_allocr = NULL;
  }
  if (ctx.compute_ctx) {
    ggml_free(ctx.compute_ctx);
    ctx.compute_ctx = NULL;
  }
  if (ctx.compute_buffer) {
    ggml_backend_buffer_free(ctx.compute_buffer);
    ctx.compute_buffer = NULL;
  }
}

void BaseModel::InitializeBackend() {
  // initialize the backend
#ifdef GGML_USE_METAL
  fprintf(stderr, "%s: using Metal backend\n", __func__);
  ctx.backend = ggml_backend_metal_init();
  if (!ctx.backend) {
    fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
  }
#endif
#ifdef GGML_USE_VULKAN
  fprintf(stderr, "%s: using Vulkan backend\n", __func__);
  ctx.backend = ggml_backend_vk_init(0);
  if (!ctx.backend) {
    fprintf(stderr, "%s: ggml_backend_vulkan_init() failed\n", __func__);
  }
#endif

  // if there aren't GPU Backends fallback to CPU backend
  if (!ctx.backend) {
    fprintf(stderr, "%s: using CPU backend\n", __func__);
    ctx.backend = ggml_backend_cpu_init();
  }
}

struct ggml_cgraph *BaseModel::CommonBatchForwardSetup(
    const std::vector<Encoding> &batch, bool normalize, int pooling_method) {
  Clear();

  // build compute graph
  auto graph = BuildGraph(batch, normalize, pooling_method);

  // alloc graph
  ctx.compute_allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.backend));
  ggml_gallocr_alloc_graph(ctx.compute_allocr, graph);

  auto buffer_size = ggml_gallocr_get_buffer_size(ctx.compute_allocr, 0);
  printf("compute buffer size: %.2f MB\n", buffer_size / 1024.0 / 1024.0);

  // run the computation
  int n_threads = 1;
  if (ggml_backend_is_cpu(ctx.backend)) {
    ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
  }

  ggml_backend_graph_compute(ctx.backend, graph);

  return graph;
}

std::vector<std::vector<float>> BaseModel::ExtractResults(
    struct ggml_cgraph *graph, int batch_size, int hidden_size) {
  std::vector<std::vector<float>> ret;

  // in this example, output tensor is always the last tensor in the graph
  auto result = ggml_graph_node(graph, -1);

  // Ensure the tensor has a valid buffer
  if (!result || !result->buffer) {
    fprintf(stderr, "%s: result tensor or buffer is null\n", __func__);
    Clear();
    return ret;
  }

  float *result_data = (float *)malloc(ggml_nbytes(result));

  // because the tensor data is stored in device buffer, we need to copy it back
  // to RAM
  ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));

  for (int j = 0; j < result->ne[1] /* rows */; j++) {
    std::vector<float> emb;
    for (int i = 0; i < result->ne[0] /* cols */; i++) {
      emb.push_back(result_data[j * result->ne[0] + i]);
    }
    ret.push_back(emb);
  }

  free(result_data);
  Clear();

  return ret;
}
}  // namespace embeddings
