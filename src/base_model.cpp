#include "base_model.h"

#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "utils.h"

namespace embeddings {

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

ggml_tensor *BaseModel::get_tensor(ggml_context *ctx, const std::string &name) {
  ggml_tensor *tensor = ggml_get_tensor(ctx, name.c_str());
  if (!tensor) {
    fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name.c_str());
    throw std::runtime_error("tensor not found: " + name);
  }
  return tensor;
}

struct ggml_cgraph* BaseModel::CommonBatchForwardSetup(const std::vector<Encoding> &batch, 
                                       bool normalize, int pooling_method) {
  Clear();
  
  // build compute graph
  auto graph = BuildGraph(batch, normalize, pooling_method);

  // alloc graph
  ctx.compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.backend));
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

std::vector<std::vector<float>> BaseModel::ExtractResults(struct ggml_cgraph *graph, 
                                                        int batch_size, int hidden_size) {
  std::vector<std::vector<float>> ret;

  // in this example, output tensor is always the last tensor in the graph
  auto result = ggml_graph_node(graph, -1);
  
  // 确保张量有有效的缓冲区
  if (!result || !result->buffer) {
    fprintf(stderr, "%s: result tensor or buffer is null\n", __func__);
    Clear();
    return ret;
  }
  
  float *result_data = (float *)malloc(ggml_nbytes(result));
  
  // because the tensor data is stored in device buffer, we need to copy it back to RAM
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

void BaseModel::LoadModel(const std::string &gguf_model) {
  // 这是一个通用的模型加载框架，子类应该重写这个方法
  struct ggml_context *ctx_ggml = NULL;

  struct gguf_init_params gguf_params = {
      /*.no_alloc = */ true,
      /*.ctx      = */ &ctx_ggml,
  };

  // open gguf file
  struct gguf_context *ctx_gguf = gguf_init_from_file(gguf_model.c_str(), gguf_params);
  if (!ctx_gguf) {
    fprintf(stderr, "%s: failed to load model from %s. Does this file exist?\n",
            __func__, gguf_model.c_str());
    throw std::runtime_error("failed to load model file");
  }

  // get generic model info
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  const int n_kv = gguf_get_n_kv(ctx_gguf);
  const int ftype = get_u32(ctx_gguf, KEY_FTYPE);
  const int alignment = gguf_get_alignment(ctx_gguf);
  const int version = gguf_get_version(ctx_gguf);
  const std::string ftype_str = get_ftype(ftype);
  const std::string description = get_str(ctx_gguf, KEY_DESCRIPTION);
  const std::string name = get_str(ctx_gguf, KEY_NAME);
  arch = get_str(ctx_gguf, KEY_ARCHITECTURE);

  fprintf(stderr, "\n");
  fprintf(stderr, "%s: GGUF\n", __func__);
  fprintf(stderr, "%s: model name:   %s\n", __func__, name.c_str());
  fprintf(stderr, "%s: architecture: %s\n", __func__, arch.c_str());
  fprintf(stderr, "%s: description:  %s\n", __func__, description.c_str());
  fprintf(stderr, "%s: ftype:        %s\n", __func__, ftype_str.c_str());
  fprintf(stderr, "%s: version:      %d\n", __func__, version);
  fprintf(stderr, "%s: alignment:    %d\n", __func__, alignment);
  fprintf(stderr, "%s: n_tensors:    %d\n", __func__, n_tensors);
  fprintf(stderr, "%s: n_kv:         %d\n", __func__, n_kv);
  fprintf(stderr, "\n");

  // 在这里子类应该加载特定的权重和配置

  // clean up
  ggml_free(ctx_ggml);
  gguf_free(ctx_gguf);
}

}  // namespace embeddings
