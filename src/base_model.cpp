#include "base_model.h"

#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "utils.h"

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace embeddings {
namespace {

int get_num_threads() {
  unsigned int detected = std::thread::hardware_concurrency();
  int max_threads = detected > 0 ? static_cast<int>(detected) : 1;
  if (const char *env = std::getenv("EMBEDDINGS_CPP_THREADS")) {
    int value = std::atoi(env);
    if (value > 0) {
      return std::min(value, max_threads);
    }
  }

  return max_threads;
}

int get_blas_num_threads(int cpu_threads) {
  const char *env = std::getenv("EMBEDDINGS_CPP_BLAS_THREADS");
  if (env) {
    int value = std::atoi(env);
    if (value > 0) {
      return std::min(value, cpu_threads);
    }
  }

  // Snowflake fp32 uses many medium-sized GEMMs; OpenBLAS scales well up to a
  // point and then loses time in thread coordination.
  return std::min(cpu_threads, 6);
}

bool should_log_compute_buffer() {
  const char *env = std::getenv("EMBEDDINGS_CPP_LOG_COMPUTE_BUFFER");
  return env && std::atoi(env) != 0;
}

bool should_profile() {
  const char *env = std::getenv("EMBEDDINGS_CPP_PROFILE");
  return env && std::atoi(env) != 0;
}

bool should_print_graph() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GRAPH_PRINT");
  return env && std::atoi(env) != 0;
}

bool should_use_blas(int model_ftype) {
  const char *env = std::getenv("EMBEDDINGS_CPP_BLAS");
  if (env) {
    return std::atoi(env) != 0;
  }
  (void)model_ftype;
  return false;
}

const char *requested_backend() {
  return std::getenv("EMBEDDINGS_CPP_BACKEND");
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

BaseModel::BaseModel(const std::string &gguf_model) : model_path(gguf_model) {}

BaseModel::~BaseModel() {
  Clear();
  if (ctx.compute_allocr) {
    ggml_gallocr_free(ctx.compute_allocr);
    ctx.compute_allocr = NULL;
  }
  if (ctx.compute_sched) {
    ggml_backend_sched_free(ctx.compute_sched);
    ctx.compute_sched = NULL;
  }
  if (ctx.weights_buffer) {
    ggml_backend_buffer_free(ctx.weights_buffer);
    ctx.weights_buffer = NULL;
  }
  if (ctx.blas_backend) {
    ggml_backend_free(ctx.blas_backend);
    ctx.blas_backend = NULL;
  }
  if (ctx.fallback_backend) {
    ggml_backend_free(ctx.fallback_backend);
    ctx.fallback_backend = NULL;
  }
  if (ctx.ctx_data) {
    ggml_free(ctx.ctx_data);
    ctx.ctx_data = NULL;
  }
  if (ctx.backend) {
    ggml_backend_free(ctx.backend);
    ctx.backend = NULL;
  }
  if (hparams) {
    delete hparams;
    hparams = nullptr;
  }
}

void BaseModel::Load() { LoadModelImpl(this->model_path); }

std::vector<float> BaseModel::Forward(const TokenizedInput &enc, bool normalize,
                                      PoolingMethod pooling_method) {
  std::vector<TokenizedInput> batch = {enc};
  return BatchForward(batch, normalize, pooling_method)[0];
}

std::vector<std::vector<float>> BaseModel::BatchForward(
    const std::vector<TokenizedInput> &batch, bool normalize,
    PoolingMethod pooling_method) {
  if (batch.empty()) {
    return {};
  }
  const auto t0 = Clock::now();
  auto graph = CommonBatchForwardSetup(batch, normalize, pooling_method);
  const auto t1 = Clock::now();
  auto result = ExtractResults(graph, batch.size(), hparams->hidden_size);
  const auto t2 = Clock::now();
  if (should_profile()) {
    fprintf(stderr,
            "profile,total_batch_forward_ms=%.3f,setup_compute_ms=%.3f,extract_ms=%.3f,batch=%zu,hidden=%d\n",
            elapsed_ms(t0, t2), elapsed_ms(t0, t1), elapsed_ms(t1, t2),
            batch.size(), hparams->hidden_size);
  }
  return result;
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
  model_ftype = get_u32(ctx_gguf, KEY_FTYPE);
  fprintf(stderr, "%s: model name:   %s\n", __func__,
          get_str(ctx_gguf, KEY_NAME).c_str());
  fprintf(stderr, "%s: architecture: %s\n", __func__, arch.c_str());
  fprintf(stderr, "%s: ftype:        %s (%d)\n", __func__,
          get_ftype(model_ftype).c_str(), model_ftype);
  fprintf(stderr, "%s: desc: %s\n", __func__,
          get_str(ctx_gguf, KEY_DESCRIPTION).c_str());

  // 2. Call virtual functions to let subclasses load specific hyperparameters
  LoadHyperparameters(ctx_gguf);

  // 3. Initialize backend
  InitializeBackend();

  // 4. General tensor loading process
  {
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
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
    const char *cpu_repack_env = std::getenv("EMBEDDINGS_CPP_CPU_REPACK");
    const bool use_cpu_repack =
        (cpu_repack_env ? std::atoi(cpu_repack_env) != 0 : model_ftype != 0) &&
        !ctx.blas_backend && ggml_backend_is_cpu(ctx.backend);
    ggml_backend_buffer_type_t weight_buft =
        use_cpu_repack
            ? ggml_backend_cpu_repack_buffer_type()
            : (ggml_backend_is_cpu(ctx.backend)
                   ? ggml_backend_cpu_buffer_type()
                   : ggml_backend_get_default_buffer_type(ctx.backend));
    const size_t weight_alignment = ggml_backend_buft_get_alignment(weight_buft);
    size_t buffer_size = weight_alignment;
    for (int i = 0; i < n_tensors; ++i) {
      struct ggml_tensor *cur =
          ggml_get_tensor(ctx.ctx_data, gguf_get_tensor_name(ctx_gguf, i));
      buffer_size +=
          GGML_PAD(ggml_backend_buft_get_alloc_size(weight_buft, cur),
                   weight_alignment);
    }
    fprintf(stderr, "%s: model size = %.2f MB / num tensors = %d\n", __func__,
            buffer_size / (1024.0 * 1024.0), n_tensors);
    ctx.weights_buffer = ggml_backend_buft_alloc_buffer(weight_buft, buffer_size);
    fprintf(stderr, "%s: weights buffer: %s (repack=%d)\n", __func__,
            ggml_backend_buft_name(weight_buft), use_cpu_repack ? 1 : 0);
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

  // 5. Call virtual functions to let subclasses set their own pointers based on
  // loaded tensors
  LoadTensors();

  // 6. Cleanup
  gguf_free(ctx_gguf);
  ggml_free(ctx_ggml);
}

void BaseModel::Clear() {
  if (ctx.compute_sched) {
    ggml_backend_sched_free(ctx.compute_sched);
    ctx.compute_sched = NULL;
  }
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
  const char *backend_name = requested_backend();

#ifdef GGML_USE_WEBGPU
  if (!ctx.backend && backend_name && std::strcmp(backend_name, "webgpu") == 0) {
    fprintf(stderr, "%s: trying WebGPU backend\n", __func__);
    ctx.backend = ggml_backend_webgpu_init();
    if (!ctx.backend) {
      fprintf(stderr, "%s: ggml_backend_webgpu_init() failed\n", __func__);
    }
  }
#endif

  // initialize the backend
#ifdef GGML_USE_METAL
  if (!ctx.backend && (!backend_name || std::strcmp(backend_name, "metal") == 0)) {
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    ctx.backend = ggml_backend_metal_init();
    if (!ctx.backend) {
      fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
  }
#endif
#ifdef GGML_USE_VULKAN
  if (!ctx.backend && (!backend_name || std::strcmp(backend_name, "vulkan") == 0)) {
    fprintf(stderr, "%s: using Vulkan backend\n", __func__);
    ctx.backend = ggml_backend_vk_init(0);
    if (!ctx.backend) {
      fprintf(stderr, "%s: ggml_backend_vulkan_init() failed\n", __func__);
    }
  }
#endif

  // if there aren't GPU Backends fallback to CPU backend
  if (!ctx.backend) {
    fprintf(stderr, "%s: using CPU backend\n", __func__);
    ctx.backend = ggml_backend_cpu_init();
  }

  if (ctx.backend && !ggml_backend_is_cpu(ctx.backend)) {
    ctx.fallback_backend = ggml_backend_cpu_init();
    if (ctx.fallback_backend) {
      fprintf(stderr, "%s: using CPU fallback backend\n", __func__);
    } else {
      fprintf(stderr, "%s: ggml_backend_cpu_init() fallback failed\n", __func__);
    }
  }

#ifdef GGML_USE_BLAS
  if (ctx.backend && ggml_backend_is_cpu(ctx.backend) &&
      should_use_blas(model_ftype)) {
    ctx.blas_backend = ggml_backend_blas_init();
    if (ctx.blas_backend) {
      fprintf(stderr, "%s: using BLAS backend for supported ops\n", __func__);
    } else {
      fprintf(stderr, "%s: ggml_backend_blas_init() failed\n", __func__);
    }
  }
#endif
}

struct ggml_cgraph *BaseModel::CommonBatchForwardSetup(
    const std::vector<TokenizedInput> &batch, bool normalize,
    PoolingMethod pooling_method) {
  Clear();

  // build compute graph
  const auto t0 = Clock::now();
  auto graph = BuildGraph(batch, normalize, pooling_method);
  const auto t1 = Clock::now();

  // alloc graph
  if (!ctx.blas_backend && !ctx.fallback_backend) {
    ctx.compute_allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.backend));
    ggml_gallocr_alloc_graph(ctx.compute_allocr, graph);
  }
  const auto t2 = Clock::now();

  if (should_log_compute_buffer() && ctx.compute_allocr) {
    auto buffer_size = ggml_gallocr_get_buffer_size(ctx.compute_allocr, 0);
    fprintf(stderr, "compute buffer size: %.2f MB\n",
            buffer_size / 1024.0 / 1024.0);
  }

  // run the computation
  const int n_threads = get_num_threads();
  if (ggml_backend_is_cpu(ctx.backend)) {
    ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
  }
  if (ctx.fallback_backend) {
    ggml_backend_cpu_set_n_threads(ctx.fallback_backend, n_threads);
  }
#ifdef GGML_USE_BLAS
  if (ctx.blas_backend) {
    ggml_backend_blas_set_n_threads(ctx.blas_backend,
                                    get_blas_num_threads(n_threads));
  }
#endif

  enum ggml_status compute_status = GGML_STATUS_SUCCESS;
  if (ctx.blas_backend) {
    ggml_backend_t backends[] = {ctx.blas_backend, ctx.backend};
    ggml_backend_buffer_type_t bufts[] = {
        ggml_backend_get_default_buffer_type(ctx.backend),
        ggml_backend_get_default_buffer_type(ctx.backend),
    };
    ctx.compute_sched =
        ggml_backend_sched_new(backends, bufts, 2, ggml_graph_size(graph),
                               false, true);
    compute_status = ggml_backend_sched_graph_compute(ctx.compute_sched, graph);
  } else if (ctx.fallback_backend) {
    ggml_backend_t backends[] = {ctx.backend, ctx.fallback_backend};
    ctx.compute_sched =
        ggml_backend_sched_new(backends, nullptr, 2, ggml_graph_size(graph),
                               false, true);
    compute_status = ggml_backend_sched_graph_compute(ctx.compute_sched, graph);
  } else {
    compute_status = ggml_backend_graph_compute(ctx.backend, graph);
  }
  const auto t3 = Clock::now();

  if (compute_status != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("ggml graph compute failed");
  }

  if (should_print_graph()) {
    ggml_graph_print(graph);
  }

  if (should_profile()) {
    fprintf(stderr,
            "profile,build_graph_ms=%.3f,alloc_graph_ms=%.3f,compute_ms=%.3f,nodes=%d,batch=%zu,blas=%d\n",
            elapsed_ms(t0, t1), elapsed_ms(t1, t2), elapsed_ms(t2, t3),
            ggml_graph_n_nodes(graph), batch.size(), ctx.blas_backend ? 1 : 0);
  }

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

  ret.reserve(result->ne[1]);
  for (int j = 0; j < result->ne[1] /* rows */; j++) {
    const float *begin = result_data + j * result->ne[0];
    ret.emplace_back(begin, begin + result->ne[0]);
  }

  free(result_data);
  Clear();

  return ret;
}
}  // namespace embeddings
