#include "gte.h"

#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#include <fstream>

#include "tokenizer.h"
#include "utils.h"

namespace embeddings {

GteBertModel::GteBertModel(const std::string &gguf_model) {
  struct ggml_context *ctx_ggml = NULL;

  struct gguf_init_params gguf_params = {
      /*.no_alloc = */ true,
      /*.ctx      = */ &ctx_ggml,
  };

  // open gguf file
  struct gguf_context *ctx_gguf =
      gguf_init_from_file(gguf_model.c_str(), gguf_params);
  if (!ctx_gguf) {
    fprintf(stderr,
            "%s: failed to load GTE model from %s. Does this file exist?\n",
            __func__, gguf_model.c_str());
    return;
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
  fprintf(stderr, "%s: architecture:   %s\n", __func__, arch.c_str());
  fprintf(stderr, "%s: description:  %s\n", __func__, description.c_str());
  fprintf(stderr, "%s: GGUF version: %d\n", __func__, version);
  fprintf(stderr, "%s: alignment:    %d\n", __func__, alignment);
  fprintf(stderr, "%s: n_tensors:    %d\n", __func__, n_tensors);
  fprintf(stderr, "%s: n_kv:         %d\n", __func__, n_kv);
  fprintf(stderr, "%s: ftype:        %s\n", __func__, ftype_str.c_str());
  fprintf(stderr, "\n");

  hparams = GteBertConfig();
  // load hparams
  {
    hparams.vocab_size = get_u32(ctx_gguf, "vocab_size");
    hparams.hidden_size = get_u32(ctx_gguf, "hidden_size");
    hparams.intermediate_size = get_u32(ctx_gguf, "intermediate_size");
    hparams.num_attention_heads = get_u32(ctx_gguf, "num_attention_heads");
    hparams.num_hidden_layers = get_u32(ctx_gguf, "num_hidden_layers");
    hparams.layer_norm_eps = get_f32(ctx_gguf, "layer_norm_eps");
    hparams.rope_theta = get_f32(ctx_gguf, "rope_theta");

    fprintf(stderr, "%s: MODEL\n", __func__);
    fprintf(stderr, "%s: vocab_size        = %d\n", __func__,
            hparams.vocab_size);
    fprintf(stderr, "%s: hidden_size         = %d\n", __func__,
            hparams.hidden_size);
    fprintf(stderr, "%s: intermediate_size = %d\n", __func__,
            hparams.intermediate_size);
    fprintf(stderr, "%s: num_attention_heads         = %d\n", __func__,
            hparams.num_attention_heads);
    fprintf(stderr, "%s: num_hidden_layers        = %d\n", __func__,
            hparams.num_hidden_layers);
    fprintf(stderr, "%s: layer_norm_eps = %g\n", __func__,
            hparams.layer_norm_eps);
    fprintf(stderr, "%s: rope_theta = %g\n", __func__, hparams.rope_theta);
    fprintf(stderr, "\n");
  }

  // init backend
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

  // model tensor sizing
  size_t buffer_size = 32 * 1024;  // TODO: need some extra room??
  {
    for (int i = 0; i < n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      const size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
      struct ggml_tensor *cur = ggml_get_tensor(ctx_ggml, name);
      size_t tensor_size = ggml_nbytes(cur);
      buffer_size += tensor_size;

      fprintf(stderr,
              "%s: tensor[%d]: type = %s, n_dims = %d, name = %s, offset=%zu, "
              "type=%d\n",
              __func__, i, ggml_type_name(cur->type), ggml_n_dims(cur),
              cur->name, offset, cur->type);
    }
  }

  // load tensors
  {
    // host buffer for CUDA loading
    std::vector<uint8_t> read_buf;

    // context params for tensors
    struct ggml_init_params ggml_params = {
        /*.mem_size =*/(n_tensors + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc =*/true,
    };

    // create context for tensors
    ctx.ctx_data = ggml_init(ggml_params);
    if (!ctx.ctx_data) {
      fprintf(stderr, "%s: ggml_init() failed\n", __func__);
      throw "";
    }

    // open model gguf file
    auto fin = std::ifstream(gguf_model, std::ios::binary);
    if (!fin) {
      fprintf(stderr, "cannot open model file for loading tensors\n");
      throw "";
    }

    // add tensors to our context
    for (int i = 0; i < n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      struct ggml_tensor *ten = ggml_get_tensor(ctx_ggml, name);
      struct ggml_tensor *cur = ggml_dup_tensor(ctx.ctx_data, ten);
      ggml_set_name(cur, name);
    }

    // create params buffer and allocr
    ctx.weights_buffer = ggml_backend_alloc_buffer(ctx.backend, buffer_size);
    auto alloc = ggml_tallocr_new(ctx.weights_buffer);

    // loop over tensors and load in
    for (int i = 0; i < n_tensors; ++i) {
      // do the actual allocation on the backend
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      struct ggml_tensor *cur = ggml_get_tensor(ctx.ctx_data, name);
      ggml_tallocr_alloc(&alloc, cur);

      // seek to the tensor data in the file
      const size_t offset =
          gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
      fin.seekg(offset, std::ios::beg);
      if (!fin) {
        fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__, name);
        throw "";
      }

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

  // use get_tensors to populate bert_model
  {
    // embeddings weights
    embeddings.word_embeddings =
        get_tensor(ctx.ctx_data, "embeddings.word_embeddings.weight");
    embeddings.token_type_embeddings =
        get_tensor(ctx.ctx_data, "embeddings.token_type_embeddings.weight");
    embeddings.LayerNorm_w =
        get_tensor(ctx.ctx_data, "embeddings.LayerNorm.weight");
    embeddings.LayerNorm_b =
        get_tensor(ctx.ctx_data, "embeddings.LayerNorm.bias");

    // layers
    layers.resize(hparams.num_hidden_layers);
    for (int i = 0; i < hparams.num_hidden_layers; ++i) {
      auto &layer = layers[i];
      std::string pre = "encoder.layer." + std::to_string(i) + ".";

      // attention
      layer.qkv_proj_w =
          get_tensor(ctx.ctx_data, pre + "attention.qkv_proj.weight");
      layer.qkv_proj_b =
          get_tensor(ctx.ctx_data, pre + "attention.qkv_proj.bias");
      layer.o_proj_w =
          get_tensor(ctx.ctx_data, pre + "attention.o_proj.weight");
      layer.o_proj_b = get_tensor(ctx.ctx_data, pre + "attention.o_proj.bias");

      layer.attn_ln_w = get_tensor(ctx.ctx_data, pre + "attn_ln.weight");
      layer.attn_ln_b = get_tensor(ctx.ctx_data, pre + "attn_ln.bias");

      // ff
      layer.up_gate_proj_w =
          get_tensor(ctx.ctx_data, pre + "mlp.up_gate_proj.weight");
      layer.down_proj_w =
          get_tensor(ctx.ctx_data, pre + "mlp.down_proj.weight");
      layer.down_proj_b = get_tensor(ctx.ctx_data, pre + "mlp.down_proj.bias");

      layer.mlp_ln_w = get_tensor(ctx.ctx_data, pre + "mlp_ln.weight");
      layer.mlp_ln_b = get_tensor(ctx.ctx_data, pre + "mlp_ln.bias");
    }
  }
  // free metadata
  ggml_free(ctx_ggml);
  gguf_free(ctx_gguf);
}

std::vector<float> GteBertModel::Forward(const Encoding &enc, bool normalize,
                                         int pooling_method) {
  std::vector<Encoding> batch = {enc};
  // return BatchForward(batch, pooling_method)[0];
  std::vector<float> empty;
  return empty;
}

std::vector<std::vector<float>> GteBertModel::BatchForward(
    const std::vector<Encoding> &batch, bool normalize, int pooling_method) {
  Clear();
  // build compute graph
  auto graph = BuildGraph(batch, normalize, pooling_method);

  // alloc graph
  ctx.compute_allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.backend));
  ggml_gallocr_alloc_graph(ctx.compute_allocr, graph);

  auto bufferss_size = ggml_gallocr_get_buffer_size(ctx.compute_allocr, 0);

  printf("compute buffer size: %.2f MB\n", bufferss_size / 1024.0 / 1024.0);

  ggml_backend_graph_compute(ctx.backend, graph);

  std::vector<std::vector<float>> ret(batch.size(), std::vector<float>(0));

  // in this example, output tensor is always the last tensor in the graph
  auto result = ggml_graph_node(graph, -1);
  float *result_data = (float *)malloc(ggml_nbytes(result));
  // because the tensor data is stored in device buffer, we need to copy it
  // / back / to RAM
  ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
  for (int j = 0; j < result->ne[1] /* rows */; j++) {
    std::vector<float> emb;
    for (int i = 0; i < result->ne[0] /* cols */; i++) {
      emb.push_back(result_data[j * result->ne[0] + i]);
    }
    ret[j] = emb;
  }

  Clear();

  return ret;
}

void GteBertModel::Clear() {
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

// reference from llama2
// https://github.com/ggerganov/llama.cpp/blob/master/ggml.c#L12175
static void apply_rope_inplace(struct ggml_tensor *q, struct ggml_tensor *k,
                               int pos, int n_rot, float theta) {
  GGML_ASSERT(q->ne[0] == k->ne[0]);
  GGML_ASSERT(q->ne[1] == k->ne[1]);
  GGML_ASSERT(q->ne[2] == k->ne[2]);
  GGML_ASSERT(q->ne[3] == k->ne[3]);

  const int n_elem = q->ne[0];
  const int n_head = q->ne[2];
  const int n_batch = q->ne[3];

  // const float scale = 1.0f / powf(10000.0f, rot / (float)n_elem);

  float *q_data = (float *)q->data;
  float *k_data = (float *)k->data;

  const size_t q_row_size = q->nb[1];
  const size_t k_row_size = k->nb[1];

  for (int b = 0; b < n_batch; ++b) {
    for (int h = 0; h < n_head; ++h) {
      float *q_ptr = q_data + b * q->nb[3] + h * q->nb[2];
      float *k_ptr = k_data + b * k->nb[3] + h * k->nb[2];

      for (int i = 0; i < n_elem; i += 2) {
        const float q0 = q_ptr[i * q->nb[0]];
        const float q1 = q_ptr[(i + 1) * q->nb[0]];

        const float k0 = k_ptr[i * k->nb[0]];
        const float k1 = k_ptr[(i + 1) * k->nb[0]];

        const float inv_freq = theta * ((float)pos);
        const float fC = cosf(inv_freq);
        const float fS = sinf(inv_freq);

        q_ptr[i * q->nb[0]] = q0 * fC - q1 * fS;
        q_ptr[(i + 1) * q->nb[0]] = q0 * fS + q1 * fC;

        k_ptr[i * k->nb[0]] = k0 * fC - k1 * fS;
        k_ptr[(i + 1) * k->nb[0]] = k0 * fS + k1 * fC;
      }
    }
  }
}

struct ggml_cgraph *GteBertModel::BuildGraph(const std::vector<Encoding> &batch,
                                             bool normalize,
                                             int pooling_method) {
  const int n_embd = hparams.hidden_size;
  const int n_layer = hparams.num_hidden_layers;
  const int n_head = hparams.num_attention_heads;
  const float layer_norm_eps = hparams.layer_norm_eps;
  const int d_head = n_embd / n_head;

  const int B = batch.size();
  const int L = batch[0].ids.size();
  const float theta = hparams.rope_theta;

  size_t ctx_size =
      GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead();

  struct ggml_init_params params = {
      ctx_size,
      NULL,
      true,
  };

  ctx.compute_ctx = ggml_init(params);
  ctx.compute_graph_ctx = ggml_init(params);
  struct ggml_context *ctx0 = ctx.compute_graph_ctx;

  struct ggml_cgraph *gf = ggml_new_graph(ctx0);

  struct ggml_tensor *input_ids =
      ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, B * L);
  struct ggml_tensor *token_types =
      ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, B * L);
  struct ggml_tensor *pos =
      ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, L);
  struct ggml_tensor *pool =
      ggml_new_tensor_3d(ctx.compute_ctx, GGML_TYPE_F32, L, 1, B);
  ctx.compute_buffer =
      ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);

  std::vector<int32_t> ids, types(B * L, 0);
  for (auto &b : batch) {
    ids.insert(ids.end(), b.ids.begin(), b.ids.end());
  }
  ggml_backend_tensor_set(input_ids, ids.data(), 0,
                          ids.size() * sizeof(int32_t));
  ggml_backend_tensor_set(token_types, types.data(), 0,
                          types.size() * sizeof(int32_t));

  // create RoPE position indices [0, 1, 2, ..., L-1]
  std::vector<int32_t> pos_data(L);
  for (int i = 0; i < L; ++i) pos_data[i] = i;
  ggml_backend_tensor_set(pos, pos_data.data(), 0, sizeof(int32_t) * L);

  float *out_mask = (float *)malloc(sizeof(float) * B * L);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < L; ++i) {
      out_mask[b * L + i] =
          pooling_method == 0 ? (i == 0 ? 1.f : 0.f) : 1.f / L;
    }
  }
  ggml_backend_tensor_set(pool, out_mask, 0, sizeof(float) * B * L);

  struct ggml_tensor *emb =
      ggml_get_rows(ctx0, embeddings.word_embeddings, input_ids);
  emb = ggml_add(
      ctx0, emb,
      ggml_get_rows(ctx0, embeddings.token_type_embeddings, token_types));
  emb = ggml_reshape_3d(ctx0, emb, n_embd, L, B);
  emb = ggml_norm_inplace(ctx0, emb, layer_norm_eps);
  emb = ggml_add(ctx0, ggml_mul(ctx0, emb, embeddings.LayerNorm_w),
                 embeddings.LayerNorm_b);

  struct ggml_tensor *inpL = emb;
  for (int il = 0; il < n_layer; il++) {
    const auto &layer = layers[il];
    struct ggml_tensor *cur = inpL;

    struct ggml_tensor *qkv = ggml_add(
        ctx0, ggml_mul_mat(ctx0, layer.qkv_proj_w, cur), layer.qkv_proj_b);

    struct ggml_tensor *q_layer = ggml_cont(
        ctx0, ggml_view_3d(ctx0, qkv, n_embd, L, B, qkv->nb[1], qkv->nb[2], 0));
    struct ggml_tensor *k_layer =
        ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, n_embd, L, B, qkv->nb[1],
                                     qkv->nb[2], n_embd * qkv->nb[0]));
    struct ggml_tensor *v_layer =
        ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, n_embd, L, B, qkv->nb[1],
                                     qkv->nb[2], 2 * n_embd * qkv->nb[0]));

    // Reshape q_layer and k_layer to 4D tensors before applying
    // ggml_rope_custom_inplace D, H, L, B
    q_layer = ggml_reshape_4d(ctx0, q_layer, d_head, n_head, L, B);
    k_layer = ggml_reshape_4d(ctx0, k_layer, d_head, n_head, L, B);
    v_layer = ggml_reshape_4d(ctx0, v_layer, d_head, n_head, L, B);

    // Apply ggml_rope_custom_inplace after reshaping
    q_layer = ggml_rope_custom_inplace(ctx0, q_layer, pos, d_head, 0, L, theta,
                                       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_layer = ggml_rope_custom_inplace(ctx0, k_layer, pos, d_head, 0, L, theta,
                                       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    q_layer = ggml_cont(
        ctx0, ggml_permute(ctx0, q_layer, 0, 2, 1,
                           3));  // D, H, L, B -> [D, L, H, B] {64, 5, 12, 1}
    k_layer = ggml_cont(ctx0, ggml_permute(ctx0, k_layer, 0, 2, 1,
                                           3));  // {64, 5, 12, 1}
    v_layer = ggml_cont(ctx0, ggml_permute(ctx0, v_layer, 1, 2, 0,
                                           3));  // D, H, L, B -> [H, L, D, B]

    // Debug: Print dimensions of k_layer and q_layer
    fprintf(stderr, "k_layer dimensions: [%d, %d, %d, %d]\n", k_layer->ne[0],
            k_layer->ne[1], k_layer->ne[2], k_layer->ne[3]);
    fprintf(stderr, "q_layer dimensions: [%d, %d, %d, %d]\n", q_layer->ne[0],
            q_layer->ne[1], q_layer->ne[2], q_layer->ne[3]);

    // Perform matrix multiplication after fixing dimensions
    struct ggml_tensor *scores = ggml_mul_mat(ctx0, k_layer, q_layer);
    scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)d_head));
    scores = ggml_soft_max(ctx0, scores);

    struct ggml_tensor *attn =
        ggml_mul_mat(ctx0, v_layer, scores);  // D, L, H, B
    attn =
        ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));  // -> D, H, L, B
    attn = ggml_reshape_3d(ctx0, attn, n_embd, L, B);           // -> E, L, B

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.o_proj_w, attn),
                   layer.o_proj_b);
    cur = ggml_add(ctx0, inpL, cur);
    cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.attn_ln_w), layer.attn_ln_b);

    const int in_feature = n_embd;
    const int hidden_features = hparams.intermediate_size;  // 3072

    struct ggml_tensor *norm2_res = cur;

    // 1. gated_layers
    // {768, 6144, 1, 1} * {768, 5 , 1, 1} = {6144, 5, 1, 1}
    struct ggml_tensor *gate_up = ggml_mul_mat(ctx0, layer.up_gate_proj_w, cur);
    // 2. Split gated and non-gated parts
    struct ggml_tensor *gate =
        ggml_view_2d(ctx0, gate_up, hidden_features, cur->ne[1], gate_up->nb[1],
                     0);  // {3072, 5, 1, 1}
    struct ggml_tensor *up =
        ggml_view_2d(ctx0, gate_up, hidden_features, cur->ne[1], gate_up->nb[1],
                     hidden_features * gate_up->nb[0]);
    // 3. Activation function (GELU) // {3072, 5, 1, 1}
    gate = ggml_cont(ctx0, gate);
    gate = ggml_gelu(ctx0, gate);
    // 4. Element-wise multiplication // {3072, 5, 1, 1}
    cur = ggml_mul(ctx0, gate, up);
    // 5. wo (linear transformation)
    struct ggml_tensor *ffn = ggml_add(
        ctx0, ggml_mul_mat(ctx0, layer.down_proj_w, cur), layer.down_proj_b);

    cur = ggml_add(ctx0, norm2_res, ffn);
    cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.mlp_ln_w), layer.mlp_ln_b);

    inpL = cur;
  }

  // C:\Users\chuxd\repos\embeddings.cpp\ggml\src\ggml.c:2692:
  // GGML_ASSERT(!ggml_is_transposed(a)) failed
  inpL = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inpL)), pool);
  inpL = ggml_reshape_2d(ctx0, inpL, n_embd, B);

  if (normalize) {
    inpL = ggml_rms_norm(ctx0, inpL, layer_norm_eps);
    inpL = ggml_scale_inplace(ctx0, inpL, 1.0f / sqrtf((float)n_embd));
  }

  ggml_build_forward_expand(gf, inpL);
  return gf;
}

GteEmbedding::GteEmbedding(const std::string &hf_token_json,
                           const std::string &gguf_model) {
  tok = new Tokenizer(hf_token_json);
  model = new GteBertModel(gguf_model);
}

std::vector<float> GteEmbedding::Encode(const std::string &text, bool normalize,
                                        int pooling_method) {
  std::vector<std::string> batch = {text};
  return BatchEncode(batch, normalize, pooling_method)[0];
}

std::vector<std::vector<float>> GteEmbedding::BatchEncode(
    const std::vector<std::string> &batch, bool normalize, int pooling_method) {
  auto encodings = tok->EncodeBatch(batch);
  auto embeddings = model->BatchForward(encodings, normalize, pooling_method);
  return embeddings;
}

}  // namespace embeddings
