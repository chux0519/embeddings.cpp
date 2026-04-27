#include "gte.h"

#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "tokenizer.h"
#include "utils.h"

namespace embeddings {

static bool use_flash_attn_ext() {
  const char *env = std::getenv("EMBEDDINGS_CPP_FLASH_ATTN");
  if (!env) {
#ifdef __EMSCRIPTEN__
    const char *backend = std::getenv("EMBEDDINGS_CPP_BACKEND");
    if (backend && std::strcmp(backend, "webgpu") == 0) {
      return false;
    }
#endif
    return true;
  }
  return std::atoi(env) != 0;
}

static bool use_gte_fused_rope_layout() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_FUSED_ROPE_LAYOUT");
  if (!env) {
    return true;
  }
  return std::atoi(env) != 0;
}

static int gte_fused_v_min_tokens() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_FUSED_V_MIN_TOKENS");
  if (!env) {
    return 0;
  }
  return std::atoi(env);
}

static int gte_fused_geglu_min_tokens() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_FUSED_GEGLU_MIN_TOKENS");
  if (!env) {
    return 0;
  }
  return std::atoi(env);
}

static bool use_gte_fused_norm() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_FUSED_NORM");
  if (!env) {
    const char *backend = std::getenv("EMBEDDINGS_CPP_BACKEND");
    if (backend && std::strcmp(backend, "webgpu") == 0) {
      return true;
    }
    return false;
  }
  return std::atoi(env) != 0;
}

static bool use_gte_grouped_attention() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_GROUPED_ATTN");
  if (!env) {
    return true;
  }
  return std::atoi(env) != 0;
}

static int gte_grouped_attention_max_tokens() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_GROUPED_ATTN_MAX_TOKENS");
  if (!env) {
    return 96;
  }
  return std::atoi(env);
}

static float gte_grouped_attention_max_overhead() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_GROUPED_ATTN_MAX_OVERHEAD");
  if (!env) {
    return 1.8f;
  }
  return std::strtof(env, nullptr);
}

static bool use_gte_length_sorted_pack() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_LENGTH_SORT");
  if (!env) {
    return true;
  }
  return std::atoi(env) != 0;
}

static bool use_gte_fused_linear() {
  const char *env = std::getenv("EMBEDDINGS_CPP_GTE_FUSED_LINEAR");
  if (!env) {
    const char *backend = std::getenv("EMBEDDINGS_CPP_BACKEND");
    if (backend && std::strcmp(backend, "webgpu") == 0) {
      return false;
    }
    return true;
  }
  return std::atoi(env) != 0;
}

static bool use_inplace_graph_ops() {
  const char *backend = std::getenv("EMBEDDINGS_CPP_BACKEND");
  return !backend || std::strcmp(backend, "webgpu") != 0;
}

static ggml_tensor *gte_add(struct ggml_context *ctx, ggml_tensor *a,
                            ggml_tensor *b) {
  return use_inplace_graph_ops() ? ggml_add_inplace(ctx, a, b)
                                 : ggml_add(ctx, a, b);
}

static ggml_tensor *gte_mul(struct ggml_context *ctx, ggml_tensor *a,
                            ggml_tensor *b) {
  return use_inplace_graph_ops() ? ggml_mul_inplace(ctx, a, b)
                                 : ggml_mul(ctx, a, b);
}

static ggml_tensor *gte_scale(struct ggml_context *ctx, ggml_tensor *a,
                              float scale) {
  return use_inplace_graph_ops() ? ggml_scale_inplace(ctx, a, scale)
                                 : ggml_scale(ctx, a, scale);
}

static ggml_tensor *gte_gelu(struct ggml_context *ctx, ggml_tensor *a) {
  return use_inplace_graph_ops() ? ggml_gelu_inplace(ctx, a)
                                 : ggml_gelu(ctx, a);
}

static bool use_gte_fused_linear_for_shape(const ggml_tensor *weight) {
  if (!use_gte_fused_linear()) {
    return false;
  }

  const char *mode = std::getenv("EMBEDDINGS_CPP_GTE_FUSED_LINEAR_MODE");
  if (!mode || std::strcmp(mode, "all") == 0) {
    return true;
  }
  if (std::strcmp(mode, "attn") == 0) {
    return weight->ne[0] == 768 && (weight->ne[1] == 768 || weight->ne[1] == 2304);
  }
  if (std::strcmp(mode, "mlp") == 0) {
    return (weight->ne[0] == 768 && weight->ne[1] == 6144) ||
           (weight->ne[0] == 3072 && weight->ne[1] == 768);
  }
  return std::atoi(mode) != 0;
}

static ggml_tensor *gte_linear(struct ggml_context *ctx, ggml_tensor *weight,
                               ggml_tensor *inp, ggml_tensor *bias) {
  if (use_gte_fused_linear_for_shape(weight) && weight->type == GGML_TYPE_F32 &&
      inp->type == GGML_TYPE_F32 && (bias == nullptr || bias->type == GGML_TYPE_F32)) {
    return ggml_gte_linear(ctx, weight, inp, bias);
  }

  auto *out = ggml_mul_mat(ctx, weight, inp);
  return bias ? ggml_add(ctx, out, bias) : out;
}

struct AttentionGroup {
  int begin_batch = 0;
  int end_batch = 0;
  int token_offset = 0;
  int tokens = 0;
  bool needs_mask = false;
};

static std::vector<AttentionGroup> build_attention_groups(
    const std::vector<int> &valid_token_counts,
    const std::vector<int> &token_offsets, bool grouped_attention) {
  std::vector<AttentionGroup> groups;
  const int B = valid_token_counts.size();
  const int max_tokens = std::max(1, gte_grouped_attention_max_tokens());
  const float max_overhead =
      std::max(1.0f, gte_grouped_attention_max_overhead());

  for (int b = 0; b < B;) {
    if (valid_token_counts[b] == 0) {
      ++b;
      continue;
    }

    AttentionGroup group;
    group.begin_batch = b;
    group.end_batch = b + 1;
    group.token_offset = token_offsets[b];
    group.tokens = valid_token_counts[b];
    int64_t block_cost =
        (int64_t)valid_token_counts[b] * valid_token_counts[b];

    if (grouped_attention) {
      int next = b + 1;
      while (next < B) {
        const int next_tokens = valid_token_counts[next];
        if (next_tokens == 0) {
          group.end_batch = next + 1;
          ++next;
          continue;
        }

        const int merged_tokens = group.tokens + next_tokens;
        const int64_t merged_block_cost =
            block_cost + (int64_t)next_tokens * next_tokens;
        const float overhead =
            (float)((int64_t)merged_tokens * merged_tokens) /
            (float)merged_block_cost;
        if (merged_tokens > max_tokens || overhead > max_overhead) {
          break;
        }

        group.tokens = merged_tokens;
        group.end_batch = next + 1;
        block_cost = merged_block_cost;
        ++next;
      }
    }

    int non_empty = 0;
    for (int i = group.begin_batch; i < group.end_batch; ++i) {
      non_empty += valid_token_counts[i] > 0 ? 1 : 0;
    }
    group.needs_mask = non_empty > 1;
    groups.push_back(group);
    b = group.end_batch;
  }

  return groups;
}

static ggml_tensor *ggml_norm_affine_inplace(struct ggml_context *ctx,
                                             struct ggml_tensor *x,
                                             struct ggml_tensor *weight,
                                             struct ggml_tensor *bias,
                                             float eps) {
  if (use_gte_fused_norm()) {
    return ggml_gte_norm_affine(ctx, x, weight, bias, eps);
  }
  x = use_inplace_graph_ops() ? ggml_norm_inplace(ctx, x, eps)
                              : ggml_norm(ctx, x, eps);
  x = gte_mul(ctx, x, weight);
  x = gte_add(ctx, x, bias);
  return x;
}

// === helper for rotary embedding cache ===
static std::pair<std::vector<float>, std::vector<float>> build_rope_cache(
    const std::vector<int32_t> &positions, int dim, float rope_theta) {
  const int half_dim = dim / 2;

  std::vector<float> inv_freq(half_dim);
  for (int i = 0; i < half_dim; ++i) {
    inv_freq[i] = 1.0f / powf(rope_theta, (float)i / half_dim);
  }

  std::vector<float> cos_data(positions.size() * dim);
  std::vector<float> sin_data(positions.size() * dim);

  for (size_t token = 0; token < positions.size(); ++token) {
    const int pos = positions[token];
    for (int i = 0; i < half_dim; ++i) {
      float val = pos * inv_freq[i];
      float cos_val = cosf(val);
      float sin_val = sinf(val);
      // duplicate like PyTorch cat((freqs, freqs))
      cos_data[token * dim + i] = cos_val;
      cos_data[token * dim + half_dim + i] = cos_val;
      sin_data[token * dim + i] = sin_val;
      sin_data[token * dim + half_dim + i] = sin_val;
    }
  }

  return {cos_data, sin_data};
}

// === rope apply using cache ===
ggml_tensor *ggml_apply_rotary_pos_emb(
    struct ggml_context *ctx,
    struct ggml_tensor *x,           // [D, L, H, B]
    struct ggml_tensor *cos_tensor,  // same shape or broadcastable
    struct ggml_tensor *sin_tensor   // same shape or broadcastable
) {
  const int d = x->ne[0];
  GGML_ASSERT(d % 2 == 0);
  const int half_d = d / 2;

  // view x1 = x[:d/2], x2 = x[d/2:]
  struct ggml_tensor *x1 =
      ggml_view_4d(ctx, x, half_d, x->ne[1], x->ne[2], x->ne[3], x->nb[1],
                   x->nb[2], x->nb[3], 0);

  struct ggml_tensor *x2 =
      ggml_view_4d(ctx, x, half_d, x->ne[1], x->ne[2], x->ne[3], x->nb[1],
                   x->nb[2], x->nb[3], half_d * x->nb[0]);  // offset
  x2 = ggml_cont(ctx, x2);

  struct ggml_tensor *neg_x2 = ggml_scale(ctx, x2, -1.0f);
  struct ggml_tensor *rotated = ggml_concat(ctx, neg_x2, x1, 0);  // [-x2, x1]

  // final = x * cos + rotated * sin
  struct ggml_tensor *cos_shaped =
      ggml_reshape_4d(ctx, cos_tensor,
                      cos_tensor->ne[0],  // D
                      1,                  // H (will be broadcasted)
                      cos_tensor->ne[1],  // L
                      1                   // B (will be broadcasted)
      );                                  // => [D, H=1, L, B=1]
  struct ggml_tensor *sin_shaped =
      ggml_reshape_4d(ctx, sin_tensor,
                      sin_tensor->ne[0],  // D
                      1,                  // H (will be broadcasted)
                      sin_tensor->ne[1],  // L
                      1                   // B (will be broadcasted)
      );                                  // => [D, H=1, L, B=1]
  struct ggml_tensor *out = ggml_add(ctx, ggml_mul(ctx, x, cos_shaped),
                                     ggml_mul(ctx, rotated, sin_shaped));

  return out;
}

GteBertModel::GteBertModel(const std::string &gguf_model)
    : BaseModel(gguf_model) {}

void GteBertModel::LoadHyperparameters(struct gguf_context *ctx_gguf) {
  auto hparams = new GteBertConfig();
  // load hparams
  hparams->vocab_size = get_u32(ctx_gguf, KEY_VOCAB_SIZE);
  hparams->max_position_embeddings =
      get_u32(ctx_gguf, KEY_MAX_POSITION_EMBEDDING);
  hparams->hidden_size = get_u32(ctx_gguf, KEY_HIDDEN_SIZE);
  hparams->intermediate_size = get_u32(ctx_gguf, KEY_INTERMEDIATE_SIZE);
  hparams->num_attention_heads = get_u32(ctx_gguf, KEY_NUM_ATTENTION_HEADS);
  hparams->num_hidden_layers = get_u32(ctx_gguf, KEY_NUM_HIDDEN_LAYERS);
  hparams->layer_norm_eps = get_f32(ctx_gguf, KEY_LAYER_NORM_EPS);
  hparams->rope_theta = get_f32(ctx_gguf, "rope_theta");

  this->hparams = hparams;

  fprintf(stderr, "%s: MODEL\n", __func__);
  fprintf(stderr, "%s: vocab_size        = %d\n", __func__,
          hparams->vocab_size);
  fprintf(stderr, "%s: hidden_size         = %d\n", __func__,
          hparams->hidden_size);
  fprintf(stderr, "%s: intermediate_size = %d\n", __func__,
          hparams->intermediate_size);
  fprintf(stderr, "%s: num_attention_heads         = %d\n", __func__,
          hparams->num_attention_heads);
  fprintf(stderr, "%s: num_hidden_layers        = %d\n", __func__,
          hparams->num_hidden_layers);
  fprintf(stderr, "%s: layer_norm_eps = %g\n", __func__,
          hparams->layer_norm_eps);
  fprintf(stderr, "%s: rope_theta = %g\n", __func__, hparams->rope_theta);
  fprintf(stderr, "\n");
}

void GteBertModel::LoadTensors() {
  embeddings.word_embeddings =
      get_tensor(ctx.ctx_data, "embeddings.word_embeddings.weight");
  embeddings.token_type_embeddings =
      get_tensor(ctx.ctx_data, "embeddings.token_type_embeddings.weight");
  embeddings.LayerNorm_w =
      get_tensor(ctx.ctx_data, "embeddings.LayerNorm.weight");
  embeddings.LayerNorm_b =
      get_tensor(ctx.ctx_data, "embeddings.LayerNorm.bias");

  // layers
  layers.resize(hparams->num_hidden_layers);
  for (int i = 0; i < hparams->num_hidden_layers; ++i) {
    auto &layer = layers[i];
    std::string pre = "encoder.layer." + std::to_string(i) + ".";

    // attention
    layer.qkv_proj_w =
        get_tensor(ctx.ctx_data, pre + "attention.qkv_proj.weight");
    layer.qkv_proj_b =
        get_tensor(ctx.ctx_data, pre + "attention.qkv_proj.bias");
    layer.o_proj_w = get_tensor(ctx.ctx_data, pre + "attention.o_proj.weight");
    layer.o_proj_b = get_tensor(ctx.ctx_data, pre + "attention.o_proj.bias");

    layer.attn_ln_w = get_tensor(ctx.ctx_data, pre + "attn_ln.weight");
    layer.attn_ln_b = get_tensor(ctx.ctx_data, pre + "attn_ln.bias");

    // ff
    layer.up_gate_proj_w =
        get_tensor(ctx.ctx_data, pre + "mlp.up_gate_proj.weight");
    layer.down_proj_w = get_tensor(ctx.ctx_data, pre + "mlp.down_proj.weight");
    layer.down_proj_b = get_tensor(ctx.ctx_data, pre + "mlp.down_proj.bias");

    layer.mlp_ln_w = get_tensor(ctx.ctx_data, pre + "mlp_ln.weight");
    layer.mlp_ln_b = get_tensor(ctx.ctx_data, pre + "mlp_ln.bias");
  }
}

struct ggml_cgraph *GteBertModel::BuildGraph(
    const std::vector<TokenizedInput> &batch, bool normalize,
    PoolingMethod pooling_method) {
  auto gte_hparams = static_cast<GteBertConfig *>(this->hparams);
  const int D = gte_hparams->hidden_size;
  const int L = batch[0].ids.size();
  const int B = batch.size();
  const int n_layer = gte_hparams->num_hidden_layers;
  const int n_head = gte_hparams->num_attention_heads;
  const int d_head = D / n_head;
  const float theta = gte_hparams->rope_theta;

  std::vector<int32_t> token_ids;
  std::vector<int32_t> token_types;
  std::vector<int32_t> rope_positions;
  std::vector<int> valid_token_counts(B, 0);
  bool has_empty_row = false;
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < L; ++i) {
      if (batch[b].attention_mask[i]) {
        valid_token_counts[b]++;
      }
    }
    has_empty_row |= valid_token_counts[b] == 0;
  }
  std::vector<int> pack_order(B);
  for (int b = 0; b < B; ++b) {
    pack_order[b] = b;
  }
  if (use_gte_length_sorted_pack()) {
    std::stable_sort(pack_order.begin(), pack_order.end(),
                     [&](int lhs, int rhs) {
                       return valid_token_counts[lhs] < valid_token_counts[rhs];
                     });
  }
  token_ids.reserve(B * L);
  token_types.reserve(B * L);
  rope_positions.reserve(B * L);
  std::vector<int> token_offsets(B, 0);
  std::vector<int> packed_valid_token_counts;
  std::vector<int> packed_token_offsets;
  packed_valid_token_counts.reserve(B);
  packed_token_offsets.reserve(B);
  for (int order_index = 0; order_index < B; ++order_index) {
    const int b = pack_order[order_index];
    token_offsets[b] = token_ids.size();
    packed_token_offsets.push_back(token_ids.size());
    packed_valid_token_counts.push_back(valid_token_counts[b]);
    for (int i = 0; i < L; ++i) {
      if (batch[b].attention_mask[i]) {
        token_ids.push_back(batch[b].ids[i]);
        token_types.push_back(0);
        rope_positions.push_back(i);
      }
    }
  }
  const int N = token_ids.size();
  std::vector<int32_t> cls_positions;
  if (pooling_method == PoolingMethod::CLS) {
    cls_positions.reserve(B);
    for (int b = 0; b < B; ++b) {
      cls_positions.push_back(valid_token_counts[b] > 0 ? token_offsets[b]
                                                         : -1);
    }
  }
  const bool flash_attn = use_flash_attn_ext();
  const auto attention_groups = build_attention_groups(
      packed_valid_token_counts, packed_token_offsets,
      flash_attn && use_gte_grouped_attention());

  size_t ctx_size =
      B * GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
      ggml_graph_overhead();
  struct ggml_init_params params = {ctx_size, NULL, true};
  ctx.compute_ctx = ggml_init(params);
  ctx.compute_graph_ctx = ggml_init(params);
  struct ggml_context *ctx0 = ctx.compute_graph_ctx;
  struct ggml_cgraph *gf =
      ggml_new_graph_custom(ctx0, B * GGML_DEFAULT_GRAPH_SIZE, false);

  if (N == 0) {
    auto *zero_pooled =
        ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, D, B);
    ctx.compute_buffer =
        ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);
    std::vector<float> zero_data(D * B, 0.0f);
    ggml_backend_tensor_set(zero_pooled, zero_data.data(), 0,
                            zero_data.size() * sizeof(float));

    auto *pooled = zero_pooled;
    if (normalize) {
      pooled = ggml_rms_norm(ctx0, pooled, gte_hparams->layer_norm_eps);
      pooled = gte_scale(ctx0, pooled, 1.0f / sqrtf((float)D));
    }
    ggml_build_forward_expand(gf, pooled);
    return gf;
  }

  auto *input_ids = ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, N);
  auto *token_type_ids =
      ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, N);
  auto *rope_cos =
      ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, d_head, N);
  auto *rope_sin =
      ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, d_head, N);
  auto *cls_position_ids =
      pooling_method == PoolingMethod::CLS
          ? ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, B)
          : nullptr;
  std::vector<ggml_tensor *> attention_masks;
  std::vector<std::vector<ggml_fp16_t>> attention_mask_data;
  attention_masks.reserve(attention_groups.size());
  attention_mask_data.reserve(attention_groups.size());
  for (const auto &group : attention_groups) {
    if (!group.needs_mask) {
      attention_masks.push_back(nullptr);
      continue;
    }
    attention_masks.push_back(ggml_new_tensor_4d(
        ctx.compute_ctx, GGML_TYPE_F16, group.tokens, group.tokens, 1, 1));
    auto &mask = attention_mask_data.emplace_back(
        group.tokens * group.tokens, ggml_fp32_to_fp16(-INFINITY));
    for (int b = group.begin_batch; b < group.end_batch; ++b) {
      const int valid_tokens = packed_valid_token_counts[b];
      if (valid_tokens == 0) {
        continue;
      }
      const int local_begin = packed_token_offsets[b] - group.token_offset;
      for (int q = 0; q < valid_tokens; ++q) {
        for (int k = 0; k < valid_tokens; ++k) {
          mask[(local_begin + q) * group.tokens + local_begin + k] =
              ggml_fp32_to_fp16(0.0f);
        }
      }
    }
  }
  auto *zero_pooled =
      has_empty_row && pooling_method != PoolingMethod::CLS
          ? ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, D, 1)
          : nullptr;

  // Allocate backend memory for all tensors
  ctx.compute_buffer =
      ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);

  ggml_backend_tensor_set(input_ids, token_ids.data(), 0,
                          token_ids.size() * sizeof(int32_t));
  ggml_backend_tensor_set(token_type_ids, token_types.data(), 0,
                          token_types.size() * sizeof(int32_t));
  auto [rope_cos_data, rope_sin_data] =
      build_rope_cache(rope_positions, d_head, theta);
  ggml_backend_tensor_set(rope_cos, rope_cos_data.data(), 0,
                          rope_cos_data.size() * sizeof(float));
  ggml_backend_tensor_set(rope_sin, rope_sin_data.data(), 0,
                          rope_sin_data.size() * sizeof(float));
  if (cls_position_ids) {
    ggml_backend_tensor_set(cls_position_ids, cls_positions.data(), 0,
                            cls_positions.size() * sizeof(int32_t));
  }
  int mask_data_index = 0;
  for (size_t i = 0; i < attention_masks.size(); ++i) {
    if (!attention_masks[i]) {
      continue;
    }
    const auto &mask = attention_mask_data[mask_data_index++];
    ggml_backend_tensor_set(attention_masks[i], mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));
  }
  if (zero_pooled) {
    std::vector<float> zero_data(D, 0.0f);
    ggml_backend_tensor_set(zero_pooled, zero_data.data(), 0,
                            zero_data.size() * sizeof(float));
  }

  auto *emb = ggml_get_rows(ctx0, embeddings.word_embeddings, input_ids);
  emb = gte_add(ctx0, emb,
                ggml_get_rows(ctx0, embeddings.token_type_embeddings,
                              token_type_ids));
  emb = ggml_cont(ctx0, ggml_reshape_2d(ctx0, emb, D, N));
  emb = ggml_norm_affine_inplace(ctx0, emb, embeddings.LayerNorm_w,
                                 embeddings.LayerNorm_b,
                                 gte_hparams->layer_norm_eps);

  auto *inpL = emb;
  const bool fused_rope_layout = use_gte_fused_rope_layout();
  const bool fused_v_layout =
      flash_attn && N >= gte_fused_v_min_tokens();
  const bool fused_geglu = N >= gte_fused_geglu_min_tokens();
  for (int il = 0; il < n_layer; ++il) {
    const auto &layer = layers[il];
    auto *qkv = gte_linear(ctx0, layer.qkv_proj_w, inpL, layer.qkv_proj_b);

    ggml_tensor *q = nullptr;
    ggml_tensor *k = nullptr;
    ggml_tensor *v_flash = nullptr;
    ggml_tensor *v_scores = nullptr;
    if (fused_rope_layout) {
      auto *qkv_flash = ggml_gte_qkv_rope(ctx0, qkv, rope_cos, rope_sin, D,
                                          n_head, fused_v_layout);
      q = ggml_view_4d(ctx0, qkv_flash, d_head, N, n_head, 1,
                       qkv_flash->nb[1], qkv_flash->nb[2],
                       qkv_flash->nb[3], 0);
      k = ggml_view_4d(ctx0, qkv_flash, d_head, N, n_head, 1,
                       qkv_flash->nb[1], qkv_flash->nb[2],
                       qkv_flash->nb[3], qkv_flash->nb[3]);
      if (fused_v_layout) {
        v_flash = ggml_view_4d(ctx0, qkv_flash, d_head, N, n_head, 1,
                               qkv_flash->nb[1], qkv_flash->nb[2],
                               qkv_flash->nb[3], 2 * qkv_flash->nb[3]);
      }
    } else {
      auto *v = ggml_reshape_4d(
          ctx0,
          ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1],
                                       2 * D * qkv->nb[0])),
          d_head, n_head, N, 1);
      q = ggml_reshape_4d(
          ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1], 0)),
          d_head, n_head, N, 1);
      k = ggml_reshape_4d(
          ctx0,
          ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1],
                                       D * qkv->nb[0])),
          d_head, n_head, N, 1);

      q = ggml_apply_rotary_pos_emb(ctx0, q, rope_cos, rope_sin);
      k = ggml_apply_rotary_pos_emb(ctx0, k, rope_cos, rope_sin);

      q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
      k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
      if (flash_attn) {
        v_flash = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));
      } else {
        v_scores = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));
      }
    }
    if (flash_attn && v_flash == nullptr) {
      auto *v = ggml_reshape_4d(
          ctx0,
          ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1],
                                       2 * D * qkv->nb[0])),
          d_head, n_head, N, 1);
      v_flash = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));
    }
    if (!flash_attn && v_scores == nullptr) {
      auto *v = ggml_reshape_4d(
          ctx0,
          ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1],
                                       2 * D * qkv->nb[0])),
          d_head, n_head, N, 1);
      v_scores = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));
    }

    std::vector<struct ggml_tensor *> batch_attn;
    batch_attn.reserve(attention_groups.size());
    for (size_t ig = 0; ig < attention_groups.size(); ++ig) {
      const auto &group = attention_groups[ig];
      const int group_tokens = group.tokens;
      const int token_offset = group.token_offset;

      auto *qb = ggml_view_4d(ctx0, q, d_head, group_tokens, n_head, 1,
                              q->nb[1], q->nb[2], q->nb[3],
                              token_offset * q->nb[1]);
      auto *kb = ggml_view_4d(ctx0, k, d_head, group_tokens, n_head, 1,
                              k->nb[1], k->nb[2], k->nb[3],
                              token_offset * k->nb[1]);

      struct ggml_tensor *attn = nullptr;
      if (flash_attn) {
        auto *vb = ggml_view_4d(ctx0, v_flash, d_head, group_tokens, n_head, 1,
                                v_flash->nb[1], v_flash->nb[2],
                                v_flash->nb[3], token_offset * v_flash->nb[1]);
        attn = ggml_flash_attn_ext(ctx0, qb, kb, vb, attention_masks[ig],
                                   1.0f / sqrtf((float)d_head), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_cont(ctx0, attn);
      } else {
        auto *vb = ggml_view_4d(ctx0, v_scores, group_tokens, d_head, n_head, 1,
                                v_scores->nb[1], v_scores->nb[2],
                                v_scores->nb[3],
                                token_offset * v_scores->nb[0]);
        auto *score = ggml_mul_mat(ctx0, kb, qb);
        score = gte_scale(ctx0, score, 1.0f / sqrtf((float)d_head));
        score = ggml_soft_max(ctx0, score);

        attn = ggml_mul_mat(ctx0, vb, score);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
      }
      batch_attn.push_back(ggml_reshape_2d(ctx0, attn, D, group_tokens));
    }

    auto *attn = batch_attn[0];
    for (size_t b = 1; b < batch_attn.size(); ++b) {
      attn = ggml_concat(ctx0, attn, batch_attn[b], 1);
    }

    auto *proj = gte_linear(ctx0, layer.o_proj_w, attn, layer.o_proj_b);
    auto *res = gte_add(ctx0, proj, inpL);
    res = ggml_norm_affine_inplace(ctx0, res, layer.attn_ln_w,
                                   layer.attn_ln_b,
                                   gte_hparams->layer_norm_eps);

    const int hidden_features = gte_hparams->intermediate_size;
    struct ggml_tensor *up_gate =
        gte_linear(ctx0, layer.up_gate_proj_w, res, nullptr);
    struct ggml_tensor *gated_states = nullptr;
    if (fused_geglu) {
      gated_states = ggml_gte_geglu(ctx0, up_gate, hidden_features);
    } else {
      struct ggml_tensor *up_state =
          ggml_view_2d(ctx0, up_gate, hidden_features, res->ne[1],
                       up_gate->nb[1], 0);
      struct ggml_tensor *gate = ggml_view_2d(
          ctx0, up_gate, hidden_features, res->ne[1], up_gate->nb[1],
          hidden_features * up_gate->nb[0]);
      gate = ggml_cont(ctx0, gate);
      gate = gte_gelu(ctx0, gate);
      gated_states = ggml_mul(ctx0, gate, up_state);
    }
    struct ggml_tensor *ffn =
        gte_linear(ctx0, layer.down_proj_w, gated_states, layer.down_proj_b);

    inpL = gte_add(ctx0, ffn, res);
    inpL = ggml_norm_affine_inplace(ctx0, inpL, layer.mlp_ln_w, layer.mlp_ln_b,
                                    gte_hparams->layer_norm_eps);
  }

  struct ggml_tensor *pooled = nullptr;
  if (pooling_method == PoolingMethod::CLS) {
    pooled = ggml_gte_cls_pool(ctx0, inpL, cls_position_ids);
  } else {
    std::vector<struct ggml_tensor *> batch_pooled;
    batch_pooled.reserve(B);
    for (int b = 0; b < B; ++b) {
      const int valid_tokens = valid_token_counts[b];
      if (valid_tokens > 0) {
        auto *batch_tokens =
            ggml_view_2d(ctx0, inpL, D, valid_tokens, inpL->nb[1],
                         token_offsets[b] * inpL->nb[1]);
        auto *mean = ggml_mean(ctx0, batch_tokens);
        batch_pooled.push_back(ggml_reshape_2d(ctx0, mean, D, 1));
      } else {
        batch_pooled.push_back(zero_pooled);
      }
    }

    pooled = batch_pooled[0];
    for (int b = 1; b < B; ++b) {
      pooled = ggml_concat(ctx0, pooled, batch_pooled[b], 1);
    }
  }

  if (normalize) {
    pooled = ggml_rms_norm(ctx0, pooled, gte_hparams->layer_norm_eps);
    pooled = gte_scale(ctx0, pooled, 1.0f / sqrtf((float)D));
  }

  ggml_build_forward_expand(gf, pooled);
  return gf;
}

}  // namespace embeddings
