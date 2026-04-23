#include "gte.h"

#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#include <cstdlib>
#include <fstream>

#include "tokenizer.h"
#include "utils.h"

namespace embeddings {

static bool use_flash_attn_ext() {
  const char *env = std::getenv("EMBEDDINGS_CPP_FLASH_ATTN");
  if (!env) {
    return true;
  }
  return std::atoi(env) != 0;
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
  token_ids.reserve(B * L);
  token_types.reserve(B * L);
  rope_positions.reserve(B * L);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < L; ++i) {
      if (batch[b].attention_mask[i]) {
        token_ids.push_back(batch[b].ids[i]);
        token_types.push_back(0);
        rope_positions.push_back(i);
        valid_token_counts[b]++;
      }
    }
    has_empty_row |= valid_token_counts[b] == 0;
  }
  const int N = token_ids.size();

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
      pooled = ggml_scale_inplace(ctx0, pooled, 1.0f / sqrtf((float)D));
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
  auto *zero_pooled =
      has_empty_row ? ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, D, 1)
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
  if (zero_pooled) {
    std::vector<float> zero_data(D, 0.0f);
    ggml_backend_tensor_set(zero_pooled, zero_data.data(), 0,
                            zero_data.size() * sizeof(float));
  }

  auto *emb = ggml_get_rows(ctx0, embeddings.word_embeddings, input_ids);
  emb = ggml_add(
      ctx0, emb,
      ggml_get_rows(ctx0, embeddings.token_type_embeddings, token_type_ids));
  emb = ggml_cont(ctx0, ggml_reshape_2d(ctx0, emb, D, N));
  emb = ggml_norm_inplace(ctx0, emb, gte_hparams->layer_norm_eps);
  emb = ggml_add(ctx0, ggml_mul(ctx0, emb, embeddings.LayerNorm_w),
                 embeddings.LayerNorm_b);

  auto *inpL = emb;
  for (int il = 0; il < n_layer; ++il) {
    const auto &layer = layers[il];
    auto *qkv = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.qkv_proj_w, inpL),
                         layer.qkv_proj_b);

    auto *q = ggml_reshape_4d(
        ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1], 0)),
        d_head, n_head, N, 1);
    auto *k = ggml_reshape_4d(
        ctx0,
        ggml_cont(ctx0,
                  ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1], D * qkv->nb[0])),
        d_head, n_head, N, 1);
    auto *v = ggml_reshape_4d(
        ctx0,
        ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, D, N, qkv->nb[1],
                                     2 * D * qkv->nb[0])),
        d_head, n_head, N, 1);

    q = ggml_apply_rotary_pos_emb(ctx0, q, rope_cos, rope_sin);
    k = ggml_apply_rotary_pos_emb(ctx0, k, rope_cos, rope_sin);

    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
    auto *v_flash = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));
    v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

    std::vector<struct ggml_tensor *> batch_attn;
    batch_attn.reserve(B);
    int token_offset = 0;
    for (int b = 0; b < B; ++b) {
      const int valid_tokens = valid_token_counts[b];
      if (valid_tokens == 0) {
        continue;
      }

      auto *qb = ggml_view_4d(ctx0, q, d_head, valid_tokens, n_head, 1,
                              q->nb[1], q->nb[2], q->nb[3],
                              token_offset * q->nb[1]);
      auto *kb = ggml_view_4d(ctx0, k, d_head, valid_tokens, n_head, 1,
                              k->nb[1], k->nb[2], k->nb[3],
                              token_offset * k->nb[1]);

      struct ggml_tensor *attn = nullptr;
      if (use_flash_attn_ext()) {
        auto *vb = ggml_view_4d(ctx0, v_flash, d_head, valid_tokens, n_head, 1,
                                v_flash->nb[1], v_flash->nb[2],
                                v_flash->nb[3], token_offset * v_flash->nb[1]);
        attn = ggml_flash_attn_ext(ctx0, qb, kb, vb, nullptr,
                                   1.0f / sqrtf((float)d_head), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_cont(ctx0, attn);
      } else {
        auto *vb = ggml_view_4d(ctx0, v, valid_tokens, d_head, n_head, 1,
                                v->nb[1], v->nb[2], v->nb[3],
                                token_offset * v->nb[0]);
        auto *score = ggml_mul_mat(ctx0, kb, qb);
        score = ggml_scale_inplace(ctx0, score, 1.0f / sqrtf((float)d_head));
        score = ggml_soft_max(ctx0, score);

        attn = ggml_mul_mat(ctx0, vb, score);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
      }
      batch_attn.push_back(ggml_reshape_2d(ctx0, attn, D, valid_tokens));
      token_offset += valid_tokens;
    }

    auto *attn = batch_attn[0];
    for (size_t b = 1; b < batch_attn.size(); ++b) {
      attn = ggml_concat(ctx0, attn, batch_attn[b], 1);
    }

    auto *proj = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.o_proj_w, attn),
                          layer.o_proj_b);
    auto *res = ggml_add(ctx0, inpL, proj);
    res = ggml_norm_inplace(ctx0, res, gte_hparams->layer_norm_eps);
    res = ggml_add(ctx0, ggml_mul(ctx0, res, layer.attn_ln_w), layer.attn_ln_b);

    const int hidden_features = gte_hparams->intermediate_size;
    struct ggml_tensor *up_gate = ggml_mul_mat(ctx0, layer.up_gate_proj_w, res);
    struct ggml_tensor *up_state =
        ggml_view_2d(ctx0, up_gate, hidden_features, res->ne[1], up_gate->nb[1],
                     0);
    struct ggml_tensor *gate =
        ggml_view_2d(ctx0, up_gate, hidden_features, res->ne[1], up_gate->nb[1],
                     hidden_features * up_gate->nb[0]);
    gate = ggml_cont(ctx0, gate);
    gate = ggml_gelu(ctx0, gate);
    struct ggml_tensor *gated_states = ggml_mul(ctx0, gate, up_state);
    struct ggml_tensor *ffn =
        ggml_add(ctx0, ggml_mul_mat(ctx0, layer.down_proj_w, gated_states),
                 layer.down_proj_b);

    inpL = ggml_add(ctx0, res, ffn);
    inpL = ggml_norm_inplace(ctx0, inpL, gte_hparams->layer_norm_eps);
    inpL = ggml_add(ctx0, ggml_mul(ctx0, inpL, layer.mlp_ln_w), layer.mlp_ln_b);
  }

  std::vector<struct ggml_tensor *> batch_pooled;
  batch_pooled.reserve(B);
  int token_offset = 0;
  for (int b = 0; b < B; ++b) {
    const int valid_tokens = valid_token_counts[b];
    if (valid_tokens > 0) {
      auto *batch_tokens = ggml_view_2d(
          ctx0, inpL, D, valid_tokens, inpL->nb[1], token_offset * inpL->nb[1]);
      if (pooling_method == PoolingMethod::CLS) {
        batch_pooled.push_back(
            ggml_view_2d(ctx0, batch_tokens, D, 1, batch_tokens->nb[1], 0));
      } else {
        auto *mean = ggml_mean(ctx0, batch_tokens);
        batch_pooled.push_back(ggml_reshape_2d(ctx0, mean, D, 1));
      }
      token_offset += valid_tokens;
    } else {
      batch_pooled.push_back(zero_pooled);
    }
  }

  auto *pooled = batch_pooled[0];
  for (int b = 1; b < B; ++b) {
    pooled = ggml_concat(ctx0, pooled, batch_pooled[b], 1);
  }

  if (normalize) {
    pooled = ggml_rms_norm(ctx0, pooled, gte_hparams->layer_norm_eps);
    pooled = ggml_scale_inplace(ctx0, pooled, 1.0f / sqrtf((float)D));
  }

  ggml_build_forward_expand(gf, pooled);
  return gf;
}

}  // namespace embeddings
