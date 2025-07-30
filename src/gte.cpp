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

// Helper: scatter [D, N] -> [D, L, B] using indices
static struct ggml_tensor *ggml_scatter_rows_3d(
    struct ggml_context *ctx,
    struct ggml_tensor *src,              // [D, N]
    const std::vector<int32_t> &indices,  // length N
    int B, int L) {                       // target shape
  // Ensure src is contiguous first
  src = ggml_cont(ctx, src);

  const int D = src->ne[0];
  const int N = src->ne[1];
  struct ggml_tensor *dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, B);

  // Initialize dst to zeros
  dst = ggml_set_zero(dst);

  // Note: This function will be called during graph execution,
  // so we can't directly access tensor data here.
  // We need to implement this as a custom GGML operation or use existing
  // operations. For now, let's create a zero tensor and note that this needs to
  // be implemented properly.

  return dst;
}

// === helper for rotary embedding cache ===
static std::pair<std::vector<float>, std::vector<float>> build_rope_cache(
    int max_pos, int dim, float rope_theta) {
  const int half_dim = dim / 2;

  std::vector<float> inv_freq(half_dim);
  for (int i = 0; i < half_dim; ++i) {
    inv_freq[i] = 1.0f / powf(rope_theta, (float)i / half_dim);
  }

  std::vector<float> cos_data(max_pos * dim);
  std::vector<float> sin_data(max_pos * dim);

  for (int pos = 0; pos < max_pos; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      float val = pos * inv_freq[i];
      float cos_val = cosf(val);
      float sin_val = sinf(val);
      // duplicate like PyTorch cat((freqs, freqs))
      cos_data[pos * dim + i] = cos_val;
      cos_data[pos * dim + half_dim + i] = cos_val;
      sin_data[pos * dim + i] = sin_val;
      sin_data[pos * dim + half_dim + i] = sin_val;
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
  struct ggml_tensor *cos_broadcast = ggml_repeat(ctx, cos_shaped, x);
  struct ggml_tensor *sin_broadcast = ggml_repeat(ctx, sin_shaped, rotated);
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

  std::vector<int32_t> indices;
  std::vector<int32_t> unpadded_ids;
  std::vector<float> pad_mask_data(B * L, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < L; ++i) {
      if (batch[b].attention_mask[i]) {
        indices.push_back(b * L + i);
        unpadded_ids.push_back(batch[b].ids[i]);
        pad_mask_data[b * L + i] = 1.0f;
      }
    }
  }
  const int N = unpadded_ids.size();

  size_t ctx_size =
      GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead();
  struct ggml_init_params params = {ctx_size, NULL, true};
  ctx.compute_ctx = ggml_init(params);
  ctx.compute_graph_ctx = ggml_init(params);
  struct ggml_context *ctx0 = ctx.compute_graph_ctx;
  struct ggml_cgraph *gf = ggml_new_graph(ctx0);

  // Tensors
  auto *input_ids = ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, N);
  auto *pad_mask =
      ggml_new_tensor_4d(ctx.compute_ctx, GGML_TYPE_F32, 1, L, 1, B);
  auto *minus_one = ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_F32, 1);

  // RoPE cache tensors
  auto *rope_cos =
      ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, d_head, N);
  auto *rope_sin =
      ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, d_head, N);

  // Allocate backend memory for all tensors
  ctx.compute_buffer =
      ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);

  // Now set tensor data
  ggml_backend_tensor_set(input_ids, unpadded_ids.data(), 0,
                          N * sizeof(int32_t));
  ggml_backend_tensor_set(pad_mask, pad_mask_data.data(), 0,
                          sizeof(float) * B * L);
  float m1 = -1.0f;
  ggml_backend_tensor_set(minus_one, &m1, 0, sizeof(float));

  // RoPE cache
  auto [rope_cos_data, rope_sin_data] = build_rope_cache(N, d_head, theta);
  ggml_backend_tensor_set(rope_cos, rope_cos_data.data(), 0,
                          rope_cos_data.size() * sizeof(float));
  ggml_backend_tensor_set(rope_sin, rope_sin_data.data(), 0,
                          rope_sin_data.size() * sizeof(float));

  // Embedding
  auto *emb =
      ggml_get_rows(ctx0, embeddings.word_embeddings, input_ids);  // [D, N]
  emb = ggml_cont(ctx0, ggml_reshape_2d(ctx0, emb, D, N));
  emb = ggml_norm_inplace(ctx0, emb, gte_hparams->layer_norm_eps);
  emb = ggml_add(ctx0, ggml_mul(ctx0, emb, embeddings.LayerNorm_w),
                 embeddings.LayerNorm_b);

  // Attention mask
  auto *attn_mask = ggml_mul_mat(ctx0, pad_mask, pad_mask);  // [L, L, 1, B]
  attn_mask = ggml_add(ctx0, attn_mask, minus_one);
  attn_mask = ggml_scale_inplace(ctx0, attn_mask, 100000.0f);

  // Encoder (single QKV block version)
  auto *inpL = emb;  // [D, N]
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

    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));  // D, L, H, 1
    k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
    v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

    auto *score = ggml_mul_mat(ctx0, k, q);
    score = ggml_scale_inplace(ctx0, score, 1.0f / sqrtf((float)d_head));
    score = ggml_soft_max(ctx0, score);

    auto *attn = ggml_mul_mat(ctx0, v, score);
    attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx0, attn, D, N);

    auto *proj = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.o_proj_w, attn),
                          layer.o_proj_b);
    auto *res = ggml_add(ctx0, inpL, proj);
    res = ggml_norm_inplace(ctx0, res, gte_hparams->layer_norm_eps);
    res = ggml_add(ctx0, ggml_mul(ctx0, res, layer.attn_ln_w), layer.attn_ln_b);

    // 1. gated_layers
    // {768, 6144, 1, 1} * {768, 5 , 1, 1} = {6144, 5, 1, 1}
    const int hidden_features = gte_hparams->intermediate_size;  // 3072
    struct ggml_tensor *up_gate = ggml_mul_mat(ctx0, layer.up_gate_proj_w, res);
    // 2. Split gated and non-gated parts
    struct ggml_tensor *up_state =
        ggml_view_2d(ctx0, up_gate, hidden_features, res->ne[1], up_gate->nb[1],
                     0);  // {3072, 5, 1, 1}
    struct ggml_tensor *gate =
        ggml_view_2d(ctx0, up_gate, hidden_features, res->ne[1], up_gate->nb[1],
                     hidden_features * up_gate->nb[0]);
    // 3. Activation function (GELU) // {3072, 5, 1, 1}
    gate = ggml_cont(ctx0, gate);
    gate = ggml_gelu(ctx0, gate);
    // 4. Element-wise multiplication // {3072, 5, 1, 1}
    struct ggml_tensor *gated_states = ggml_mul(ctx0, gate, up_state);
    // 5. wo (linear transformation)
    struct ggml_tensor *ffn =
        ggml_add(ctx0, ggml_mul_mat(ctx0, layer.down_proj_w, gated_states),
                 layer.down_proj_b);

    inpL = ggml_add(ctx0, res, ffn);
    inpL = ggml_norm_inplace(ctx0, inpL, gte_hparams->layer_norm_eps);
    inpL = ggml_add(ctx0, ggml_mul(ctx0, inpL, layer.mlp_ln_w), layer.mlp_ln_b);
  }

  // Since we have unpadded tokens [D, N], we need to do pooling per batch
  // For simplicity, let's do mean pooling across all valid tokens per batch

  // Create per-batch pooled results
  std::vector<struct ggml_tensor *> batch_pooled;
  int token_offset = 0;

  for (int b = 0; b < B; ++b) {
    // Count valid tokens for this batch
    int valid_tokens = 0;
    for (int i = 0; i < L; ++i) {
      if (batch[b].attention_mask[i]) {
        valid_tokens++;
      }
    }

    if (valid_tokens > 0) {
      // Extract tokens for this batch [D, valid_tokens]
      auto *batch_tokens = ggml_view_2d(
          ctx0, inpL, D, valid_tokens, inpL->nb[1], token_offset * inpL->nb[1]);

      // Pool based on method
      struct ggml_tensor *batch_pooled_result;
      if (pooling_method == PoolingMethod::CLS) {
        // Use first token (CLS)
        batch_pooled_result =
            ggml_view_2d(ctx0, batch_tokens, D, 1, batch_tokens->nb[1], 0);
      } else {
        // Mean pooling
        batch_pooled_result = ggml_mean(ctx0, batch_tokens);
        batch_pooled_result = ggml_reshape_2d(ctx0, batch_pooled_result, D, 1);
      }

      batch_pooled.push_back(batch_pooled_result);
      token_offset += valid_tokens;
    } else {
      // If no valid tokens, create zero tensor
      auto *zero_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, 1);
      zero_tensor = ggml_set_zero(zero_tensor);
      batch_pooled.push_back(zero_tensor);
    }
  }

  // Concatenate all batch results [D, B]
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
