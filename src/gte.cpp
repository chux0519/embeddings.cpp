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

// === PATCHED: helper for rotary embedding cache ===
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

// === PATCHED: rope apply using cache ===
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
  hparams->max_position_embeddings = get_u32(ctx_gguf, KEY_MAX_POSITION_EMBEDDING);
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

struct ggml_cgraph *GteBertModel::BuildGraph(const std::vector<TokenizedInput> &batch,
                                             bool normalize,
                                             PoolingMethod pooling_method) {
  auto gte_hparams = dynamic_cast<GteBertConfig *>(this->hparams);
  if (!gte_hparams) {
    throw std::runtime_error("Incorrect hparams type for BertModel");
  }
  const int n_embd = gte_hparams->hidden_size;
  const int n_layer = gte_hparams->num_hidden_layers;
  const int n_head = gte_hparams->num_attention_heads;
  const float layer_norm_eps = gte_hparams->layer_norm_eps;
  const int max_position_embeddings = gte_hparams->max_position_embeddings;
  const int d_head = n_embd / n_head;

  const int B = batch.size();
  const int L = batch[0].ids.size();
  const float theta = gte_hparams->rope_theta;

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
  struct ggml_tensor *pool =
      ggml_new_tensor_3d(ctx.compute_ctx, GGML_TYPE_F32, L, 1, B);
  struct ggml_tensor *pos =
      ggml_new_tensor_1d(ctx.compute_ctx, GGML_TYPE_I32, L);
  struct ggml_tensor *rope_cos =
      ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, d_head, L);
  struct ggml_tensor *rope_sin =
      ggml_new_tensor_2d(ctx.compute_ctx, GGML_TYPE_F32, d_head, L);
  struct ggml_tensor *pad_mask = ggml_new_tensor_4d(
      ctx.compute_ctx, GGML_TYPE_F32, 1, L, 1, B);
  struct ggml_tensor *minus_one = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_F32, 1);  // for attention mask
  ctx.compute_buffer =
      ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);

  // TODO: we should unpad all the inputs then restore them after the forward pass

  std::vector<int32_t> ids(B * L, 0);
  for (auto &b : batch) {
    // unpad inputs
    // ids.insert(ids.end(), b.ids.begin(), b.ids.begin() + b.no_pad_len);
    ids.insert(ids.end(), b.ids.begin(), b.ids.end());
  }
  auto [rope_cos_data, rope_sin_data] = build_rope_cache(L, d_head, theta);
  // create RoPE position indices [0, 1, 2, ..., L-1]
  // TODO: could be different for each batch, but we assume the same
  std::vector<int32_t> pos_data(L);
  for (int i = 0; i < L; ++i) pos_data[i] = i;

  ggml_backend_tensor_set(pos, pos_data.data(), 0, sizeof(int32_t) * L);
  ggml_backend_tensor_set(input_ids, ids.data(), 0,
                          ids.size() * sizeof(int32_t));
  ggml_backend_tensor_set(rope_cos, rope_cos_data.data(), 0,
                          rope_cos_data.size() * sizeof(float));
  ggml_backend_tensor_set(rope_sin, rope_sin_data.data(), 0,
                          rope_sin_data.size() * sizeof(float));

  float *pad_mask_data = (float *)malloc(ggml_nbytes(pad_mask));
  float *out_mask = (float *)malloc(sizeof(float) * B * L);
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < L; ++i) {
    pad_mask_data[b * L + i] =
          static_cast<float>(batch[b].attention_mask[i]);
      out_mask[b * L + i] =
          pooling_method == PoolingMethod::CLS ? (i == 0 ? 1.f : 0.f) : 1.f / L;
    }
  }
  float m1 = -1.0f;
  ggml_backend_tensor_set(minus_one, &m1, 0, sizeof(m1));
  ggml_backend_tensor_set(pad_mask, pad_mask_data, 0, ggml_nbytes(pad_mask));
  ggml_backend_tensor_set(pool, out_mask, 0, sizeof(float) * B * L);
  free(out_mask);
  free(pad_mask_data);

  // TODO: we should use the unpadded inputs
  struct ggml_tensor *emb =
      ggml_get_rows(ctx0, embeddings.word_embeddings, input_ids);
  emb = ggml_reshape_3d(ctx0, emb, n_embd, L, B);
  emb = ggml_norm_inplace(ctx0, emb, layer_norm_eps);
  emb = ggml_add(ctx0, ggml_mul(ctx0, emb, embeddings.LayerNorm_w),
                 embeddings.LayerNorm_b);

struct ggml_tensor *attn_mask =
      ggml_mul_mat(ctx0, pad_mask, pad_mask);        // [L, L, 1, B]
  attn_mask = ggml_add(ctx0, attn_mask, minus_one);  // result -0
  attn_mask = ggml_scale_inplace(ctx0, attn_mask,
                                 100000.0f);  // FIXME: 1e3 will cause overflow?

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

    q_layer = ggml_apply_rotary_pos_emb(ctx0, q_layer, rope_cos, rope_sin);
    k_layer = ggml_apply_rotary_pos_emb(ctx0, k_layer, rope_cos, rope_sin);

    q_layer = ggml_cont(
        ctx0, ggml_permute(ctx0, q_layer, 0, 2, 1,
                           3));  // D, H, L, B -> [D, L, H, B] {64, 5, 12, 1}
    k_layer = ggml_cont(ctx0, ggml_permute(ctx0, k_layer, 0, 2, 1,
                                           3));  // {64, 5, 12, 1}
    v_layer = ggml_cont(ctx0, ggml_permute(ctx0, v_layer, 1, 2, 0,
                                           3));  // D, H, L, B -> [H, L, D, B]

    // Perform matrix multiplication after fixing dimensions
    struct ggml_tensor *scores = ggml_mul_mat(ctx0, k_layer, q_layer);
    scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)d_head));
    scores = ggml_add(ctx0, scores, attn_mask);
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
    const int hidden_features = gte_hparams->intermediate_size;  // 3072

    struct ggml_tensor *norm2_res = cur;

    // 1. gated_layers
    // {768, 6144, 1, 1} * {768, 5 , 1, 1} = {6144, 5, 1, 1}
    struct ggml_tensor *up_gate = ggml_mul_mat(ctx0, layer.up_gate_proj_w, cur);
    // 2. Split gated and non-gated parts
    struct ggml_tensor *up_state =
        ggml_view_2d(ctx0, up_gate, hidden_features, cur->ne[1], up_gate->nb[1],
                     0);  // {3072, 5, 1, 1}
    struct ggml_tensor *gate =
        ggml_view_2d(ctx0, up_gate, hidden_features, cur->ne[1], up_gate->nb[1],
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

    cur = ggml_add(ctx0, norm2_res, ffn);
    cur = ggml_norm_inplace(ctx0, cur, layer_norm_eps);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.mlp_ln_w), layer.mlp_ln_b);

    inpL = cur;
  }

  // TODO: restore inputL here, we should use ggml functions

  inpL = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inpL)), pool);
  inpL = ggml_reshape_2d(ctx0, inpL, n_embd, B);

  if (normalize) {
    inpL = ggml_rms_norm(ctx0, inpL, layer_norm_eps);
    inpL = ggml_scale_inplace(ctx0, inpL, 1.0f / sqrtf((float)n_embd));
  }

  ggml_build_forward_expand(gf, inpL);
  return gf;
}

}  // namespace embeddings
