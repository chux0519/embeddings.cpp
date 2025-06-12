#include "jina_bert.h"

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

std::vector<float> get_slopes_power_of_2(int nheads) {
  float start = pow(2, -(pow(2, -(log2(nheads) - 3))));
  float ratio = start;
  std::vector<float> slopes(nheads);
  for (int i = 0; i < nheads; ++i) {
    slopes[i] = start * pow(ratio, i);
  }
  return slopes;
}

// Function to calculate ALiBi slopes
std::vector<float> get_alibi_slopes(int nheads) {
  if (log2(nheads) == (int)log2(nheads)) {  // Check if nheads is a power of 2
    return get_slopes_power_of_2(nheads);
  } else {
    int closest_power_of_2 = pow(2, floor(log2(nheads)));
    std::vector<float> slopes_power_of_2 =
        get_slopes_power_of_2(closest_power_of_2);
    std::vector<float> slopes_recursive =
        get_alibi_slopes(2 * closest_power_of_2);
    std::vector<float> slopes_recursive_subset;

    // Extract every other element and limit the size
    for (size_t i = 0; i < slopes_recursive.size(); i += 2) {
      slopes_recursive_subset.push_back(slopes_recursive[i]);
    }
    slopes_recursive_subset.resize(nheads - closest_power_of_2);

    // Concatenate the two vectors
    slopes_power_of_2.insert(slopes_power_of_2.end(),
                             slopes_recursive_subset.begin(),
                             slopes_recursive_subset.end());
    return slopes_power_of_2;
  }
}

std::vector<float> get_alibi_data(int nheads, int seq_len) {
  // Get the slopes
  std::vector<float> slopes = get_alibi_slopes(nheads);

  // Calculate the size of the ALiBi data
  size_t alibi_data_size = (size_t)nheads * seq_len * seq_len;

  // Create a vector to store the ALiBi data, initialized to 0
  std::vector<float> alibi_data(alibi_data_size, 0.0f);  // Initialize to 0

  // Fill the ALiBi data
  for (int h = 0; h < nheads; ++h) {
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        alibi_data[h * seq_len * seq_len + i * seq_len + j] =
            -slopes[h] * abs(i - j);
      }
    }
  }

  return alibi_data;
}

JinaBertModel::JinaBertModel(const std::string &gguf_model)
    : BaseModel(gguf_model) {}

void JinaBertModel::LoadHyperparameters(struct gguf_context *ctx_gguf) {
  auto hparams = new JinaBertConfig();

  hparams->vocab_size = get_u32(ctx_gguf, KEY_VOCAB_SIZE);
  hparams->hidden_size = get_u32(ctx_gguf, KEY_HIDDEN_SIZE);
  hparams->num_hidden_layers = get_u32(ctx_gguf, KEY_NUM_HIDDEN_LAYERS);
  hparams->num_attention_heads = get_u32(ctx_gguf, KEY_NUM_ATTENTION_HEADS);
  hparams->intermediate_size = get_u32(ctx_gguf, KEY_INTERMEDIATE_SIZE);
  hparams->type_vocab_size = get_u32(ctx_gguf, KEY_TYPE_VOCAB_SIZE);
  hparams->pad_token_id = get_u32(ctx_gguf, KEY_PAD_TOKEN_ID);
  hparams->layer_norm_eps = get_f32(ctx_gguf, KEY_LAYER_NORM_EPS);

  this->hparams = hparams;
  
  fprintf(stderr, "%s: MODEL\n", __func__);
  fprintf(stderr, "%s: vocab_size        = %d\n", __func__,
          hparams->vocab_size);
  fprintf(stderr, "%s: hidden_size         = %d\n", __func__,
          hparams->hidden_size);
  fprintf(stderr, "%s: num_hidden_layers         = %d\n", __func__,
          hparams->num_hidden_layers);
  fprintf(stderr, "%s: num_attention_heads         = %d\n", __func__,
          hparams->num_attention_heads);
  fprintf(stderr, "%s: intermediate_size = %d\n", __func__,
          hparams->intermediate_size);
  fprintf(stderr, "%s: type_vocab_size         = %d\n", __func__,
          hparams->type_vocab_size);
  fprintf(stderr, "%s: pad_token_id        = %d\n", __func__,
          hparams->pad_token_id);
  fprintf(stderr, "%s: layer_norm_eps = %g\n", __func__,
          hparams->layer_norm_eps);
  fprintf(stderr, "\n");
}

void JinaBertModel::LoadTensors() {
  embeddings.word_embeddings =
      get_tensor(ctx.ctx_data, "transformer.embeddings.word_embeddings.weight");
  embeddings.token_type_embeddings = get_tensor(
      ctx.ctx_data, "transformer.embeddings.token_type_embeddings.weight");
  embeddings.ln_e_w = get_tensor(ctx.ctx_data, "transformer.emb_ln.weight");
  embeddings.ln_e_b = get_tensor(ctx.ctx_data, "transformer.emb_ln.bias");

  // layers
  layers.resize(hparams->num_hidden_layers);
  for (int i = 0; i < hparams->num_hidden_layers; ++i) {
    auto &layer = layers[i];
    std::string pre = "transformer.encoder.layers." + std::to_string(i) + ".";

    // attention
    layer.Wqkv_w = get_tensor(ctx.ctx_data, pre + "mixer.Wqkv.weight");
    layer.Wqkv_b = get_tensor(ctx.ctx_data, pre + "mixer.Wqkv.bias");

    layer.o_w = get_tensor(ctx.ctx_data, pre + "mixer.out_proj.weight");
    layer.o_b = get_tensor(ctx.ctx_data, pre + "mixer.out_proj.bias");

    layer.norm1_w = get_tensor(ctx.ctx_data, pre + "norm1.weight");
    layer.norm1_b = get_tensor(ctx.ctx_data, pre + "norm1.bias");

    // ff
    layer.mlp_gated_layers_w =
        get_tensor(ctx.ctx_data, pre + "mlp.gated_layers.weight");

    layer.mlp_out_w = get_tensor(ctx.ctx_data, pre + "mlp.wo.weight");
    layer.mlp_out_b = get_tensor(ctx.ctx_data, pre + "mlp.wo.bias");

    layer.norm2_w = get_tensor(ctx.ctx_data, pre + "norm2.weight");
    layer.norm2_b = get_tensor(ctx.ctx_data, pre + "norm2.bias");
  }
}

struct ggml_cgraph *JinaBertModel::BuildGraph(
    const std::vector<Encoding> &batch, bool normalize,
    PoolingMethod pooling_method) {
  auto jina_hparams = dynamic_cast<JinaBertConfig *>(this->hparams);
  if (!jina_hparams) {
    throw std::runtime_error("Incorrect hparams type for JinaBertModel");
  }
  const int n_embd = jina_hparams->hidden_size;
  const int n_layer = jina_hparams->num_hidden_layers;
  const int n_head = jina_hparams->num_attention_heads;
  const float layer_norm_eps = jina_hparams->layer_norm_eps;
  const int d_head = n_embd / n_head;  // E = D * H

  int n_batch_size = batch.size();           // B
  int cur_batch_size = batch[0].ids.size();  // L

  size_t ctx_size =
      GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead();

  // alloc `ggml_context` to store tensor data
  struct ggml_init_params params0 = {
      /*.mem_size   =*/ctx_size,
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true,
  };

  // initialze computational graph
  ctx.compute_ctx = ggml_init(params0);
  // embeddings = word_embeddings + token_type_embeddings
  struct ggml_tensor *token_layer = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_I32, cur_batch_size * n_batch_size);
  struct ggml_tensor *token_types = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_I32, cur_batch_size * n_batch_size);
  struct ggml_tensor *pad_mask = ggml_new_tensor_4d(
      ctx.compute_ctx, GGML_TYPE_F32, 1, cur_batch_size, 1, n_batch_size);
  struct ggml_tensor *pooler =
      ggml_new_tensor_3d(ctx.compute_ctx, GGML_TYPE_F32, cur_batch_size, 1,
                         n_batch_size);  // the avg pooler
  struct ggml_tensor *minus_one = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_F32, 1);  // for attention mask
  struct ggml_tensor *alibi_bias =
      ggml_new_tensor_4d(ctx.compute_ctx, GGML_TYPE_F32, cur_batch_size,
                         cur_batch_size, n_head, 1);

  ctx.compute_buffer =
      ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);

  // Copy tensor data from main memory (RAM) to backend buffer
  int32_t *token_layer_data = (int32_t *)malloc(ggml_nbytes(token_layer));
  int32_t *token_types_data = (int32_t *)malloc(ggml_nbytes(token_types));
  float *pad_mask_data = (float *)malloc(ggml_nbytes(pad_mask));
  float *pooler_data = (float *)malloc(ggml_nbytes(pooler));
  float m1 = -1.0f;

  for (int ba = 0; ba < n_batch_size; ba++) {
    for (int i = 0; i < cur_batch_size; i++) {
      int cur_len = batch[ba].ids.size();
      if (cur_len != cur_batch_size) {
        throw "batch should be padded before building";
      }

      token_layer_data[ba * cur_batch_size + i] = batch[ba].ids[i];
      pad_mask_data[ba * cur_batch_size + i] =
          static_cast<float>(batch[ba].attention_mask[i]);
      if (pooling_method == PoolingMethod::CLS) {
        // [CLS] is the first token, we only need the first one, for the later
        // mulmat
        pooler_data[ba * cur_batch_size + i] = (i == 0 ? 1 : 0);
      } else if (pooling_method == PoolingMethod::MEAN) {
        // default to use mean pooling
        pooler_data[ba * cur_batch_size + i] =
            (i < batch[ba].no_pad_len
                 ? 1 / static_cast<float>(batch[ba].no_pad_len)
                 : 0.0);
      } else {
        throw "unknow pooling method";
      }

      token_types_data[ba * cur_batch_size + i] = 0;
    }
  }

  auto alibi_bias_data = get_alibi_data(n_head, cur_batch_size);

  ggml_backend_tensor_set(alibi_bias, alibi_bias_data.data(), 0,
                          ggml_nbytes(alibi_bias));
  ggml_backend_tensor_set(token_layer, token_layer_data, 0,
                          ggml_nbytes(token_layer));
  ggml_backend_tensor_set(token_types, token_types_data, 0,
                          ggml_nbytes(token_types));
  ggml_backend_tensor_set(pad_mask, pad_mask_data, 0, ggml_nbytes(pad_mask));
  ggml_backend_tensor_set(pooler, pooler_data, 0, ggml_nbytes(pooler));
  ggml_backend_tensor_set(minus_one, &m1, 0, sizeof(m1));
  free(token_layer_data);
  free(token_types_data);
  free(pad_mask_data);
  free(pooler_data);

  // Create a `ggml_cgraph` for forward operation
  struct ggml_init_params params1 = {
      /*.mem_size   =*/ctx_size,
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true,  // the tensors will be allocated later by
                              // ggml_gallocr_alloc_graph()
  };
  ctx.compute_graph_ctx = ggml_init(params1);
  struct ggml_context *ctx_cgraph = ctx.compute_graph_ctx;

  struct ggml_cgraph *gf = ggml_new_graph(ctx_cgraph);

  // outer product the padding mask to kill off outside
  struct ggml_tensor *attn_mask =
      ggml_mul_mat(ctx_cgraph, pad_mask, pad_mask);        // [L, L, 1, B]
  attn_mask = ggml_add(ctx_cgraph, attn_mask, minus_one);  // result -0
  attn_mask = ggml_scale_inplace(ctx_cgraph, attn_mask, 10000.0f);

  // get various embedding components
  struct ggml_tensor *inpL = ggml_get_rows(
      ctx_cgraph, embeddings.word_embeddings, token_layer);  // [E, L * B]

  inpL = ggml_add(
      ctx_cgraph,
      ggml_get_rows(ctx_cgraph, embeddings.token_type_embeddings, token_types),
      inpL);
  inpL = ggml_reshape_3d(ctx_cgraph, inpL, n_embd, cur_batch_size,
                         n_batch_size);  // [E, L, B]
  // embed layer norm
  inpL = ggml_norm_inplace(ctx_cgraph, inpL, layer_norm_eps);
  inpL = ggml_add(ctx_cgraph, ggml_mul(ctx_cgraph, inpL, embeddings.ln_e_w),
                  embeddings.ln_e_b);  // [E, L, B]

  // layers
  for (int il = 0; il < n_layer; il++) {
    struct ggml_tensor *cur = inpL;

    // self-attention
    {
      // extract Q K V
      struct ggml_tensor *qkv = cur;
      qkv =
          ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].Wqkv_w, qkv),
                   layers[il].Wqkv_b);  // {2304, L, 1, 1}

      struct ggml_tensor *q_layer = ggml_cont(
          ctx_cgraph,
          ggml_view_3d(ctx_cgraph, qkv, n_embd, cur_batch_size, n_batch_size,
                       qkv->nb[1], qkv->nb[2], 0));  // [E, L, B]
      struct ggml_tensor *k_layer = ggml_cont(
          ctx_cgraph,
          ggml_view_3d(ctx_cgraph, qkv, n_embd, cur_batch_size, n_batch_size,
                       qkv->nb[1], qkv->nb[2], n_embd * qkv->nb[0]));
      struct ggml_tensor *v_layer = ggml_cont(
          ctx_cgraph,
          ggml_view_3d(ctx_cgraph, qkv, n_embd, cur_batch_size, n_batch_size,
                       qkv->nb[1], qkv->nb[2], 2 * n_embd * qkv->nb[0]));

      // Reshape into {64, 12, L, 1}
      q_layer = ggml_reshape_4d(ctx_cgraph, q_layer, d_head, n_head,
                                cur_batch_size, n_batch_size);
      k_layer = ggml_reshape_4d(ctx_cgraph, k_layer, d_head, n_head,
                                cur_batch_size, n_batch_size);
      v_layer = ggml_reshape_4d(ctx_cgraph, v_layer, d_head, n_head,
                                cur_batch_size, n_batch_size);

      q_layer = ggml_cont(
          ctx_cgraph,
          ggml_permute(ctx_cgraph, q_layer, 0, 2, 1,
                       3));  // D, H, L, B -> [D, L, H, B] {64, 5, 12, 1}
      k_layer = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, k_layer, 0, 2, 1,
                                                   3));  // {64, 5, 12, 1}
      v_layer =
          ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, v_layer, 1, 2, 0,
                                             3));  // D, H, L, B -> [H, L, D, B]

      struct ggml_tensor *attention_scores;
      attention_scores = ggml_mul_mat(
          ctx_cgraph, k_layer, q_layer);  // [L, L, n_head, B] {5, 5, 12, 1}
      attention_scores = ggml_scale_inplace(ctx_cgraph, attention_scores,
                                            1.0f / sqrt((float)d_head));

      attention_scores = ggml_add(ctx_cgraph, attention_scores, attn_mask);
      attention_scores = ggml_add(ctx_cgraph, attention_scores, alibi_bias);
      struct ggml_tensor *attention_probs = ggml_soft_max(
          ctx_cgraph, attention_scores);  // [L, L, n_head, B] {5, 5, 12, 1}

      struct ggml_tensor *attention_output = ggml_mul_mat(
          ctx_cgraph, v_layer, attention_probs);  // [d_head, L, n_head, B]
      attention_output =
          ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, attention_output, 0, 2,
                                             1, 3));  // -> [D, H, L, B]

      cur = ggml_reshape_3d(ctx_cgraph, attention_output, n_embd,
                            cur_batch_size, n_batch_size);  // [E, L, B]
    }

    // attention output
    cur = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].o_w, cur),
                   layers[il].o_b);
    // residual connection
    cur = ggml_add(ctx_cgraph, cur, inpL);
    // attention layer norm
    cur = ggml_norm_inplace(ctx_cgraph, cur, layer_norm_eps);
    cur = ggml_add(ctx_cgraph, ggml_mul(ctx_cgraph, cur, layers[il].norm1_w),
                   layers[il].norm1_b);

    // store for later
    struct ggml_tensor *norm2_res = cur;

    // GLUMLP
    int in_features = n_embd;
    int hidden_features = layers[il].mlp_gated_layers_w->ne[1] / 2;

    // 1. gated_layers
    // {768, 6144, 1, 1} * {768, 5 , 1, 1} = {6144, 5, 1, 1}
    struct ggml_tensor *gated_layers =
        ggml_mul_mat(ctx_cgraph, layers[il].mlp_gated_layers_w, cur);

    // 2. Split gated and non-gated parts
    struct ggml_tensor *gated =
        ggml_view_2d(ctx_cgraph, gated_layers, hidden_features, cur->ne[1],
                     gated_layers->nb[1], 0);  // {3072, 5, 1, 1}
    struct ggml_tensor *non_gated = ggml_view_2d(
        ctx_cgraph, gated_layers, hidden_features, cur->ne[1],
        gated_layers->nb[1], hidden_features * gated_layers->nb[0]);
    gated = ggml_cont(ctx_cgraph, gated);

    // 3. Activation function (GELU) // {3072, 5, 1, 1}
    gated = ggml_gelu(ctx_cgraph, gated);
    // 4. Element-wise multiplication // {3072, 5, 1, 1}
    cur = ggml_mul(ctx_cgraph, gated, non_gated);

    // 6. wo (linear transformation)
    struct ggml_tensor *glumlp_out = ggml_add(
        ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].mlp_out_w, cur),
        layers[il].mlp_out_b);

    cur = ggml_add(ctx_cgraph, glumlp_out, norm2_res);
    // output layer norm
    cur = ggml_norm_inplace(ctx_cgraph, cur, layer_norm_eps);
    cur = ggml_add(ctx_cgraph, ggml_mul(ctx_cgraph, cur, layers[il].norm2_w),
                   layers[il].norm2_b);
    // on to next layer
    inpL = cur;
  }

  // pooler
  inpL = ggml_mul_mat(ctx_cgraph,
                      ggml_cont(ctx_cgraph, ggml_transpose(ctx_cgraph, inpL)),
                      pooler);  // [ 1, E, B ]
  inpL = ggml_reshape_2d(ctx_cgraph, inpL, n_embd, n_batch_size);  // [E, B]

  // l2 normalize
  if (normalize) {
    inpL = ggml_rms_norm(ctx_cgraph, inpL, layer_norm_eps);  // [E, B]
    inpL = ggml_scale_inplace(
        ctx_cgraph, inpL,
        1.0f / sqrt((float)n_embd));  // [E, B] (since rms_norm does
                                      // mean instead of sum)
  }

  // final output
  ggml_tensor *output = inpL;

  // build the graph
  ggml_build_forward_expand(gf, output);

  // return complete graph
  return gf;
}

}  // namespace embeddings
