#include "bert.h"

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

BertModel::BertModel(const std::string &gguf_model) : BaseModel(gguf_model) {}

// NEW: Implement LoadHyperparameters, containing only model-specific loading
// logic
void BertModel::LoadHyperparameters(struct gguf_context *ctx_gguf) {
  auto params = new BertConfig();

  params->vocab_size = get_u32(ctx_gguf, KEY_VOCAB_SIZE);
  params->max_position_embeddings =
      get_u32(ctx_gguf, KEY_MAX_POSITION_EMBEDDING);
  params->hidden_size = get_u32(ctx_gguf, KEY_HIDDEN_SIZE);
  params->intermediate_size = get_u32(ctx_gguf, KEY_INTERMEDIATE_SIZE);
  params->num_attention_heads = get_u32(ctx_gguf, KEY_NUM_ATTENTION_HEADS);
  params->num_hidden_layers = get_u32(ctx_gguf, KEY_NUM_HIDDEN_LAYERS);
  params->layer_norm_eps = get_f32(ctx_gguf, KEY_LAYER_NORM_EPS);

  // Assign to the base class's hparams pointer
  this->hparams = params;

  // Print information
  fprintf(stderr, "%s: vocab_size        = %d\n", __func__, params->vocab_size);
  fprintf(stderr, "%s: max_position_embeddings   = %d\n", __func__,
          params->max_position_embeddings);
  fprintf(stderr, "%s: hidden_size         = %d\n", __func__,
          params->hidden_size);
  fprintf(stderr, "%s: num_hidden_layers   = %d\n", __func__,
          params->num_hidden_layers);
  fprintf(stderr, "\n");
}

void BertModel::LoadTensors() {
  // embeddings weights
  embeddings.word_embeddings =
      get_tensor(ctx.ctx_data, "embeddings.word_embeddings.weight");
  embeddings.token_type_embeddings =
      get_tensor(ctx.ctx_data, "embeddings.token_type_embeddings.weight");
  embeddings.position_embeddings =
      get_tensor(ctx.ctx_data, "embeddings.position_embeddings.weight");
  embeddings.ln_e_w = get_tensor(ctx.ctx_data, "embeddings.LayerNorm.weight");
  embeddings.ln_e_b = get_tensor(ctx.ctx_data, "embeddings.LayerNorm.bias");

  // pooler
  embeddings.pooler_e_w = get_tensor(ctx.ctx_data, "pooler.dense.weight");
  embeddings.pooler_e_b = get_tensor(ctx.ctx_data, "pooler.dense.bias");

  // layers
  layers.resize(hparams->num_hidden_layers);
  for (int i = 0; i < hparams->num_hidden_layers; ++i) {
    auto &layer = layers[i];
    std::string pre = "encoder.layer." + std::to_string(i) + ".";

    // attention
    layer.q_w = get_tensor(ctx.ctx_data, pre + "attention.self.query.weight");
    layer.q_b = get_tensor(ctx.ctx_data, pre + "attention.self.query.bias");
    layer.k_w = get_tensor(ctx.ctx_data, pre + "attention.self.key.weight");
    layer.k_b = get_tensor(ctx.ctx_data, pre + "attention.self.key.bias");
    layer.v_w = get_tensor(ctx.ctx_data, pre + "attention.self.value.weight");
    layer.v_b = get_tensor(ctx.ctx_data, pre + "attention.self.value.bias");

    layer.o_w = get_tensor(ctx.ctx_data, pre + "attention.output.dense.weight");
    layer.o_b = get_tensor(ctx.ctx_data, pre + "attention.output.dense.bias");

    layer.ln_att_w =
        get_tensor(ctx.ctx_data, pre + "attention.output.LayerNorm.weight");
    layer.ln_att_b =
        get_tensor(ctx.ctx_data, pre + "attention.output.LayerNorm.bias");

    // ff
    layer.ff_i_w = get_tensor(ctx.ctx_data, pre + "intermediate.dense.weight");
    layer.ff_i_b = get_tensor(ctx.ctx_data, pre + "intermediate.dense.bias");

    layer.ff_o_w = get_tensor(ctx.ctx_data, pre + "output.dense.weight");
    layer.ff_o_b = get_tensor(ctx.ctx_data, pre + "output.dense.bias");

    layer.ln_out_w = get_tensor(ctx.ctx_data, pre + "output.LayerNorm.weight");
    layer.ln_out_b = get_tensor(ctx.ctx_data, pre + "output.LayerNorm.bias");
  }
}

struct ggml_cgraph *BertModel::BuildGraph(const std::vector<Encoding> &batch,
                                          bool normalize,
                                          PoolingMethod pooling_method) {
  // Safely cast from base class pointer back to derived class config type
  auto bert_hparams = dynamic_cast<BertConfig *>(this->hparams);
  if (!bert_hparams) {
    throw std::runtime_error("Incorrect hparams type for BertModel");
  }
  // extract model params
  const int n_embd = bert_hparams->hidden_size;
  const int n_layer = bert_hparams->num_hidden_layers;
  const int n_max_tokens = bert_hparams->max_position_embeddings;
  const int n_head = bert_hparams->num_attention_heads;
  const float layer_norm_eps = bert_hparams->layer_norm_eps;
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
  // embeddings = word_embeddings + token_type_embeddings + position_embeddings
  struct ggml_tensor *token_layer = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_I32, cur_batch_size * n_batch_size);
  struct ggml_tensor *token_types = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_I32, cur_batch_size * n_batch_size);
  struct ggml_tensor *pad_mask = ggml_new_tensor_4d(
      ctx.compute_ctx, GGML_TYPE_F32, 1, cur_batch_size, 1, n_batch_size);
  struct ggml_tensor *positions = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_I32, cur_batch_size * n_batch_size);
  struct ggml_tensor *pooler =
      ggml_new_tensor_3d(ctx.compute_ctx, GGML_TYPE_F32, cur_batch_size, 1,
                         n_batch_size);  // the avg pooler
  struct ggml_tensor *minus_one = ggml_new_tensor_1d(
      ctx.compute_ctx, GGML_TYPE_F32, 1);  // for attention mask
  ctx.compute_buffer =
      ggml_backend_alloc_ctx_tensors(ctx.compute_ctx, ctx.backend);

  // Copy tensor data from main memory (RAM) to backend buffer
  int32_t *token_layer_data = (int32_t *)malloc(ggml_nbytes(token_layer));
  int32_t *token_types_data = (int32_t *)malloc(ggml_nbytes(token_types));
  float *pad_mask_data = (float *)malloc(ggml_nbytes(pad_mask));
  int32_t *pos_data = (int32_t *)malloc(ggml_nbytes(positions));
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

      if (arch == ARCH_XLMROBERTA) {
        // start from pad_token_id + 1, pad token is ignored
        pos_data[ba * cur_batch_size + i] =
            (i < batch[ba].no_pad_len ? i + 2 : 0);
      } else {
        pos_data[ba * cur_batch_size + i] = i;
      }
    }
  }

  ggml_backend_tensor_set(token_layer, token_layer_data, 0,
                          ggml_nbytes(token_layer));
  ggml_backend_tensor_set(token_types, token_types_data, 0,
                          ggml_nbytes(token_types));
  ggml_backend_tensor_set(pad_mask, pad_mask_data, 0, ggml_nbytes(pad_mask));
  ggml_backend_tensor_set(positions, pos_data, 0, ggml_nbytes(positions));
  ggml_backend_tensor_set(pooler, pooler_data, 0, ggml_nbytes(pooler));
  ggml_backend_tensor_set(minus_one, &m1, 0, sizeof(m1));

  free(token_layer_data);
  free(token_types_data);
  free(pad_mask_data);
  free(pos_data);
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
  attn_mask = ggml_scale_inplace(ctx_cgraph, attn_mask,
                                 100000.0f);  // FIXME: 1e3 will cause overflow?

  // get various embedding components
  struct ggml_tensor *inpL = ggml_get_rows(
      ctx_cgraph, embeddings.word_embeddings, token_layer);  // [E, L * B]
  inpL = ggml_add(
      ctx_cgraph,
      ggml_get_rows(ctx_cgraph, embeddings.token_type_embeddings, token_types),
      inpL);
  inpL = ggml_add(
      ctx_cgraph,
      ggml_get_rows(ctx_cgraph, embeddings.position_embeddings, positions),
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
      // extract Q
      struct ggml_tensor *Q = cur;
      Q = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].q_w, Q),
                   layers[il].q_b);  // [E, L, B]
      Q = ggml_reshape_4d(ctx_cgraph, Q, d_head, n_head, cur_batch_size,
                          n_batch_size);  // [D, H, L, B]
      Q = ggml_cont(ctx_cgraph,
                    ggml_permute(ctx_cgraph, Q, 0, 2, 1, 3));  // [D, L, H, B]

      // extract K
      struct ggml_tensor *K = cur;
      K = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].k_w, K),
                   layers[il].k_b);  // [E, L, B]
      K = ggml_reshape_4d(ctx_cgraph, K, d_head, n_head, cur_batch_size,
                          n_batch_size);  // [D, H, L, B]
      K = ggml_cont(ctx_cgraph,
                    ggml_permute(ctx_cgraph, K, 0, 2, 1, 3));  // [D, L, H, B]

      // extract V
      struct ggml_tensor *V = cur;
      V = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].v_w, V),
                   layers[il].v_b);  // [E, L, B]
      V = ggml_reshape_4d(ctx_cgraph, V, d_head, n_head, cur_batch_size,
                          n_batch_size);  // [D, H, L, B]
      V = ggml_cont(ctx_cgraph,
                    ggml_permute(ctx_cgraph, V, 1, 2, 0, 3));  // [H, L, D, B]

      // scaled attention
      struct ggml_tensor *KQ =
          ggml_mul_mat(ctx_cgraph, K, Q);  // -> [L, L, H, B]
      KQ = ggml_scale_inplace(ctx_cgraph, KQ, 1.0f / sqrt((float)d_head));
      KQ = ggml_add(ctx_cgraph, KQ, attn_mask);
      KQ = ggml_soft_max(ctx_cgraph, KQ);

      // get weighted values
      struct ggml_tensor *KQV =
          ggml_mul_mat(ctx_cgraph, V, KQ);  // -> [D, L, H, B]
      KQV = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, KQV, 0, 2, 1,
                                               3));  // -> [D, H, L, B]

      // copy back to input (E = D * H)
      cur = ggml_reshape_3d(ctx_cgraph, KQV, n_embd, cur_batch_size,
                            n_batch_size);  // [E, L, B]
    }

    // attention output
    cur = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].o_w, cur),
                   layers[il].o_b);

    // residual connection
    cur = ggml_add(ctx_cgraph, cur, inpL);

    // attention layer norm
    cur = ggml_norm_inplace(ctx_cgraph, cur, layer_norm_eps);
    cur = ggml_add(ctx_cgraph, ggml_mul(ctx_cgraph, cur, layers[il].ln_att_w),
                   layers[il].ln_att_b);

    // store for later
    struct ggml_tensor *att_output = cur;

    // feed forward steps
    cur = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].ff_i_w, cur),
                   layers[il].ff_i_b);
    cur = ggml_gelu(ctx_cgraph, cur);
    cur = ggml_add(ctx_cgraph, ggml_mul_mat(ctx_cgraph, layers[il].ff_o_w, cur),
                   layers[il].ff_o_b);

    // attentions bypass the intermediate layer
    cur = ggml_add(ctx_cgraph, att_output, cur);

    // output layer norm
    cur = ggml_norm_inplace(ctx_cgraph, cur, layer_norm_eps);
    cur = ggml_add(ctx_cgraph, ggml_mul(ctx_cgraph, cur, layers[il].ln_out_w),
                   layers[il].ln_out_b);

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
