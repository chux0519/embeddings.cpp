#include "transformer.h"
#include "ggml.h"

namespace embeddings {

struct LayerNormParams {
    struct ggml_tensor* weight;
    struct ggml_tensor* bias;
    float eps;
};

void LayerNorm::Init(const LayerNormParams& params) {
    weight = params.weight;
    bias = params.bias;
    eps = params.eps;
}

struct ggml_tensor* LayerNorm::Forward(struct ggml_tensor* const& input) {
    // Layer normalization
    auto normalized = ggml_norm(ctx, input, eps);
    auto scaled = ggml_mul(ctx, normalized, weight);
    return ggml_add(ctx, scaled, bias);
}

struct MSAParams {
    struct ggml_tensor* q_w;
    struct ggml_tensor* q_b;
    struct ggml_tensor* k_w;
    struct ggml_tensor* k_b;
    struct ggml_tensor* v_w;
    struct ggml_tensor* v_b;
    struct ggml_tensor* o_w;
    struct ggml_tensor* o_b;
    int n_head;
    int n_embd;
};

void MultiHeadSelfAttention::Init(const MSAParams& params) {
    q_weight = params.q_w;
    q_bias = params.q_b;
    k_weight = params.k_w;
    k_bias = params.k_b;
    v_weight = params.v_w;
    v_bias = params.v_b;
    o_weight = params.o_w;
    o_bias = params.o_b;
    num_heads = params.n_head;
    hidden_size = params.n_embd;
    head_dim = hidden_size / num_heads;
}

struct ggml_tensor* MultiHeadSelfAttention::Forward(struct ggml_tensor* const& input) {
    // Project to Q, K, V
    auto Q = ggml_add(ctx, ggml_mul_mat(ctx, q_weight, input), q_bias);
    auto K = ggml_add(ctx, ggml_mul_mat(ctx, k_weight, input), k_bias);
    auto V = ggml_add(ctx, ggml_mul_mat(ctx, v_weight, input), v_bias);

    // Reshape and transpose for attention
    Q = ggml_reshape_4d(ctx, Q, head_dim, num_heads, Q->ne[1], Q->ne[2]);
    K = ggml_reshape_4d(ctx, K, head_dim, num_heads, K->ne[1], K->ne[2]);
    V = ggml_reshape_4d(ctx, V, head_dim, num_heads, V->ne[1], V->ne[2]);

    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // [D, L, H, B]
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3)); // [D, L, H, B]
    V = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3)); // [L, D, H, B]

    // Scaled dot-product attention
    auto scores = ggml_mul_mat(ctx, K, Q); // [L, L, H, B]
    scores = ggml_scale_inplace(ctx, scores, 1.0f / sqrt(float(head_dim)));
    scores = ggml_soft_max(ctx, scores);

    // Combine with values
    auto attention = ggml_mul_mat(ctx, V, scores); // [D, L, H, B]
    attention = ggml_cont(ctx, ggml_permute(ctx, attention, 0, 2, 1, 3));
    attention = ggml_reshape_3d(ctx, attention, hidden_size, attention->ne[2], attention->ne[3]);

    // Output projection
    return ggml_add(ctx, ggml_mul_mat(ctx, o_weight, attention), o_bias);
}

struct FFNParams {
    struct ggml_tensor* i_w;
    struct ggml_tensor* i_b;
    struct ggml_tensor* o_w;
    struct ggml_tensor* o_b;
};

void FeedForwardNetwork::Init(const FFNParams& params) {
    intermediate_weight = params.i_w;
    intermediate_bias = params.i_b;
    output_weight = params.o_w;
    output_bias = params.o_b;
}

struct ggml_tensor* FeedForwardNetwork::Forward(struct ggml_tensor* const& input) {
    // First linear layer + GELU
    auto intermediate = ggml_add(ctx, ggml_mul_mat(ctx, intermediate_weight, input), intermediate_bias);
    intermediate = ggml_gelu(ctx, intermediate);

    // Second linear layer
    return ggml_add(ctx, ggml_mul_mat(ctx, output_weight, intermediate), output_bias);
}

struct TransformerParams {
    MSAParams attention;
    FFNParams ffn;
    LayerNormParams attn_ln;
    LayerNormParams ffn_ln;
};

void TransformerEncoder::Init(const TransformerParams& params) {
    // Initialize sub-components
    attention.Init(params.attention);
    ffn.Init(params.ffn);
    attention_ln.Init(params.attn_ln);
    ffn_ln.Init(params.ffn_ln);
}

struct ggml_tensor* TransformerEncoder::Forward(struct ggml_tensor* const& input) {
    // Self attention block
    auto attn_out = attention.Forward(input);
    auto residual = ggml_add(ctx, attn_out, input);
    auto norm_attn = attention_ln.Forward(residual);

    // Feed forward block
    auto ffn_out = ffn.Forward(norm_attn);
    residual = ggml_add(ctx, ffn_out, norm_attn);
    return ffn_ln.Forward(residual);
}

} // namespace embeddings
