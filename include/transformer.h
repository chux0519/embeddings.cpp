#pragma once

#include <ggml.h>
#include <string>
#include <vector>

namespace embeddings {

// Forward declarations of parameter structs
struct LayerNormParams;
struct MSAParams;
struct FFNParams;
struct TransformerParams;

// Base class template for all modules
template<typename InitParams, typename ForwardInput, typename ForwardOutput>
class Module {
protected:
    struct ggml_context* ctx = nullptr;  // Context for compute operations

public:
    virtual ~Module() = default;
    virtual void Init(const InitParams& params) = 0;
    virtual ForwardOutput Forward(const ForwardInput& input) = 0;
    
    // Set compute context
    void SetContext(struct ggml_context* compute_ctx) {
        ctx = compute_ctx;
    }
};

// Layer normalization module
class LayerNorm : public Module<LayerNormParams, struct ggml_tensor*, struct ggml_tensor*> {
private:
    struct ggml_tensor* weight;
    struct ggml_tensor* bias;
    float eps;

public:
    void Init(const LayerNormParams& params) override;
    struct ggml_tensor* Forward(struct ggml_tensor* const& input) override;
};

// Multi-head self attention module
class MultiHeadSelfAttention : public Module<MSAParams, struct ggml_tensor*, struct ggml_tensor*> {
private:
    struct ggml_tensor* q_weight;
    struct ggml_tensor* q_bias;
    struct ggml_tensor* k_weight;
    struct ggml_tensor* k_bias;
    struct ggml_tensor* v_weight;
    struct ggml_tensor* v_bias;
    struct ggml_tensor* o_weight;
    struct ggml_tensor* o_bias;
    int num_heads;
    int hidden_size;
    int head_dim;

public:
    void Init(const MSAParams& params) override;
    struct ggml_tensor* Forward(struct ggml_tensor* const& input) override;
};

// Feed forward network module
class FeedForwardNetwork : public Module<FFNParams, struct ggml_tensor*, struct ggml_tensor*> {
private:
    struct ggml_tensor* intermediate_weight;
    struct ggml_tensor* intermediate_bias;
    struct ggml_tensor* output_weight;
    struct ggml_tensor* output_bias;

public:
    void Init(const FFNParams& params) override;
    struct ggml_tensor* Forward(struct ggml_tensor* const& input) override;
};

// Transformer encoder module
class TransformerEncoder : public Module<TransformerParams, struct ggml_tensor*, struct ggml_tensor*> {
private:
    MultiHeadSelfAttention attention;
    FeedForwardNetwork ffn;
    LayerNorm attention_ln;
    LayerNorm ffn_ln;

public:
    void Init(const TransformerParams& params) override;
    struct ggml_tensor* Forward(struct ggml_tensor* const& input) override;
    
    // Override SetContext to propagate to sub-modules
    void SetContext(struct ggml_context* compute_ctx) {
        ctx = compute_ctx;
        attention.SetContext(compute_ctx);
        ffn.SetContext(compute_ctx);
        attention_ln.SetContext(compute_ctx);
        ffn_ln.SetContext(compute_ctx);
    }
};

} // namespace embeddings
