#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "bert.h"
#include "ggml.h"
#include "gguf.h"
#include "gte.h"
#include "jina_bert.h"

static std::map<std::string, enum ggml_type> ggml_type_map = {
    {"f32", GGML_TYPE_F32},   {"f16", GGML_TYPE_F16},
    {"q4_0", GGML_TYPE_Q4_0}, {"q4_1", GGML_TYPE_Q4_1},
    {"q5_0", GGML_TYPE_Q5_0}, {"q5_1", GGML_TYPE_Q5_1},
    {"q8_0", GGML_TYPE_Q8_0}, {"q2_k", GGML_TYPE_Q2_K},
    {"q3_k", GGML_TYPE_Q3_K}, {"q4_k", GGML_TYPE_Q4_K},
    {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K},
    {"q8_k", GGML_TYPE_Q8_K},
};

static enum ggml_type ggml_type_from_str(const std::string& s) {
  auto it = ggml_type_map.find(s);
  if (it == ggml_type_map.end()) {
    throw std::runtime_error("Invalid ggml type string: " + s);
  }
  return it->second;
}

static bool should_quantize_tensor(const std::string& name,
                                   const ggml_tensor* tensor) {
  // This used to be a regex, but <regex> has an extreme cost to compile times.
  bool quantize =
      name.rfind("weight") == name.size() - 6;  // ends with 'weight'?
  // quantize only 2D and 3D tensors (experts)
  quantize &= (ggml_n_dims(tensor) >= 2);
  // do not quantize norm tensors
  quantize &= name.find("embeddings.LayerNorm.weight") == std::string::npos;
  quantize &= name.find("mlp.up_gate_proj.weight") == std::string::npos;
  quantize &= name.find("attn_ln.weight") == std::string::npos;
  quantize &= name.find("mlp_ln.weight") == std::string::npos;
  return quantize;
}

static void write_gte_hparams(gguf_context* ctx,
                              const embeddings::GteBertConfig& hparams) {
  gguf_set_val_u32(ctx, "vocab_size", hparams.vocab_size);
  gguf_set_val_u32(ctx, "max_position_embeddings",
                   hparams.max_position_embeddings);
  gguf_set_val_u32(ctx, "hidden_size", hparams.hidden_size);
  gguf_set_val_u32(ctx, "intermediate_size", hparams.intermediate_size);
  gguf_set_val_u32(ctx, "num_attention_heads", hparams.num_attention_heads);
  gguf_set_val_u32(ctx, "num_hidden_layers", hparams.num_hidden_layers);
  gguf_set_val_f32(ctx, "layer_norm_eps", hparams.layer_norm_eps);
  gguf_set_val_f32(ctx, "rope_theta", hparams.rope_theta);
}