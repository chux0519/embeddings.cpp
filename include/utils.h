#pragma once

#include "ggml.h"

#include <string>
#include <vector>

namespace embeddings {
std::string load_bytes_from_file(const std::string &path);

void print_encode_result(const std::vector<int> &ids);
void print_tensors(const std::vector<float> &tensors);

int get_key_idx(const gguf_context *ctx, const char *key);
int32_t get_i32(const gguf_context *ctx, const std::string &key);
uint32_t get_u32(const gguf_context *ctx, const std::string &key);
float get_f32(const gguf_context *ctx, const std::string &key);
std::string get_str(const gguf_context *ctx, const std::string &key,
                    const std::string &def = "");
struct ggml_tensor *get_tensor(struct ggml_context *ctx,
                               const std::string &name);
std::string get_ftype(int ftype);
} // namespace embeddings