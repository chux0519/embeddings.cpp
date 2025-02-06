#include "utils.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace embeddings {
std::string load_bytes_from_file(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

void print_encode_result(const std::vector<int> &ids) {
  std::cout << "[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

void print_tensors(const std::vector<float> &tensors) {
  std::cout << "[";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << tensors[i];
  }
  std::cout << "]" << std::endl;
}

int get_key_idx(const gguf_context *ctx, const char *key) {
  int i = gguf_find_key(ctx, key);
  if (i == -1) {
    fprintf(stderr, "%s: key %s not found in file\n", __func__, key);
    throw;
  }

  return i;
}
int32_t get_i32(const gguf_context *ctx, const std::string &key) {
  const int i = get_key_idx(ctx, key.c_str());

  return gguf_get_val_i32(ctx, i);
}

uint32_t get_u32(const gguf_context *ctx, const std::string &key) {
  const int i = get_key_idx(ctx, key.c_str());

  return gguf_get_val_u32(ctx, i);
}

float get_f32(const gguf_context *ctx, const std::string &key) {
  const int i = get_key_idx(ctx, key.c_str());

  return gguf_get_val_f32(ctx, i);
}

std::string get_str(const gguf_context *ctx, const std::string &key,
                    const std::string &def) {
  const int i = gguf_find_key(ctx, key.c_str());
  if (i == -1) {
    return def;
  }
  return gguf_get_val_str(ctx, i);
}

struct ggml_tensor *get_tensor(struct ggml_context *ctx,
                               const std::string &name) {
  struct ggml_tensor *cur = ggml_get_tensor(ctx, name.c_str());
  if (!cur) {
    fprintf(stderr, "%s: unable to find tensor %s\n", __func__, name.c_str());
    throw;
  }

  return cur;
}

std::string get_ftype(int ftype) {
  return ggml_type_name(static_cast<ggml_type>(ftype));
}

std::string to_lowercase(const std::string &str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 ::tolower);
  return lower_str;
}
} // namespace embeddings
