#include <algorithm>  // For std::min/std::max
#include <cassert>
#include <cinttypes>  // For PRId64
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gguf_utils.h"  // Header file we created earlier
#include "utils.h"       // Your get_u32, get_f32, etc. functions

// Helper function borrowed from llama.cpp to copy kv pairs from one gguf context to another
void gguf_kv_set_from_ctx(gguf_context *ctx_out, gguf_context *ctx_in, int i) {
  const char *key = gguf_get_key(ctx_in, i);
  enum gguf_type type = gguf_get_kv_type(ctx_in, i);

  switch (type) {
    case GGUF_TYPE_UINT8:
      gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_in, i));
      break;
    case GGUF_TYPE_INT8:
      gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_in, i));
      break;
    case GGUF_TYPE_UINT16:
      gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_in, i));
      break;
    case GGUF_TYPE_INT16:
      gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_in, i));
      break;
    case GGUF_TYPE_UINT32:
      gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_in, i));
      break;
    case GGUF_TYPE_INT32:
      gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_in, i));
      break;
    case GGUF_TYPE_FLOAT32:
      gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_in, i));
      break;
    case GGUF_TYPE_BOOL:
      gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_in, i));
      break;
    case GGUF_TYPE_STRING:
      gguf_set_val_str(ctx_out, key, gguf_get_val_str(ctx_in, i));
      break;
    case GGUF_TYPE_ARRAY: {
      const void *data = gguf_get_arr_data(ctx_in, i);
      enum gguf_type arr_type = gguf_get_arr_type(ctx_in, i);
      uint64_t n = gguf_get_arr_n(ctx_in, i);
      gguf_set_arr_data(ctx_out, key, arr_type, data, n);
    } break;
    case GGUF_TYPE_UINT64:
      gguf_set_val_u64(ctx_out, key, gguf_get_val_u64(ctx_in, i));
      break;
    case GGUF_TYPE_INT64:
      gguf_set_val_i64(ctx_out, key, gguf_get_val_i64(ctx_in, i));
      break;
    case GGUF_TYPE_FLOAT64:
      gguf_set_val_f64(ctx_out, key, gguf_get_val_f64(ctx_in, i));
      break;
    default:
      break;
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s <input_model.gguf> <output_model.gguf> <qtype>\n",
            argv[0]);
    fprintf(stderr, "  qtype: e.g., 'q4_k', 'q6_k', 'q8_0'\n");
    return 1;
  }

  const std::string fname_inp = argv[1];
  const std::string fname_out = argv[2];
  const std::string qtype_str = argv[3];

  enum ggml_type out_type;
  try {
    out_type = ggml_type_from_str(qtype_str);
  } catch (const std::runtime_error &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }

  if (!ggml_is_quantized(out_type)) {
    fprintf(stderr, "Error: Target type %s is not a quantized type.\n",
            qtype_str.c_str());
    return 1;
  }

  printf("Input model:  %s\n", fname_inp.c_str());
  printf("Output model: %s\n", fname_out.c_str());
  printf("Target type:  %s\n\n", ggml_type_name(out_type));

  // 1. Open the input GGUF file and load metadata
  struct ggml_context *ctx_meta = nullptr;  // This context is only used to save tensor metadata
  struct gguf_init_params params = {
      /*.no_alloc = */ true,
      /*.ctx      = */ &ctx_meta,
  };
  struct gguf_context *ctx_in = gguf_init_from_file(fname_inp.c_str(), params);
  if (!ctx_in) {
    fprintf(stderr, "Failed to open %s\n", fname_inp.c_str());
    return 1;
  }

  // 2. Create GGUF writer for output
  struct gguf_context *ctx_out = gguf_init_empty();

  // First, register all tensor metadata to ctx_out
  int64_t n_tensors = gguf_get_n_tensors(ctx_in);
  for (int64_t i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_in, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_meta, name);
    gguf_add_tensor(ctx_out, tensor);
  }

  // 3. Copy metadata
  const int n_kv = gguf_get_n_kv(ctx_in);
  for (int i = 0; i < n_kv; ++i) {
    const char *key = gguf_get_key(ctx_in, i);
    // Update file type
    if (strcmp(key, "general.file_type") == 0) {
      gguf_set_val_u32(ctx_out, key, out_type);
    } else {
      gguf_kv_set_from_ctx(ctx_out, ctx_in, i);
    }
  }
  // Add quantization version
  gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);

  // Open input file to read tensor data
  FILE *f_in = fopen(fname_inp.c_str(), "rb");
  if (!f_in) {
    fprintf(stderr, "Failed to open %s for reading tensor data\n",
            fname_inp.c_str());
    gguf_free(ctx_in);
    ggml_free(ctx_meta);
    gguf_free(ctx_out);
    return 1;
  }

  // 1. First quantize and synchronize meta, collect new data and new sizes for all tensors
  std::vector<std::vector<char>> tensor_datas(n_tensors);
  std::vector<size_t> tensor_sizes(n_tensors);
  std::vector<float> work_f32;
  size_t max_q_size = 0;
  for (int i = 0; i < n_tensors; ++i) {
    struct ggml_tensor *tensor =
        ggml_get_tensor(ctx_meta, gguf_get_tensor_name(ctx_in, i));
    if (should_quantize_tensor(gguf_get_tensor_name(ctx_in, i), tensor)) {
      int nrows = tensor->ne[1];
      int row_size = tensor->ne[0];
      int qk = ggml_blck_size(out_type);
      int qtype_size = ggml_type_size(out_type);
      size_t q_size = qtype_size * row_size / qk * nrows;
      if (q_size > max_q_size) max_q_size = q_size;
    }
  }
  std::vector<char> work_q(max_q_size);
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_in, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_meta, name);
    enum ggml_type new_type =
        should_quantize_tensor(name, tensor) ? out_type : tensor->type;
    size_t nbytes_tensor = ggml_nbytes(tensor);
    std::vector<char> data(nbytes_tensor);
    fseek(f_in,
          gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, i),
          SEEK_SET);
    fread(data.data(), 1, data.size(), f_in);
    void *new_data = nullptr;
    size_t new_size = nbytes_tensor;
    printf("[%4d/%4d] %-40s | type: %-6s -> %-6s ... ", i + 1, n_tensors, name,
           ggml_type_name(tensor->type), ggml_type_name(new_type));
    fflush(stdout);
    if (should_quantize_tensor(name, tensor)) {
      const size_t nelements = ggml_nelements(tensor);
      work_f32.resize(nelements);
      if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((ggml_fp16_t *)data.data(), work_f32.data(),
                              nelements);
      } else {
        memcpy(work_f32.data(), data.data(), data.size());
      }
      int row_size = tensor->ne[0];
      int qk = ggml_blck_size(new_type);
      int qtype_size = ggml_type_size(new_type);
      int64_t min_chunk = 32 * 512;
      int64_t rows_per_chunk =
          std::max<int64_t>(1, (min_chunk + row_size - 1) / row_size);
      while ((rows_per_chunk * row_size) % qk != 0) ++rows_per_chunk;
      const int64_t chunk_size = rows_per_chunk * row_size;
      const int64_t nchunk = (nelements + chunk_size - 1) / chunk_size;
      size_t q_size = 0;
      for (int64_t chunk_idx = 0; chunk_idx < nchunk; ++chunk_idx) {
        int64_t offset = chunk_idx * chunk_size;
        int64_t cur_size = std::min<int64_t>(chunk_size, nelements - offset);
        int64_t n_chunk_rows = cur_size / row_size;
        q_size += ggml_quantize_chunk(
            new_type, work_f32.data() + offset,
            (uint8_t *)work_q.data() + offset * qtype_size / qk, 0,
            n_chunk_rows, row_size, nullptr);
      }
      tensor_datas[i].assign(work_q.data(), work_q.data() + q_size);
      tensor_sizes[i] = q_size;
      gguf_set_tensor_type(ctx_out, name, new_type);
      GGML_ASSERT(gguf_get_tensor_size(
                      ctx_out, gguf_find_tensor(ctx_out, name)) == q_size);
      printf("size: %8.2f MB -> %8.2f MB\n",
             ggml_nbytes(tensor) / 1024.0 / 1024.0, q_size / 1024.0 / 1024.0);
    } else {
      tensor_datas[i] = std::move(data);
      tensor_sizes[i] = nbytes_tensor;
      printf("size: %8.2f MB -> %8.2f MB\n",
             ggml_nbytes(tensor) / 1024.0 / 1024.0,
             nbytes_tensor / 1024.0 / 1024.0);
    }
    gguf_set_tensor_type(ctx_out, name, new_type);
    gguf_set_tensor_data(ctx_out, name, tensor_datas[i].data());
  }

  // 2. Write header/meta (meta is already synchronized at this point)
  // ---- llama.cpp style sequential write ----
  size_t meta_size = gguf_get_meta_size(ctx_out);
  std::ofstream fout(fname_out, std::ios::binary);
  if (!fout) {
    fprintf(stderr, "Failed to open %s for writing tensor data\n",
            fname_out.c_str());
    gguf_free(ctx_in);
    ggml_free(ctx_meta);
    gguf_free(ctx_out);
    return 1;
  }
  // First write meta_size zeros as placeholders
  std::vector<uint8_t> zeros(meta_size, 0);
  fout.write(reinterpret_cast<const char *>(zeros.data()), meta_size);

  size_t total_size_orig = 0;
  size_t total_size_new = 0;
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_in, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_meta, name);
    size_t new_size = tensor_sizes[i];
    void *new_data = tensor_datas[i].data();
    fout.write(reinterpret_cast<const char *>(new_data), new_size);
    size_t pad = GGML_PAD(new_size, GGUF_DEFAULT_ALIGNMENT) - new_size;
    if (pad > 0) {
      std::vector<uint8_t> pad_zeros(pad, 0);
      fout.write(reinterpret_cast<const char *>(pad_zeros.data()), pad);
    }
    total_size_orig += ggml_nbytes(tensor);
    total_size_new += new_size;
  }
  // Write the actual meta/header
  fout.seekp(0);
  std::vector<uint8_t> meta_buf(meta_size);
  gguf_get_meta_data(ctx_out, meta_buf.data());
  fout.write(reinterpret_cast<const char *>(meta_buf.data()), meta_size);
  fout.close();

  printf("\nQuantization successful!\n");
  printf("Original model size: %.2f MB\n", total_size_orig / 1024.0 / 1024.0);
  printf("Quantized model size: %.2f MB\n", total_size_new / 1024.0 / 1024.0);

  // 6. Cleanup
  gguf_free(ctx_in);
  ggml_free(ctx_meta);
  gguf_free(ctx_out);

  return 0;
}