#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "huggingface_tokenizer.h"

typedef std::vector<int32_t> tokens;
typedef std::vector<tokens> tokens_batch;

typedef tokenizers::HFEncoding encoding;

namespace embeddings {

struct TokenizedInput {
  std::vector<int32_t> ids;
  std::vector<int32_t> attention_mask;
  size_t no_pad_len;
};

class Tokenizer {
 public:
  Tokenizer(const std::string &path);
  Tokenizer(const std::string &json_blob, bool /* is_blob */);

  TokenizedInput Encode(const std::string &, bool add_special_tokens = true);
  std::vector<TokenizedInput> EncodeBatch(const std::vector<std::string> &,
                                    bool add_special_tokens = true);
  std::string Decode(const tokens &, bool skip_special_tokens = true);

  tokenizers::HFTokenizer *GetFastTokenizer();

 private:
  tokenizers::HFTokenizer *tok;
};
}  // namespace embeddings