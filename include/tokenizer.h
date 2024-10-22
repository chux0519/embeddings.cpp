#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "huggingface_tokenizer.h"

typedef std::vector<int32_t> tokens;
typedef std::vector<tokens> tokens_batch;

typedef tokenizers::HFEncoding encoding;

namespace embeddings {
class Tokenizer {
public:
  Tokenizer(const std::string &path);

  encoding Encode(const std::string &);
  std::vector<encoding> EncodeBatch(const std::vector<std::string> &);
  std::string Decode(const tokens &);

  tokenizers::HFTokenizer *GetFastTokenizer();

private:
  tokenizers::HFTokenizer *tok;
};
} // namespace embeddings