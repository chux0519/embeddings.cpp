#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tokenizers_cpp.h"

typedef std::vector<int32_t> tokens;
typedef std::vector<tokens> tokens_batch;

namespace embeddings {
class Tokenizer {
public:
  Tokenizer(const std::string &path);

  tokens Encode(const std::string &);
  tokens_batch EncodeBatch(const std::vector<std::string> &);
  std::string Decode(const tokens &);

  tokenizers::Tokenizer *GetFastTokenizer();

private:
  tokenizers::Tokenizer *tok;
};
} // namespace embeddings