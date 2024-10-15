#pragma once

#include <string>
#include <vector>

#include "tokenizers_cpp.h"

namespace embeddings {
class Tokenizer {
public:
  Tokenizer(const std::string &path);

  std::vector<int> Encode(const std::string &);
  std::string Decode(const std::vector<int> &);

  tokenizers::Tokenizer *GetFastTokenizer();

private:
  tokenizers::Tokenizer *tok;
};
} // namespace embeddings