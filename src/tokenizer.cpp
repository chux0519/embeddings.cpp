#include "tokenizer.h"
#include "utils.h"

namespace embeddings {
Tokenizer::Tokenizer(const std::string &path) {
  auto blob = LoadBytesFromFile(path);
  auto tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  tok = tok_.release();
}

std::vector<int> Tokenizer::Encode(const std::string &text) {
  return tok->Encode(text);
}

std::string Tokenizer::Decode(const std::vector<int> &ids) {
  return tok->Decode(ids);
}

tokenizers::Tokenizer *Tokenizer::GetFastTokenizer() { return tok; }

} // namespace embeddings
