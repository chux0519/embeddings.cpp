#include "tokenizer.h"
#include "utils.h"

namespace embeddings {
Tokenizer::Tokenizer(const std::string &path) {
  auto blob = load_bytes_from_file(path);
  auto tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  tok = tok_.release();
}

tokens Tokenizer::Encode(const std::string &text) {
  return tok->Encode(text, true);
}

tokens_batch Tokenizer::EncodeBatch(const std::vector<std::string> &texts) {
  return tok->EncodeBatch(texts, true);
}

std::string Tokenizer::Decode(const tokens &ids) {
  return tok->Decode(ids, true);
}

tokenizers::Tokenizer *Tokenizer::GetFastTokenizer() { return tok; }

} // namespace embeddings
