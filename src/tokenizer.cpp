#include "tokenizer.h"
#include "utils.h"
#include <cstddef>
#include <cstdio>
#include <iostream>

namespace embeddings {
Tokenizer::Tokenizer(const std::string &path) {
  auto blob = load_bytes_from_file(path);
  auto tok_ = tokenizers::HFTokenizer::FromBlobJSON(blob);
  tok = tok_.release();
}

encoding Tokenizer::Encode(const std::string &text) {
  std::vector<std::string> texts = {text};
  return EncodeBatch(texts)[0];
}

std::vector<encoding>
Tokenizer::EncodeBatch(const std::vector<std::string> &texts) {
  auto results = tok->EncodeBatch(texts, true);
  int max_size = 0;
  for (auto enc : results) {
    for (size_t pos = 0; pos < enc.attention_mask.size(); pos++) {
      if (enc.attention_mask[pos] == 0) {
        if (pos > max_size)
          max_size = pos;
        break;
      }
    }
  }

  for (size_t i = 0; i < results.size(); i++) {
    results[i].attention_mask.resize(max_size);
    results[i].ids.resize(max_size);
  }
  return results;
}

std::string Tokenizer::Decode(const tokens &ids) {
  return tok->Decode(ids, true);
}

tokenizers::HFTokenizer *Tokenizer::GetFastTokenizer() { return tok; }

} // namespace embeddings
