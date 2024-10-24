#include "tokenizer.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace embeddings {

Tokenizer::Tokenizer(const std::string &path) {
  auto blob = load_bytes_from_file(path);
  auto tok_ = tokenizers::HFTokenizer::FromBlobJSON(blob);
  tok = tok_.release();
}

encoding Tokenizer::Encode(const std::string &text, bool add_special_tokens) {
  std::vector<std::string> texts = {text};
  return EncodeBatch(texts, add_special_tokens)[0];
}

std::vector<encoding>
Tokenizer::EncodeBatch(const std::vector<std::string> &texts,
                       bool add_special_tokens) {
  auto results = tok->EncodeBatch(texts, add_special_tokens);
  if (results.size() <= 1) {
    return results;
  }

  auto size0 = results[0].ids.size();
  bool is_same_size = true;
  for (size_t i = 1; i < results.size(); ++i) {
    if (results[i].ids.size() != size0) {
      is_same_size = false;
      break;
    }
  }

  if (is_same_size) {
    // some model always returns full length, like text2vec-base-multilingual
    // shrink to the max size in the batch
    size_t max_size = 0;
    for (auto enc : results) {
      for (size_t pos = 0; pos < enc.attention_mask.size(); pos++) {
        if (enc.attention_mask[pos] == 0) {
          if (pos > max_size)
            max_size = pos;
          break;
        }
      }
    }

    if (max_size > 0) {
      for (size_t i = 0; i < results.size(); i++) {
        results[i].attention_mask.resize(max_size);
        results[i].ids.resize(max_size);
      }
    }
  } else {
    // we should pad them to the same length using <PAD>, and set the attention
    // mask to 0 on the padded position
    size_t max_size = 0;
    for (auto enc : results) {
      if (enc.ids.size() > max_size) {
        max_size = enc.ids.size();
      }
    }

    for (size_t i = 0; i < results.size(); i++) {
      size_t cur_size = results[i].ids.size();
      int32_t pad_id = 0;
      auto added_tokens = tok->GetAddedTokens();
      for (size_t t = 0; t < added_tokens.size(); t++) {
        std::string lower_word = to_lowercase(added_tokens[t].content);
        if (lower_word.find("pad") != std::string::npos) {
          pad_id = added_tokens[t].id;
        }
      }

      if (cur_size < max_size) {
        results[i].attention_mask.resize(max_size);
        results[i].ids.resize(max_size);
        for (size_t j = cur_size; j < max_size; ++j) {
          results[i].attention_mask[j] = 0;
          results[i].ids[j] = pad_id;
        }
      }
    }
  }

  return results;
}

std::string Tokenizer::Decode(const tokens &ids, bool skip_special_tokens) {
  return tok->Decode(ids, skip_special_tokens);
}

tokenizers::HFTokenizer *Tokenizer::GetFastTokenizer() { return tok; }

} // namespace embeddings
