#include "tokenizer.h"
#include "utils.h"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace embeddings;

void TestTokenizer(const std::string &tokenizer_file) {
  auto tok = Tokenizer(tokenizer_file);

  for (auto atk : tok.GetFastTokenizer()->GetAddedTokens()) {
    std::cout << atk.content << ": " << atk.id << std::endl;
  }
  // Check #1. Encode and Decode
  std::vector<std::string> prompts = {"你好，今天天气怎么样？",
                                      "What's the weather like today?"};
  auto res = tok.EncodeBatch(prompts);
  for (size_t i = 0; i < res.size(); ++i) {
    auto encoding = res[i];
    std::string decoded_prompt = tok.Decode(encoding.ids, false);
    std::cout << "prompt=\"" << prompts[i] << "\"" << std::endl;
    std::cout << "ids: ";
    print_encode_result(encoding.ids);
    std::cout << "attention mask: ";
    print_encode_result(encoding.attention_mask);
    std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
  }

  // Check #2. IdToToken and TokenToId
  std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 4, 33, 34, 130, 131};
  for (auto id : ids_to_test) {
    auto token = tok.GetFastTokenizer()->IdToToken(id);
    auto id_new = tok.GetFastTokenizer()->TokenToId(token);
    std::cout << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new
              << std::endl;
    assert(id == id_new);
  }

  // Check #3. GetVocabSize
  auto vocab_size = tok.GetFastTokenizer()->GetVocabSize();
  std::cout << "vocab_size=" << vocab_size << std::endl;

  std::cout << std::endl;
}

int main() {
  TestTokenizer("models/bge-base-zh-v1.5.tokenizer.json");
  TestTokenizer("models/text2vec-base-multilingual.tokenizer.json");
  return 0;
}