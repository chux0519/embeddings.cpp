#include "tokenizer.h"
#include "utils.h"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace embeddings;

void TestTokenizer(Tokenizer tok, bool print_vocab = false,
                   bool check_id_back = true) {
  // Check #1. Encode and Decode
  std::vector<std::string> prompts = {"如何更换花呗绑定银行卡",
                                      "What is the capital of Canada?"};
  auto res = tok.EncodeBatch(prompts);
  for (size_t i = 0; i < res.size(); ++i) {
    auto encoding = res[i];
    std::string decoded_prompt = tok.Decode(encoding.ids);
    std::cout << "prompt=\"" << prompts[i] << "\"" << std::endl;
    std::cout << "ids: ";
    print_encode_result(encoding.ids);
    std::cout << "attention mask: ";
    print_encode_result(encoding.attention_mask);
    std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
    assert(decoded_prompt == prompts[i]);
  }

  // Check #2. IdToToken and TokenToId
  std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 4, 33, 34, 130, 131, 250001};
  for (auto id : ids_to_test) {
    auto token = tok.GetFastTokenizer()->IdToToken(id);
    auto id_new = tok.GetFastTokenizer()->TokenToId(token);
    std::cout << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new
              << std::endl;
    if (check_id_back) {
      assert(id == id_new);
    }
  }

  // Check #3. GetVocabSize
  auto vocab_size = tok.GetFastTokenizer()->GetVocabSize();
  std::cout << "vocab_size=" << vocab_size << std::endl;

  std::cout << std::endl;
}

void HuggingFaceTokenizerExample() {
  std::cout << "Tokenizer: Huggingface" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  const std::string tokenizer_path = "models/tokenizer.json";

  auto tok = Tokenizer(tokenizer_path);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  std::cout << "Load time: " << duration << " ms" << std::endl;

  TestTokenizer(tok, false, true);
}

int main() {
  HuggingFaceTokenizerExample();
  return 0;
}