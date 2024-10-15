#include "tokenizer.h"
#include "utils.h"

#include <cassert>
#include <chrono>
#include <iostream>

using namespace embeddings;

void TestTokenizer(Tokenizer tok, bool print_vocab = false,
                   bool check_id_back = true) {
  // Check #1. Encode and Decode
  //   std::string prompt = "What is the  capital of Canada?";
  std::string prompt = "今天天气怎么样？";
  std::vector<int> ids = tok.Encode(prompt);
  std::string decoded_prompt = tok.Decode(ids);
  PrintEncodeResult(ids);
  std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
  assert(decoded_prompt == prompt);

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

  const std::string tokenizer_path =
      "/home/yongsheng/.cache/huggingface/hub/"
      "models--shibing624--text2vec-base-multilingual/snapshots/"
      "6633dc49e554de7105458f8f2e96445c6598e9d1/tokenizer.json";

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