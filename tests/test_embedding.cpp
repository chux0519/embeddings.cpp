#include "bert.h"
#include "utils.h"
#include <cstddef>
#include <iostream>

using namespace embeddings;

void TestEmbedding() {
  auto model = Embedding("models/tokenizer.json",
                         "models/text2vec-base-multilingual.fp16.gguf");
  std::vector<std::string> prompts = {
      "如何更换花呗绑定银行卡", "How to replace the Huabei bundled bank card"};
  auto res = model.BatchEncode(prompts);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

int main() { TestEmbedding(); }