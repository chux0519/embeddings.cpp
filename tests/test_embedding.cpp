#include "bert.h"
#include "utils.h"
#include <cstddef>
#include <iostream>

using namespace embeddings;

void TestEmbedding(const std::string &tokenizer_file,
                   const std::string &model_file) {
  auto model = Embedding(tokenizer_file, model_file);
  std::vector<std::string> prompts = {"你好，今天天气怎么样？",
                                      "What's the weather like today?"};
  auto res = model.BatchEncode(prompts);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

int main() {
  TestEmbedding("models/bge-base-zh-v1.5.tokenizer.json",
                "models/bge-base-zh-v1.5.fp16.gguf");

  TestEmbedding("models/text2vec-base-multilingual.tokenizer.json",
                "models/text2vec-base-multilingual.fp16.gguf");
}