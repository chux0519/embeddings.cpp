#include "../bert.h"
#include "../utils.h"
#include <cstddef>
#include <iostream>

using namespace embeddings;

void TestEmbedding(const std::string &tokenizer_file,
                   const std::string &model_file, bool normalize,
                   int pooling_method) {
  auto model = Embedding(tokenizer_file, model_file);
  std::vector<std::string> prompts = {"你好，今天天气怎么样？"};
  auto res = model.BatchEncode(prompts, normalize, pooling_method);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

int main() {
  TestEmbedding("models/text2vec-base-multilingual.tokenizer.json",
                "models/text2vec-base-multilingual.fp16.gguf", true,
                POOLING_METHOD_MEAN);
  TestEmbedding("models/bge-base-zh-v1.5.tokenizer.json",
                "models/bge-base-zh-v1.5.fp16.gguf", true, POOLING_METHOD_CLS);
  TestEmbedding("models/bge-m3.tokenizer.json", "models/bge-m3.fp16.gguf", true,
                POOLING_METHOD_CLS);
}