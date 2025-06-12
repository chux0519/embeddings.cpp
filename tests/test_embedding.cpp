#include <cstddef>
#include <iostream>

#include "base_model.h"
#include "bert.h"
#include "gte.h"
#include "jina_bert.h"
#include "utils.h"

using namespace embeddings;

void TestEmbedding(const std::string &tokenizer_file,
                   const std::string &model_file, bool normalize,
                   PoolingMethod pooling_method) {
  auto model = Embedding(tokenizer_file, model_file);
  std::vector<std::string> prompts = {"你好，今天天气怎么样？"};
  auto res = model.BatchEncode(prompts, normalize, pooling_method);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

void TestJinaEmbedding(const std::string &tokenizer_file,
                       const std::string &model_file, bool normalize,
                       PoolingMethod pooling_method) {
  auto model = JinaEmbedding(tokenizer_file, model_file);
  std::vector<std::string> prompts = {"A blue cat"};
  auto res = model.BatchEncode(prompts, normalize, pooling_method);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

void TestGteEmbedding(const std::string &tokenizer_file,
                      const std::string &model_file, bool normalize,
                      PoolingMethod pooling_method) {
  auto model = GteEmbedding(tokenizer_file, model_file);
  std::vector<std::string> prompts = {"A blue cat"};
  auto res = model.BatchEncode(prompts, normalize, pooling_method);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

int main() {
  // TestEmbedding("models/text2vec-base-multilingual.tokenizer.json",
  //               "models/text2vec-base-multilingual.fp16.gguf", true,
  //               POOLING_METHOD_MEAN);
  // TestEmbedding("models/bge-base-zh-v1.5.tokenizer.json",
  //               "models/bge-base-zh-v1.5.fp16.gguf", true,
  //               POOLING_METHOD_CLS);
  // TestEmbedding("models/bge-m3.tokenizer.json", "models/bge-m3.fp16.gguf",
  // true,
  //               POOLING_METHOD_CLS);
  TestGteEmbedding("models/snowflake-arctic-embed-m-v2.0.tokenizer.json",
                   "models/snowflake-arctic-embed-m-v2.0.fp16.gguf", true,
                   PoolingMethod::CLS);
}