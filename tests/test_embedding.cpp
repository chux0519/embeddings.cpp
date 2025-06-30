#include <cstddef>
#include <iostream>

#include "base_model.h"
#include "bert.h"
#include "gte.h"
#include "jina_bert.h"
#include "utils.h"
#include "factory.h"

using namespace embeddings;

void TestEmbedding(const std::string &model_file, bool normalize,
                   PoolingMethod pooling_method) {
  auto model = create_embedding(model_file);
  std::vector<std::string> prompts = {"你好，今天天气怎么样？"};
  auto res = model->batch_encode(prompts, normalize, pooling_method);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}


int main() {
  TestEmbedding("models/snowflake-arctic-embed-m-v2.0.fp16.gguf", true,
                   PoolingMethod::CLS);
}