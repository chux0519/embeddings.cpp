#include <cstddef>
#include <iostream>

#include "base_model.h"
#include "bert.h"
#include "gte.h"
#include "jina_bert.h"
#include "utils.h"
#include "factory.h"

using namespace embeddings;

void TestTokenizer(const std::string &model_file, std::vector<std::string> prompts) {
  auto model = create_embedding(model_file);
  auto res = model->batch_tokenize(prompts);
  for (size_t i = 0; i < res.size(); ++i) {
    auto encoding = res[i];
    std::cout << "prompt=\"" << prompts[i] << "\"" << std::endl;
    std::cout << "ids: ";
    print_encode_result(encoding.ids);
    std::cout << "attention mask: ";
    print_encode_result(encoding.attention_mask);
  }
}


int main() {
  TestTokenizer("models/snowflake-arctic-embed-m-v2.0.fp16.gguf", {"A cute cat.", "A cute cat"});
  return 0;
}
