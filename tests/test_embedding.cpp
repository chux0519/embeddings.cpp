#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>

#include "base_model.h"
#include "bert.h"
#include "gte.h"
#include "jina_bert.h"
#include "tokenizer.h"
#include "utils.h"
#include "factory.h"

using namespace embeddings;

void TestEmbedding(const std::string &model_file, bool normalize,
                   PoolingMethod pooling_method) {
  auto model = create_embedding(model_file);
  std::vector<std::string> prompts = {"A blue cat.", "A blue cat"};
  auto res = model->batch_encode(prompts, normalize, pooling_method);
  for (size_t i = 0; i < prompts.size(); i++) {
    std::cout << "prompt: " << prompts[i] << std::endl;
    print_tensors(res[i]);
  }
}

void TestFullyMaskedGteBatch(const std::string &model_file) {
  auto model = create_model_from_gguf(model_file);
  model->Load();

  TokenizedInput empty_a{{0, 0}, {0, 0}, 0};
  TokenizedInput empty_b{{0, 0}, {0, 0}, 0};
  auto res = model->BatchForward({empty_a, empty_b}, true, PoolingMethod::CLS);

  assert(res.size() == 2);
  for (const auto &embedding : res) {
    assert(embedding.size() == 768);
    for (float value : embedding) {
      assert(std::isfinite(value));
      assert(std::abs(value) < 1e-6f);
    }
  }
}

int main() {
  TestEmbedding("models/snowflake-arctic-embed-m-v2.0.fp16.gguf", true,
                   PoolingMethod::CLS);
  TestFullyMaskedGteBatch("models/snowflake-arctic-embed-m-v2.0.fp16.gguf");
}
