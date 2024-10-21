#include "bert.h"
#include "utils.h"

using namespace embeddings;

void TestEmbedding() {
  auto model = Embedding("models/tokenizer.json",
                         "models/text2vec-base-multilingual.fp16.gguf");
  auto text = "如何更换花呗绑定银行卡";
  auto res = model.Encode(text);
  print_tensors(res);
}

int main() { TestEmbedding(); }