import os
import sys
from typing import List

try:
    from embeddings_cpp import BertEmbedding, GteEmbedding, JinaEmbedding, PoolingMethod
except ImportError:
    print("Error: Could not import the C++ bindings.")
    print("Please make sure you have compiled the project and the output is in the python path.")
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def print_embedding(embedding: List[float]):
    """Prints a truncated version of the embedding vector."""
    dim = len(embedding)
    # 打印前3个和后3个元素
    preview = ", ".join(map(str, embedding[:3])) + " ... " + ", ".join(map(str, embedding[-3:]))
    print(f"    Embedding (dim={dim}): [{preview}]")

def test_bert_embedding(model_path: str, normalize: bool, pooling_method: PoolingMethod):
    print(f"\n--- Testing BertEmbedding with {os.path.basename(model_path)} ---")
    print(f"Normalize: {normalize}, Pooling: {pooling_method.name}")
    
    # 你的 C++ 代码似乎对中文和英文使用不同的 prompt，这里统一
    prompts = ["Hello, how is the weather today?", "你好，今天天气怎么样？"]
    
    model = BertEmbedding(model_path)
    embeddings = model.batch_encode(prompts, normalize=normalize, pooling_method=pooling_method)
    
    for i, prompt in enumerate(prompts):
        print(f"  Prompt: '{prompt}'")
        print_embedding(embeddings[i])

def test_jina_embedding(model_path: str, normalize: bool, pooling_method: PoolingMethod):
    print(f"\n--- Testing JinaEmbedding with {os.path.basename(model_path)} ---")
    print(f"Normalize: {normalize}, Pooling: {pooling_method.name}")

    prompts = ["A blue cat"]
    
    model = JinaEmbedding(model_path)
    embeddings = model.batch_encode(prompts, normalize=normalize, pooling_method=pooling_method)

    for i, prompt in enumerate(prompts):
        print(f"  Prompt: '{prompt}'")
        print_embedding(embeddings[i])

def test_gte_embedding(model_path: str, normalize: bool, pooling_method: PoolingMethod):
    print(f"\n--- Testing GteEmbedding with {os.path.basename(model_path)} ---")
    print(f"Normalize: {normalize}, Pooling: {pooling_method.name}")

    prompts = ["A blue cat"]
    
    model = GteEmbedding(model_path)
    embeddings = model.batch_encode(prompts, normalize=normalize, pooling_method=pooling_method)

    for i, prompt in enumerate(prompts):
        print(f"  Prompt: '{prompt}'")
        print_embedding(embeddings[i])

if __name__ == "__main__":
    # 检查模型文件是否存在，如果不存在则跳过
    def run_if_exists(test_func, model_name, *args):
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            test_func(model_path, *args)
        else:
            print(f"\n--- Skipping test for {model_name} (file not found) ---")

    # 复现 C++ 测试中的调用
    run_if_exists(test_bert_embedding, "paraphrase-multilingual-MiniLM-L12-v2.fp16.gguf", True, PoolingMethod.MEAN)
    run_if_exists(test_bert_embedding, "text2vec-base-multilingual.fp16.gguf", True, PoolingMethod.MEAN)
    run_if_exists(test_bert_embedding, "bge-base-zh-v1.5.fp16.gguf", True, PoolingMethod.CLS)
    run_if_exists(test_bert_embedding, "bge-m3.fp16.gguf", True, PoolingMethod.CLS)
    run_if_exists(test_gte_embedding, "snowflake-arctic-embed-m-v2.0.fp16.gguf", True, PoolingMethod.CLS)
    
    # Jina 的测试用例在原 C++ 测试中没有，但为了完整性，可以添加一个
    # run_if_exists(test_jina_embedding, "path/to/your/jina-model.gguf", True, PoolingMethod.MEAN)