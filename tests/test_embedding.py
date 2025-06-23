import os
import sys

try:
    from embeddings_cpp import create_embedding, PoolingMethod
except ImportError:
    print("Error: Could not import the C++ bindings.")
    print("Please make sure you have compiled the project and the output is in the python path.")
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def run_test(model_name, prompt, normalize=True, pooling_method=PoolingMethod.MEAN):
    print(f"\n--- Testing: {model_name} ---")
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print("Model file not found. Skipping.")
        return

    try:
        # 统一的创建方式
        model = create_embedding(model_path)
        
        # 统一的使用方式
        embedding = model.encode(prompt, normalize, pooling_method)
        
        print(f"Prompt: '{prompt}'")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Embedding preview: {embedding[:4]}...")
        print("Test PASSED.")
    except Exception as e:
        print(f"Test FAILED for {model_name}: {e}")

if __name__ == "__main__":

    run_test("snowflake-arctic-embed-m-v2.0.fp16.gguf", "A cute cat.", True, PoolingMethod.CLS)