import os
import sys
from transformers import AutoTokenizer

try:
    from embeddings_cpp import create_embedding, PoolingMethod
except ImportError:
    print("Error: Could not import the C++ bindings.")
    print("Please make sure you have compiled the project and the output is in the python path.")
    sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def run_test(repo_name, model_file, prompts, normalize=True, pooling_method=PoolingMethod.MEAN):
    print(f"\n--- Testing: {model_file} ---")
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        print("Model file not found. Skipping.")
        return

    try:
        model = create_embedding(model_path)
        tokens = model.batch_tokenize(prompts, add_special_tokens=True)
        for prompt, token in zip(prompts, tokens):
            print(f"Prompt: '{prompt}'")
            print(f"Token IDs: {token.ids}")
            print(f"Attention Mask: {token.attention_mask}")
            print(f"No Pad Length: {token.no_pad_len}")
    
        tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        for prompt, input_ids in zip(prompts, inputs['input_ids']):
            print(f"Prompt: '{prompt}'")
            print(f"Input IDs: {input_ids.tolist()}")
            print(f"Attention Mask: {inputs['attention_mask'].tolist()}")
        

    except Exception as e:
        print(f"Test FAILED for {model_file}: {e}")

if __name__ == "__main__":

    prompts = [
        "你好，今天天气怎么样？",
        "What's the weather like today?",
        "The quick brown fox jumps over the lazy dog.",
        "机器学习是人工智能的一个重要分支。",
    ]
    run_test("Snowflake/snowflake-arctic-embed-m-v2.0","snowflake-arctic-embed-m-v2.0.fp16.gguf", prompts, True, PoolingMethod.CLS)