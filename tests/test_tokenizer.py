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
        # C++ tokenizer results
        model = create_embedding(model_path)
        cpp_tokens = model.batch_tokenize(prompts, add_special_tokens=True)
        
        # HuggingFace tokenizer results
        tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
        hf_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        
        # Compare results for each prompt
        print("\n=== Tokenization Comparison ===")
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: '{prompt}'")
            print("-" * 60)
            
            # C++ tokenizer output
            cpp_token = cpp_tokens[i]
            print("C++ Tokenizer:")
            print(f"  Token IDs: {cpp_token.ids}")
            print(f"  Attention Mask: {cpp_token.attention_mask}")
            print(f"  No Pad Length: {cpp_token.no_pad_len}")
            
            # HuggingFace tokenizer output
            hf_token_ids = hf_inputs['input_ids'][i].tolist()
            hf_attention_mask = hf_inputs['attention_mask'][i].tolist()
            print("HuggingFace Tokenizer:")
            print(f"  Token IDs: {hf_token_ids}")
            print(f"  Attention Mask: {hf_attention_mask}")
            
            # Comparison
            ids_match = cpp_token.ids == hf_token_ids
            mask_match = cpp_token.attention_mask == hf_attention_mask
            
            print("Comparison:")
            print(f"  Token IDs Match: {ids_match}")
            print(f"  Attention Mask Match: {mask_match}")
            
            if not ids_match:
                print(f"  Token IDs Difference:")
                print(f"    C++ length: {len(cpp_token.ids)}, HF length: {len(hf_token_ids)}")
                # Show differences in detail
                max_len = max(len(cpp_token.ids), len(hf_token_ids))
                for j in range(max_len):
                    cpp_id = cpp_token.ids[j] if j < len(cpp_token.ids) else "N/A"
                    hf_id = hf_token_ids[j] if j < len(hf_token_ids) else "N/A"
                    if cpp_id != hf_id:
                        print(f"    Position {j}: C++={cpp_id}, HF={hf_id}")
            
            if not mask_match:
                print(f"  Attention Mask Difference:")
                print(f"    C++ length: {len(cpp_token.attention_mask)}, HF length: {len(hf_attention_mask)}")
        
        # Summary
        all_ids_match = all(cpp_tokens[i].ids == hf_inputs['input_ids'][i].tolist() for i in range(len(prompts)))
        all_masks_match = all(cpp_tokens[i].attention_mask == hf_inputs['attention_mask'][i].tolist() for i in range(len(prompts)))
        
        print(f"\n=== Summary ===")
        print(f"All Token IDs Match: {all_ids_match}")
        print(f"All Attention Masks Match: {all_masks_match}")
        print(f"Overall Match: {all_ids_match and all_masks_match}")

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