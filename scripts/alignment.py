import os
import sys
import csv
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer

try:
    from embeddings_cpp import create_embedding, PoolingMethod
except ImportError:
    print("Error: Could not import the C++ bindings.")
    print("Please make sure you have compiled the project and the output is in the python path.")
    sys.exit(1)

# Configuration
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR.parent / 'models'
MODELS_TXT = SCRIPT_DIR / 'models.txt'
PROMPTS_TXT = SCRIPT_DIR / 'prompts.txt'
OUTPUT_DIR = SCRIPT_DIR / 'output'

# Default test prompts if prompts.txt doesn't exist
DEFAULT_PROMPTS = [
    "你好，今天天气怎么样？",
    "What's the weather like today?",
    "The quick brown fox jumps over the lazy dog.",
    "机器学习是人工智能的一个重要分支。"
]

# Model configurations - you may need to adjust pooling methods for specific models
MODEL_CONFIGS = {
    'BAAI/bge-m3': {'pooling': PoolingMethod.CLS},
    'BAAI/bge-base-zh-v1.5': {'pooling': PoolingMethod.CLS},
    'shibing624/text2vec-base-multilingual': {'pooling': PoolingMethod.MEAN},
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': {'pooling': PoolingMethod.MEAN},
    'Snowflake/snowflake-arctic-embed-m-v2.0': {'pooling': PoolingMethod.CLS},
}

def load_models_list() -> List[str]:
    """Load model names from models.txt"""
    if not MODELS_TXT.exists():
        print(f"Warning: {MODELS_TXT} not found. Using default model list.")
        return list(MODEL_CONFIGS.keys())
    
    with open(MODELS_TXT, 'r', encoding='utf-8') as f:
        models = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Loaded {len(models)} models from {MODELS_TXT}")
    return models

def load_test_prompts() -> List[str]:
    """Load test prompts from prompts.txt or use defaults"""
    if PROMPTS_TXT.exists():
        with open(PROMPTS_TXT, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(prompts)} prompts from {PROMPTS_TXT}")
        return prompts
    else:
        print(f"Using default prompts (create {PROMPTS_TXT} to customize)")
        return DEFAULT_PROMPTS

def get_gguf_filename(repo_name: str) -> str:
    """Generate GGUF filename from repository name"""
    # Convert repo name to filename format
    model_name = repo_name.split('/')[-1]  # Get model name after '/'
    return f"{model_name}.fp16.gguf"

def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling to hidden states"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cls_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply CLS pooling (use first token)"""
    return hidden_state[:, 0]

def run_embeddings_cpp(repo_name: str, prompts: List[str], normalize: bool = True) -> Optional[np.ndarray]:
    """Run embeddings using embeddings.cpp"""
    gguf_file = get_gguf_filename(repo_name)
    model_path = MODELS_DIR / gguf_file
    
    if not model_path.exists():
        print(f"Warning: GGUF file {gguf_file} not found for {repo_name}")
        return None
    
    try:
        print(f"Loading C++ model: {gguf_file}")
        model = create_embedding(str(model_path))
        
        # Get pooling method for this model
        pooling_method = MODEL_CONFIGS.get(repo_name, {}).get('pooling', PoolingMethod.MEAN)
        
        embeddings = model.batch_encode(prompts, normalize, pooling_method)
        return np.array(embeddings)
        
    except Exception as e:
        print(f"Error running C++ embeddings for {repo_name}: {e}")
        return None

def run_transformers(repo_name: str, prompts: List[str], normalize: bool = True) -> Optional[np.ndarray]:
    """Run embeddings using transformers library"""
    try:
        print(f"Loading transformers model: {repo_name}")
        tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo_name, trust_remote_code=True, add_pooling_layer=False, use_memory_efficient_attention=False)
        model.eval()
        
        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using device: {device}")

        # Tokenize inputs - use max_length=8192 for better compatibility
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            # Get pooling method for this model
            pooling_method = MODEL_CONFIGS.get(repo_name, {}).get('pooling', PoolingMethod.MEAN)
            
            if pooling_method == PoolingMethod.CLS:
                # For CLS pooling, use the approach from your reference code
                # This gets the CLS token directly from the model output
                embeddings = model(**inputs)[0][:, 0]
            else:
                # For mean pooling, get last hidden state and apply mean pooling
                outputs = model(**inputs)
                hidden_state = outputs.last_hidden_state
                embeddings = mean_pooling(hidden_state, inputs['attention_mask'])
            
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
            
    except Exception as e:
        print(f"Error running transformers for {repo_name}: {e}")
        return None

def calculate_mse(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two embedding sets"""
    return np.mean((embeddings1 - embeddings2) ** 2)

def calculate_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> Tuple[float, dict]:
    """Calculate average cosine similarity between two embedding sets and return debug info"""
    # Calculate cosine similarity for each pair of embeddings
    similarities = []
    debug_info = {
        'cpp_first_10': [],
        'transformers_first_10': [],
        'cpp_last_10': [],
        'transformers_last_10': [],
        'individual_similarities': []
    }
    
    for i in range(len(embeddings1)):
        # Get original vectors before normalization for debugging
        vec1_orig = embeddings1[i]
        vec2_orig = embeddings2[i]
        
        # Store first 10 and last 10 values for debugging
        if i == 0:  # Only store for first prompt to avoid too much data
            debug_info['cpp_first_10'] = vec1_orig[:10].tolist()
            debug_info['cpp_last_10'] = vec1_orig[-10:].tolist()
            debug_info['transformers_first_10'] = vec2_orig[:10].tolist()
            debug_info['transformers_last_10'] = vec2_orig[-10:].tolist()
        
        # Normalize vectors
        vec1_norm = np.linalg.norm(vec1_orig)
        vec2_norm = np.linalg.norm(vec2_orig)
        
        if vec1_norm == 0 or vec2_norm == 0:
            similarity = 0.0
        else:
            vec1 = vec1_orig / vec1_norm
            vec2 = vec2_orig / vec2_norm
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2)
        
        similarities.append(similarity)
        debug_info['individual_similarities'].append(similarity)
    
    avg_similarity = np.mean(similarities)
    return avg_similarity, debug_info

def compare_models(models: List[str], prompts: List[str]) -> List[dict]:
    """Compare embeddings between C++ and transformers implementations"""
    results = []
    
    for repo_name in models:
        print(f"\n{'='*60}")
        print(f"Testing model: {repo_name}")
        print(f"{'='*60}")
        
        # Run both implementations
        cpp_embeddings = run_embeddings_cpp(repo_name, prompts)
        transformers_embeddings = run_transformers(repo_name, prompts)
        
        if cpp_embeddings is not None and transformers_embeddings is not None:
            mse = calculate_mse(cpp_embeddings, transformers_embeddings)
            cosine_sim, debug_info = calculate_cosine_similarity(cpp_embeddings, transformers_embeddings)
            gguf_filename = get_gguf_filename(repo_name)
            
            result = {
                'repo_name': repo_name,
                'gguf_file_name': gguf_filename,
                'mse': mse,
                'cosine_similarity': cosine_sim,
                'embedding_dim': cpp_embeddings.shape[1],
                'num_prompts': len(prompts),
                'status': 'success',
                # Debug information
                'cpp_first_10': str(debug_info['cpp_first_10']),
                'transformers_first_10': str(debug_info['transformers_first_10']),
                'cpp_last_10': str(debug_info['cpp_last_10']),
                'transformers_last_10': str(debug_info['transformers_last_10']),
                'individual_similarities': str(debug_info['individual_similarities']),
                'cpp_norm': float(np.linalg.norm(cpp_embeddings[0])),
                'transformers_norm': float(np.linalg.norm(transformers_embeddings[0]))
            }
            
            print(f"MSE: {mse:.2e}")
            print(f"Cosine Similarity: {cosine_sim:.6f}")
            print(f"C++ embedding norm: {result['cpp_norm']:.6f}")
            print(f"Transformers embedding norm: {result['transformers_norm']:.6f}")
            print(f"Individual similarities: {debug_info['individual_similarities']}")
            print(f"Embedding dimension: {cpp_embeddings.shape[1]}")
            
        else:
            result = {
                'repo_name': repo_name,
                'gguf_file_name': get_gguf_filename(repo_name),
                'mse': float('nan'),
                'cosine_similarity': float('nan'),
                'embedding_dim': 'N/A',
                'num_prompts': len(prompts),
                'status': 'failed',
                'cpp_first_10': 'N/A',
                'transformers_first_10': 'N/A',
                'cpp_last_10': 'N/A',
                'transformers_last_10': 'N/A',
                'individual_similarities': 'N/A',
                'cpp_norm': 'N/A',
                'transformers_norm': 'N/A'
            }
            print("Comparison failed - missing embeddings")
        
        results.append(result)
    
    return results

def save_results_csv(results: List[dict], output_path: Path):
    """Save results to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

def save_debug_csv(results: List[dict], output_path: Path):
    """Save detailed debug information to separate CSV file"""
    debug_data = []
    for result in results:
        if result['status'] == 'success':
            # Parse the stored lists back from strings
            try:
                cpp_first_10 = eval(result['cpp_first_10'])
                transformers_first_10 = eval(result['transformers_first_10'])
                cpp_last_10 = eval(result['cpp_last_10'])
                transformers_last_10 = eval(result['transformers_last_10'])
                
                # Create debug row with individual values
                debug_row = {'repo_name': result['repo_name']}
                
                # Add first 10 values
                for i in range(10):
                    debug_row[f'cpp_first_{i+1}'] = cpp_first_10[i] if i < len(cpp_first_10) else 'N/A'
                    debug_row[f'transformers_first_{i+1}'] = transformers_first_10[i] if i < len(transformers_first_10) else 'N/A'
                
                # Add last 10 values
                for i in range(10):
                    debug_row[f'cpp_last_{i+1}'] = cpp_last_10[i] if i < len(cpp_last_10) else 'N/A'
                    debug_row[f'transformers_last_{i+1}'] = transformers_last_10[i] if i < len(transformers_last_10) else 'N/A'
                
                # Add norms and similarity info
                debug_row['cpp_norm'] = result['cpp_norm']
                debug_row['transformers_norm'] = result['transformers_norm']
                debug_row['cosine_similarity'] = result['cosine_similarity']
                debug_row['mse'] = result['mse']
                
                debug_data.append(debug_row)
            except:
                print(f"Warning: Could not parse debug data for {result['repo_name']}")
    
    if debug_data:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=debug_data[0].keys())
            writer.writeheader()
            writer.writerows(debug_data)

def save_results_markdown(results: List[dict], output_path: Path):
    """Save results to Markdown file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Embeddings Alignment Results\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## Results Summary\n\n")
        f.write("| Repository Name | GGUF File | MSE | Cosine Similarity | Status |\n")
        f.write("|----------------|-----------|-----|-------------------|--------|\n")
        
        for result in results:
            repo = result['repo_name']
            gguf = result['gguf_file_name']
            mse = f"{result['mse']:.2e}" if not np.isnan(result['mse']) else "N/A"
            cosine_sim = f"{result['cosine_similarity']:.6f}" if not np.isnan(result['cosine_similarity']) else "N/A"
            status = result['status']
            f.write(f"| {repo} | {gguf} | {mse} | {cosine_sim} | {status} |\n")
        
        # Detailed results
        f.write("\n## Detailed Results\n\n")
        for result in results:
            f.write(f"### {result['repo_name']}\n\n")
            f.write(f"- **GGUF File**: `{result['gguf_file_name']}`\n")
            mse_str = f"{result['mse']:.2e}" if not np.isnan(result['mse']) else 'N/A'
            cosine_sim_str = f"{result['cosine_similarity']:.6f}" if not np.isnan(result['cosine_similarity']) else 'N/A'
            f.write(f"- **MSE**: {mse_str}\n")
            f.write(f"- **Cosine Similarity**: {cosine_sim_str}\n")
            f.write(f"- **Embedding Dimension**: {result['embedding_dim']}\n")
            f.write(f"- **Number of Test Prompts**: {result['num_prompts']}\n")
            f.write(f"- **Status**: {result['status']}\n")
            
            # Add debug information if available
            if result['status'] == 'success':
                f.write(f"- **C++ Embedding Norm**: {result['cpp_norm']:.6f}\n")
                f.write(f"- **Transformers Embedding Norm**: {result['transformers_norm']:.6f}\n")
                f.write(f"- **Individual Similarities**: {result['individual_similarities']}\n")
            
            f.write("\n")

def main():
    """Main function to run the alignment comparison"""
    print("Embeddings Alignment Comparison Tool")
    print("=" * 50)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load models and prompts
    models = load_models_list()
    prompts = load_test_prompts()
    
    print(f"\nWill test {len(models)} models with {len(prompts)} prompts")
    
    # Run comparisons
    results = compare_models(models, prompts)
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = OUTPUT_DIR / f"alignment_results_{timestamp}.csv"
    debug_csv_path = OUTPUT_DIR / f"alignment_debug_{timestamp}.csv"
    md_path = OUTPUT_DIR / f"alignment_results_{timestamp}.md"
    
    # Save results
    save_results_csv(results, csv_path)
    save_debug_csv(results, debug_csv_path)
    save_results_markdown(results, md_path)
    
    print(f"\n{'='*60}")
    print("Results saved:")
    print(f"CSV: {csv_path}")
    print(f"Debug CSV: {debug_csv_path}")
    print(f"Markdown: {md_path}")
    print(f"{'='*60}")
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSummary: {successful}/{len(results)} models tested successfully")

if __name__ == "__main__":
    main()
    