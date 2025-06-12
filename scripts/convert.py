import sys
import torch
import os

from gguf import GGUFWriter, GGMLQuantizationType
from transformers import AutoModel, AutoTokenizer

class BaseConversionConfig:
    """
    Base class for model-specific conversion configurations.
    """
    # The architecture name to be written into the GGUF file.
    # This should match what your C++ factory expects.
    ARCHITECTURE = "base"

    def get_param_keys(self):
        """
        Returns a list of hyperparameter keys to be read from the
        Hugging Face model config and written to the GGUF file.
        """
        return [
            'vocab_size', 'hidden_size', 'num_hidden_layers',
            'num_attention_heads', 'layer_norm_eps'
        ]

    def get_tensor_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        """
        Determines the dtype for a given tensor.
        By default, all tensors use the default float type.
        Subclasses should override this for model-specific rules.
        """
        return default_dtype

    def write_extra_hparams(self, gguf_writer, hf_config):
        """
        Writes any additional, model-specific hyperparameters to the GGUF file.
        This is a hook for parameters that don't fit the common keys.
        """
        pass # Default implementation does nothing

class BertConversionConfig(BaseConversionConfig):
    ARCHITECTURE = "BertModel"

    def get_param_keys(self):
        keys = super().get_param_keys()
        keys.extend([
            'intermediate_size',
            'max_position_embeddings',
            'type_vocab_size',
            'pad_token_id'
        ])
        return keys

    def get_tensor_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        # For classic BERT, LayerNorm and bias tensors are usually f32
        if 'LayerNorm' in name or 'bias' in name:
            return torch.float32
        return default_dtype
    
class GteConversionConfig(BaseConversionConfig):
    ARCHITECTURE = "GteModel"

    def get_param_keys(self):
        keys = super().get_param_keys()
        keys.extend(['max_position_embeddings', 'intermediate_size', 'rope_theta'])
        return keys

    def get_tensor_dtype(self, name: str, default_dtype: torch.dtype) -> torch.dtype:
        # GTE models (like Snowflake) have specific layers that benefit from f32
        if 'attn_ln' in name or 'mlp_ln' in name  or 'bias' in name or 'LayerNorm' in name:
            return torch.float32
        return default_dtype


MODEL_CONFIG_MAP = {
    "Snowflake/snowflake-arctic-embed-m-v2.0": GteConversionConfig,
}

def get_config_for_repo(repo_id: str, hf_config) -> BaseConversionConfig:
    """
    Selects the appropriate conversion configuration based on the repo ID or
    the model's architecture field from its config.
    """
    if repo_id in MODEL_CONFIG_MAP:
        print(f"Using custom config for '{repo_id}'.")
        return MODEL_CONFIG_MAP[repo_id]()

    if hasattr(hf_config, 'architectures'):
        for arch in hf_config.architectures:
            arch_lower = arch.lower()
            if 'GteModel' in arch_lower: # Example
                return GteConversionConfig()

    # Fallback to a default or raise an error
    print(f"Warning: Could not determine a specific model type for '{repo_id}'. Falling back to BERT config.")
    return BertConversionConfig()

def convert_hf(repo_id, output_path, float_type='f16'):
    # convert to ggml quantization type
    if float_type not in ['f16', 'f32']:
        print(f'Float type must be f16 or f32, got: {float_type}')
        sys.exit(1)
    else:
        qtype = GGMLQuantizationType[float_type.upper()]
        dtype0 = {'f16': torch.float16, 'f32': torch.float32}[float_type]

    # load tokenizer and model
    print(f"Loading tokenizer and model for '{repo_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id, add_pooling_layer=False, trust_remote_code=True)
    print("Model and tokenizer loaded.")
    
    hf_config = model.config
    model_config = get_config_for_repo(repo_id, hf_config)

    # Start to write GGUF file
    gguf_writer = GGUFWriter(output_path, model_config.ARCHITECTURE)
    gguf_writer.add_name(repo_id)
    gguf_writer.add_description(model_config.ARCHITECTURE + ' model for embeddings.cpp')
    gguf_writer.add_file_type(qtype)
    # --- Use the config object to write hparams ---
    print("\n--- Model Config ---")
    for key in model_config.get_param_keys():
        if hasattr(hf_config, key) and getattr(hf_config, key) is not None:
            # We need a small helper to decide which gguf_writer method to call
            value = getattr(hf_config, key)

            if key =="rope_theta":
                gguf_writer.add_float32(key, value)
            else:
                if isinstance(value, bool):
                    gguf_writer.add_bool(key, value)
                elif isinstance(value, int):
                    gguf_writer.add_uint32(key, value) # Assuming uint32 is fine for most int params
                elif isinstance(value, float):
                    gguf_writer.add_float32(key, value)
                elif isinstance(value, str):
                    gguf_writer.add_string(key, value)
            print(f'{key:<24s} = {value}')
    
    # --- Use the config hook for extra hparams ---
    model_config.write_extra_hparams(gguf_writer, hf_config)

    # 1. Define standard GGUF keys for tokenizer
    KEY_TOKENIZER_FILE = "tokenizer.ggml.file"

    # 2. Get the tokenizer JSON content as a string
    # This accesses the underlying "fast" tokenizer and gets its JSON representation.
    tokenizer_json_str = tokenizer._tokenizer.to_str()
    
    # 3. Add the JSON string to the GGUF file
    print(f"Embedding tokenizer JSON ({len(tokenizer_json_str) / 1024:.2f} KB)...")
    gguf_writer.add_string(KEY_TOKENIZER_FILE, tokenizer_json_str)

    # write tensors
    print('\n--- Tensors ---')
    for name, data in model.state_dict().items():
        # Get the correct dtype from our config object
        target_dtype = model_config.get_tensor_dtype(name, dtype0)
        
        shape_str = str(list(data.shape))
        print(f'{name:64s} = {shape_str:16s} {data.dtype} → {target_dtype}')

        data = data.to(target_dtype)
        gguf_writer.add_tensor(name, data.numpy())

    # execute and close writer
    print("\nWriting GGUF file...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    # print success
    print(f'\nSuccessfully converted with {model_config.__class__.__name__}.')
    print(f'GGUF model written to {output_path}')

if __name__ == '__main__':
    # primary usage
    if len(sys.argv) < 3:
        print('Usage: convert.py repo_id output_path [float-type=f16,f32]\n')
        sys.exit(1)

    # output in the same directory as the model
    repo_id = sys.argv[1]
    output_path = sys.argv[2]

    # get float type
    if len(sys.argv) > 3:
        kwargs = {'float_type': sys.argv[3].lower()}
    else:
        kwargs = {}

    # convert to ggml
    convert_hf(repo_id, output_path, **kwargs)
