from __future__ import annotations
from embeddings_cpp._C import Embedding
from embeddings_cpp._C import Encoding
from embeddings_cpp._C import Tokenizer
from embeddings_cpp._C import Tokens
from embeddings_cpp._C import TokensBatch
from . import _C
__all__ = ['Embedding', 'Encoding', 'POOLING_METHOD_CLS', 'POOLING_METHOD_MEAN', 'Tokenizer', 'Tokens', 'TokensBatch']
POOLING_METHOD_CLS: int = 1
POOLING_METHOD_MEAN: int = 0
__version__: str = '0.1.0'
