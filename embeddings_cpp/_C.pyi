"""
A unified embedding library for various models.
"""
from __future__ import annotations
import typing
__all__ = ['CLS', 'Embedding', 'MEAN', 'PoolingMethod', 'TokenizedInput', 'create_embedding']
class Embedding:
    def batch_encode(self, texts: list[str], normalize: bool = True, pooling_method: PoolingMethod = ...) -> list[list[float]]:
        """
        Encodes a batch of strings into a list of float vectors.
        """
    def batch_tokenize(self, texts: list[str], add_special_tokens: bool = True) -> list[TokenizedInput]:
        """
        Tokenizes a batch of strings into token IDs and attention masks.
        """
    def encode(self, text: str, normalize: bool = True, pooling_method: PoolingMethod = ...) -> list[float]:
        """
        Encodes a single string into a vector of floats.
        """
    def tokenize(self, text: str, add_special_tokens: bool = True) -> TokenizedInput:
        """
        Tokenizes a single string into token IDs and attention mask.
        """
class PoolingMethod:
    """
    Members:
    
      MEAN
    
      CLS
    """
    CLS: typing.ClassVar[PoolingMethod]  # value = <PoolingMethod.CLS: 1>
    MEAN: typing.ClassVar[PoolingMethod]  # value = <PoolingMethod.MEAN: 0>
    __members__: typing.ClassVar[dict[str, PoolingMethod]]  # value = {'MEAN': <PoolingMethod.MEAN: 0>, 'CLS': <PoolingMethod.CLS: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TokenizedInput:
    @property
    def attention_mask(self) -> list[int]:
        """
        Attention mask
        """
    @attention_mask.setter
    def attention_mask(self, arg0: list[int]) -> None:
        ...
    @property
    def ids(self) -> list[int]:
        """
        Token IDs
        """
    @ids.setter
    def ids(self, arg0: list[int]) -> None:
        ...
    @property
    def no_pad_len(self) -> int:
        """
        Length without padding
        """
    @no_pad_len.setter
    def no_pad_len(self, arg0: int) -> None:
        ...
def create_embedding(gguf_path: str) -> Embedding:
    """
    Creates a ready-to-use embedding model from a GGUF file.
    """
CLS: PoolingMethod  # value = <PoolingMethod.CLS: 1>
MEAN: PoolingMethod  # value = <PoolingMethod.MEAN: 0>
