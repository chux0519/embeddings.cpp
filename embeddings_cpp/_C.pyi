"""
A unified embedding library for various models.
"""
from __future__ import annotations
import typing
__all__ = ['CLS', 'Embedding', 'MEAN', 'PoolingMethod', 'create_embedding']
class Embedding:
    def batch_encode(self, texts: list[str], normalize: bool = True, pooling_method: PoolingMethod = ...) -> list[list[float]]:
        """
        Encodes a batch of strings into a list of float vectors.
        """
    def encode(self, text: str, normalize: bool = True, pooling_method: PoolingMethod = ...) -> list[float]:
        """
        Encodes a single string into a vector of floats.
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
def create_embedding(gguf_path: str) -> Embedding:
    """
    Creates a ready-to-use embedding model from a GGUF file.
    """
CLS: PoolingMethod  # value = <PoolingMethod.CLS: 1>
MEAN: PoolingMethod  # value = <PoolingMethod.MEAN: 0>
