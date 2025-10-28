"""
embeddings.cpp Python bindings
"""
from __future__ import annotations
import typing
__all__: list[str] = ['Encoding', 'Tokenizer', 'Tokens', 'TokensBatch']
class Encoding:
    def __init__(self) -> None:
        ...
    @property
    def attention_mask(self) -> list[int]:
        """
        Attention mask for the encoding.
        """
    @attention_mask.setter
    def attention_mask(self, arg0: list[int]) -> None:
        ...
    @property
    def ids(self) -> list[int]:
        """
        Token IDs of the encoding.
        """
    @ids.setter
    def ids(self, arg0: list[int]) -> None:
        ...
class Tokenizer:
    def __init__(self, path: str) -> None:
        ...
    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes tokens into a string.
        """
    def encode(self, text: str, add_special_tokens: bool = True) -> Encoding:
        """
        Encodes a single string into tokens.
        """
    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[Encoding]:
        """
        Encodes a batch of strings into tokens.
        """
class Tokens:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self: list[int]) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self: list[int], x: int) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self: list[int], arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self: list[int], arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self: list[int], arg0: list[int]) -> bool:
        ...
    @typing.overload
    def __getitem__(self: list[int], s: slice) -> list[int]:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self: list[int], arg0: int) -> int:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[int]) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self: list[int]) -> typing.Iterator[int]:
        ...
    def __len__(self: list[int]) -> int:
        ...
    def __ne__(self: list[int], arg0: list[int]) -> bool:
        ...
    def __repr__(self: list[int]) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self: list[int], arg0: int, arg1: int) -> None:
        ...
    @typing.overload
    def __setitem__(self: list[int], arg0: slice, arg1: list[int]) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self: list[int], x: int) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self: list[int]) -> None:
        """
        Clear the contents
        """
    def count(self: list[int], x: int) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self: list[int], L: list[int]) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: list[int], L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self: list[int], i: int, x: int) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self: list[int]) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self: list[int], i: int) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self: list[int], x: int) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class TokensBatch:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self: list[list[int]]) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self: list[list[int]], x: list[int]) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self: list[list[int]], arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self: list[list[int]], arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self: list[list[int]], arg0: list[list[int]]) -> bool:
        ...
    @typing.overload
    def __getitem__(self: list[list[int]], s: slice) -> list[list[int]]:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self: list[list[int]], arg0: int) -> list[int]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: list[list[int]]) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self: list[list[int]]) -> typing.Iterator[list[int]]:
        ...
    def __len__(self: list[list[int]]) -> int:
        ...
    def __ne__(self: list[list[int]], arg0: list[list[int]]) -> bool:
        ...
    @typing.overload
    def __setitem__(self: list[list[int]], arg0: int, arg1: list[int]) -> None:
        ...
    @typing.overload
    def __setitem__(self: list[list[int]], arg0: slice, arg1: list[list[int]]) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self: list[list[int]], x: list[int]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self: list[list[int]]) -> None:
        """
        Clear the contents
        """
    def count(self: list[list[int]], x: list[int]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self: list[list[int]], L: list[list[int]]) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: list[list[int]], L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self: list[list[int]], i: int, x: list[int]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self: list[list[int]]) -> list[int]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self: list[list[int]], i: int) -> list[int]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self: list[list[int]], x: list[int]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
