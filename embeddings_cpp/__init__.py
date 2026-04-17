from __future__ import annotations

try:
    from embeddings_cpp._C import *
except ImportError:
    # Registry and packaging helpers are usable before the native extension has
    # been built. Runtime APIs such as create_embedding/load still require _C.
    pass

from embeddings_cpp.registry import ModelSpec, get_model_spec, list_models


def load(*args, **kwargs):
    from embeddings_cpp.hub import load as _load

    return _load(*args, **kwargs)


__version__ = "0.1.0"
