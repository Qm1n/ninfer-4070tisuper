"""GGUF checkpoint support for the NInfer artifact converters.

The package parses reference GGUF containers, dequantizes stored blocks with the vendored
reference math, and presents logical tensors in the Hugging Face row-major convention.
"""

from .reader import GgufFile, GgufTensorInfo

__all__ = ["GgufFile", "GgufTensorInfo"]
