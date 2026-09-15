"""Minimal GGUF container reader used by the NInfer artifact converters.

The converter reads reference GGUF checkpoints directly instead of requiring a safetensors
conversion step: this module parses the container, exposes metadata and the tensor table, and
dequantizes stored blocks with the vendored reference math (see quants.py).  Logical tensors are
presented in the Hugging Face row-major convention, so a GGUF tensor declared with dims
[in_features, out_features] is returned with shape (out_features, in_features).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from .ggml_types import GGML_QUANT_SIZES, GGMLQuantizationType
from .quants import dequantize

_MAGIC = 0x46554747
_SUPPORTED_VERSIONS = (2, 3)

_VALUE_FORMATS = {
    0: "B",   # UINT8
    1: "b",   # INT8
    2: "H",   # UINT16
    3: "h",   # INT16
    4: "I",   # UINT32
    5: "i",   # INT32
    6: "f",   # FLOAT32
    7: "?",   # BOOL
    10: "Q",  # UINT64
    11: "q",  # INT64
    12: "d",  # FLOAT64
}
_STRING_TYPE = 8
_ARRAY_TYPE = 9


@dataclass(frozen=True, slots=True)
class GgufTensorInfo:
    name: str
    dims: tuple[int, ...]
    qtype: GGMLQuantizationType
    offset: int
    elements: int
    stored_bytes: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical (out_features, in_features, ...) shape."""
        return tuple(reversed(self.dims))

    @property
    def block_size(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][0]

    @property
    def type_size(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][1]


def _read_string(data: np.ndarray, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    text = data[offset : offset + length].tobytes().decode("utf-8")
    return text, offset + int(length)


def _read_value(data: np.ndarray, offset: int) -> tuple[object, int]:
    (kind,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if kind == _STRING_TYPE:
        return _read_string(data, offset)
    if kind == _ARRAY_TYPE:
        (item_kind,) = struct.unpack_from("<I", data, offset)
        (count,) = struct.unpack_from("<Q", data, offset + 4)
        offset += 12
        items = []
        for _ in range(count):
            value, offset = _read_value_of_kind(data, offset, item_kind)
            items.append(value)
        return items, offset
    return _read_value_of_kind(data, offset, kind)


def _read_value_of_kind(data: np.ndarray, offset: int, kind: int) -> tuple[object, int]:
    if kind == _STRING_TYPE:
        return _read_string(data, offset)
    if kind == _ARRAY_TYPE:
        return _read_value(data, offset)
    fmt = _VALUE_FORMATS.get(kind)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata type {kind}")
    value = struct.unpack_from("<" + fmt, data, offset)[0]
    return value, offset + struct.calcsize(fmt)


class GgufFile:
    """Read-only view over one GGUF file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mapping = np.memmap(self.path, mode="r")
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, GgufTensorInfo] = {}
        self._parse()
        # The container family decides which layout transforms apply; the text model and the
        # separate vision projector use different architectures and different layouts.
        self.architecture = str(self.metadata.get("general.architecture", ""))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.tensors)

    def has(self, name: str) -> bool:
        return name in self.tensors

    def info(self, name: str) -> GgufTensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{name} is not present in {self.path.name}") from None

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping._mmap.close()
            self._mapping = None

    def __enter__(self) -> GgufFile:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _parse(self) -> None:
        data = self._mapping
        magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", data, 0)
        if magic != _MAGIC:
            raise ValueError(f"{self.path} is not a GGUF file")
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported GGUF version {version} in {self.path}")
        offset = 24
        for _ in range(kv_count):
            key, offset = _read_string(data, offset)
            value, offset = _read_value(data, offset)
            self.metadata[key] = value
        entries = []
        for _ in range(tensor_count):
            name, offset = _read_string(data, offset)
            (n_dims,) = struct.unpack_from("<I", data, offset)
            offset += 4
            dims = struct.unpack_from("<" + "Q" * n_dims, data, offset)
            offset += 8 * n_dims
            (raw_type,) = struct.unpack_from("<I", data, offset)
            offset += 4
            (relative,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            info = self._describe(name, dims, raw_type, relative)
            if name in self.tensors:
                raise ValueError(f"duplicate tensor {name} in {self.path}")
            self.tensors[name] = info
            entries.append(info)
        alignment = int(self.metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"invalid GGUF alignment {alignment}")
        self.alignment = alignment
        self.data_offset = (offset + alignment - 1) // alignment * alignment

    def _describe(self, name: str, dims, raw_type: int, relative: int) -> GgufTensorInfo:
        try:
            qtype = GGMLQuantizationType(int(raw_type))
        except ValueError:
            raise ValueError(f"{name}: unknown GGML type {raw_type}") from None
        elements = 1
        for dim in dims:
            elements *= int(dim)
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        if elements % block_size:
            raise ValueError(f"{name}: element count {elements} is not a multiple of {block_size}")
        stored_bytes = elements // block_size * type_size
        return GgufTensorInfo(
            name=name,
            dims=tuple(int(dim) for dim in dims),
            qtype=qtype,
            offset=int(relative),
            elements=elements,
            stored_bytes=stored_bytes,
        )

    def raw(self, name: str) -> np.ndarray:
        info = self.info(name)
        return np.frombuffer(
            self._mapping,
            dtype=np.uint8,
            count=info.stored_bytes,
            offset=self.data_offset + info.offset,
        )

    def values(self, name: str) -> np.ndarray:
        """Dequantize one tensor into float32 with the HF logical shape."""
        info = self.info(name)
        raw = self.raw(name)
        if info.qtype == GGMLQuantizationType.F32:
            return raw.view(np.float32).reshape(info.shape).astype(np.float32, copy=False)
        if info.qtype == GGMLQuantizationType.F16:
            return raw.view(np.float16).reshape(info.shape).astype(np.float32)
        if info.qtype == GGMLQuantizationType.BF16:
            words = raw.view(np.uint16).astype(np.uint32) << 16
            return words.view(np.float32).reshape(info.shape)
        _, type_size = GGML_QUANT_SIZES[info.qtype]
        byte_shape = (*info.shape[:-1], info.shape[-1] // info.block_size * type_size)
        stored = raw.reshape(byte_shape)
        return dequantize(stored, info.qtype).astype(np.float32, copy=False)
