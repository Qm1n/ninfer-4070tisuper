"""ShardReader-compatible access to GGUF checkpoints for the artifact converters.

The artifact recipes request named source tensors and inspect their declared shape, dtype, and
shard.  This reader answers those queries from one or more GGUF containers, materializing each
requested tensor on demand and dequantizing stored blocks through the vendored reference math.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import torch

from tools.convert.common.safetensors import TensorMetadata

from .names import CONV1D_COMBINE, PATCH_COMBINE, GgufEntry
from .reader import GgufFile
from .transforms import container_transform

SOURCE_DTYPE = "BF16"

# Container layouts whose stored tensors need the Hugging Face convention restored.
_TRANSFORMED_ARCHITECTURES = ("qwen35",)


class GgufSourceReader:
    """Present GGUF tensors through the safetensors recipe interface."""

    def __init__(
        self,
        files: Mapping[str, GgufFile],
        mapping: Mapping[str, GgufEntry],
    ) -> None:
        self._files = dict(files)
        self._entries = dict(mapping)
        missing = sorted(
            (name, entry.source)
            for name, entry in self._entries.items()
            if entry.source not in self._files
        )
        if missing:
            raise ValueError(f"source mapping references unknown containers: {missing[:4]}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def has(self, name: str) -> bool:
        return name in self._entries

    def metadata(self, names: Iterable[str]) -> dict[str, TensorMetadata]:
        result: dict[str, TensorMetadata] = {}
        for name in names:
            entry = self._entries[name]
            result[name] = TensorMetadata(
                name=name,
                shard=self._files[entry.source].path.name,
                shape=entry.shape,
                dtype=SOURCE_DTYPE,
            )
        return result

    def get(self, name: str) -> torch.Tensor:
        entry = self._entries[name]
        values = self._materialize(entry)
        if tuple(values.shape) != entry.shape:
            raise ValueError(f"{name}: GGUF tensor shape {tuple(values.shape)} != {entry.shape}")
        if not values.flags.writeable:
            values = np.array(values, copy=True)
        # The registered recipes describe a bfloat16 checkpoint, and the direct encoders expect
        # that dtype, so dequantized values are rounded to the same target precision.
        return torch.from_numpy(values).to(torch.bfloat16)

    def close(self) -> None:
        for gguf in self._files.values():
            gguf.close()

    def __enter__(self) -> GgufSourceReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _materialize(self, entry: GgufEntry) -> np.ndarray:
        gguf = self._files[entry.source]
        for name in entry.tensors:
            if not gguf.has(name):
                raise ValueError(f"{entry.tensors}: {name} is missing from {gguf.path.name}")
        if entry.combine == PATCH_COMBINE:
            if len(entry.tensors) != 2:
                raise ValueError("the temporal patch pair must name exactly two tensors")
            first, second = (gguf.values(name) for name in entry.tensors)
            return np.ascontiguousarray(np.stack((first, second), axis=2))
        if entry.combine == CONV1D_COMBINE:
            if len(entry.tensors) != 1:
                raise ValueError("the convolution entry must name exactly one tensor")
            stored = gguf.values(entry.tensors[0])
            stored = self._restore(gguf, entry.tensors[0], stored)
            if stored.size != int(np.prod(entry.shape)):
                raise ValueError(f"{entry.tensors[0]}: cannot reshape to {entry.shape}")
            return np.ascontiguousarray(stored.reshape(entry.shape))
        if entry.combine is not None:
            raise ValueError(f"unknown combination mode {entry.combine!r}")
        if len(entry.tensors) != 1:
            raise ValueError("multi-tensor entries require an explicit combination mode")
        stored = gguf.values(entry.tensors[0])
        return self._restore(gguf, entry.tensors[0], stored)

    @staticmethod
    def _restore(gguf: GgufFile, tensor_name: str, values: np.ndarray) -> np.ndarray:
        if gguf.architecture not in _TRANSFORMED_ARCHITECTURES:
            return values
        return container_transform(tensor_name, values)
