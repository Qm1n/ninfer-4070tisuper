"""Reference-model cache and recurrent state ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .config import CFG


class KVCache:
    def __init__(self, layers: int, capacity: int, device: torch.device, dtype: str = "bf16"):
        # // Fork: mirror the product's native signed-nibble G64 cache option.
        if dtype not in {"bf16", "int8", "i4"}:
            raise ValueError(f"kv dtype must be bf16/int8/i4, got {dtype!r}")
        if capacity <= 0:
            raise ValueError("KV capacity must be positive")
        self.layers = layers
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self.length = 0
        self._k: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._ks: dict[int, torch.Tensor] = {}
        self._vs: dict[int, torch.Tensor] = {}

    def _allocate(self, layer: int) -> None:
        if layer in self._k:
            return
        shape = (self.capacity, CFG.kv_heads, CFG.head_dim)
        if self.dtype == "bf16":
            self._k[layer] = torch.empty(shape, device=self.device, dtype=torch.bfloat16)
            self._v[layer] = torch.empty(shape, device=self.device, dtype=torch.bfloat16)
        elif self.dtype == "int8":
            self._k[layer] = torch.empty(shape, device=self.device, dtype=torch.int8)
            self._v[layer] = torch.empty(shape, device=self.device, dtype=torch.int8)
            scales = (self.capacity, CFG.kv_heads, CFG.head_dim // 64)
            self._ks[layer] = torch.empty(scales, device=self.device, dtype=torch.float16)
            self._vs[layer] = torch.empty(scales, device=self.device, dtype=torch.float16)
        else:
            # // Fork: reference storage preserves the exact two-nibbles-per-U8 physical codec.
            packed_shape = (self.capacity, CFG.kv_heads, CFG.head_dim // 2)
            self._k[layer] = torch.empty(packed_shape, device=self.device, dtype=torch.uint8)
            self._v[layer] = torch.empty(packed_shape, device=self.device, dtype=torch.uint8)
            scales = (self.capacity, CFG.kv_heads, CFG.head_dim // 64)
            self._ks[layer] = torch.empty(scales, device=self.device, dtype=torch.float16)
            self._vs[layer] = torch.empty(scales, device=self.device, dtype=torch.float16)

    @staticmethod
    def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = x.float().reshape(*x.shape[:-1], CFG.head_dim // 64, 64)
        scale = (groups.abs().amax(dim=-1) / 127.0).to(torch.float16)
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale).float()
        code = torch.round(groups / safe_scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        code = torch.where((scale == 0).unsqueeze(-1), 0, code).to(torch.int8)
        return code.reshape_as(x), scale

    @staticmethod
    def _dequantize(code: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        groups = code.float().reshape(*code.shape[:-1], CFG.head_dim // 64, 64)
        return (groups * scale.float().unsqueeze(-1)).reshape_as(code).to(torch.bfloat16)

    # // Fork: exact I4-G64 logical quantizer with FP16_RNE(amax/7) scales.
    @staticmethod
    def _quantize_i4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = x.float().reshape(*x.shape[:-1], CFG.head_dim // 64, 64)
        scale = (groups.abs().amax(dim=-1) / 7.0).to(torch.float16)
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale).float()
        code = torch.round(groups / safe_scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
        code = torch.where((scale == 0).unsqueeze(-1), 0, code).to(torch.int8)
        return code.reshape_as(x), scale

    # // Fork: low nibble is the even logical dimension, high nibble the following odd one.
    @staticmethod
    def _pack_i4(code: torch.Tensor) -> torch.Tensor:
        even = code[..., 0::2].to(torch.int16) & 0x0F
        odd = (code[..., 1::2].to(torch.int16) & 0x0F) << 4
        return (even | odd).to(torch.uint8)

    # // Fork: signed-nibble XOR/subtract decode mirrors the CUDA atom.
    @staticmethod
    def _unpack_i4(packed: torch.Tensor) -> torch.Tensor:
        low = packed.to(torch.int16) & 0x0F
        high = (packed.to(torch.int16) >> 4) & 0x0F
        low = (low ^ 0x08) - 0x08
        high = (high ^ 0x08) - 0x08
        return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        end = start + k.shape[0]
        if start < 0 or end > self.capacity or v.shape != k.shape:
            raise ValueError("KV write range or shape mismatch")
        self._allocate(layer)
        if self.dtype == "bf16":
            self._k[layer][start:end].copy_(k)
            self._v[layer][start:end].copy_(v)
        elif self.dtype == "int8":
            kc, ks = self._quantize(k)
            vc, vs = self._quantize(v)
            self._k[layer][start:end].copy_(kc)
            self._v[layer][start:end].copy_(vc)
            self._ks[layer][start:end].copy_(ks)
            self._vs[layer][start:end].copy_(vs)
        else:
            # // Fork: persist packed I4 codes and exact FP16 scales.
            kc, ks = self._quantize_i4(k)
            vc, vs = self._quantize_i4(v)
            self._k[layer][start:end].copy_(self._pack_i4(kc))
            self._v[layer][start:end].copy_(self._pack_i4(vc))
            self._ks[layer][start:end].copy_(ks)
            self._vs[layer][start:end].copy_(vs)

    def read(self, layer: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if end < 0 or end > self.capacity or layer not in self._k:
            raise ValueError("KV read range or layer mismatch")
        if self.dtype == "bf16":
            return self._k[layer][:end], self._v[layer][:end]
        if self.dtype == "int8":
            return (
                self._dequantize(self._k[layer][:end], self._ks[layer][:end]),
                self._dequantize(self._v[layer][:end], self._vs[layer][:end]),
            )
        # // Fork: decode packed signed nibbles before applying their G64 scales.
        return (
            self._dequantize(self._unpack_i4(self._k[layer][:end]), self._ks[layer][:end]),
            self._dequantize(self._unpack_i4(self._v[layer][:end]), self._vs[layer][:end]),
        )

    def rewind(self, position: int) -> None:
        if position < 0 or position > self.length:
            raise ValueError("KV rewind cannot move forward")
        self.length = position


@dataclass
class ModelState:
    device: torch.device
    capacity: int
    kv_dtype: str
    kv: KVCache = field(init=False)
    mtp_kv: KVCache = field(init=False)
    conv: list[torch.Tensor] = field(init=False)
    ssm: list[torch.Tensor] = field(init=False)
    position: int = 0
    rope_delta: int = 0
    mrope: bool = False

    def __post_init__(self) -> None:
        self.kv = KVCache(CFG.full_layers, self.capacity, self.device, self.kv_dtype)
        self.mtp_kv = KVCache(1, self.capacity, self.device, self.kv_dtype)
        self.conv = [
            torch.zeros(CFG.conv_dim, CFG.conv_width - 1, device=self.device, dtype=torch.float32)
            for _ in range(CFG.gdn_layers)
        ]
        self.ssm = [
            torch.zeros(
                1,
                CFG.gdn_v_heads,
                CFG.gdn_k_dim,
                CFG.gdn_v_dim,
                device=self.device,
                dtype=torch.float32,
            )
            for _ in range(CFG.gdn_layers)
        ]

    def snapshot(self) -> "StateSnapshot":
        return StateSnapshot(
            position=self.position,
            rope_delta=self.rope_delta,
            mrope=self.mrope,
            kv_length=self.kv.length,
            mtp_kv_length=self.mtp_kv.length,
            conv=[tensor.clone() for tensor in self.conv],
            ssm=[tensor.clone() for tensor in self.ssm],
        )

    def restore(self, snapshot: "StateSnapshot") -> None:
        if snapshot.position < 0 or snapshot.position > self.capacity:
            raise ValueError("snapshot position is outside state capacity")
        self.position = snapshot.position
        self.rope_delta = snapshot.rope_delta
        self.mrope = snapshot.mrope
        self.kv.length = snapshot.kv_length
        self.mtp_kv.length = snapshot.mtp_kv_length
        for target, source in zip(self.conv, snapshot.conv, strict=True):
            target.copy_(source)
        for target, source in zip(self.ssm, snapshot.ssm, strict=True):
            target.copy_(source)


@dataclass
class StateSnapshot:
    position: int
    rope_delta: int
    mrope: bool
    kv_length: int
    mtp_kv_length: int
    conv: list[torch.Tensor]
    ssm: list[torch.Tensor]
