from pathlib import Path

import numpy as np

from tools.convert.gguf.names import GgufEntry, PATCH_COMBINE
from tools.convert.gguf.source import GgufSourceReader
from tools.convert.gguf.transforms import (
    HEAD_VALUE_DIM,
    NUM_KEY_HEADS,
    NUM_V_PER_K,
    QK_CHANNELS,
    container_transform,
)


def _group_tiled_heads(values: np.ndarray, axis: int, head_dim: int) -> np.ndarray:
    source_heads = [
        value_slot * NUM_KEY_HEADS + key_head
        for key_head in range(NUM_KEY_HEADS)
        for value_slot in range(NUM_V_PER_K)
    ]
    indices = np.concatenate(
        [np.arange(head * head_dim, (head + 1) * head_dim) for head in source_heads]
    )
    return np.take(values, indices, axis=axis)


def test_qwen35_container_transforms_restore_checkpoint_conventions() -> None:
    norms = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    for name in (
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.48.nextn.enorm.weight",
        "blk.48.nextn.hnorm.weight",
        "blk.48.nextn.shared_head_norm.weight",
    ):
        np.testing.assert_array_equal(container_transform(name, norms), norms - 1.0)
    np.testing.assert_array_equal(container_transform("blk.0.ssm_norm.weight", norms), norms)

    value_rows = np.arange(NUM_KEY_HEADS * NUM_V_PER_K * HEAD_VALUE_DIM * 2, dtype=np.float32)
    value_rows = value_rows.reshape(NUM_KEY_HEADS * NUM_V_PER_K * HEAD_VALUE_DIM, 2)
    expected_rows = _group_tiled_heads(value_rows, axis=0, head_dim=HEAD_VALUE_DIM)
    qk = np.full((QK_CHANNELS, 2), -1.0, dtype=np.float32)
    qkv = np.concatenate((qk, value_rows), axis=0)
    expected_qkv = np.concatenate((qk, expected_rows), axis=0)
    np.testing.assert_array_equal(container_transform("blk.0.attn_qkv.weight", qkv), expected_qkv)
    np.testing.assert_array_equal(container_transform("blk.0.ssm_conv1d.weight", qkv), expected_qkv)
    np.testing.assert_array_equal(
        container_transform("blk.0.attn_gate.weight", value_rows), expected_rows
    )

    scalar_heads = np.arange(NUM_KEY_HEADS * NUM_V_PER_K, dtype=np.float32)
    expected_scalars = _group_tiled_heads(scalar_heads, axis=0, head_dim=1)
    for name in ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_dt.bias"):
        np.testing.assert_array_equal(container_transform(name, scalar_heads), expected_scalars)

    tiled_log = np.linspace(-2.0, 2.0, NUM_KEY_HEADS * NUM_V_PER_K, dtype=np.float64)
    stored_a = -np.exp(tiled_log)
    expected_a_log = _group_tiled_heads(tiled_log, axis=0, head_dim=1).astype(np.float32)
    np.testing.assert_allclose(
        container_transform("blk.0.ssm_a", stored_a), expected_a_log, rtol=0, atol=2e-7
    )

    value_columns = value_rows.T
    np.testing.assert_array_equal(
        container_transform("blk.0.ssm_out.weight", value_columns),
        _group_tiled_heads(value_columns, axis=1, head_dim=HEAD_VALUE_DIM),
    )


class _FakeGguf:
    architecture = "clip"
    path = Path("vision.gguf")

    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def values(self, name: str) -> np.ndarray:
        return self._tensors[name]


def test_vision_patch_temporal_slices_are_stacked_in_checkpoint_order() -> None:
    first = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2)
    second = first + 20
    gguf = _FakeGguf({"v.patch_embd.weight": first, "v.patch_embd.weight.1": second})
    entry = GgufEntry(
        source="vision",
        tensors=("v.patch_embd.weight", "v.patch_embd.weight.1"),
        shape=(2, 1, 2, 2, 2),
        combine=PATCH_COMBINE,
    )
    reader = GgufSourceReader({"vision": gguf}, {"visual.patch_embed.proj.weight": entry})

    actual = reader.get("visual.patch_embed.proj.weight").float().numpy()
    np.testing.assert_array_equal(actual, np.stack((first, second), axis=2))
