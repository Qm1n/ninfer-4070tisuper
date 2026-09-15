"""Undo the container-side transforms that llama.cpp applies to Qwen3.5 checkpoints.

The qwen35 GGUF layout is not a plain rename of the Hugging Face checkpoint: linear-attention
value heads are reordered from grouped order to tiled order, and the RMSNorm weights that the
runtime consumes directly are stored as the effective ``1 + w`` values.  The GGUF reader restores
the Hugging Face convention so the registered recipes stay unchanged.
"""

from __future__ import annotations

import numpy as np

# Registered Qwen3.8-27B linear-attention geometry (Qwen3.5 uses the same layout).
NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
HEAD_KEY_DIM = 128
HEAD_VALUE_DIM = 128
NUM_V_PER_K = NUM_VALUE_HEADS // NUM_KEY_HEADS
QK_CHANNELS = 2 * HEAD_KEY_DIM * NUM_KEY_HEADS

# Tensors the runtime consumes through a plain RMSNorm, stored as 1 + w.
OFFSET_NORM_SUFFIXES = (
    ".attn_norm.weight",
    ".post_attention_norm.weight",
    ".output_norm.weight",
    ".attn_q_norm.weight",
    ".attn_k_norm.weight",
    ".nextn.enorm.weight",
    ".nextn.hnorm.weight",
    ".nextn.shared_head_norm.weight",
)


def reorder_v_heads(values: np.ndarray, dim: int, head_dim: int) -> np.ndarray:
    """Restore grouped (by key head) value heads from the tiled container order.

    The container stores value heads tiled as (value position, key head); the checkpoint groups
    them as (key head, value position).  Grouping back therefore splits the axis into
    (NUM_V_PER_K, NUM_KEY_HEADS, head_dim) and transposes the first two factors.
    """

    shape = list(values.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [NUM_V_PER_K, NUM_KEY_HEADS, head_dim] + shape[dim + 1 :]
    reshaped = values.reshape(new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(np.transpose(reshaped, perm).reshape(shape))


def container_transform(tensor_name: str, values: np.ndarray) -> np.ndarray:
    """Restore Hugging Face convention for one container tensor."""

    name = tensor_name
    if name == "output_norm.weight" or name.endswith(OFFSET_NORM_SUFFIXES):
        return (values.astype(np.float32, copy=False) - 1.0).astype(np.float32)
    if name.endswith(".attn_qkv.weight"):
        qk = values[:QK_CHANNELS]
        v = values[QK_CHANNELS:]
        return np.concatenate(
            (qk, reorder_v_heads(v, 0, HEAD_VALUE_DIM)), axis=0
        )
    if name.endswith(".ssm_conv1d.weight"):
        qk = values[:QK_CHANNELS]
        v = values[QK_CHANNELS:]
        return np.concatenate(
            (qk, reorder_v_heads(v, 0, HEAD_VALUE_DIM)), axis=0
        )
    if name.endswith(".attn_gate.weight"):
        return reorder_v_heads(values, 0, HEAD_VALUE_DIM)
    if name.endswith(".ssm_alpha.weight") or name.endswith(".ssm_beta.weight"):
        return reorder_v_heads(values, 0, 1)
    if name.endswith(".ssm_dt.bias"):
        return reorder_v_heads(values, 0, 1)
    if name.endswith(".ssm_a"):
        reordered = reorder_v_heads(values.astype(np.float64, copy=False), 0, 1)
        with np.errstate(divide="raise", invalid="raise"):
            a_log = np.log(-reordered)
        return a_log.astype(np.float32)
    if name.endswith(".ssm_out.weight"):
        return reorder_v_heads(values, 1, HEAD_VALUE_DIM)
    return values
