"""Source-name mapping from the NInfer recipes onto GGUF checkpoints.

The recipes address a Hugging Face style checkpoint (``model.language_model.layers.0...``).
This module translates those names onto the llama.cpp GGUF names of the text model and the
separate vision projector, together with the logical shape each recipe declares.  Logical shapes
follow the Hugging Face row-major convention, which is the convention the GGUF reader already
presents for stored tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

TEXT_SOURCE = "model"
VISION_SOURCE = "vision"

TEXT_WIDTH = 5120
HEAD_COUNT = 24
HEAD_DIM = 256
KV_WIDTH = 1024
ATTENTION_OUTPUT = 6144
INTERMEDIATE = 17408
VOCAB_ROWS = 248320
GDN_QKV = 10240
GDN_VALUE = 6144
GDN_RANK = 48
GDN_CONV_KERNEL = 4
GDN_STATE = 128

VISION_WIDTH = 1152
VISION_INTERMEDIATE = 4304
VISION_QKV = 3456
VISION_PATCH_TOKENS = 2
VISION_PATCH = 16
VISION_POSITIONS = 2304
MERGER_WIDTH = 4608

CONV1D_COMBINE = "collapse_channel"
PATCH_COMBINE = "stack_temporal"


@dataclass(frozen=True, slots=True)
class GgufEntry:
    """One logical source tensor and the GGUF tensors that carry it."""

    source: str
    tensors: tuple[str, ...]
    shape: tuple[int, ...]
    combine: str | None = None


def _entry(
    source: str,
    tensors: str | tuple[str, ...],
    shape: tuple[int, ...],
    combine: str | None = None,
) -> GgufEntry:
    names = (tensors,) if isinstance(tensors, str) else tensors
    return GgufEntry(source=source, tensors=names, shape=shape, combine=combine)


def build_source_mapping(
    *,
    text_layers: int,
    full_attention_layers: tuple[int, ...],
    vision_layers: tuple[int, ...],
) -> dict[str, GgufEntry]:
    """Map every recipe source name onto its GGUF tensor(s)."""

    full_attention = set(full_attention_layers)
    mapping: dict[str, GgufEntry] = {
        "lm_head.weight": _entry(TEXT_SOURCE, "output.weight", (VOCAB_ROWS, TEXT_WIDTH)),
        "model.language_model.embed_tokens.weight": _entry(
            TEXT_SOURCE, "token_embd.weight", (VOCAB_ROWS, TEXT_WIDTH)
        ),
        "model.language_model.norm.weight": _entry(TEXT_SOURCE, "output_norm.weight", (TEXT_WIDTH,)),
    }

    for layer in range(text_layers):
        hf_prefix = f"model.language_model.layers.{layer}."
        gguf_prefix = f"blk.{layer}."
        mapping[hf_prefix + "input_layernorm.weight"] = _entry(
            TEXT_SOURCE, gguf_prefix + "attn_norm.weight", (TEXT_WIDTH,)
        )
        mapping[hf_prefix + "post_attention_layernorm.weight"] = _entry(
            TEXT_SOURCE, gguf_prefix + "post_attention_norm.weight", (TEXT_WIDTH,)
        )
        mapping[hf_prefix + "mlp.gate_proj.weight"] = _entry(
            TEXT_SOURCE, gguf_prefix + "ffn_gate.weight", (INTERMEDIATE, TEXT_WIDTH)
        )
        mapping[hf_prefix + "mlp.up_proj.weight"] = _entry(
            TEXT_SOURCE, gguf_prefix + "ffn_up.weight", (INTERMEDIATE, TEXT_WIDTH)
        )
        mapping[hf_prefix + "mlp.down_proj.weight"] = _entry(
            TEXT_SOURCE, gguf_prefix + "ffn_down.weight", (TEXT_WIDTH, INTERMEDIATE)
        )
        if layer in full_attention:
            mapping[hf_prefix + "self_attn.q_proj.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_q.weight",
                (HEAD_COUNT * (HEAD_DIM * 2), TEXT_WIDTH),
            )
            mapping[hf_prefix + "self_attn.k_proj.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_k.weight", (KV_WIDTH, TEXT_WIDTH)
            )
            mapping[hf_prefix + "self_attn.v_proj.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_v.weight", (KV_WIDTH, TEXT_WIDTH)
            )
            mapping[hf_prefix + "self_attn.o_proj.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_output.weight", (TEXT_WIDTH, ATTENTION_OUTPUT)
            )
            mapping[hf_prefix + "self_attn.q_norm.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_q_norm.weight", (HEAD_DIM,)
            )
            mapping[hf_prefix + "self_attn.k_norm.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_k_norm.weight", (HEAD_DIM,)
            )
        else:
            mapping[hf_prefix + "linear_attn.in_proj_qkv.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_qkv.weight", (GDN_QKV, TEXT_WIDTH)
            )
            mapping[hf_prefix + "linear_attn.in_proj_z.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "attn_gate.weight", (GDN_VALUE, TEXT_WIDTH)
            )
            mapping[hf_prefix + "linear_attn.in_proj_a.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_alpha.weight", (GDN_RANK, TEXT_WIDTH)
            )
            mapping[hf_prefix + "linear_attn.in_proj_b.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_beta.weight", (GDN_RANK, TEXT_WIDTH)
            )
            mapping[hf_prefix + "linear_attn.A_log"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_a", (GDN_RANK,)
            )
            mapping[hf_prefix + "linear_attn.dt_bias"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_dt.bias", (GDN_RANK,)
            )
            mapping[hf_prefix + "linear_attn.conv1d.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_conv1d.weight",
                (GDN_QKV, 1, GDN_CONV_KERNEL),
                CONV1D_COMBINE,
            )
            mapping[hf_prefix + "linear_attn.norm.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_norm.weight", (GDN_STATE,)
            )
            mapping[hf_prefix + "linear_attn.out_proj.weight"] = _entry(
                TEXT_SOURCE, gguf_prefix + "ssm_out.weight", (TEXT_WIDTH, GDN_VALUE)
            )

    mtp_prefix = "mtp.layers.0."
    mtp_gguf = f"blk.{text_layers}."
    mapping["mtp.fc.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "nextn.eh_proj.weight", (TEXT_WIDTH, GDN_QKV)
    )
    mapping["mtp.pre_fc_norm_embedding.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "nextn.enorm.weight", (TEXT_WIDTH,)
    )
    mapping["mtp.pre_fc_norm_hidden.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "nextn.hnorm.weight", (TEXT_WIDTH,)
    )
    mapping["mtp.norm.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "nextn.shared_head_norm.weight", (TEXT_WIDTH,)
    )
    mapping[mtp_prefix + "input_layernorm.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_norm.weight", (TEXT_WIDTH,)
    )
    mapping[mtp_prefix + "self_attn.q_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_q.weight", (HEAD_COUNT * (HEAD_DIM * 2), TEXT_WIDTH)
    )
    mapping[mtp_prefix + "self_attn.k_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_k.weight", (KV_WIDTH, TEXT_WIDTH)
    )
    mapping[mtp_prefix + "self_attn.v_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_v.weight", (KV_WIDTH, TEXT_WIDTH)
    )
    mapping[mtp_prefix + "self_attn.o_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_output.weight", (TEXT_WIDTH, ATTENTION_OUTPUT)
    )
    mapping[mtp_prefix + "self_attn.q_norm.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_q_norm.weight", (HEAD_DIM,)
    )
    mapping[mtp_prefix + "self_attn.k_norm.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "attn_k_norm.weight", (HEAD_DIM,)
    )
    mapping[mtp_prefix + "post_attention_layernorm.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "post_attention_norm.weight", (TEXT_WIDTH,)
    )
    mapping[mtp_prefix + "mlp.gate_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "ffn_gate.weight", (INTERMEDIATE, TEXT_WIDTH)
    )
    mapping[mtp_prefix + "mlp.up_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "ffn_up.weight", (INTERMEDIATE, TEXT_WIDTH)
    )
    mapping[mtp_prefix + "mlp.down_proj.weight"] = _entry(
        TEXT_SOURCE, mtp_gguf + "ffn_down.weight", (TEXT_WIDTH, INTERMEDIATE)
    )

    vision_prefix = "model.visual."
    mapping[vision_prefix + "patch_embed.proj.weight"] = _entry(
        VISION_SOURCE,
        ("v.patch_embd.weight", "v.patch_embd.weight.1"),
        (VISION_WIDTH, 3, VISION_PATCH_TOKENS, VISION_PATCH, VISION_PATCH),
        PATCH_COMBINE,
    )
    mapping[vision_prefix + "patch_embed.proj.bias"] = _entry(
        VISION_SOURCE, "v.patch_embd.bias", (VISION_WIDTH,)
    )
    mapping[vision_prefix + "pos_embed.weight"] = _entry(
        VISION_SOURCE, "v.position_embd.weight", (VISION_POSITIONS, VISION_WIDTH)
    )
    for layer in vision_layers:
        hf_layer = vision_prefix + f"blocks.{layer}."
        gguf_layer = f"v.blk.{layer}."
        for hf_suffix, gguf_suffix, shape in (
            ("attn.qkv.weight", "attn_qkv.weight", (VISION_QKV, VISION_WIDTH)),
            ("attn.qkv.bias", "attn_qkv.bias", (VISION_QKV,)),
            ("attn.proj.weight", "attn_out.weight", (VISION_WIDTH, VISION_WIDTH)),
            ("attn.proj.bias", "attn_out.bias", (VISION_WIDTH,)),
            ("mlp.linear_fc1.weight", "ffn_up.weight", (VISION_INTERMEDIATE, VISION_WIDTH)),
            ("mlp.linear_fc1.bias", "ffn_up.bias", (VISION_INTERMEDIATE,)),
            ("mlp.linear_fc2.weight", "ffn_down.weight", (VISION_WIDTH, VISION_INTERMEDIATE)),
            ("mlp.linear_fc2.bias", "ffn_down.bias", (VISION_WIDTH,)),
            ("norm1.weight", "ln1.weight", (VISION_WIDTH,)),
            ("norm1.bias", "ln1.bias", (VISION_WIDTH,)),
            ("norm2.weight", "ln2.weight", (VISION_WIDTH,)),
            ("norm2.bias", "ln2.bias", (VISION_WIDTH,)),
        ):
            mapping[hf_layer + hf_suffix] = _entry(
                VISION_SOURCE, gguf_layer + gguf_suffix, shape
            )

    merger = vision_prefix + "merger."
    for hf_suffix, gguf_name, shape in (
        ("linear_fc1.weight", "mm.0.weight", (MERGER_WIDTH, MERGER_WIDTH)),
        ("linear_fc1.bias", "mm.0.bias", (MERGER_WIDTH,)),
        ("linear_fc2.weight", "mm.2.weight", (TEXT_WIDTH, MERGER_WIDTH)),
        ("linear_fc2.bias", "mm.2.bias", (TEXT_WIDTH,)),
        ("norm.weight", "v.post_ln.weight", (VISION_WIDTH,)),
        ("norm.bias", "v.post_ln.bias", (VISION_WIDTH,)),
    ):
        mapping[merger + hf_suffix] = _entry(VISION_SOURCE, gguf_name, shape)

    return mapping
