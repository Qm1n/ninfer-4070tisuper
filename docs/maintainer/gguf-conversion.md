# GGUF conversion

The registered Qwen3.8-27B converter builds an artifact from the official Hugging Face
checkpoint. Distributed GGUF checkpoints are a different container: the same weights are stored
with llama.cpp's block formats and, for Qwen3.5 linear attention, with a container-specific tensor
layout. `tools/convert/qwen3_8_27b/convert_gguf.py` keeps the registered recipe, object plan,
numeric formats, and artifact identity, and reads the weights from the GGUF files instead.

## Inputs

| Input | Role |
| --- | --- |
| `--gguf` | Main `qwen35` container: text layers, output head, and the MTP block |
| `--mmproj` | Vision projector (`clip` architecture) that carries the 333 vision tensors |
| `--frontend` | Official frontend directory: `config.json`, the six registered frontend resources, and the tokenizer used for the draft-head shortlist |

The frontend resources are checked against the registered SHA-256 hashes, so any copy of the
official Qwen3.8-27B frontend files works (a quantized derivative that ships them unchanged is
fine).

## Usage

    python -m tools.convert.qwen3_8_27b.convert_gguf \\
      --frontend /path/to/Qwen3.8-27B \\
      --gguf /path/to/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \\
      --mmproj /path/to/mmproj-Qwen3.8-27B-BF16.gguf \\
      --out out/qwen3_8_27b_gguf.ninfer

On Windows the packaged helper resolves the interpreter and streams progress:

    .\\scripts\\windows\\convert-gguf.ps1 -Frontend <dir> -Gguf <file> -Mmproj <file> -Out <artifact>

Quantization runs on the CPU by default; pass `--device cuda` only when torch with CUDA support
is installed. The conversion needs numpy and torch, and reads the GGUF blocks with the vendored
reference dequantizers in `tools/convert/gguf/quants.py` (MIT, from llama.cpp's gguf-py).

## Container transforms

A GGUF is not a plain rename of the checkpoint. The converter restores the Hugging Face
convention before the registered recipes run:

- **Linear-attention value heads.** With 16 key heads and 48 value heads, the container tiles the
  value heads as (value position, key head) while the checkpoint groups them as (key head, value
  position). The converter re-groups `in_proj_qkv` (the value rows), `in_proj_a`, `in_proj_b`,
  `in_proj_z`, `A_log`, `dt_bias`, `conv1d` (the value channels), and `out_proj` (the input
  columns).
- **`ssm_a`.** Stored as `-exp(A_log)`; the converter restores `A_log`.
- **RMSNorm weights.** Weights the runtime consumes through a plain RMSNorm are stored as
  `1 + w`; the converter subtracts the offset. This covers `attn_norm`, `post_attention_norm`,
  `output_norm`, `attn_q_norm`, `attn_k_norm`, and the MTP norms. `ssm_norm` is a gated norm and
  keeps its raw value.
- **Vision patch embedding.** The projector stores the two temporal slices of the 3-D patch
  convolution separately; the converter stacks them back into one `(out, in, 2, 16, 16)` weight.

The transforms are applied only to `qwen35` containers, so the vision projector keeps its own
layout.

## Verification

Three checks establish that a converted artifact carries the original weights:

1. **Dequantization parity.** Dequantized blocks match llama.cpp's `gguf.quants.dequantize`
   bit-for-bit for every format the checkpoint uses (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S,
   IQ3_XXS, IQ4_XS, Q2_K, Q4_K, Q6_K, BF16, F32).
2. **Parameter recovery.** Unquantized parameters are byte-identical to the base checkpoint after
   the transforms: linear-attention `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, `conv1d`,
   `ssm_norm`, every RMSNorm, and the vision norms.
3. **Runtime behavior.** The artifact loads through the registered target and decodes at the same
   rate as the reference build of the same model.

## Limits

- Quantized weights are re-encoded into NInfer formats, so the artifact matches the min-Q4 profile
  rather than the GGUF's own bit layout (expect the same size and comparable quality).
- Vision tensors are validated structurally and by parameter recovery; a text-only build cannot
  exercise the image path end to end.
- The MTP block comes from the same container, so speculative decoding uses the checkpoint's own
  draft head.
