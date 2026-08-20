#!/usr/bin/env python3
"""llama.cpp-side optimization: surgical draft-head upgrade.

The unsloth IQ3_XXS GGUF quantizes the MTP draft layer (blk.64.*) at 3 bits,
which caps speculative acceptance at ~45% (llama.cpp measured 81/179).
NInfer's W8-precision draft reaches 64%. This script copies the IQ3_XXS file
verbatim and replaces the blk.64 2-D weights with Q8_0 quantized from the
BF16 source. Cost ~+0.27 GiB VRAM when speculating; benefit: draft passes at
8.5 bpw. Output dtype everywhere else unchanged.
"""
import sys
from pathlib import Path

sys.path.insert(1, "/home/lio/.unsloth/llama.cpp/gguf-py")
import numpy as np
import torch
import gguf
from gguf.quants import quant_shape_from_byte_shape

SRC = "/home/lio/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-UD-IQ3_XXS.gguf"
DST = "/run/media/lio/data/g/qwen3_8_27b_IQ3_XXS_q8draft.gguf"

QK = 32  # Q8_0 block


def q8_0_pack(w: np.ndarray) -> np.ndarray:
    """llama.cpp Q8_0: per-32 block, fp16 d = amax/127, codes = rint(x/d)."""
    flat = w.reshape(-1, QK)
    amax = np.abs(flat).max(axis=1)
    d = (amax / 127.0).astype(np.float16)
    df = d.astype(np.float32)
    inv = np.where(df > 0, 1.0 / np.maximum(df, 1e-30), 0.0)
    q = np.rint(flat * inv[:, None]).clip(-128, 127).astype(np.int8)
    out = np.zeros((flat.shape[0], 2 + QK), dtype=np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.ravel()


def main():
    src = gguf.GGUFReader(SRC)
    bf16 = gguf.GGUFReader("/run/media/lio/data/g/qwen3_8_27b_bf16.gguf")
    bf16_map = {t.name: t for t in bf16.tensors}

    arch = src.get_field("general.architecture").contents()
    writer = gguf.GGUFWriter(str(Path(DST).parent), arch)

    # copy KV metadata
    for name, field in src.fields.items():
        if name in {"general.architecture", "general.quantization_version"} or name.startswith("GGUF."):
            continue
        vt = field.types[0]
        try:
            val = field.contents()
        except Exception:
            continue
        if val is None:
            continue
        if vt == gguf.GGUFValueType.STRING:
            writer.add_string(name, str(val))
        elif vt == gguf.GGUFValueType.ARRAY:
            writer.add_array(name, list(val))
        elif vt == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(name, float(val))
        elif vt == gguf.GGUFValueType.FLOAT64:
            writer.add_float64(name, float(val))
        elif vt == gguf.GGUFValueType.BOOL:
            writer.add_bool(name, bool(val))
        elif vt in (gguf.GGUFValueType.UINT8, gguf.GGUFValueType.INT8, gguf.GGUFValueType.UINT16,
                    gguf.GGUFValueType.INT16, gguf.GGUFValueType.UINT32, gguf.GGUFValueType.INT32,
                    gguf.GGUFValueType.UINT64, gguf.GGUFValueType.INT64):
            intadd = {
                gguf.GGUFValueType.UINT8: writer.add_uint8, gguf.GGUFValueType.INT8: writer.add_int8,
                gguf.GGUFValueType.UINT16: writer.add_uint16, gguf.GGUFValueType.INT16: writer.add_int16,
                gguf.GGUFValueType.UINT32: writer.add_uint32, gguf.GGUFValueType.INT32: writer.add_int32,
                gguf.GGUFValueType.UINT64: writer.add_uint64, gguf.GGUFValueType.INT64: writer.add_int64,
            }[vt]
            intadd(name, int(val))

    # pass 1: register all tensor infos
    upgrade = set()
    for t in src.tensors:
        data = np.array(t.data, copy=False)
        if t.name.startswith("blk.64.") and data.ndim == 2 and t.tensor_type not in (
                gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F16):
            # upgrade target: logical shape from BF16 source
            src_t = bf16_map.get(t.name)
            assert src_t is not None, t.name
            ne0, ne1 = int(src_t.shape[0]), int(src_t.shape[1])
            assert ne0 % QK == 0, (t.name, ne0)
            nbytes = (ne0 // QK) * (2 + QK) * ne1
            # writer wants numpy order (ne1, bytes-per-ne0-row); blocks run along ne0
            writer.add_tensor_info(t.name, (ne1, (ne0 // QK) * (2 + QK)), np.dtype(np.uint8),
                                   nbytes, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            upgrade.add(t.name)
        else:
            # reader data.shape is (ne1, bytes-per-ne0-row): exactly what
            # quant_shape_from_byte_shape expects for quantized raw dtypes.
            writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes,
                                   raw_dtype=t.tensor_type)

    writer.write_header_to_file(path=DST)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # pass 2: stream data
    n_up = 0
    for t in src.tensors:
        if t.name in upgrade:
            src_t = bf16_map[t.name]
            b = np.array(src_t.data, copy=True).view(np.uint16)
            w = torch.from_numpy(b).view(torch.bfloat16).float().numpy()
            writer.write_tensor_data(q8_0_pack(w))
            n_up += 1
        else:
            writer.write_tensor_data(np.array(t.data, copy=True))
    writer.close()
    print(f"upgraded {n_up} draft tensors to Q8_0 -> {DST}")


if __name__ == "__main__":
    main()
