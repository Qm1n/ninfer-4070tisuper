#!/usr/bin/env python3
"""Apply the fork's groupwise-int quant math to the BF16 GGUF and write an
F16 GGUF (dequantized weights) for exact PPL evaluation with llama-perplexity.

Quant math == tools/convert/common/quantize.py: max-abs symmetric RTN,
fp16 group scales, group=64. Output stored F16 (products q*scale16 are exact
to ~0.05%; identical store rounding across all layouts, so comparisons and
the bf16-source reference stay clean).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import torch
import sys as _sys
_sys.path.insert(1, "/home/lio/.unsloth/llama.cpp/gguf-py")
import gguf

FORMATS = {"bf16": None, "q3": (-4, 3), "q4": (-8, 7), "q5": (-16, 15), "q6": (-32, 31)}
GROUP = 64

LAYOUTS = {
    "v2":     {"mlp": "q4", "gdn_qk": "q4", "gdn_vz": "q5", "gdn_out": "q4",
               "fa_qk": "q4", "fa_gv": "q5", "fa_out": "q4", "embed": "q6", "head": "q6"},
    "minq4":  {"mlp": "q4", "gdn_qk": "q4", "gdn_vz": "q4", "gdn_out": "q4",
               "fa_qk": "q4", "fa_gv": "q4", "fa_out": "q4", "embed": "q4", "head": "q4"},
    "q3mlp":  {"mlp": "q3", "gdn_qk": "q4", "gdn_vz": "q4", "gdn_out": "q4",
               "fa_qk": "q4", "fa_gv": "q4", "fa_out": "q4", "embed": "q4", "head": "q4"},
    "q3all":  {"mlp": "q3", "gdn_qk": "q3", "gdn_vz": "q3", "gdn_out": "q3",
               "fa_qk": "q3", "fa_gv": "q3", "fa_out": "q3", "embed": "q3", "head": "q3"},
}

FA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63}


def fmt_for(name, layout):
    """Return list of (axis, slices_or_None, fmt) — slices restrict dim1 ranges.

    GGUF layout is [in, out] for blk weights; groups run along in (axis 0),
    64 consecutive in-rows per group (matches NInfer per-output-row K groups).
    Fused output dim splits per family:
      FA:  attn_q out=[query(6144)|gate(6144)], attn_k->fa_qk, attn_v->fa_gv
      GDN: attn_qkv out=[qk(4096)|v(6144)], attn_gate->gdn_vz, ssm_out->gdn_out
    """
    if name == "token_embd.weight":
        return [(None, None, layout["embed"])]
    if name == "output.weight":
        return [(None, None, layout["head"])]
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return []
    L, sub = int(m.group(1)), m.group(2)
    fa = L in FA_LAYERS
    if sub.startswith("ffn_"):
        return [(None, None, layout["mlp"])]
    if fa:
        if sub == "attn_q.weight":      # rows = [query 6144 | gate 6144]
            return [(None, (0, 6144), layout["fa_qk"]), (None, (6144, 12288), layout["fa_gv"])]
        if sub == "attn_k.weight":
            return [(None, None, layout["fa_qk"])]
        if sub == "attn_v.weight":
            return [(None, None, layout["fa_gv"])]
        if sub == "attn_output.weight":
            return [(None, None, layout["fa_out"])]
        return []
    # GDN layer
    if sub == "attn_qkv.weight":        # rows = [q|k 4096 | v 6144]
        return [(None, (0, 4096), layout["gdn_qk"]), (None, (4096, 10240), layout["gdn_vz"])]
    if sub == "attn_gate.weight":       # z
        return [(None, None, layout["gdn_vz"])]
    if sub == "ssm_out.weight":
        return [(None, None, layout["gdn_out"])]
    return []


def quant_dequant_axis(w: np.ndarray, axis: int, qmin: int, qmax: int) -> np.ndarray:
    """Group-64 max-abs symmetric RTN with fp16 scales, along `axis`."""
    w = np.ascontiguousarray(w)
    K = w.shape[axis]
    assert K % GROUP == 0, f"K={K} not divisible by {GROUP}"
    if axis == 0:
        g = w.reshape(K // GROUP, GROUP, *w.shape[1:])
        reduce_axis = 1
    else:
        g = w.reshape(w.shape[0], K // GROUP, GROUP, *w.shape[2:]) if w.ndim == 2 else None
        assert g is not None, "axis-1 grouping only for 2-D"
        reduce_axis = 2
    amax = np.abs(g).max(axis=reduce_axis, keepdims=True)
    np.maximum(amax, 1e-12, out=amax)
    scale16 = (amax / qmax).astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(g / scale16), qmin, qmax)
    return (q * scale16).reshape(w.shape)


def copy_kv(reader, writer):
    skip = {"general.architecture", "general.quantization_version"}
    for name, field in reader.fields.items():
        if name in skip or name.startswith("GGUF."):
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
        elif vt in (gguf.GGUFValueType.UINT8,):
            writer.add_uint8(name, int(val))
        elif vt in (gguf.GGUFValueType.INT8,):
            writer.add_int8(name, int(val))
        elif vt in (gguf.GGUFValueType.UINT16,):
            writer.add_uint16(name, int(val))
        elif vt in (gguf.GGUFValueType.INT16,):
            writer.add_int16(name, int(val))
        elif vt in (gguf.GGUFValueType.UINT32,):
            writer.add_uint32(name, int(val))
        elif vt in (gguf.GGUFValueType.INT32,):
            writer.add_int32(name, int(val))
        elif vt in (gguf.GGUFValueType.UINT64,):
            writer.add_uint64(name, int(val))
        elif vt in (gguf.GGUFValueType.INT64,):
            writer.add_int64(name, int(val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/run/media/lio/data/g/qwen3_8_27b_bf16.gguf")
    ap.add_argument("--layout", choices=LAYOUTS)
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        reader = gguf.GGUFReader(args.src)
        for t in reader.tensors:
            print(t.name, t.shape, t.tensor_type)
        return
    if not args.layout or not args.out:
        ap.error("--layout and --out are required unless --list")

    reader = gguf.GGUFReader(args.src)
    if args.list:
        for t in reader.tensors:
            print(t.name, t.shape, t.tensor_type)
        return

    layout = LAYOUTS[args.layout]
    arch = reader.get_field("general.architecture").contents()
    writer = gguf.GGUFWriter(str(Path(args.out).parent), arch)
    copy_kv(reader, writer)


    n_q = 0
    # pass 1: register tensor infos (must happen in NO_FILE state)
    raw_tensors = {}
    f32_tensors = set()
    for t in reader.tensors:
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            f32_tensors.add(t.name)
        if t.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32):
            # small control tensors stored packed (Q8_0 etc.): copy verbatim
            raw_bytes = np.array(t.data, copy=True)
            shape = gguf.quant_shape_from_byte_shape(tuple(int(d) for d in reversed(t.shape)), t.tensor_type)
            writer.add_tensor_info(t.name, shape, raw_bytes.dtype, raw_bytes.nbytes,
                                   raw_dtype=t.tensor_type)
            raw_tensors[t.name] = raw_bytes
            continue
        shape = tuple(int(d) for d in reversed(t.shape))
        if t.name in f32_tensors:
            # GDN control tensors must stay F32 (CPU binary-op type mixing)
            writer.add_tensor_info(t.name, shape, np.dtype(np.float32),
                                   int(np.prod(shape)) * 4, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(t.name, shape, np.dtype(np.uint16),
                                   int(np.prod(shape)) * 2, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.write_header_to_file(path=args.out)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    # pass 2: quantize and stream data
    for t in reader.tensors:
        if t.name in raw_tensors:
            writer.write_tensor_data(raw_tensors[t.name])
            continue
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            b = np.array(t.data, copy=True).view(np.uint16)          # (N, K)
            w = torch.from_numpy(b).view(torch.bfloat16).float().numpy()
        else:                                                        # F32
            w = np.array(t.data, copy=True)
        w = np.ascontiguousarray(w)                                  # [N, K]
        for _axis, dim1_range, fmt in fmt_for(t.name, layout):
            if not fmt or not FORMATS[fmt]:
                continue
            qmin, qmax = FORMATS[fmt]
            if dim1_range is not None and w.ndim == 2:
                lo, hi = dim1_range
                w[lo:hi, :] = quant_dequant_axis(w[lo:hi, :], 1, qmin, qmax)
            else:
                w = quant_dequant_axis(w, 1, qmin, qmax)
            n_q += 1
        if t.name in f32_tensors:
            writer.write_tensor_data(w.astype(np.float32))
        else:
            bf = torch.from_numpy(w).to(torch.bfloat16).view(torch.uint16).numpy()
            writer.write_tensor_data(bf)

    writer.close()
    print(f"[{args.layout}] quantized {n_q} tensors -> {args.out}")


if __name__ == "__main__":
    main()
