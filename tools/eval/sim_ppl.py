#!/usr/bin/env python3
"""Simulated-PPL evaluator for NInfer A5000 fork weight layouts.

Applies the fork's exact groupwise-int quantization math (max-abs symmetric
round-to-nearest, fp16 group scales — same as tools/convert/common/quantize.py)
to every quantized tensor family of a layout, then measures LM loss of the
resulting dequantized model on Wikitext-2 test chunks (same 50x512 protocol as
the llama-perplexity IQ3_XXS anchor: PPL 6.2569).

Runs on CPU with torch, no NInfer build required. Quality gate BEFORE kernels.

Layouts are expressed per family; 'bf16' leaves the tensor untouched.
"""
import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

BF16_DIR = Path("/run/media/lio/data/g/qwen3.8-27b-bf16")
WIKI = Path("/tmp/wiki.test.raw")

FAMILY_TEMPLATES = {
    # family -> list of (source tensor name template, needs_transpose)
    "mlp_gate_up": ["model.language_model.layers.{L}.mlp.gate_proj.weight",
                    "model.language_model.layers.{L}.mlp.up_proj.weight"],
    "mlp_down":    ["model.language_model.layers.{L}.mlp.down_proj.weight"],
    "gdn_qk":      ["model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight"],
    "gdn_vz":      ["model.language_model.layers.{L}.linear_attn.in_proj_z.weight"],
    "gdn_out":     ["model.language_model.layers.{L}.linear_attn.out_proj.weight"],
    "fa_qk":       ["model.language_model.layers.{L}.self_attn.q_proj.weight"],
    "fa_gv":       ["model.language_model.layers.{L}.self_attn.v_proj.weight"],
    "fa_out":      ["model.language_model.layers.{L}.self_attn.o_proj.weight"],
}
FA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

FORMATS = {  # qmin, qmax, group
    "bf16": None,
    "q4": (-8, 7, 64),
    "q5": (-16, 15, 64),
    "q6": (-32, 31, 64),
    "q3": (-4, 3, 64),
}

LAYOUTS = {
    # current v2 artifact
    "v2":            {"mlp_gate_up": "q4", "mlp_down": "q4", "gdn_qk": "q4", "gdn_vz": "q5",
                      "gdn_out": "q4", "fa_qk": "q4", "fa_gv": "q5", "fa_out": "q4",
                      "embed": "q6", "head": "q6"},
    # min-Q4 target
    "minq4":         {"mlp_gate_up": "q4", "mlp_down": "q4", "gdn_qk": "q4", "gdn_vz": "q4",
                      "gdn_out": "q4", "fa_qk": "q4", "fa_gv": "q4", "fa_out": "q4",
                      "embed": "q4", "head": "q4"},
    # codex recommended 10.93 GiB layout
    "q3mlp":         {"mlp_gate_up": "q3", "mlp_down": "q3", "gdn_qk": "q4", "gdn_vz": "q4",
                      "gdn_out": "q4", "fa_qk": "q4", "fa_gv": "q4", "fa_out": "q4",
                      "embed": "q4", "head": "q4"},
    # endpoints-only sensitivity probe
    "v2_q4endpoints": {"mlp_gate_up": "q4", "mlp_down": "q4", "gdn_qk": "q4", "gdn_vz": "q5",
                      "gdn_out": "q4", "fa_qk": "q4", "fa_gv": "q5", "fa_out": "q4",
                      "embed": "q4", "head": "q4"},
}


def quant_dequant(w: torch.Tensor, fmt: str) -> torch.Tensor:
    spec = FORMATS[fmt]
    if spec is None:
        return w
    qmin, qmax, group = spec
    flat = w.detach().float().flatten()
    n = flat.numel()
    pad = (-n) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    scale16 = (g.abs().amax(dim=1) / qmax).clamp_min(1e-12).to(torch.float16)  # fp16 scales, as stored
    scale = scale16.float()
    q = torch.round(g / scale[:, None]).clamp_(qmin, qmax)
    dq = (q * scale[:, None]).view(-1)[:n]
    return dq.view(w.shape).to(w.dtype)


class Patched:
    """Loads shards lazily, quantizing on first touch, caching quantized tensors."""

    def __init__(self, layout: dict):
        self.layout = layout
        self.cache: dict[str, torch.Tensor] = {}
        self.index = json.loads((BF16_DIR / "model.safetensors.index.json").read_text())["weight_map"]
        self.shard_cache: dict[str, dict] = {}

    def tensor(self, name: str) -> torch.Tensor:
        if name in self.cache:
            return self.cache[name]
        shard = self.index[name]
        if shard not in self.shard_cache:
            self.shard_cache[shard] = load_file(str(BF16_DIR / shard), device="cpu")
            if len(self.shard_cache) > 2:  # keep at most 2 shards resident (~6GB)
                self.shard_cache.pop(next(iter(self.shard_cache)))
        t = self.shard_cache[shard][name]
        self.cache[name] = self.quantize_if_mapped(name, t)
        return self.cache[name]

    def quantize_if_mapped(self, name: str, t: torch.Tensor) -> torch.Tensor:
        # endpoints
        if name == "model.language_model.embed_tokens.weight":
            return quant_dequant(t, self.layout["embed"]) if self.layout.get("embed") else t
        if name == "lm_head.weight":
            return quant_dequant(t, self.layout["head"]) if self.layout.get("head") else t
        if ".layers." not in name:
            return t
        L = int(name.split(".layers.")[1].split(".")[0])
        for fam, templates in FAMILY_TEMPLATES.items():
            fmt = self.layout.get(fam)
            if not fmt or fmt == "bf16":
                continue
            is_fa_layer = L in FA_LAYERS
            is_gdn_fam = fam.startswith("gdn")
            if is_gdn_fam and is_fa_layer:
                continue
            if fam.startswith("fa") and not is_fa_layer:
                continue
            for tmpl in templates:
                if name == tmpl.format(L=L):
                    return quant_dequant(t, fmt)
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, choices=LAYOUTS)
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(16)
    layout = LAYOUTS[args.layout]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(BF16_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(BF16_DIR), torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    store = Patched(layout)

    # swap in quantized weights
    n_swapped = 0
    for name, param in list(model.named_parameters()):
        base = name
        t = store.tensor(base) if base in store.index else None
        if t is None:
            # tie fallback: lm_head may be tied off config
            continue
        if t.data_ptr() != param.data_ptr():
            with torch.no_grad():
                param.copy_(t)
            n_swapped += 1
    print(f"[{args.layout}] swapped {n_swapped} parameter tensors", file=sys.stderr)

    ids = tok(open(WIKI).read(), return_tensors="pt").input_ids[0]
    n_ctx = args.ctx
    total_tokens = args.chunks * n_ctx
    ids = ids[: total_tokens + 1]

    model.eval()
    nll = 0.0
    count = 0
    with torch.no_grad():
        for i in range(args.chunks):
            chunk = ids[i * n_ctx : (i + 1) * n_ctx + 1]
            logits = model(chunk[:-1].unsqueeze(0)).logits[0].float()
            lp = F.log_softmax(logits, dim=-1)
            tgt = chunk[1:]
            nll += -lp.gather(1, tgt.unsqueeze(1)).sum().item()
            count += n_ctx
            ppl = math.exp(nll / count)
            print(f"[{args.layout}] chunk {i+1}/{args.chunks} running PPL={ppl:.4f}",
                  file=sys.stderr, flush=True)
            gc.collect()

    ppl = math.exp(nll / count)
    line = f"{args.layout}: PPL = {ppl:.4f} over {count} tokens"
    print(line)
    if args.out:
        Path(args.out).write_text(json.dumps({"layout": args.layout, "ppl": ppl,
                                              "tokens": count}) + "\n")


if __name__ == "__main__":
    main()
