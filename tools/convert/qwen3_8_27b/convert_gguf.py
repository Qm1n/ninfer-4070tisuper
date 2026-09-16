"""Convert a Qwen3.8-27B GGUF checkpoint pair into one NInfer artifact.

The registered converter builds the artifact from an official safetensors checkpoint.  This entry
point keeps the same recipe, object plan, numeric formats, and artifact identity, but reads the
weights from the distributed GGUF files instead: the text and MTP tensors from the main file and
the vision tower from the matching mmproj projector.

Canonical invocation::

    python -m tools.convert.qwen3_8_27b.convert_gguf \\
      --frontend /path/to/Qwen3.8-27B \\
      --gguf /path/to/Qwen3.8-27B-...gguf \\
      --mmproj /path/to/mmproj-Qwen3.8-27B-BF16.gguf \\
      --out out/qwen3_8_27b.ninfer

The frontend directory supplies config.json, the six registered frontend resources, and the
tokenizer used for the draft-head shortlist; those files must match the official hashes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.gguf.names import GgufEntry, build_source_mapping
from tools.convert.gguf.reader import GgufFile
from tools.convert.gguf.source import GgufSourceReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.recipe import preflight_source_reader
from tools.convert.qwen3_6_27b import convert as qwen3_6_convert
from tools.convert.qwen3_6_27b import draft_head, recipe
from tools.convert.qwen3_6_27b import inventory as text_inventory

from . import convert as registered
from . import inventory

RECIPE_ID = "qwen3_8_27b-gguf-v1"


@dataclass(frozen=True, slots=True)
class SourcePlan:
    text_layers: int
    full_attention_layers: tuple[int, ...]
    vision_layers: tuple[int, ...]
    mapping: dict[str, GgufEntry]


@dataclass(frozen=True, slots=True)
class GgufPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[object, ...]
    draft: draft_head.DraftHeadContext
    object_plan: object
    plan: SourcePlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def plan_sources(text_gguf: GgufFile, vision_gguf: GgufFile) -> SourcePlan:
    """Derive the layer schedule from the container and map every recipe source."""

    metadata = text_gguf.metadata
    block_count = int(metadata.get("qwen35.block_count", 0))
    nextn = int(metadata.get("qwen35.nextn_predict_layers", 0))
    if block_count <= 0:
        raise ValueError("the GGUF metadata does not declare qwen35.block_count")
    text_layers = block_count - nextn
    expected_layers = len(inventory.FULL_ATTENTION_LAYERS) + len(inventory.GDN_LAYERS)
    if text_layers != expected_layers:
        raise ValueError(
            f"the GGUF declares {text_layers} text layers, but the registered recipe expects "
            f"{expected_layers}"
        )
    stable = metadata.get("general.architecture")
    if stable != "qwen35":
        raise ValueError(f"unexpected GGUF architecture {stable!r}")

    full_attention = tuple(inventory.FULL_ATTENTION_LAYERS)
    vision_layers = tuple(text_inventory.VISION_LAYERS)
    for layer in range(text_layers):
        has_attention = text_gguf.has(f"blk.{layer}.attn_q.weight")
        expects_attention = layer in full_attention
        if has_attention != expects_attention:
            raise ValueError(
                f"layer {layer}: the GGUF attention tensors do not match the registered schedule"
            )
    for layer in vision_layers:
        if not vision_gguf.has(f"v.blk.{layer}.attn_qkv.weight"):
            raise ValueError(f"the projector is missing vision layer {layer}")

    mapping = build_source_mapping(
        text_layers=text_layers,
        full_attention_layers=full_attention,
        vision_layers=vision_layers,
    )
    required = set(recipe.source_requirements())
    mapped = set(mapping)
    if required != mapped:
        missing = sorted(required - mapped)[:6]
        extra = sorted(mapped - required)[:6]
        raise ValueError(f"GGUF source mapping mismatch: missing={missing} extra={extra}")

    return SourcePlan(
        text_layers=text_layers,
        full_attention_layers=full_attention,
        vision_layers=vision_layers,
        mapping=mapping,
    )


def preflight_conversion(
    frontend_dir: str | Path,
    reader: GgufSourceReader,
    plan: SourcePlan,
) -> GgufPreflight:
    model = Path(frontend_dir)
    registered.preflight_inventory()
    config = family_conversion.load_json(model / "config.json")
    config_summary = qwen3_6_convert.validate_config(config)
    source_preflight = preflight_source_reader(reader, recipe.RECIPE_SPECS)
    resources = registered.load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = registered.build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return GgufPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source_preflight,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
        plan=plan,
    )


def convert(
    frontend_dir: str | Path,
    gguf_path: str | Path,
    mmproj_path: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)

    with GgufFile(gguf_path) as text_gguf, GgufFile(mmproj_path) as vision_gguf:
        plan = plan_sources(text_gguf, vision_gguf)
        reader = GgufSourceReader(
            {"model": text_gguf, "vision": vision_gguf},
            plan.mapping,
        )
        preflight = preflight_conversion(frontend_dir, reader, plan)
        print(
            f"preflight complete: {len(preflight.object_plan.objects)} objects, "
            f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
            flush=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        resources = {resource.name: resource.data for resource in preflight.resources}
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = registered.materialize_tensor(spec, reader, preflight.draft)
                    payload = registered.encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 25 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "frontend": str(frontend_dir),
        "gguf": str(gguf_path),
        "mmproj": str(mmproj_path),
        "out": str(out_path),
        "device": requested_device,
    }
    report = registered.build_conversion_report(
        model_dir=frontend_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report["recipe_id"] = RECIPE_ID
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    convert(args.frontend, args.gguf, args.mmproj, args.out, device=args.device)


if __name__ == "__main__":
    main()
