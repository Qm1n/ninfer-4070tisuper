codex
# Engineering conclusion

The strongest first production target is:

- all current Q4 text-core matrices → `Q3G64_F16S`;
- current Q5 attention/GDN secondary operands → Q4;
- token embedding and full output head → Q4.

That lands at **10.931 GiB** for resident text-core weights, down **3.424 GiB** from 14.355 GiB. It preserves Q4 rather than Q3 for the especially exposed vocabulary endpoints and the current Q5 operands.

A full symmetric-Q3 conversion reaches **10.215 GiB**, but the extra 0.716 GiB saving is unlikely to justify quantizing every endpoint and sensitive projection to three bits before model-quality measurements.

One correction to the context estimate: the repository’s INT8-G64 cache is **33,792 bytes/token**, not 32 KiB, once FP16 K/V scale planes are included. The recommended 10.931 GiB layout therefore has a theoretical weight-only capacity of about **161k tokens**, before CUDA context, graph, workspace, state, page rounding, and allocator overhead. At 100k tokens it leaves about **1.92 GiB** for those non-KV allocations.

## 1. Baseline sanity check

There are 64 main text layers:

- 16 full-attention layers: `3, 7, …, 63`;
- 48 GDN layers;
- the “65th layer” is the separate optional MTP layer and is not in the 14.355 GiB text-core baseline.

The converter authority is [`inventory.py`](/home/lio/ninfer-a5000/tools/convert/qwen3_8_27b/inventory.py:36), primarily `_a5000_endpoint_and_down()`.

| Text-core family | Format | Elements | Resident GiB |
|---|---:|---:|---:|
| Embedding + output head | Q6 | 2,542,796,800 | 1.8501 |
| 64 MLP gate/up | Q4 | 11,408,506,880 | 5.6445 |
| 64 MLP down | Q4 | 5,704,253,440 | 2.8223 |
| FA query/key | Q4 | 587,202,560 | 0.2905 |
| FA gate/value | Q5 | 587,202,560 | 0.3589 |
| FA output | Q4 | 503,316,480 | 0.2490 |
| GDN query/key | Q4 | 1,006,632,960 | 0.4980 |
| GDN value/z | Q5 | 3,019,898,880 | 1.8457 |
| GDN output | Q4 | 1,509,949,440 | 0.7471 |
| Norms/control tensors | BF16/FP32 | — | 0.0489 |
| **Total** | | | **14.3551** |

Text-core object counts are 256 Q4 tensors, 64 Q5 tensors, two Q6 tensors, 353 BF16 tensors, and 96 FP32 tensors.

The row-split byte rates are:

| Format | Bytes/element | Effective bits |
|---|---:|---:|
| Q4G64-F16S | `4/8 + 2/64 = 0.53125` | 4.25 |
| Q5G64-F16S | `5/8 + 2/64 = 0.65625` | 5.25 |
| Q6G64-F16S | `6/8 + 2/64 = 0.78125` | 6.25 |
| Q3G64-F16S | `3/8 + 2/64 = 0.40625` | 3.25 |
| Q3G128-F16S | `3/8 + 2/128 = 0.390625` | 3.125 |
| Q3G64-E4M3S | `3/8 + 1/64 = 0.390625` | 3.125 |
| Q3G128-E4M3S | `3/8 + 1/128 = 0.3828125` | 3.0625 |

All current text dimensions are compatible with 128-element K padding, so these totals do not hide significant tail padding.

## 2. Ranked reduction approaches

### 1. Output head Q6 → Q4

**Value:** 0.296 GiB.  
**Effort:** low, roughly 1–2 days including qualification.

No new GEMM implementation is required. [`q4_dispatch.cpp`](/home/lio/ninfer-a5000/src/ops/linear/q4/q4_dispatch.cpp:8) simply does not admit `{N=248320,K=5120}` today. Existing generic kernels accept runtime row counts:

- `launch_q4_gemv_r4_w1_direct` for `T=1`;
- generic Q4 SIMT for small `T`;
- generic Q4 MMA for larger verification/prefill widths.

Do not use `launch_q4_draft_head_small_t`: it is compiled around the 131072-row `Q4DraftHeadGeometry`.

Touch:

- `_a5000_endpoint_and_down()` in the Qwen3.8 inventory;
- split [`endpoint_format()`](/home/lio/ninfer-a5000/src/targets/qwen3_6_27b/impl/load/bindings.cpp:35) into embedding/head format decisions;
- `select_q4_a16_launch()` in `q4_dispatch.cpp`;
- Q4 linear tests for `{248320,5120}` at `T=1..8` and representative verification widths.

Risk is primarily speed: a 248320-row head is bandwidth-dominant, so the R4W1/SIMT/MMA crossover must be measured on A5000.

### 2. Token embedding Q6 → Q4

**Value:** 0.296 GiB.  
**Effort:** low-medium, roughly 2–3 days.

There is no existing Q4 gather route. [`embedding()`](/home/lio/ninfer-a5000/src/ops/wrapper/embedding.cpp:197) accepts BF16, Q6, W8, and FP8 only. [`embed_gather.cu`](/home/lio/ninfer-a5000/src/ops/launcher/embed_gather.cu:70) and [`embed_gather.cuh`](/home/lio/ninfer-a5000/src/ops/kernel/embed_gather.cuh:1) likewise have no Q4 decoder.

This needs a gather kernel, but not a GEMM kernel:

- add `require_q4_metadata()`;
- add `embed_gather_q4_launch()`;
- add grouped Q4 gather decoding two signed nibbles per byte;
- add embedding tests at real `D=5120`.

The gather touches only selected rows, so this is a relatively small kernel project.

### 3. Disable optional resident feature packages when unused

**Value:** up to approximately 1.028 GiB.  
**Effort:** none; already supported.

Current optional inventories are approximately:

- optimized proposal head: 0.3325 GiB;
- MTP package: 0.4203 GiB;
- Vision package: 0.2754 GiB.

[`bindings.cpp`](/home/lio/ninfer-a5000/src/targets/qwen3_6_27b/impl/load/bindings.cpp:449) already makes these validate-only unless the startup feature requires device placement. The 14.355 GiB baseline already excludes them, so this is operational guidance rather than further baseline saving.

### 4. Current Q5 operands → Q4

**Value:** 0.4199 GiB:

- GDN `value_z`: 0.3516 GiB;
- attention `gate_value`: 0.0684 GiB.

**Effort:** medium, approximately 4–7 days.

This is not an inventory-only flip. The public split APIs explicitly require Q4 first and Q5 second:

- [`attn_input_proj()`](/home/lio/ninfer-a5000/src/ops/wrapper/attn_input_proj.cpp:221);
- [`gdn_input_proj()`](/home/lio/ninfer-a5000/src/ops/wrapper/gdn_input_proj.cpp:720);
- GDN snapshot/record entry points at lines 910 and 965.

Small-T and fused paths are similarly fixed:

- `q4_q5_attn_input_small_t.cu`;
- `q4_q5_gdn_input_independent.cu`;
- `q4_q5_gdn_input_conv_snapshot.cu`.

The large-T grouped mechanism in [`rowsplit_grouped_mma.cuh`](/home/lio/ninfer-a5000/src/ops/common/rowsplit_grouped_mma.cuh:15) can already mark each job as Q4 or Q5, but wrapper admission and small-T/fused kernels prevent Q4/Q4 use.

The clean implementation is a Q4/Q4 semantic leaf family, reusing the grouped Q4 codec for both parents. Merely swapping formats gives poor returns:

- GDN qk Q4→Q5 and value_z Q5→Q4 saves only **0.2344 GiB**;
- attention operands have equal shape, so swapping them saves **zero**.

### 5. Q3 MLP gate/up first

**Value:** 1.3281 GiB.  
**Effort:** high.

This is the best Q3 value/effort slice because gate/up alone accounts for 11.4 billion elements. It requires:

- the complete Q3 artifact/codec foundation;
- a Q3 `linear_swiglu` family corresponding to `src/ops/linear_swiglu/q4/`;
- wrapper and planning admission in `linear_swiglu.cpp`.

It avoids Q3 attention/GDN input kernels initially.

### 6. Q3 MLP down

**Value:** 0.6641 GiB.  
**Effort:** medium-high after the Q3 foundation exists.

Implement `src/ops/linear_add/q3/` by following the current Q4 residual path. The fork’s [`q4_linear_add.cu`](/home/lio/ninfer-a5000/src/ops/linear_add/q4/q4_linear_add.cu:1) deliberately avoids the specialized partial-dot GEMV for K=6144/17408. Q3 should not blindly clone that limitation: qualify a correct wide-K `T=1` path and retain SIMT/MMA routes for larger T.

Q3 on both MLP families saves **1.9922 GiB**, producing 12.363 GiB before other reductions.

### 7. Q3 remaining current-Q4 text matrices

**Additional value after Q3 MLP:** 0.4199 GiB.  
**Effort:** high.

This adds Q3 attention query/key, GDN query/key, and attention/GDN output projections. It therefore requires:

- Q3/Q4 or Q3/Q5 attention input;
- Q3/Q4 or Q3/Q5 GDN input;
- Q3 output `linear_add`;
- snapshot and record routes.

All current Q4 text matrices converted to Q3 produce **11.943 GiB** while leaving Q5/Q6 unchanged.

### 8. Recommended combined layout

Current Q4→Q3, Q5→Q4, Q6 endpoints→Q4:

- **10.931 GiB**
- **3.424 GiB saved**

This is the first layout that cleanly reaches the stated range without putting Q3 on every sensitive operand.

A useful lower-effort staging point is Q3 only for both MLP matrices, plus Q4 endpoints and Q4/Q4 input projections: **11.351 GiB**.

### 9. Everything quantized as Q3G64-F16S

**Value:** 4.140 GiB total saving.  
**Result:** 10.215 GiB.  
**Effort:** highest; quality risk materially higher.

This needs Q3 embedding and the 248320-row Q3 output head in addition to every fused family.

A max-abs symmetric Q3 quantizer must not be treated as quality-equivalent to an IQ3_XXS GGUF merely because the byte counts are similar. The encoding and quantization strategy are different. It needs real checkpoint perplexity/task evaluation.

### 10. Larger groups or smaller scales

For the recommended mixed layout:

| Q3 variant on the former-Q4 matrices | Text-core GiB |
|---|---:|
| G64, FP16 scales | 10.931 |
| G128, FP16 scales | 10.629 |
| G64, E4M3 scales | 10.629 |
| G128, E4M3 scales | 10.479 |

These are poor first milestones:

- G128 saves only 0.302 GiB in the recommended mix, while every Q3 kernel must change scale indexing. The grouped MMA currently asserts `BK=group_size=64`.
- E4M3 scales save the same amount, but introduce a new scale dtype, conversion path, exact codec oracle, and possible SM86 decode overhead.
- combining both saves only another 0.151 GiB.

A quality-first Q3G32-F16S alternative is also possible: all quantized text matrices would be about **10.997 GiB**, but group-32 kernels are another distinct route.

### 11. Compress optional MTP/draft/Vision weights

Lower priority because these are startup-optional and outside the base total.

- Draft Q4→Q3 saves about 0.083 GiB.
- MTP W8→Q4 could save about 0.210 GiB, but requires admissions for its five exact shapes and affected semantic leaf families.
- Vision is only 0.275 GiB total; disabling it when unused is much better value than adding Vision-specific Q3 paths.

### 12. Approaches that do not provide a coherent win

- **Main-head shortlist:** changes generation semantics; details below.
- **Tie embedding and output head:** current recipes read distinct `embed_tokens.weight` and `lm_head.weight`. Aliasing is valid only if the checkpoint tensors are proven identical.
- **Drop the unused 243 vocabulary rows:** saves only about 1.9 MiB at Q6 and breaks the registered 248320-row shape.
- **FP8/W8 endpoints:** both are larger than Q6/Q4.
- **Lossless artifact compression:** reduces disk size, not device residency, unless every kernel decodes the compressed representation directly.
- **CPU/layer streaming:** major scheduler/materializer surgery and catastrophic per-token PCIe traffic.
- **Per-row/per-tensor scales:** could eliminate much of the roughly 0.78 GiB group-scale overhead, but substantially changes quantization error and all kernels.
- **IQ/codebook, sparse, low-rank, pruning, layer dropping, Q2:** entirely new formats/model transformations, calibration or retraining, and new kernels. These are much larger projects than symmetric Q3 and do not preserve the present checkpoint computation.

## 3. Concrete Q3G64-F16S design

### Storage encoding

Use the existing `RowSplitK128V1` plane model, but add format-specific Q3 geometry:

```text
64 signed codes:
    low-2-bit plane: 16 bytes
    sign/high plane: 8 bytes
    FP16 scale:      2 bytes
total:              26 bytes/group
```

For code `q ∈ [-4,3]`:

```text
u = q & 7
low2 = u & 3
sign = (u >> 2) & 1
decode: q = (u ^ 4) - 4
```

The converter should follow the current symmetric-style convention:

```text
scale = max(abs(group)) / 3
q = clamp(round(weight / scale), -4, 3)
```

A practical lane mapping for a warp decoding two adjacent values is:

- low byte: `low2[group*16 + lane/2]`;
- nibble shift: `(lane & 1) * 4`;
- sign byte: `high[group*8 + lane/4]`;
- sign shift: `(lane & 3) * 2`.

Keeping a 32-byte nibble plane would make decode easier but would consume exactly Q4 storage and defeat Q3.

### Artifact and core touch points

Add `Q3G64_F16S` to:

- [`QType`](/home/lio/ninfer-a5000/src/core/tensor.h:25);
- [`NumericFormat`](/home/lio/ninfer-a5000/src/artifact/reader.h:17);
- `parse_format()` in `src/artifact/reader.cpp`;
- `quant_geometry()`, `format_name()`, and `row_split_geometry()` in [`storage_layouts.cpp`](/home/lio/ninfer-a5000/src/artifact/storage_layouts.cpp:36);
- `storage_layout_for()`, `qtype_for()`, and `row_split_weight()` in [`typed_binding.cpp`](/home/lio/ninfer-a5000/src/artifact/typed_binding.cpp:13).

`Weight.qdata`, `Weight.qhigh`, `group_size`, and `scale_dtype` already have enough expressiveness.

Python/converter changes:

- add the format in [`numeric.py`](/home/lio/ninfer-a5000/tools/artifact/numeric.py:1);
- change [`layouts.py`](/home/lio/ninfer-a5000/tools/artifact/layouts.py:182):
  - `row_split_geometry()`;
  - `_pack_codes()` with a dedicated compact Q3 packer;
  - `_unpack_codes()`;
  - `_high_indices()`;
  - `_EAGER_DEQUANTIZERS`;
  - exact assemble/split/gather plane checks;
- current `_pack_low_nibbles()` and `_pack_high_bits(bits-4)` cannot represent Q3;
- the generic quantizer in `tools/convert/common/quantize.py` can otherwise reuse the existing max-abs/clamp flow after Q3 supplies `qmin=-4`, `qmax=3`.

Target changes:

- add `Q3` and role assignments to `tools/convert/qwen3_8_27b/inventory.py`;
- change exact formats in `bind_groupwise_text_layers()` and endpoint binding;
- make [`row_view()`](/home/lio/ninfer-a5000/src/targets/qwen3_6_27b/impl/load/bindings.cpp:138) geometry-aware. It currently hard-codes 32 low bytes/group and only Q5/Q6 high-plane widths, so it is incorrect for Q3’s 16+8 layout.

### Generic linear family

Create `src/ops/linear/q3/` mirroring the Q4 tree:

- `q3_rowsplit_storage.cuh`;
- `q3_rowsplit_gemv.{cu,cuh}`;
- `q3_rowsplit_gemm_simt.{cu,cuh}`;
- `q3_rowsplit_gemm_mma.{cu,cuh}`;
- optional endpoint-specialized small-T files;
- `q3_dispatch.{cpp,h}`;
- `q3_launch.h`.

Clone from Q4:

- dispatch and launch structure;
- tile/scheduling configurations;
- accumulation and output policy;
- generic SIMT/MMA scaffolding.

Clone from Q5:

- distinct `qdata`/`qhigh` validation;
- separate high-plane staging;
- dual-plane load structure.

Do not clone either decoder literally:

- Q4 assumes 32 code bytes and no high plane;
- Q5 assumes 32+8 bytes;
- Q3 is 16+8 bytes.

Add Q3 branches to `dispatch_linear()` and `linear_workspace_capacity_bytes()` in [`linear.cpp`](/home/lio/ninfer-a5000/src/ops/linear/linear.cpp:78).

### Semantic fused families

A production model route also needs:

1. `src/ops/linear_swiglu/q3/`
   - GEMV, small-T, MMA, plan, and wrapper admission.

2. `src/ops/linear_add/q3/`
   - residual epilogue across decode and batched widths.

3. `src/ops/attn_input_proj/q3_q4/`
   - small-T fixed projection;
   - grouped MMA;
   - plan and wrapper admission.

4. `src/ops/gdn_input_proj/q3_q4/`
   - ordinary projection;
   - snapshot fused route;
   - record fused route;
   - materialized fallbacks and workspace plan.

5. Q3 or Q4 embedding gather, depending on the selected endpoint format.

[`rowsplit_grouped_mma.cuh`](/home/lio/ninfer-a5000/src/ops/common/rowsplit_grouped_mma.cuh:15) is currently described and implemented as a closed Q4/Q5 mechanism. Either extend its job codec from a `bool q5` to an explicit Q3/Q4/Q5 codec and parameterize plane sizes, or create a Q3-specific grouped mechanism. The existing hard-coded `Cr[...,32]` prevents a simple enum-only extension.

Add every CUDA/C++ source explicitly to [`src/CMakeLists.txt`](/home/lio/ninfer-a5000/src/CMakeLists.txt).

### Tests and documentation

Required focused coverage:

- exact Q3 pack/unpack/encoded-size tests in artifact Python tests;
- C++ reader/materialized-weight geometry tests;
- independent FP32 dequantized linear oracle at all real shapes;
- Q3 `linear_swiglu` and `linear_add`;
- Q3/Q4 attention input;
- GDN ordinary, snapshot, and record transitions;
- embedding gather if used;
- A5000 route profiling at `T=1..8`, the 16/17 dispatch boundary, and representative prefill/verify widths.

Update:

- `docs/maintainer/tensor-formats.md`;
- `docs/maintainer/storage-layouts.md`;
- `docs/maintainer/artifact-container.md`;
- `docs/maintainer/qwen3.8-27b-artifact.md`;
- affected semantic headers under `include/ninfer/ops/`.

### Honest effort

For one experienced CUDA engineer:

| Work | Estimate |
|---|---:|
| Artifact format, converter, exact codecs/tests | 3–5 days |
| Generic Q3 linear + embedding/head admission | 4–7 days |
| Q3 SwiGLU and residual-add leaf families | 5–8 days |
| Attention and GDN, including snapshot/record | 6–10 days |
| Binding, docs, numerical/model/performance qualification | 5–8 days |
| **Production-quality total** | **23–38 engineering days** |

A format plus generic-linear microbenchmark prototype is a 1–2 week task. A complete model route that retains current fused scheduling and performance is realistically **4–7 weeks**, not a small format addition.

## 4. Head, embedding, and shortlist answers

- **Output head below Q6 without a new GEMM kernel:** yes. Q4 only needs a new exact dispatch admission and qualification.
- **Embedding below Q6 without a GEMM kernel:** yes, but it needs a new Q4/Q3 gather decoder. No such route exists today.
- **Reuse the draft shortlist for the main head:** not under the current semantics.

[`proposal_argmax()`](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/text_context_impl.h:551) uses the 131072-row head only for speculative proposals, which are subsequently verified by the full target. Ordinary decoding, target verification, and prefill sampling call the full `output_head` at lines 669, 727, 1173, and in [`sample_from_hidden()`](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/text_prefill_impl.h:138).

A main-head shortlist can omit the true argmax/top-k token. Penalties can also promote a token outside the shortlist. An exact fallback cannot know that it needs to run without evaluating excluded rows or having a certified bound, neither of which exists. Converting this from a draft optimization into the public main head would therefore be a model/serving semantic change plus sampling/remap redesign.

## 5. Capacity projections

These are weight-only upper bounds on a nominal 16 GiB device.

| Text weights | GiB | Raw tokens at 32 KiB | Raw tokens at actual 33,792 B |
|---|---:|---:|---:|
| Current | 14.355 | 53,901 | 52,267 |
| Q4 endpoints | 13.763 | 73,301 | 71,080 |
| All current Q4 roles → Q3 | 11.943 | 132,941 | 128,912 |
| **Recommended Q3/Q4/Q4 mix** | **10.931** | **166,101** | **161,067** |
| All Q3G64-F16S | 10.215 | 189,561 | 183,816 |
| All Q3G128-F16S | 9.824 | 202,373 | 196,241 |

The scale-inclusive KV equation, confirmed by [`paged-kv-cache.md`](/home/lio/ninfer-a5000/docs/maintainer/paged-kv-cache.md:357), is:

```text
2 × 16 layers × 8 heads × 128 code bytes
+ 2 × 16 × 8 × (128/64) × 2 scale bytes
= 33,792 bytes/token
```

Thus:

- 100k tokens need **3.147 GiB** of KV payload;
- recommended weights leave **5.069 GiB** before KV;
- after 100k KV, approximately **1.922 GiB** remains for runtime allocations.

The allocator works in 64-token pages and device residency also includes CUDA, persistent state, workspace, graph allowance, and alignment, so the raw 161k number is not an executable `max_context`. The 100k goal is plausible with the recommended layout, but must be validated through `MemorySummary`, not inferred only from subtraction.

One incidental consistency issue: the Qwen3.8 inventory module says the endpoints are W8 and its helper docstring says output remains Q5, while the actual map is Q6 endpoints and Q4 `*/output`. Those comments should be corrected with the inventory change.

No files were modified; this was a static repository review plus exact inventory arithmetic.

**