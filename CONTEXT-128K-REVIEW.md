codex
# Engineering report: 131,072-token context on A5000

## Bottom line

The technically credible route is:

1. Implement a native INT4-G64 paged KV cache.
2. Target 131,072-token plain decoding first.
3. Treat 131,072-token MTP3 as a separate memory-optimization phase.

INT4-G64 is sufficient for plain mode with approximately 273–282 MiB of margin under the supplied measurements. It is not sufficient for MTP3: that configuration remains about 290 MiB short.

INT4-G128 saves only another 64 MiB plain / 68 MiB with MTP3 and carries greater quality risk. It does not solve MTP3 by itself.

KV host offload is a no-go. Every full-attention layer reads its complete history; no sliding-window path exists for the 27B text model.

All calculations below assume `max_concurrency=1`, Vision disabled, MTP draft window 3 where stated.

## Exact memory arithmetic

The actual registered geometry is 4 KV heads × 256 dimensions, not 8 × 128. The total element count is identical:

```text
16 layers × 4 heads × 256 dims × K/V = 32,768 logical values/token
```

| Format | Exact bytes/token | 131,072-token Text KV |
|---|---:|---:|
| INT8-G64 | 33,792 B = 33 KiB | 4,429,185,024 B = 4.125 GiB |
| INT4-G64 | 17,408 B = 17 KiB | 2,281,701,376 B = 2.125 GiB |
| INT4-G128 | 16,896 B = 16.5 KiB | 2,214,592,512 B = 2.0625 GiB |

Therefore the supplied 2.19 GiB estimate for INT4-G64 is conservative; the exact result is 2.125 GiB.

For MTP3, the runtime allocates a seventeenth KV layer and one additional 64-token MTP page:

| Format | MTP3 total KV at 131,072 |
|---|---:|
| INT4-G64 | 2,424,377,344 B = 2.257877 GiB |
| INT4-G128 | 2,353,072,128 B = 2.191469 GiB |

Deriving fixed non-KV cost from the supplied rounded capacity observations:

- Plain fixed reserve: approximately 0.513–0.522 GiB.
- MTP3 fixed reserve: approximately 0.4954 GiB.

Resulting fit:

| Configuration | Required runtime budget | Available | Result |
|---|---:|---:|---|
| Plain INT4-G64 | 2.638–2.647 GiB | 2.92 GiB | Fits by 273–282 MiB |
| Plain INT4-G128 | 2.576–2.584 GiB | 2.92 GiB | Fits by 336–344 MiB |
| MTP3 INT4-G64 | 2.7533 GiB | 2.47 GiB | Short by 290 MiB |
| MTP3 INT4-G128 | 2.6869 GiB | 2.47 GiB | Short by 222 MiB |

These inferred fixed figures are based on two-decimal measurements. The authoritative decomposition is already exposed through `MemorySummary`: sequence capacity, workspace capacity, KV payload, graph allowance, and graph-observed bytes are reported separately in [main.cpp](/home/lio/ninfer-a5000/apps/cli/main.cpp:180).

## 1. Native INT4-G64 KV

### Current implementation boundaries

INT8-G64 is hard-coded through the complete route:

- Public format enum: [types.h](/home/lio/ninfer-a5000/include/ninfer/types.h:26)
- Cache view and physical plane descriptions: [paged_kv_cache.h](/home/lio/ninfer-a5000/src/core/paged_kv_cache.h:23)
- Four-plane INT8 layout planning: [decoder_state.cpp](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/state/decoder_state.cpp:14)
- Target planner mapping: [layouts_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/layouts_impl.h:689)
- Cache validation and shapes: [gqa_attention.cpp](/home/lio/ninfer-a5000/src/ops/wrapper/gqa_attention.cpp:54)
- Codec: [gqa_attention_kv_quant.cuh](/home/lio/ninfer-a5000/src/ops/kernel/gqa_attention_kv_quant.cuh:21)
- Decode kernel: [gqa_attention_decode_i8.cuh](/home/lio/ninfer-a5000/src/ops/kernel/gqa_attention_decode_i8.cuh:57)
- Prefill/fill kernels: [gqa_attention_prefill_i8.cuh](/home/lio/ninfer-a5000/src/ops/kernel/gqa_attention_prefill_i8.cuh:80)
- Decode and prefill dispatch: [gqa_attention_decode.cu](/home/lio/ninfer-a5000/src/ops/launcher/gqa_attention_decode.cu:114), [gqa_attention_prefill.cu](/home/lio/ninfer-a5000/src/ops/launcher/gqa_attention_prefill.cu:61)

The existing decode route quantizes Q to Q8-G64, keeps K as INT8 for `mma.sync ... s8`, and dequantizes only V to BF16. That is the performance property worth preserving.

### Recommended physical representation

Do not add a fractional `DType::I4`; tensor byte sizing assumes integral element sizes in [dtype.h](/home/lio/ninfer-a5000/src/core/dtype.h:8).

Use:

- Semantic encoding: `Int4Group64`
- Physical code planes: `DType::U8`
- Code shape: `[128, 64, kv_heads, physical_pages]`
- Scale planes: FP16 `[4, 64, kv_heads, physical_pages]`
- Signed symmetric codes `[-7, 7]`
- Scale `FP16_RNE(amax / 7)`
- Two signed nibbles per byte

Introduce an internal encoding enum—such as `PagedKVEncoding`—instead of overloading `DType::U8`. Thread that through `PagedKVLayerView`, `PagedKVBatchLayerView`, `DecoderStateSpec`, `SequencePlanImpl`, workspace planning, validators, and launch dispatch.

### Kernel design

The best SM86 route is Q8 × unpacked-K4, not native Q4 × Q4:

- Keep the existing on-chip Q8-G64 path.
- Load packed K4/V4 using half the DRAM traffic.
- Unpack K nibbles into INT8 shared memory.
- Reuse the existing `mma_s8` QK path.
- Unpack V directly into BF16/FP16 shared memory.
- Reuse the existing BF16/FP16 PV MMA.

The shared-memory arena can remain approximately unchanged:

```text
Current: K-i8 1D + V-i8 1D + V-bf16 2D = 4D bytes/key
INT4:    K-packed 0.5D + V-packed 0.5D + K-i8 1D + V-bf16 2D = 4D
```

The existing weight atom in [q4_rowsplit_storage.cuh](/home/lio/ninfer-a5000/src/ops/linear/q4/q4_rowsplit_storage.cuh:18) contains reusable signed-nibble decoding. It is not directly reusable as a KV atom because it assumes the weight row-split layout and scale addressing. Reuse the bit-level method in a KV-specific codec.

There is no suitable native signed-INT4 MMA helper elsewhere in this tree. Moving Q to INT4 would increase attention-logit error and is the wrong first implementation.

`gqa_attention_geometry.cuh` needs no head-geometry change. Its 85-split cap and page-ID envelope already cover 131,072 and 262,144 keys. Format-specific split/CTA tuning belongs in the launcher.

### Mandatory validation work

Update:

- Exact INT4 encode/decode oracle and A1/A2/A3 tests in [test_gqa_attention.cpp](/home/lio/ninfer-a5000/tests/ops/test_gqa_attention.cpp:323)
- Runtime plane inventory tests in [test_runtime_mechanisms.cpp](/home/lio/ninfer-a5000/tests/targets/qwen3_6/test_runtime_mechanisms.cpp:68)
- Python reference codec in [test_reference_ops.py](/home/lio/ninfer-a5000/tests/targets/qwen3_6_27b/test_reference_ops.py:44)
- Decode/prefill and append benchmarks
- CLI/server parsing, request logs, memory summaries, and documentation

Quality is the primary risk. llama.cpp Q4 KV is useful precedent, but q4_0 normally has finer block granularity than the proposed G64 route. At 256 dimensions, G64 supplies four scales per head; G128 supplies only two.

Qualification should include:

- Direct FP64 attention oracle at representative activation ranges.
- Exact append codec verification.
- Perplexity against INT8-G64.
- Long-context retrieval/NIAH at 32K, 64K, and 128K.
- MTP acceptance-rate comparison.
- End-to-end decode throughput at short and long frontiers.

If G64 quality fails, test G32 before G128. INT4-G32 is 18 KiB/token and still fits plain 128K with roughly 150 MiB margin.

## 2. Fixed VRAM and shaving opportunities

### Largest persistent allocation: GDN state

The runtime reserves two GDN state slots per concurrency lane: current frontier plus rewrite checkpoint, as defined in [linear_state_slots.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/linear_state_slots.h:9).

For one slot:

```text
Conv BF16:      48 × 10,240 × 3 × 2 B       =   2,949,120 B
Recurrent FP32: 48 × 128 × 128 × 48 × 4 B   = 150,994,944 B
Total                                           146.8125 MiB
```

Two slots cost 293.625 MiB per concurrency lane. At C=8 this alone is 2.294 GiB.

The hot current slot cannot be offloaded without adding roughly 288 MiB/token of bidirectional PCIe traffic and breaking graph-stable device addressing.

The rewrite checkpoint is different. It is only captured at a prompt checkpoint and restored during prefix reuse; current code copies it device-to-device in [text_context_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/text_context_impl.h:1284) and [program_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/program_impl.h:521).

Moving that checkpoint exactly to an existing `PinnedHostBuffer` would save 146.8125 MiB per lane with no steady-state decode traffic. It would add approximately one 147 MiB D2H copy at capture and one H2D copy at restore.

This is the strongest decode-neutral persistent-state reduction.

### Workspace

The workspace is phase-reused, not additive. It is planned in [layouts_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/layouts_impl.h:227).

Important findings:

- GQA small-T workspace stops growing once the 85-split cap is reached.
- One-token 27B attention partials at 85 splits are only about 1.01 MiB.
- No attention workspace scales linearly to 131,072.
- GDN chunk scratch is 8.523 MiB at chunk 128 and 4.262 MiB at chunk 64.
- Including persistent prefill roots, the text-prefill workspace peak is approximately 21.6 MiB at chunk 128 and 10.8 MiB at chunk 64 for the groupwise profile.
- Q4 projection scratch is not a hidden hundreds-of-MiB allocation for this route.

`--prefill-chunk 64` is mechanically plausible:

- GQA prefill tiles are 64.
- GDN’s hard chunk size is 64.
- Groupwise Q4 kernels admit T=64.
- The only direct blocker found is the target-level 128 alignment check in [layouts_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/layouts_impl.h:538) and duplicated target constant.

Expected saving is only around 11 MiB including `prefill_hidden`. It doubles the number of prefill rounds and may reduce prefill throughput, but it should not affect decode speed.

### CUDA Graph reservation

Graph memory is not owned by `graph_impl.h`; that file only launches or captures graphs. The reservation belongs to [layouts_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/layouts_impl.h:637):

- Plain: 12 MiB × concurrency.
- MTP: 82 MiB per topology class × concurrency for long contexts.

Actual driver consumption is measured by `cudaMemGetInfo` and reported separately in [program_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/program_impl.h:1384).

Calibrating the A5000 allowance to `observed + safety margin` is worthwhile. Disabling graphs is not acceptable without proving unchanged decode throughput. If the supplied MTP budget already used `--no-cuda-graph`, this lever has zero remaining value.

## 3. KV host offload

No-go.

Decode computes:

```cpp
window = last_pos + 1;
```

and partitions the entire `[0, window)` range in [gqa_attention_decode_i8.cuh](/home/lio/ninfer-a5000/src/ops/kernel/gqa_attention_decode_i8.cuh:178). Prefill similarly derives all key blocks from the absolute maximum query position.

At 128K, current INT8 KV traffic is:

- 4.125 GiB per target forward.
- Approximately 264 MiB per full-attention layer.
- 4.383 GiB including the MTP layer.

Even at an optimistic 25 GiB/s effective PCIe rate, 4.125 GiB requires 165 ms/token before weights or computation: at most 6.1 tok/s. A laptop link or UVA faulting behavior can readily produce the observed approximately 3.7 tok/s.

Layer prefetch, double buffering, or page streaming changes latency overlap but not total PCIe traffic.

The in-tree cyclic cache and SWA implementation are DFlash-only, BF16, head-dimension 128, and fixed to a 4096 window in [swa.cpp](/home/lio/ninfer-a5000/src/ops/wrapper/swa.cpp:16). Qwen3.8-27B does not support DFlash. Applying SWA or attention sinks to its full-attention layers would change model semantics, not deliver an exact 128K context.

## 4. Other details

### Scale compression

Current scales are hard-coded FP16 in the layout, validators, launchers, and kernels. An “INT8 scale” is not a direct dtype substitution: a real positive scale needs either:

- FP8 E4M3,
- logarithmic encoding, or
- an additional secondary scale/metadata level.

Halving scale storage saves:

- 512 B/token
- 64 MiB plain at 128K
- Approximately 68 MiB with MTP3

That is useful but not enough alone. On SM86 it also adds software conversion and dynamic-range/underflow risk. No-go for the first implementation; keep as an MTP contingency.

### Capacity and concurrency semantics

`--max-context` is a per-request logical ceiling. `--kv-capacity` is an aggregate physical page pool, not automatically `max_context × max_concurrency`.

The planner allows:

```text
minimum pages = max(ceil(max_context / 64), concurrency)
maximum pages = concurrency × ceil(max_context / 64)
```

as shown in [layouts_impl.h](/home/lio/ninfer-a5000/src/targets/qwen3_6/impl/runtime/layouts_impl.h:548).

Thus C requests share the physical capacity. To guarantee all C requests can simultaneously reach 131,072, the pool must actually contain C × 2,048 pages.

131,072 is exactly divisible by 64:

- Logical pages: 2,048.
- Payload rounding waste: zero.
- Block table at C=1: 8 KiB.
- MTP3 adds exactly one physical page per concurrency lane.

## Ranked plan

| Rank | Item | Value | Effort | Risk | Recommendation |
|---|---|---:|---:|---|---|
| 1 | Native INT4-G64 KV, Q8×unpacked-K4 | Saves exactly 2.0 GiB at 128K; plain fits | 10–14 engineer-days | Medium performance, high quality | **GO** |
| 2 | Calibrate CUDA Graph allowance to A5000 observed bytes | Up to 82 MiB MTP, low effort | 0.5–1 day | Low if margin retained | **GO**, conditional on graphs being enabled |
| 3 | Store rewrite-checkpoint GDN state in pinned host | Saves 146.8125 MiB/lane | 3–5 days | Prefix capture/restore latency and lifecycle complexity | **GO for MTP phase** |
| 4 | Admit prefill chunk 64 | Roughly 11 MiB | 1–2 days | Prefill throughput | **GO only as adjunct** |
| 5 | INT4-G128 | Additional 64/68 MiB | 1–2 days after G64 | Higher long-context quality risk | **CONDITIONAL GO** |
| 6 | One-byte scale codec | Additional 64/68 MiB | 3–5 days | Codec quality and SM86 conversion cost | **NO-GO initially** |
| 7 | KV host offload | Cannot preserve speed | Large | Fundamental PCIe lower bound | **NO-GO** |
| 8 | FP32 GDN state reduction or hot-state offload | 72–147 MiB | Large | Numerical contract or decode-speed regression | **NO-GO** |

For MTP3, the best-case combination illustrates the remaining problem:

```text
Initial INT4-G64 shortfall                       ~290 MiB
Host rewrite checkpoint                          -147 MiB
Prefill chunk 64                                  -11 MiB
Remove entire MTP graph allowance                 -82 MiB
Remaining                                          ~50 MiB
Switch G64 -> G128                                 -68 MiB
Theoretical final margin                           ~18 MiB
```

That only works if the entire 82 MiB graph allowance is recoverable. With graphs retained, observed graph memory must be exceptionally small; if the baseline already disabled graphs, the combination does not fit.

## Final recommendation

- **GO:** build and qualify INT4-G64 for a 131,072-token plain profile.
- **NO-GO:** claim 131,072-token MTP3 from INT4 alone.
- **Conditional:** pursue MTP3 only after recording exact `sequence.capacity`, `workspace.capacity`, and `CUDA Graph observed/allowance` for the A5000. The likely route is host-resident rewrite checkpoints plus either G128 or compressed scales, but strict no-decode-regression is not established until CUDA Graph retention and end-to-end MTP throughput are measured.

No files were modified, and the unavailable GPU driver prevented independent runtime measurements; the conclusions use the supplied measurements and the live source tree.

Review complete