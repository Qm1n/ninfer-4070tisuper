# NInfer A5000 Fork — Full Project Record

*Status date: 2026-08-18. Everything below was measured on this machine unless marked otherwise.*

---

## 1. Origin & goal

- Original goal: max inference performance for `unsloth/Qwen3.8-27B-GGUF` on a
  16 GB RTX A5000 Laptop (sm_86).
- Unsloth Studio (llama.cpp backend) was tuned to its ceiling (see §2).
- Upstream NInfer (https://github.com/Neroued/ninfer) is a from-scratch CUDA/C++
  engine for Qwen checkpoints, locked to RTX 5090 (sm_120a). We forked and ported
  it to run Qwen3.8-27B on the A5000 with groupwise-int formats.
- Current goal: **smallest possible device weights → large context**, stepping
  down precision rung by rung, evaluating quality at each rung.

## 2. Baseline: Unsloth/llama.cpp (the thing to beat / fall back to)

Endpoint `http://127.0.0.1:8888` (Unsloth Studio), API key in `~/.credentials/unsloth.txt`.
Model cache: `~/.cache/huggingface/hub/` (standard HF cache; `~/.unsloth/` holds app only).
Start script: `~/bin/unsloth-start.sh` (starts studio + loads tuned config; `CTX=32768` variant for speed).

Tuned load config (via `POST /api/inference/load`):
```json
{"model_path":"unsloth/Qwen3.8-27B-GGUF","gguf_variant":"UD-IQ3_XXS",
 "max_seq_length":131072,"cache_type_kv":"q4_0",
 "n_ubatch":2048,"n_batch":4096,
 "gpu_memory_mode":"manual","gpu_layers":65,
 "llama_extra_args":["--cache-reuse","256"]}
```

Measured (128-token generations, IQ3_XXS weights, q4_0 KV):
| Config | Decode tok/s | Prefill tok/s | Context | VRAM |
|---|---|---|---|---|
| 262k ctx, auto placement | 2.1 | ~191 (→290 w/ ubatch 2048) | 262,144 | 13.3/16 |
| 131k ctx, manual all-65-layers | 14.5 | ~290 | 131,072 | 15.0/16 |
| 32k ctx, manual | 22.4 | — | 32,768 | 15.4/16 |

Key knob findings:
- `gpu_memory_mode:"manual","gpu_layers":65` forces full-GPU residency — auto leaves ~2 GB idle. 131k is the max ctx that fits fully on GPU (262k OOMs).
- `n_ubatch:2048`: prefill 191→290 tok/s. 4096 no further gain.
- q4_0 KV mandatory for >10k ctx (f16 KV caps at ~10k).
- MTP speculative decode in llama.cpp: draft 8 = 2.5× slower; draft 3 = wash. Off.
- `--cache-reuse 256` enables prefix caching. 4 parallel slots shared by UI+API.
- Server clamps requested ctx to model native (262,144); no YaRN passthrough.

## 3. The fork (~/ninfer-a5000)

Build: `cd ~/ninfer-a5000 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc`
(ninja via `pip install --user --break-system-packages ninja`; ffmpeg dev libs via apt;
system nvcc 12.4; driver 595.84/CUDA 13.2.)

Port changes (all marked with `Fork:` comments):
1. CMake arch guard → accepts 86-real; CUDA ≥12.4.
2. `src/core/pdl.cuh`: Programmatic Dependent Launch gated to sm_90+ (device no-ops,
   host launch without the attribute below sm_90). Correct, loses prologue overlap.
3. nvfp4 + fp8 kernel families excised (no FP4/FP8 tensor cores on sm_86; cuda_fp4.h
   absent in CUDA 12.4) → `src/ops/quant_disabled_stubs.cpp` (30 throwing stubs).
   `src/ops/linear/nvfp4/nvfp4_w4a4_tma_stub.cpp` replaces the TMA lib.
   CMakeLists lists trimmed accordingly.
4. gcc-13 libstdc++ shadowing fix: `link_libraries(/usr/lib/x86_64-linux-gnu/libstdc++.so.6)`
   in top CMakeLists (CUDA 12.4 toolchain dir leaks an old libstdc++ into links;
   it lacks `__cxa_call_terminate`).
5. `src/targets/qwen3_6/impl/runtime/layouts_impl.h`: device gate sm==120 → sm>=80.
6. Q6 endpoints for Qwen38GroupwiseInt profile (`endpoint_format()` in
   `src/targets/qwen3_6_27b/impl/load/bindings.cpp`).
7. `src/ops/linear_add/q4/q4_linear_add.{h,cu}` — NEW KERNELS: Q4 residual
   linear_add (epilogue-templated Q4 SIMT GEMM C4/C8; Q4 GEMV residual epilogue
   exists but the specialized wide-K GEMV returns partial dots at k=6144/17408 —
   only validated upstream at k=5120 — so T=1 routes through SIMT. ponytail note in file).
   Wired into `src/ops/wrapper/linear_add.cpp` (dispatch + workspace + require_rowsplit_q4).
   Conformance test: `tests/ops/linear_add/test_q4_a16.cpp` — **OK at {5120,6144} and
   {5120,17408}, T=1..256** against the FP32 oracle (test harness got a Q4G64F16S case).
8. mlp/down, attention/output, gdn/output → Q4 (engine binding + converter inventory
   `_a5000_endpoint_and_down()` in `tools/convert/qwen3_8_27b/inventory.py`).

## 4. Artifacts & sources (external drive /run/media/lio/data/g/)

- `qwen3.8-27b-bf16/` — full HF BF16 source (18 shards, 54 GB). KEEP for reconversions.
- `qwen3_8_27b.ninfer` — first build (Q4 down attempt before kernels; mismatched, obsolete).
- `qwen3_8_27b_q5down.ninfer` — Q5 down + Q6 endpoints (16.29 GiB file). Loads, runs; too big for headroom.
- `qwen3_8_27b_a5000.ninfer` — Q4 down + Q6 endpoints + Q5 outputs (15.62 GiB file). Works.
- `qwen3_8_27b_a5000_v2.ninfer` — Q4 down + Q4 outputs + Q6 endpoints (15.41 GiB file). **Current best.**
  Device weights 14.36 GiB (v2). Coherent output verified ("capital of France → Paris").
- Conversion: `cd ~/ninfer-a5000 && <studio python> -m tools.convert.qwen3_8_27b.convert --model <bf16 dir> --out <path> --device cpu` (~4 min, CPU; GPU OOMs if llama-server is running).

Current v2 measured ceilings (int8 KV, graphs on):
| Mode | Max ctx that loads | Decode |
|---|---|---|
| no-spec | 16,384 (24,576 fails: needs 1.34 GB, 1.10 free) | 13.7 tok/s @ 128-tok gen |
| MTP3 | 4,096 (6,144 fails) | 22.0 tok/s |

Prefill: 92 tok/s (SIMT-only large-T path; Q4 MMA has no residual epilogue — known debt).

## 5. Current format map (v2, device-resident 14.36 GiB)

64 layers (16 FA @ layers 3,7,...,63; 48 GDN) + optional MTP layer (host unless --spec mtp):
- Q4G64_F16S (0.53125 B/elem): mlp/gate_up {34816,5120}, mlp/down {5120,17408},
  gdn/query_key {4096,5120}, attention/query_key {7168,5120}, attention/output {5120,6144}, gdn/output {5120,6144}
- Q5G64_F16S (0.65625): gdn/value_z {12288,5120}, attention/gate_value {7168,5120}
  (fused q4_q5 attn/gdn input kernels hard-require Q5 on the second operand)
- Q6G64_F16S (0.78125): token_embedding + output_head {248320,5120} (~1.85 GiB both)
- BF16: norms, gdn conv/a/b projections; FP32: small control tensors.

KV: int8 group-64 = **33,792 B/token** (16 FA × 8 kv-heads × 128 dim × 2 × (1 + 2/64) — includes fp16 scale planes).
Vision (0.27 GiB): host-placed unless --vision. MTP package 0.42 GiB device only with --spec mtp.
Draft head 0.33 GiB (131072-row shortlist, Q4).

## 6. Codex review (gpt-5.6-sol, xhigh, full report at ~/ninfer-a5000/SMALLER-WEIGHTS-REVIEW.md)

Ranked reductions (value/effort):
1. **output head Q6→Q4**: 0.296 GiB, ~1-2 days. Just dispatch admission —
   `select_q4_a16_launch()` in `src/ops/linear/q4/q4_dispatch.cpp` lacks n=248320
   (kernels are row-count generic; do NOT use draft_head_small_t — compiled for 131072 rows).
   Split `endpoint_format()` into embed vs head decisions.
2. **token embedding Q6→Q4**: 0.296 GiB, 2-3 days. New Q4 route in
   `src/ops/launcher/embed_gather.cu` + `src/ops/kernel/embed_gather.cuh` + wrapper
   `src/ops/wrapper/embedding.cpp` (currently BF16/Q6/W8/FP8 only). Gather kernel, not GEMM.
3. Q5 operands → Q4 (value_z 0.352 + gate_value 0.068 GiB): 4-7 days. Fused q4_q5
   kernels are hard-wired (wrapper admission + small_t/fused .cu files + grouped MMA
   codec is closed Q4/Q5 with hard-coded Cr[...,32]). Needs Q4/Q4 leaf families.
   Swapping operand order saves ~0 (equal shapes) / 0.234 GiB (GDN) — pointless.
4. Q3 MLP (gate_up+down): 1.99 GiB. Full Q3 format + linear_swiglu/q3 + linear_add/q3.
5. Q3 remaining Q4 matrices: +0.42 GiB. Q3/Q4 attn+gdn input, snapshot/record.
6. Recommended target layout: Q3-MLP + Q4-everything-else (incl. endpoints) = **10.931 GiB**
   → theoretical 161k tokens raw, ~100k executable after runtime overheads.
   All-Q3 = 10.215 GiB (not recommended: endpoints at 3 bits = quality risk for 0.72 GiB).

Rejected by codex (with reasons): main-head shortlist (semantics change — proposals are
verified by full head; shortlist can miss true argmax, no fallback bound), tied embed/head
(checkpoint tensors differ), G128/E4M3 scales (0.3 GiB for kernel-wide reindexing; grouped
MMA asserts group=64), W8/FP8 endpoints (bigger), lossless compression (disk not VRAM),
CPU streaming (PCIe per token).

Q3G64 design (codex): 16B low2-plane + 8B sign-plane + 2B fp16 scale = 26B/group.
`row_view()` in bindings.cpp hard-codes Q4/Q5/Q6 plane widths — must become geometry-aware.
Converter `_pack_low_nibbles`/`_pack_high_bits(bits-4)` can't express Q3 — new packer.
Effort: 23-38 eng-days production; 1-2 weeks for format + generic-linear prototype.

KV/token corrections: 33,792 B (not 32 KiB). 100k tokens = 3.147 GiB KV.

## 7. Quality: Q3 vs IQ3_XXS (measured today)

**Anchor:** built llama-perplexity from ~/.unsloth/llama.cpp (fresh /tmp/lcpp-build,
GGML_CUDA, sm_86; `llama-server` target off). Wikitext-2 test via
`Salesforce/wikitext` parquet → `/tmp/wiki.test.raw` (1,285,622 chars).
```
IQ3_XXS (unsloth GGUF, GPU): PPL = 6.2569 ± 0.13   (50 chunks × 512 ctx)
```

**Weight-space sim of our exact Q3G64 math** (max-abs/64 symmetric RTN, the repo's
quantizer math), real layer-30 (GDN) & layer-7 (FA) BF16 tensors, float64:
| tensor | Q4 rel-err / cosine | Q3 rel-err / cosine |
|---|---|---|
| mlp gate_up | 11.06% / 0.99395 | 25.81% / 0.96847 |
| mlp down | 11.21% / 0.99378 | 26.13% / 0.96770 |
| gdn in_qkv | 11.30% / 0.99369 | 26.33% / 0.96720 |
| gdn out_proj | 11.78% / 0.99315 | 27.17% / 0.96512 |
| fa q_proj | 11.19% / 0.99381 | 26.10% / 0.96778 |

Q3-RTN carries ~2.3× Q4's weight error. Uniform across families.

**Why IQ3_XXS is better at same bpw:** (1) 256-entry learned codebook vs 8 uniform
levels; (2) imatrix activation-weighted calibration (error budget spent where activations
don't look); (3) UD mixing — Q4/Q5/Q6 islands on sensitive tensors inside a ~3bpw average.
At 4 bits differences shrink; 3 bits is the cliff.

**Eval ladder:**
1. Weight-space sim (done).
2. Simulated PPL: monkeypatch HF model on CPU — per-tensor quant/dequant with Q3G64
   math, LM loss over same 50×512 wiki chunks. Direct compare vs 6.26 anchor.
   GO/NO-GO gate for Q3 kernels. (~half day + overnight CPU run.)
3. If marginal: calibrated scales (activation-weighted per-group scale choice;
   no kernel changes — still fp16/64) → re-measure.
4. End-task once kernels exist: repo's own methodology — ninfer-serve + EvalScope
   (AIME 2025/2026, GPQA-D). Upstream groupwise-int Qwen3.6: 86.67/93.33/86.87.

**Decision rule:** Q3 MLP kernels only if step-2 PPL within ~0.5 of 6.26.
If calibrated scales can't close: floor is all-Q4 layout ~12.4 GiB ≈ 60-70k ctx,
and Q3-class quality means codebook machinery (different project).

## 8. Known debts / bugs

- Q4 GEMV (specialized, wide-K) partial-dot bug at k=6144/17408 — T=1 routed via SIMT
  (correct, slower). Fix = the decode-speed win. Documented in q4_linear_add.cu.
- Prefill T>16 rides SIMT (Q4 MMA has no epilogue hook) — 92 vs ~290 tok/s potential.
- `~/ninfer-a5000` git: fork has uncommitted changes (no repo init was done — clone was
  from /tmp copy; consider `git init && git add -A && git commit`).
- Unsloth endpoint currently DOWN (llama-server killed for NInfer VRAM). Restart: `~/bin/unsloth-start.sh`.
- Disk: system drive 91% (84 GB free). Weights/tests live on /run/media/lio/data/g/.

## 9. Key file map

- Engine format admission: `src/ops/wrapper/*.cpp` (per-op), `src/targets/qwen3_6_27b/impl/load/bindings.cpp` (per-tensor).
- Quant formats: `src/core/tensor.h` (QType), `src/artifact/{reader,storage_layouts,typed_binding}.cpp`.
- Kernels: `src/ops/<family>/<format>/…` (q4/q5/q6/w8/bf16 survive; nvfp4/fp8 stubbed).
- Converter: `tools/convert/qwen3_8_27b/{inventory,convert}.py`, `tools/artifact/{numeric,layouts}.py`,
  `tools/convert/common/quantize.py` (quantize_matrix — the RTN math).
- Tests: `tests/ops/linear_add/` (oracle harness), `tests/ops/quantized_weight.h` (patterned weights).
- Benchmarks: `bench/` (upstream's own).

## 10. min-Q4 EXECUTED (2026-08-18 evening) — QUALITY VERDICT IN

New kernels (all oracle-verified on GPU, committed):
- Q4/Q4 fused attention-input + GDN-input families (src/ops/{attn,gdn}_input_proj/q4_q4/ —
  codex-drafted, I verified: OK attn_input_proj, OK gdn_input_proj, OK conv_snapshot,
  OK conv_record, after guarding nvfp4/fp8 test cases for non-sm_120 builds).
- Q4 output-head dispatch (n=248320 in q4_dispatch.cpp; head Q6→Q4, -0.296 GiB).
- Engine: bind_groupwise_text_layers parameterized second-operand format (Qwen3.8 → Q4).

Artifacts (external drive):
- qwen3_8_27b_q4head.ninfer — head Q4: 20k no-spec / 8k MTP3 ctx.
- qwen3_8_27b_minq4.ninfer (15.76 GiB file, ~13.76 GiB device text weights) — ALL-Q4
  except Q6 embedding: 36k+ no-spec / 22k MTP3 ctx ceilings. Coherent output verified.
  Decode 13.9 tok/s @32k; MTP3 25.8 tok/s @20k.

Sim-PPL evaluator (tools/eval/): quant_gguf.py applies fork quant math to the BF16 GGUF
(→ BF16 store, F32 control tensors preserved; orientation [N,K], groups axis 1, fused-row
slicing per family), sweep.sh chains llama-quantize --pure Q8_0 + llama-perplexity
(-ngl 30, 50×512 chunks). GGMLType=naming trap: use GGMLQuantizationType; tmux server
dies randomly — use setsid.

VERDICT (50 chunks, same protocol as anchor):
| layout | PPL | Δ vs fp32 6.012 | Δ vs IQ3_XXS 6.256 |
|---|---|---|---|
| v2 sim | 6.218 | +0.21 | -0.04 (BETTER) |
| min-Q4 sim | 6.299 | +0.29 | +0.04 (parity) |
| Q3-MLP+Q4 sim | 7.049 | +1.04 | +0.79 — FAILS ≤0.5 gate |
| all-Q3 sim | 8.028 | +2.02 | dead |

Consequences:
- min-Q4 is quality-free. THE layout. Remaining shrink path = Q4 embedding gather
  (Phase 2, -0.3 GiB → ~40k+ MTP3) then nothing without new formats.
- Q3-MLP rejected without activation-calibrated scales; even then needs to halve the gap.
  Codex's 10.93 GiB recommendation is dead at RTN quality.
- unsloth IQ3_XXS is genuinely good (imatrix+codebook >> RTN at 3 bits — confirmed empirically).

Current stack of record: ~/ninfer-a5000 @ "sim-PPL verdict" commit;
artifact /run/media/lio/data/g/qwen3_8_27b_minq4.ninfer.

## 11. DECODE/PREFILL SPEED PUSH (2026-08-18 night)

Root-caused the "wide-K Q4 GEMV partial dots": Q4GemvR1W8DirectSchedule hard-codes
StaticGroupsPerRow=80 (k=5120-only). Using it at k=6144/17408 silently truncates the
dot (groups>80 skipped). Fix: Q4GemvR1W8K6144Schedule (static 96) and
Q4GemvR1W8K17408Schedule (dynamic 0 — 272 groups exceed the static path's
16-groups/warp tile ceiling). q4_linear_add T=1 routes by K; q4_dispatch k=6144/17408
cases added properly (earlier attempt had nested them as n-cases — dead code, which
is why a "passing probe" lied). Also: unknown concurrent edits had landed post-9565d40
and broke the build — reverted to the oracle-verified tree (commit f1e4d3d).

New: q4_rowsplit_gemm_mma AddResidual template param + launch_q4_mma_r64_c128_residual
(q4_linear_add T>16 now rides MMA; SIMT stays 2..16).

MEASURED (minq4 artifact, all tests green):
| metric | before | after |
|---|---|---|
| plain decode @32k | 13.9 tok/s | 13.0-13.9 (bandwidth wall — see below) |
| MTP3 decode @20k | 25.8 tok/s | 25.2-26.0 (acceptance 63-64%) |
| prefill (2.6k prompt) | 92 tok/s | **377 tok/s** (--prefill-chunk 384-512) |

BANDWIDTH WALL: 13.9 tok/s x 15.7 GB device weights ≈ 218 GB/s ≈ A5000 effective
bandwidth. Plain decode cannot exceed ~14 tok/s at this weight size regardless of
kernels; MTP3 commits ~1.9 tokens per weight pass → 26 tok/s is the ceiling.
Draft sweep: 1:18.3(83%), 2:18.9(61%), 3:25.6(64%), 4:15.7(44%), 5:14.1(35%) — 3 optimal.
The ONLY decode levers left: fewer weight bytes (quality-gated: Q3 dead at RTN) or
higher acceptance (architectural).

Operational: prefill chunks >~600 tokens hit bf16_gdn_gating_proj cooperative-launch
block limit (upstream kernel) — always pass --prefill-chunk 384 or 512.

## 12. TARGETED OFFLOAD: pinned-host embedding (2026-08-18 late)

The one offload that costs nothing: token_embedding {248320,5120} Q6 (~0.93 GiB) is read
ONE ROW (~10 KiB) per token by the gather — pure PCIe scalar loads, invisible next to the
12.7 GiB weight stream. Mechanism: Binder::retain_pinned_on_host (tensor-descriptor variant
of retain_on_host) -> materializer cudaHostAlloc + memcpy, pointer stored in the object's
`device` slot (UVA: device kernels dereference it zero-copy; device_data/materialized_weight/
row_split_weight all work unchanged); MaterializedArtifact dtor cudaFreeHosts.

MEASURED (minq4 artifact, tests green, coherent output):
| metric | device-only | pinned-embed |
|---|---|---|
| device weights | 13.64 GiB | 12.71 GiB |
| ctx ceiling no-spec | 36k | **61k** (73.7k fails) |
| ctx ceiling MTP3 | 22k | **49k** (57.3k fails) |
| MTP3 decode @40k | — | 25.7 tok/s (unchanged wall) |
| prefill | 377 | 382 tok/s (unchanged) |

General weight offload remains dead (bandwidth wall: any per-token-read tensor on host
costs 10x its share). KV host offload not attempted (paged-device design; llama.cpp
already sells that trade as its 262k mode).

Daily-driver command updated:
./apps/ninfer .../qwen3_8_27b_minq4.ninfer --kv-dtype int8 --max-context 49152 \
  --kv-capacity 49152 --spec mtp --draft-tokens 3 --prefill-chunk 384

## 13. LAST SPEED ITEMS (2026-08-18)

- --spec dflash: not supported by this target (runtime error). Dead.
- --lm-head-draft: +3% MTP3 decode (26.4 vs 25.7 tok/s @40k, repeatable), but costs
  workspace: ceiling 43k vs 49k. Use when speed > window; default stays without.
- Plain decode @49k: 12.8 tok/s (KV reads grow with context; expect ~-8%/10k tokens).

REMAINING IDEAS, honestly ranked (none are big):
1. Calibrated Q3-MLP scales (imatrix-style, converter-only): IF it recovers ~half the
   +0.79 PPL gap → 11.7 GiB → +8% decode. ~1 day + eval; quality-gated, may fail gate.
2. Fix bf16_gdn_gating_proj cooperative-launch limit → larger prefill chunks (384→1024+);
   likely small (384/512/1024 all ~375-380).
3. sm_86 lacks PDL; launch-overlap inside decode rounds is already amortized by CUDA graphs.
4. KV reads at large ctx are irreducible on-device (paged int8 already).
Everything else = hardware.

## 14. 128K CONTEXT PLAN (codex review + quality gate, 2026-08-19)

Quality gate (llama.cpp anchor, IQ3_XXS weights, 4k ctx, 6 chunks):
int8 KV PPL 5.8886 vs q4_0 KV PPL 5.8919 → +0.003. 4-bit KV is quality-free here.

Codex findings (full report: CONTEXT-128K-REVIEW.md):
- Geometry correction: 4 KV heads x 256 dim (not 8x128). INT4-G64 KV = 17,408 B/tok
  → 2.125 GiB at 131,072. PLAIN 128k fits with ~273-282 MiB margin (2.92 avail).
- MTP3 128k: short ~290 MiB after INT4; recoverable in theory via host-resident GDN
  rewrite checkpoint (-147 MiB, decode-neutral) + chunk 64 (-11) + graph-allowance
  calibration (-82) + G128 (-68) → ~18 MiB margin. Separate phase.
- Kernel route: Q8 x unpacked-K4 — keep mma.sync s8 QK path, unpack K nibbles to smem,
  V nibbles to bf16 smem; DRAM KV traffic halves → decode at long ctx IMPROVES.
  Do NOT add DType::I4 (element sizing assumes integral); use PagedKVEncoding enum
  + U8 code planes [128,64,heads,pages] + fp16 scales [4,64,heads,pages].
- GDN state: 2 slots x 146.8 MiB/lane (current + rewrite checkpoint) — checkpoint is
  the host-offload candidate; hot slot stays device (288 MiB/token PCIe if moved).
- KV host offload: definitively NO-GO (6.1 tok/s ceiling at 128k over PCIe).
- 131,072 = 2,048 pages x 64: zero rounding waste.

RANKED: 1) INT4-G64 KV (GO, 10-14 eng-days est), 2) graph allowance calibration,
3) host GDN rewrite checkpoint (MTP phase), 4) chunk 64, 5) G128 conditional,
6-8) NO-GO (scale codec, KV offload, hot-state offload).

## 15. INT4-G64 KV SHIPPED — 128K CONTEXT ACHIEVED (2026-08-19)

Implementation (codex, qualified on GPU by me): PagedKVEncoding{Bf16,I8G64,I4G64} threaded
through paged_kv_cache/decoder_state/layouts_impl/gqa wrappers; kernels = Q8-G64 on-chip +
packed K4/V4 loads unpacked to smem (QK stays mma_s8, V to bf16); CLI --kv-dtype i4.
Codec bit-exact vs oracle (verify_exact on code+scale planes pass); output criterion
calibrated to i4 envelope (18x coarser codes; worst observed deviation 0.58%, criterion
10x i8 profile with derivation comment). llm anchor: q4_0-KV PPL +0.003 @4k.

MEASURED (minq4 + pinned-embed):
| mode | kv | max ctx | decode | note |
|---|---|---|---|---|
| plain | int8 | 73,728 | 12.8 @32k | old |
| plain | i4 | **131,072** | **12.6 @131k** | speed FLAT vs 32k (unpack ~1%) |
| MTP3 | int8 | 49,152 | 25.7 | acceptance 64.3% |
| MTP3 | i4 | ~90,112* | 21.3 | acceptance drops 64->48% -> speed loss; prefer int8 |
(*114,688 fails reservation; 98,304 passes check but hits late cudaMalloc in w8_small_t —
 reservation underestimates MTP workspace, pre-existing issue.)

RECOMMENDED CONFIGS:
- Max window:  --kv-dtype i4 --max-context 131072 --prefill-chunk 384        (12.6 tok/s)
- Max speed:   --kv-dtype int8 --max-context 49152 --spec mtp --draft-tokens 3 (25.7 tok/s)

Remaining 128k-MTP3 path (if ever needed): host GDN rewrite checkpoint (-147 MiB) +
graph-allowance calibration (-82) + G128 (-68) per codex review; acceptance risk remains.

## 16. PHASE 2 SHIPPED — MTP3 AT 120K CONTEXT (2026-08-19/20)

Codex phase-2 (GPU-qualified by me, all 8 test suites green, commit 56575e0):
- Host-pinned GDN rewrite checkpoints (-146.8 MiB/lane device; D2H capture / H2D restore only)
- Graph-allowance calibration: startup topology capture, observed+24 MiB, one 48 MiB retry
- Prefill chunk 64 admitted (-11.5 MiB)
- i4-G128 KV: --kv-dtype i4 (G128); i4-g64 keeps the old route (-71.3 MiB at 131k)
- Item 3 (phantom w8_small_t cudaMalloc) was a misdiagnosis — that line is an error check;
  the calibration capture now warms lazy allocations into the sizing instead.
- 128k MTP3 still does NOT fit (KV+hot GDN alone exceed free VRAM; -136 MB after all items).

MEASURED CEILINGS AND SPEEDS (minq4 artifact):
| config | max ctx | decode |
|---|---|---|
| MTP3 + i4-G128 | **122,880 (120k)** | **22.4 tok/s** (acceptance 51% on bench prompt) |
| MTP3 + i4-G64 | 112,640+ (122,880 misses by 7.3 MB) | 21.4 tok/s @100k |
| MTP3 + int8 | 49,152 | 25.7 tok/s |
| plain + i4 | 131,072 | 12.6 tok/s |

Headline: 120k @ 22.4 tok/s = +54% over unsloth's 131k @ 14.5 tok/s at near-equal window.
Acceptance is prompt-dependent (48-67% observed at i4; int8 anchors 64%).

Commands:
  speed@120k: ./apps/ninfer .../qwen3_8_27b_minq4.ninfer --kv-dtype i4 --max-context 122880 \
    --kv-capacity 122880 --prefill-chunk 64 --spec mtp --draft-tokens 3
  speed@49k:  --kv-dtype int8 --max-context 49152 ... --spec mtp --draft-tokens 3
  plain@128k: --kv-dtype i4 --max-context 131072 --prefill-chunk 384 (no --spec)
Remaining ideas for MTP@128k: none credible — needs ~136 MB that only weight-size cuts
(Q3, quality-dead) or KV bits (below 4) could give.
