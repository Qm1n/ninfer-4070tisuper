# Qwen3.8-27B on a 16 GB RTX A5000: the full record

*How we went from 2.1 tok/s to 22.4 tok/s at 120k context, with quality held at
IQ3_XXS parity, and why unsloth/llama.cpp tops out where it does.*

Hardware throughout: one laptop — RTX A5000 16 GB (sm_86, ~220 GB/s effective
memory bandwidth measured, ~16 GB/s over PCIe), 64 GB system RAM, 16-core CPU.
Everything below was measured on this machine, most of it multiple times.

---

## 0. The physics that decided everything

You cannot reason about decode speed without this one equation:

```
decode tok/s  ≈  memory bandwidth  /  bytes of weights read per token
```

Every token requires reading every weight once. Our GPU streams ~218 GB/s
effective (measured, both engines agree within 10%). So:

| Weights resident | Theoretical max (plain) | Measured (plain) |
|---|---:|---:|
| 11.9 GB (unsloth IQ3_XXS) | ~18.3 tok/s | 14.5 |
| 13.6 GB (our min-Q4) | ~16.0 tok/s | 13.9 |

Two consequences that shaped every decision after:

1. **Any weight on the CPU pays 10–14×** (PCIe vs VRAM). Offloading 3 GB costs
   ~187 ms/token — we measured exactly this: unsloth at 262k context runs
   2.1 tok/s because ~3 GB spills to host. "Optimized engine with offload" is
   an oxymoron for tensors read every token.
2. **Speculative decoding (MTP) is the only way past the wall**, because each
   weight read can commit more than one token. If acceptance is a, you get
   roughly 1+a extra tokens per pass. That's why 25.7 tok/s was reachable with
   *more* weight bytes than unsloth's.

Everything else in this document is engineering to serve that equation:
shrink weights without quality loss (→ bigger KV budget), shrink KV bytes per
token without quality loss (→ bigger window), raise acceptance, and never pay
PCIe in the decode loop.

---

## 1. Baseline: tuning unsloth (llama.cpp) properly

The endpoint is Unsloth Studio (`http://127.0.0.1:8888`) spawning a managed
`llama-server` with the cached `unsloth/Qwen3.8-27B-GGUF` (UD-IQ3_XXS, 11.9 GB).

The stock load gave 262k context at 2.1 tok/s. Diagnosis via `/api/inference/*`
and `/api/system`: auto layer placement left ~2 GB VRAM idle and spilled the
rest to CPU. The tuned load (now in `~/bin/unsloth-start.sh`):

```json
{"gguf_variant":"UD-IQ3_XXS","max_seq_length":131072,
 "cache_type_kv":"q4_0","n_ubatch":2048,"n_batch":4096,
 "gpu_memory_mode":"manual","gpu_layers":65,
 "llama_extra_args":["--cache-reuse","256"]}
```

What each knob does and measured effect:

- `gpu_memory_mode:"manual","gpu_layers":65` — force all 65 layers on GPU.
  131,072 is the largest context where this fits. 2.1 → 14.5 tok/s. This one
  knob was worth ~7×.
- `cache_type_kv:"q4_0"` — with f16 KV only ~10k context fits; q4_0 KV is
  mandatory for large windows (and later proved quality-neutral, see §5).
- `n_ubatch:2048` — prefill 191 → ~290 tok/s (bigger GPU-side prompt batches).
- `--cache-reuse 256` — KV-shifting prefix cache: agent-style conversations
  skip re-prefilling the shared prefix.
- 262k is impossible fully on GPU (OOMs); the server silently clamps requested
  context to the model's native 262,144; no YaRN passthrough.

Result: **131k @ 14.5 tok/s decode, ~290 tok/s prefill.** This is the number
to beat. We also tried: VRAM budget 1.0 (marginal), draft-8 MTP (2.5× slower),
draft-3 MTP at this config (1.88 tok/s — measured while CPU-spilled, later
understood, see §6).

---

## 2. The NInfer fork: why and how it runs at all

NInfer (github.com/Neroued/ninfer) is a from-scratch CUDA/C++ engine for
specific Qwen checkpoints, published with RTX 5090 numbers (267–1,314 tok/s
decode with MTP3, 5k–15k tok/s prefill). It hard-requires sm_120a (Blackwell),
CUDA 13.1, and 32 GB VRAM — none of which we have. The point of the fork:
hand-written kernels, CUDA-graph-captured MTP rounds, and per-shape scheduling
that llama.cpp cannot express. The bet was that this machinery beats llama.cpp's
generic graph even on weaker hardware.

Porting work (all marked `// Fork:` in `~/ninfer-a5000`):

| Change | Why |
|---|---|
| CMake arch guard accepts `86-real`; CUDA ≥ 12.4 | build at all |
| `src/core/pdl.cuh`: Programmatic Dependent Launch gated to sm_90+ | sm_86 lacks PDL; plain stream launch is correct, loses overlap |
| nvfp4 + fp8 kernel families excised; 30 host stubs throw on use | sm_86 has no FP4/FP8 tensor cores; `cuda_fp4.h` doesn't exist in CUDA 12.4 |
| `link_libraries(...libstdc++.so.6)` | CUDA 12.4's gcc-13 dir leaks an older libstdc++ into links; missing `__cxa_call_terminate` |
| Device gate `sm()==120` → `sm()>=80` | runtime refusal was hardcoded |
| gcc-13/ninja/ffmpeg deps via pip+apt | build environment |

The groupwise-int kernel families (Q4/Q5/Q6/W8/BF16) compiled for sm_86 with
zero changes — the port cost days, not weeks, because the quant formats we
needed were portable; only the exotic ones weren't.

---

## 3. The model artifact: shrinking weights without quality loss

NInfer doesn't read GGUF. It reads its own container built from HF safetensors
by `tools/convert`. The official Qwen3.8-27B artifact is 16.96 GiB — doesn't
fit 16 GB. So we downloaded the BF16 source (54 GB, kept on the external
drive) and built custom artifacts, iterating the per-tensor format map until
quality/size/speed balanced.

### What the model actually is (matters for every choice)

64 main layers: 48 are GDN (gated-delta-net recurrent — **no KV cache, but
147 MiB fixed state per layer-slot**), 16 are full attention (4 KV heads ×
256 dim — **that's why KV is cheap here**). Plus an optional MTP layer (65th)
and a vision tower (host-resident when unused; we're text-only).

### Kernel-gated format evolution

The engine hard-codes which (op, shape, format) combinations exist. Each step
below required the kernel to admit it *before* the converter could write it:

| Artifact | Format map (device) | File | Context ceilings (int8 KV, plain/MTP3) | What it needed |
|---|---|---|---|---|
| q5down | Q4 down + Q6 endpoints + Q5 everything else | 16.29 GiB | ~8k / ~2k | nothing (loads) |
| a5000 | + Q4 `mlp/down` via new kernels | 15.62 GiB | 10k / 2k | **new Q4 residual linear_add** (see below) |
| v2 | + Q4 attention/GDN outputs | 15.41 GiB | 16k / 4k | same kernels, more shapes |
| q4head | + Q4 output head | 16.21 GiB | 20k / 8k | one dispatch entry (`n=248320` in `q4_dispatch.cpp`) |
| **minq4** | **all Q4 except Q6 embedding** | 15.76 GiB | 36k / 22k | **Q4/Q4 fused input kernels** (codex) |
| minq4 + pinned embed | same, embedding in host RAM | — | **73k / 49k** | pinned-host placement (see §4) |

The final device-resident weights: **12.71 GiB** (13.64 GiB of Q4 layers +
1.85 GiB Q6 embedding → 0.93 GiB embedding moved to host).

### Kernels we wrote

1. **Q4 residual linear_add** (`src/ops/linear_add/q4/`): the MLP
   down-projection fuses "matmul + residual add". Upstream only had Q5/BF16/W8
   variants. We cloned the Q4 SIMT GEMM with a read-modify-write epilogue
   (each fragment element has one owning thread — race-free by construction)
   and later added the same epilogue to the Q4 MMA kernel for large token
   counts. Oracle-verified: `OK Q4_A16 LinearAdd` at both real shapes,
   T=1..256.
2. **Q4/Q4 fused attention/GDN input projections** (`src/ops/{attn,gdn}_input_proj/q4_q4/`):
   these kernels take two weight parents; upstream required (Q4, Q5). Codex
   drafted the Q4/Q4 twins; I qualified them against the FP32 oracle — four
   suites: `OK attn_input_proj / gdn_input_proj / conv_snapshot / conv_record`.
3. **Wide-K GEMV fix**: the specialized decode GEMV
   (`Q4GemvR1W8DirectSchedule`) hard-codes `StaticGroupsPerRow=80`, i.e. valid
   only at K=5120. Used at our K=6144/17408 it silently summed only the first
   80 groups — outputs looked fluent but wrong (caught by the oracle, then
   root-caused with per-warp instrumentation: GPU value = sum minus warp 4).
   Fix: K-specialized schedules (static-96 for K=6144; dynamic for K=17408,
   since 272 groups exceed the static tile ceiling).
4. **INT4 KV kernels** (see §5).

### The quality harness — the thing that made shrinking safe

Before writing kernels for a smaller format, we needed to know what it costs.
We built a simulator that applies **the exact quant math of the converter**
(max-abs symmetric round-to-nearest, fp16 group-64 scales) to the real BF16
weights, writes a GGUF, and scores it with `llama-perplexity` on Wikitext-2
(50 chunks × 512 tokens — same protocol as the unsloth anchor):

| Layout (simulated) | PPL | Δ vs fp32 (6.012) | Δ vs unsloth IQ3_XXS (6.256) |
|---|---:|---:|---:|
| v2 (Q4+Q5+Q6) | 6.218 | +0.21 | **−0.04 (better)** |
| **min-Q4 (all Q4)** | **6.299** | **+0.29** | +0.04 (parity) |
| Q3 MLP + Q4 rest | 7.049 | +1.04 | +0.79 — fails gate |
| all-Q3 | 8.028 | +2.02 | dead |

Conclusions that saved weeks: **min-Q4 is quality-free; Q3 by round-to-nearest
is dead.** unsloth's IQ3_XXS is genuinely good — imatrix calibration + learned
codebooks matter exactly at 3 bits (we also measured the weight-error ratio:
Q3-RTN carries 2.3× Q4's error). Our Q4 floor is where naive quant stops; going
lower needs unsloth's calibration machinery, a different project.

---

## 4. Targeted offload: the one kind that's free

The embedding (`token_embedding`, 248,320 × 5120, Q6, ~0.93 GiB) is **not**
read wholesale per token — the gather reads one row (~10 KB). That's free over
PCIe. So: `cudaHostAlloc` the embedding, keep it in pinned host memory, and let
kernels dereference it via UVA (zero-copy). ~60 lines: a `retain_pinned_on_host`
binder path, materializer support, pointer stored in the device slot so every
existing view builder works unchanged.

Device weights 13.64 → 12.71 GiB; context ceilings jumped 36k→73k plain,
22k→49k MTP3. Decode speed unchanged (the per-token PCIe traffic is ~10 KB).

This is the *only* profitable offload on this machine. Everything else is read
every token (weights) or every token's whole history (KV), and both die on
PCIe arithmetic — we kept re-proving this (KV host-offload at 128k would cost
≥165 ms/token; maximum ~6 tok/s before weights).

---

## 5. KV cache: 33,792 → 16,896 bytes per token

Full-attention KV cost: 16 layers × 4 heads × 256 dim × 2 (K,V). The engine's
int8 group-64 cache = 33,792 B/token (codes + fp16 scale planes). At 128k
that's 4.13 GiB — didn't fit next to 12.71 GiB of weights.

Two gates before writing anything:

1. **Quality (llama.cpp anchor)**: same model, 4k context, int8 KV vs q4_0 KV:
   PPL 5.8886 vs 5.8919 — **+0.003**. 4-bit KV is a rounding error at this
   KV geometry. (Makes sense: per-group scales over 256-dim keys.)
2. **Engineering design** (codex review): keep Q quantized on-chip as int8,
   load packed K4/V4 (half the DRAM traffic), unpack nibbles in shared memory,
   keep the `mma_s8` QK path and bf16 PV path — so long-context decode gets
   *faster*, not slower, from the smaller cache.

Implementation: a `PagedKVEncoding` enum threaded through the KV stack (no
fractional DType — the tensor sizing assumes whole bytes; codes live in U8
planes, two signed nibbles per byte, scale = fp16(amax/7)). Codec verified
**bit-exact** against the oracle; output tolerance calibrated with derivation
(codes 18× coarser than int8; worst observed deviation 0.58%).

Results: plain 131,072 @ 12.6 tok/s, **speed flat from 32k to 131k** (halved
KV traffic cancels the growing KV reads). Phase 2 later added **i4-G128**
(`--kv-dtype i4`; g64 kept as `i4-g64`): scale plane halves again,
−71 MiB at 131k, exact oracle added, four bytes cheaper per token
(16,896 B/token).

---

## 6. Speculative decoding (MTP): where the real speed came from

The artifact contains a Qwen MTP head (an extra layer trained to draft).
`--spec mtp --draft-tokens 3` drafts 3 tokens and verifies them in one batched
pass. On our engine: acceptance ~64% (int8 KV), giving 25.7 tok/s @ 49k —
**+87% over plain**, at the same bandwidth wall.

Measured draft-depth sweep (acceptance / tok/s): 1 → 83%/18.3, 2 → 61%/18.9,
**3 → 64%/25.6**, 4 → 44%/15.7, 5 → 35%/14.1. Three is optimal; beyond it
acceptance decays faster than drafts pay.

**Why llama.cpp can't do this on this model — measured, not assumed.** We gave
llama.cpp every fair chance:

- Discovered that with speculation enabled, its llama-server **crashes with
  flash-attn and silently retries with FA off** — every earlier "MTP is a wash"
  measurement was running crippled kernels. Also the draft context OOMs unless
  `batch ≤ 512`.
- Properly configured: acceptance 45%, 20.9 tok/s — still below its own 22.4
  baseline. Root cause: the GGUF quantizes the draft head to IQ3_XXS with
  everything else.
- **Fixed that too**: surgical GGUF surgery (`tools/eval/upgrade_draft_head.py`)
  — copy the file, re-quantize only the 8 draft-layer tensors
  (`blk.64.*`) from the BF16 source to Q8_0 (+0.25 GB). Acceptance rose to
  62% — matching our engine, proving the diagnosis.
- **And it still lost** (20.5–21.3 tok/s at every draft depth). The residual
  is structural: llama.cpp's speculative rounds carry ~15–25 ms fixed
  orchestration per round for this GDN-hybrid model (recurrent-state
  checkpointing across 48 GDN layers, eager graph launches, KV rollback).
  Our engine pays ~1 ms because the rounds are captured CUDA graphs with
  replayable SSM state. That difference is the whole 25.7-vs-21 gap.

So: llama.cpp on this hardware/artifact is maxed at ~14.5–22.4 tok/s; its MTP
path cannot win here. The Q8-draft GGUF remains on the external drive as a
useful artifact, just not for llama.cpp's spec path.

---

## 7. Prefill: 92 → 377 tok/s

The prefill bottleneck was the same missing kernel families: large-T prompts
rode the SIMT GEMM because the Q4 MMA kernel had no residual epilogue. Adding
`AddResidual` as a template parameter on the Q4 MMA kernel (plus the earlier
wide-K GEMV fix) took prefill from 92 to 377 tok/s at ~2.6k-token prompts.

One upstream landmine documented: prefill chunks ≳600 tokens crash the
`bf16_gdn_gating_proj` kernel's cooperative launch (too many blocks) — always
pass `--prefill-chunk 384` (or 64, admitted in phase 2).

---

## 8. Phase 2: MTP at 120k — the memory endgame

MTP3 at 128k initially failed its reservation check by 413 MiB. The codex
review itemized where every byte goes; we implemented four savings:

| Change | Mechanism | Saving |
|---|---|---:|
| **Host GDN checkpoints** | Each lane holds two 146.8 MiB state slots (hot frontier + rewrite checkpoint). The checkpoint is only written at prompt capture and read at prefix restore — pinned host memory, D2H/H2D only. Decode loop untouched. | −146.8 MiB |
| **Graph-allowance calibration** | Fixed 82 MiB/topology reservation replaced by a startup capture that measures actual graph bytes, reserves observed + 24 MiB, retries once at +48 MiB on overrun. Also warms lazy allocations into sizing (kills the late-OOM class we'd blamed on a phantom malloc — that diagnosis was wrong; the referenced line was an error-check, not an allocation). | ~−60 MiB |
| Prefill chunk 64 admitted | alignment check 128 → 64; kernels already supported it | −11.5 MiB |
| **i4-G128 KV** | scale rows 4 → 2 per head-page (see §5) | −71.3 MiB @131k |

Result ladder (MTP3, min-Q4): 102,400 ✓ (G64, +92 MiB margin) → 112,640 ✓ →
**122,880 ✓ (G128)** → 126,976 fails by 14 MB → 131,072 needs ~136 MiB that
only weight cuts (Q3: quality-dead) or sub-4-bit KV could give. That's closed.

Acceptance note: i4 KV costs some acceptance on some prompts (48–67% observed;
int8 anchors 64%). It's prompt-dependent, and even the low end beat the wall.

---

## 9. Final numbers and how to reproduce

| Config | Context | Decode | Quality (PPL) |
|---|---:|---:|---|
| unsloth/llama.cpp tuned (`~/bin/unsloth-start.sh`) | 131,072 | 14.5 tok/s | 6.256 (IQ3_XXS) |
| unsloth, 32k variant | 32,768 | 22.4 tok/s | same |
| **Fork: MTP3 + i4-G128** | **122,880** | **22.4 tok/s** | 6.299 (min-Q4) |
| Fork: MTP3 + int8 | 49,152 | 25.7 tok/s | 6.299 (+int8 KV) |
| Fork: plain + i4 | 131,072 | 12.6 tok/s | 6.299 (+i4 KV ≈ +0.003) |
| Fork: prefill | — | 377 tok/s | — |

```bash
# 120k @ 22.4 tok/s
~/ninfer-a5000/build/apps/ninfer /run/media/lio/data/g/qwen3_8_27b_minq4.ninfer \
  --kv-dtype i4 --max-context 122880 --kv-capacity 122880 \
  --prefill-chunk 64 --spec mtp --draft-tokens 3 --greedy --prompt "..."

# 49k @ 25.7 tok/s            # 131k @ 12.6 (no speculation)
#  --kv-dtype int8 ... 49152 --spec mtp --draft-tokens 3 --prefill-chunk 384
#  --kv-dtype i4  ... 131072 --prefill-chunk 384
```

Quality chain, all measured: fp32 6.012 → min-Q4 6.299 (+0.29, all of it from
4-bit weights) → +i4 KV ≈ +0.003. Anchored on both sides (unsloth 6.256 as
the parity bar, fp32 as the ceiling).

## 10. What's left on the table, honestly

- **Plain decode cannot exceed ~14 tok/s** at this weight size — measured
  bandwidth wall, engine-independent.
- **MTP acceptance beyond ~64%** is a property of the model's draft head, not
  the harness.
- **128k + MTP3**: needs ~136 MiB that only Q3 weights (dead at RTN quality,
  §3) or sub-4-bit KV (below llama.cpp's own floor) could provide.
- **Better than 25.7 tok/s** at any context: requires fewer weight bytes at
  IQ3_XXS-quality — i.e., implementing imatrix/codebook quantization, a
  multi-week project we deliberately didn't start.
- Everything else — PDL, fp8 paths, dflash — is sm_89+/sm_120 hardware.

The gap between our 12.71 GiB and unsloth's 11.9 GB is the entire remaining
speed difference at equal windows: 12.71/11.9 ≈ 1.07, and 22.4/14.5 ≈ 1.5+
comes from MTP that only our engine can run efficiently on this model class.

## 11. Where everything lives

- `~/ninfer-a5000` — the fork (commits through `d074d5c`; every change marked
  `// Fork:`). `PROJECT-STATE.md` is the running log with every measurement.
  `SMALLER-WEIGHTS-REVIEW.md` and `CONTEXT-128K-REVIEW.md` are the codex
  engineering reviews.
- Artifacts on `/run/media/lio/data/g/`: `qwen3_8_27b_minq4.ninfer` (daily),
  BF16 source (54 GB, keep for reconversions), `qwen3_8_27b_bf16.gguf`,
  `qwen3_8_27b_IQ3_XXS_q8draft.gguf` (llama.cpp experiment).
- `~/bin/unsloth-start.sh` — unsloth daily driver.
- Quality harness: `tools/eval/{quant_gguf.py,sweep.sh,upgrade_draft_head.py}`
  + `/tmp/lcpp-build/bin/llama-perplexity` + `/tmp/wiki.test.raw`.
- Tests: 8 oracle suites, all green at ship (`gqa_attention`,
  `linear_add_q4`, `linear_q4`, `attn_input_proj`, `gdn_input_proj` ×3,
  `runtime_mechanisms`).
