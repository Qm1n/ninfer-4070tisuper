# Windows / Ada (sm_89) port

This branch ports the 16 GB sm_86 fork to a **native Windows build for Ada GPUs (sm_89)**, so the
min-Q4 artifact runs on an RTX 4070 Ti SUPER class card without WSL, Docker, or a Linux host.

Verified on: Windows 10 (10.0.26200), RTX 4070 Ti SUPER 16 GB (driver 595.97), CUDA Toolkit 12.8,
Visual Studio 2022 Build Tools (MSVC 14.44), Ninja.

## Requirements

- Windows 10/11 x64 with a matching NVIDIA display driver.
- CUDA Toolkit 12.4 or newer with nvcc on PATH. Verified with 12.8.93.
- Visual Studio 2022 Build Tools with the C++ workload; MSVC 14.4x is verified.
- CMake 3.28+ and Ninja (the copies shipped inside Build Tools are enough).

Only the CUDA runtime DLL (cudart64_12.dll) is needed at run time; it lives in the toolkit bin
directory, so put that directory on PATH before starting ninfer.exe.

## Build

    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_TESTING=OFF
    cmake --build build --parallel

Artifacts land in build/apps/ninfer.exe and build/apps/ninfer-serve.exe.

NINFER_ENABLE_MEDIA defaults to OFF on Windows. With it off the engine builds without FFmpeg and
libcurl: text generation, serving, MTP decoding, and the artifact loader are complete, image and
video input report a clear "unavailable in this build" error, and remote media URLs are rejected.
Set -DNINFER_ENABLE_MEDIA=ON on a host that provides both through pkg-config.

## Measured performance

The groupwise-int artifact on a 16 GB RTX 4070 Ti SUPER, greedy sampling, one request at a time.
Decode is a 256-token generation; prefill covers the prompt only.

| Profile | Context | KV | Prompt | Prefill | Decode | MTP acceptance | Free after startup |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| plain | 8,192 | int8 | 24 tok | 162 tok/s | 34.1 tok/s | - | 1.33 GiB |
| MTP3 | 16,384 | int8 | 24 tok | 157 tok/s | 73.0 tok/s | 60.0% | 0.30 GiB |
| MTP3 | 32,768 | int4 | 24 tok | 160 tok/s | 70.8 tok/s | 55.6% | 0.31 GiB |
| plain | 32,768 | int4 | 12.4k tok | 1,085 tok/s | 33.1 tok/s | - | 1.14 GiB |
| plain | 32,768 | int4 | 18.5k tok | 1,059 tok/s | 32.6 tok/s | - | 1.16 GiB |
| MTP3 | 32,768 | int4 | 12.4k tok | 1,052 tok/s | 61.3 tok/s | 58.3% | 0.36 GiB |
| plain | 49,152 | int4 | 8 tok | - | 34.0 tok/s | - | 0.92 GiB |
| plain | 65,536 | int4 | 8 tok | - | 34.1 tok/s | - | 0.65 GiB |
| plain | 98,304 | int4 | 8 tok | - | 32.9 tok/s | - | 0.12 GiB |

Short prompts report a low prefill rate because fixed launch and graph overhead dominates a handful
of tokens; the same engine sustains roughly 1.0-1.1k tok/s once a prompt has thousands of tokens.
Decode stays in a 33-34 tok/s band from 8k to 96k context with int4 KV, and the MTP draft window of
three tokens roughly doubles it whenever acceptance stays above 55%.

Weight residency is 12.71 GiB plain and 13.47 GiB with the MTP package, leaving 1.1-1.9 GiB free
after weights depending on what else the desktop uses. MTP adds a 940 MiB runtime reservation, so
automatic KV sizing refuses those profiles; pass an explicit --kv-capacity. With int4 KV the
largest context that fits here is between 96k (works, 0.12 GiB free) and 112k (rejected); 131k
needs 2.49 GB of runtime capacity against 2.19 GB available after weights.

### Prefill chunking

The engine default of a 1024-token prefill chunk exceeds the cooperative-launch limit of the GDN
gating projection on this card: prompts above roughly 3k tokens fail with
cudaErrorCooperativeLaunchTooLarge. The chunk has to come down, and larger chunks are also faster,
so 512 is the practical choice.

| --prefill-chunk | Prefill (3,151-token prompt) |
| ---: | ---: |
| 64 | 456 tok/s |
| 128 | 863 tok/s |
| 256 | 985 tok/s |
| 512 | 1,061 tok/s |
| 1024 (default) | fails |

Both Windows launch scripts pass --prefill-chunk 512 for this reason.

## Differences from the sm_86 fork

- CMakeLists.txt: the absolute libstdc++ link applies to UNIX only, and FFmpeg/libcurl become
  optional through NINFER_ENABLE_MEDIA.
- src/CMakeLists.txt: selects the text-only decode/acquire stubs when media is disabled, links
  nvtx3 only when the toolkit provides it, and defines UTF8PROC_STATIC for the in-tree C copy.
- src/artifact/reader.cpp: Win32 file mapping and positioned reads replace mmap/pread; the
  zero-copy payload spans and the 4096-byte direct-read contract are unchanged.
- src/runtime/support/host_platform.h (new): terminal, process, and calendar queries for both
  platforms.
- src/media/decode/decode_unsupported.cpp and src/product/media_acquire/acquire_unsupported.cpp
  (new): media stubs that keep local paths, inline data, and byte sources working.
- third_party/nvtx_shim/: no-op NVTX header used when the toolkit ships no NVTX (CUDA 12.8 for
  Windows ships neither the headers nor the import library).
- src/ops/linear/w8/: the exact-small-T and medium-T split-K kernels stage through opt-in dynamic
  shared memory because their schedules need 50-99 KiB, above the 48 KiB static limit that ptxas
  and nvlink enforce on every architecture. Schedules above the 99 KiB per-block budget that
  Ampere and Ada allow throw a clear error instead of failing the build.
- src/ops q4 small-T, rowsplit GEMV, linear-add, and GDN-input kernels: constexpr dim3 becomes
  const dim3 (the nvcc front end rejects the former).
- src/targets/qwen3_6/impl/runtime/api_impl.h: the move operations of SequencePlanner,
  RequestBasePlan, and RequestPlan are defined explicitly; MSVC does not emit out-of-class
  defaulted specializations of those members.

## Known limits

- Vision input requires a build with FFmpeg and libcurl (NINFER_ENABLE_MEDIA=ON).
- NVTX ranges record nothing when the toolkit has no NVTX.
- The widest W8 medium/large tile schedules cannot launch on a 99 KiB-per-block card.
