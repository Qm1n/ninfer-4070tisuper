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

qwen3_8_27b_minq4.ninfer (Q4-g64 text weights, Q6 embedding) on a 16 GB RTX 4070 Ti SUPER, greedy
sampling, single request:

| Profile | Context | KV cache | Decode | Prefill | Free after startup |
| --- | ---: | --- | ---: | ---: | ---: |
| plain | 8,192 | int8 group-64 | 34.2 tok/s | 309 tok/s | 1.34 GiB |
| MTP3 | 16,384 | int8 group-64 | 81.0 tok/s | 326 tok/s | 0.29 GiB |
| MTP3 | 32,768 | int4 group-128 | 75.1 tok/s | 296 tok/s | 0.45 GiB |

MTP acceptance was 71.7% (16K, int8) and 63.6% (32K, int4) at a draft window of three tokens.
Weight residency is 12.71 GiB plain and 13.47 GiB with the MTP package loaded, leaving roughly
1.1-1.9 GiB free after weights. MTP additionally reserves 940 MiB of runtime memory, so automatic
KV sizing refuses configurations that would leave no headroom; pass an explicit --kv-capacity for
MTP profiles.

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
