#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int kN, int kK>
struct Q4SwiGluGeometry {
    static constexpr int kIntermediate      = kN / 2;
    static constexpr int kGroupK            = 64;
    static constexpr int kGroups            = kK / kGroupK;
    static constexpr int kBytesPerGroup     = 32;
    static constexpr int kVecBytes          = 16;
    static constexpr int kGroupsPerWarpTile = 16;
    static constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
    static constexpr int kWarpsPerBlock     = 4;
    static constexpr int kBlockThreads      = kWarpsPerBlock * 32;
    static constexpr int kPairsPerBlock     = kWarpsPerBlock;
    static constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
    static constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
    static_assert(kIntermediate % kPairsPerBlock == 0);
    static_assert(kBytesPerGroup == 2 * kVecBytes);
    static_assert(kGroups % kGroupsPerWarpTile == 0);
    static_assert(kVecsPerWarpTile == 32);
};

using SwiGlu27B = Q4SwiGluGeometry<34816, 5120>;
using SwiGlu9B  = Q4SwiGluGeometry<24576, 4096>;

template <class G>
struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = G::kK;
    static constexpr int kGroupsPerRow = G::kK / G::kGroupK;
};

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? G::kIntermediate : 0);
    }
};

template <class G>
struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * G::kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * G::kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

template <class G>
using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class G, std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher<G>, sizeof...(Offsets)>{
        &launch_small_t_active<G, 2 + static_cast<int>(Offsets)>...};
}

template <class G>
constexpr auto kSmallTLaunchers = make_small_t_launchers<G>(std::make_index_sequence<31>{});

template <class G, int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    constexpr int kBlocks = G::kIntermediate / Q4SwiGluSmallTRows::kOutputRowsPerCta;
    const Q4SwiGluSmallTEpilogue<G> epilogue{static_cast<__nv_bfloat16*>(out.data)};
    q4_small_t_mma_kernel<Q4SwiGluSmallTGeometry<G>, TileCols, ActiveCols, Q4SwiGluSmallTEpilogue,
                          Q4SwiGluSmallTRows>
        <<<kBlocks, Q4DraftSmallTSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            epilogue, Q4SwiGluSmallTRows{});
    CUDA_CHECK(cudaGetLastError());
}


template <class G>
__device__ __forceinline__ void q4_issue_pair_tile(
    uint4 (*__restrict__ s_code)[G::kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * G::kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * G::kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * G::kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

template <class G>
__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[G::kK];
    __shared__ uint4 code_tile[G::kWarpsPerBlock][kStages][2][G::kVecsPerWarpTile];
    __shared__ uint4 scale_tile[G::kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < G::kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * G::kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * G::kGroups * G::kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * G::kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + G::kIntermediate) * G::kGroups * G::kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + G::kIntermediate) * G::kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < G::kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < G::kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < G::kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < G::kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * G::kBytesPerGroup + lane]);
            const int gate_q0   = sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = sign_extend<4>(gate_packed >> 4);
            const int up_packed = static_cast<int>(up_codes[tile_group * G::kBytesPerGroup + lane]);
            const int up_q0     = sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = sign_extend<4>(up_packed >> 4);
            const int k0        = (tile * G::kGroupsPerWarpTile + tile_group) * G::kGroupK + lane * 2;
            const float2 xv     = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc            = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc            = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc              = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc              = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

} // namespace

template <class G>
void q4_linear_swiglu_gemv_pair_launch_geo(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (w.n != G::kN || w.k != G::kK || w.padded_shape[1] != G::kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV weight geometry is not admitted");
    }
    const int grid = G::kIntermediate / G::kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<G><<<grid, G::kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.k == 5120) { q4_linear_swiglu_gemv_pair_launch_geo<SwiGlu27B>(x, w, out, stream); }
    else if (w.k == 4096) { q4_linear_swiglu_gemv_pair_launch_geo<SwiGlu9B>(x, w, out, stream); }
    else { throw std::invalid_argument("q4 linear_swiglu GEMV geometry is not admitted"); }
}

template <class G>
void q4_linear_swiglu_small_t_exact_launch_geo(const Tensor& x, const Weight& w, Tensor& out,
                                               cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    kSmallTLaunchers<G>[static_cast<std::size_t>(x.ne[1] - 2)](x, w, out, stream);
}

void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (w.k == 5120) { q4_linear_swiglu_small_t_exact_launch_geo<SwiGlu27B>(x, w, out, stream); }
    else if (w.k == 4096) { q4_linear_swiglu_small_t_exact_launch_geo<SwiGlu9B>(x, w, out, stream); }
    else { throw std::invalid_argument("Q4 LinearSwiGLU geometry is not admitted"); }
}

} // namespace ninfer::ops::detail
