#include "ops/linear_add/q4/q4_linear_add.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Q4SimtR8C4ResidualSchedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4SimtR8C8ResidualSchedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

// out[row] = bf16(out[row] + value); one thread owns each row (SplitOutput off).
struct Q4GemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                               int row, float value) const {
        static_assert(!SplitOutput, "Q4 linear_add GEMV is contiguous-only");
        (void)out_tail;
        out[row] = __float2bfloat16(__bfloat162float(out[row]) + value);
    }
};

// Column-major residual store; each (row, col) is owned by exactly one thread.
struct Q4SimtResidualEpilogue {
    template <bool SplitOutput, int SplitRow, int Cols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t out_ld,
               std::int32_t out_tail_ld, std::int32_t row, std::int32_t col0,
               std::int32_t active_cols, const float (&values)[Cols]) const {
        static_assert(!SplitOutput, "Q4 linear_add SIMT is contiguous-only");
        (void)out_tail;
        (void)out_tail_ld;
#pragma unroll
        for (int col = 0; col < Cols; ++col) {
            if (col >= active_cols) { continue; }
            __nv_bfloat16* slot = out + static_cast<std::int64_t>(col0 + col) * out_ld + row;
            *slot               = __float2bfloat16(__bfloat162float(*slot) + values[col]);
        }
    }
};

void launch_gemv_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                          cudaStream_t stream) {
    // Fork: StaticGroupsPerRow must equal k/64 — the base R1W8 schedule is
    // hard-coded for k=5120. Route by K to the correct specialization.
    const std::int32_t rows = residual_out.ne[0];
    const std::int32_t k    = x.ne[0];
    const auto launch = [&]<class Schedule>() {
        const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
        // Fork: nvcc's front end rejects a constexpr dim3 (the type is not literal to it).
        const dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
        q4_rowsplit_gemv_kernel<Schedule, false, 0, Q4GemvResidualEpilogue>
            <<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales),
                static_cast<__nv_bfloat16*>(residual_out.data), nullptr, rows, k);
        CUDA_CHECK(cudaGetLastError());
    };
    switch (k) {
    case 6144:  launch.operator()<Q4GemvR1W8K6144Schedule>();  return;
    case 17408: launch.operator()<Q4GemvR1W8K17408Schedule>(); return;
    default:    launch.operator()<Q4GemvR1W8DirectSchedule>(); return;  // k == 5120
    }
}

template <class Schedule, bool Full>
void launch_simt_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                          cudaStream_t stream) {
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, false, 0, Q4SimtResidualEpilogue>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), nullptr, out_ld, 0, rows, k, cols,
            padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void route_simt(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const bool full = (residual_out.ne[0] % Schedule::kRowsPerCta) == 0 &&
                      ((x.ne[0] / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    // ponytail: large-T prefill rides the SIMT kernel (Q4 MMA has no residual
    // epilogue); add one if prefill throughput matters.
    for_each_token_slice(x.ne[1], Schedule::kColsPerTile,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice      = x.slice(1, offset, count);
                             Tensor residual_slice     = residual_out.slice(1, offset, count);
                             if (full) {
                                 launch_simt_residual<Schedule, true>(x_slice, w, residual_slice,
                                                                     stream);
                             } else {
                                 launch_simt_residual<Schedule, false>(x_slice, w, residual_slice,
                                                                      stream);
                             }
                         });
}

bool supported_shape(const Weight& w) noexcept {
    return (w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) ||
           (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408);
}

}  // namespace

void q4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    (void)ws;
    if (!supported_shape(w)) {
        throw std::invalid_argument("q4 linear_add: unsupported exact shape");
    }
    if (x.ne[1] == 1) {
        launch_gemv_residual(x, w, residual_out, stream);
        return;
    }
    if (x.ne[1] <= 8) {
        route_simt<Q4SimtR8C4ResidualSchedule>(x, w, residual_out, stream);
    } else if (x.ne[1] > 16) {
        // Fork: large-T prefill through the residual MMA (2.5x SIMT throughput).
        launch_q4_mma_r64_c128_residual(x, w, residual_out, stream);
    } else {
        route_simt<Q4SimtR8C8ResidualSchedule>(x, w, residual_out, stream);
    }
}

std::size_t q4_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    (void)rows;
    (void)k;
    (void)padded_k;
    (void)min_cols;
    (void)max_cols;
    return 0;  // residual epilogues read/write the destination directly
}

}  // namespace ninfer::ops::detail
