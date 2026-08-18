#include "ops/attn_input_proj/q4_q4/q4_q4_attn_input_kernels.h"

// Fork: Q4/Q4 twin of the fixed small-T attention projection path. Templated on
// the parent/split/hidden geometry so peer targets admit their own dimensions.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <std::int32_t kParentRows, std::int32_t kSplitRow, std::int32_t kHidden>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& head, Tensor& tail,
                    cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(head.data),
        static_cast<__nv_bfloat16*>(tail.data), kParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t kParentRows, std::int32_t kSplitRow, std::int32_t kHidden, class Schedule,
          bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& head, Tensor& tail,
                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(head.data), static_cast<__nv_bfloat16*>(tail.data),
            head.ne[0], tail.ne[0], kParentRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t kParentRows, std::int32_t kSplitRow, std::int32_t kHidden, class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& head, Tensor& tail,
                          cudaStream_t stream) {
    const bool full = (kParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<kParentRows, kSplitRow, kHidden, Schedule, true>(x, weight, head, tail,
                                                                        stream);
    } else {
        launch_q4_simt<kParentRows, kSplitRow, kHidden, Schedule, false>(x, weight, head, tail,
                                                                         stream);
    }
}

template <std::int32_t kParentRows, std::int32_t kSplitRow, std::int32_t kHidden>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& head, Tensor& tail,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv<kParentRows, kSplitRow, kHidden>(x, weight, head, tail, stream);
        return;
    }
    if (x.ne[1] <= 7 || (x.ne[1] >= 9 && x.ne[1] <= 15)) {
        launch_q4_simt_route<kParentRows, kSplitRow, kHidden, Q4AttnSimtR8C4Schedule>(
            x, weight, head, tail, stream);
        return;
    }
    if (x.ne[1] == 8 || x.ne[1] == 16) {
        launch_q4_simt_route<kParentRows, kSplitRow, kHidden, Q4AttnSimtR8C8Schedule>(
            x, weight, head, tail, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q4 attention split-output requires T in [1,16]");
}

void launch_q4_dispatch(const Tensor& x, const Weight& weight, Tensor& head, Tensor& tail,
                         cudaStream_t stream) {
    if (weight.n == 7168 && weight.k == 5120) {
        launch_q4<7168, 6144, 5120>(x, weight, head, tail, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 4096) {
        launch_q4<5120, 4096, 4096>(x, weight, head, tail, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q4 attention split-output geometry is not admitted");
}

} // namespace

void q4_q4_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4_dispatch(x, query_key_weight, q, k, stream);
    launch_q4_dispatch(x, gate_value_weight, gate, v, stream);
}

} // namespace ninfer::ops::detail
