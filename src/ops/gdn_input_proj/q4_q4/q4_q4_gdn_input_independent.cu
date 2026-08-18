#include "ops/gdn_input_proj/q4_q4/q4_q4_gdn_input_kernels.h"

// Fork: Q4/Q4 twin of the independent small-T GDN projection path.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

void launch_qk_gemv(const Tensor& x, const Weight& weight, Tensor& qk, cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, kQkRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

void launch_value_z_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kValueRows><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), kValueZRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_qk_simt(const Tensor& x, const Weight& weight, Tensor& qk, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(qk.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, out_ld, 0, kQkRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_value_z_simt(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream) {
    const std::int32_t cols     = x.ne[1];
    const std::int32_t value_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t z_ld     = static_cast<std::int32_t>(z.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kValueRows>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data), value_ld,
            z_ld, kValueZRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_qk_simt_route(const Tensor& x, const Weight& weight, Tensor& qk, cudaStream_t stream) {
    const bool full = (kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_qk_simt<Schedule, true>(x, weight, qk, stream);
    } else {
        launch_qk_simt<Schedule, false>(x, weight, qk, stream);
    }
}

template <class Schedule>
void launch_value_z_simt_route(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                               cudaStream_t stream) {
    const bool full = (kValueZRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_value_z_simt<Schedule, true>(x, weight, value, z, stream);
    } else {
        launch_value_z_simt<Schedule, false>(x, weight, value, z, stream);
    }
}

void launch_qk(const Tensor& x, const Weight& weight, Tensor& qk, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_qk_gemv(x, weight, qk, stream);
    } else if (x.ne[1] <= 4) {
        launch_qk_simt_route<Q4GdnSimtR8C4Schedule>(x, weight, qk, stream);
    } else if (x.ne[1] <= 16) {
        launch_qk_simt_route<Q4GdnSimtR8C8Schedule>(x, weight, qk, stream);
    } else {
        throw std::invalid_argument("Q4/Q4 GDN independent launch requires T in [1,16]");
    }
}

void launch_value_z(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_value_z_gemv(x, weight, value, z, stream);
    } else if (x.ne[1] <= 4) {
        launch_value_z_simt_route<Q4GdnSimtR8C4Schedule>(x, weight, value, z, stream);
    } else if (x.ne[1] <= 16) {
        launch_value_z_simt_route<Q4GdnSimtR8C8Schedule>(x, weight, value, z, stream);
    } else {
        throw std::invalid_argument("Q4/Q4 GDN independent launch requires T in [1,16]");
    }
}

void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Schedule = Q4GdnSimtR8C4Schedule;
    const dim3 value_z_grid(kValueZRows / Schedule::kRowsPerCta, 1u, 1u);
    const dim3 qk_grid(kQkRows / Schedule::kRowsPerCta, 1u, 1u);
    const std::int32_t value_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t z_ld     = static_cast<std::int32_t>(z.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t qk_ld    = static_cast<std::int32_t>(qk.nb[1] / sizeof(__nv_bfloat16));

    q4_rowsplit_gemm_simt_kernel<Schedule, true, true, kValueRows, Q4SimtStoreEpilogue, true, false>
        <<<value_z_grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data), value_ld,
            z_ld, kValueZRows, kHidden, 4, value_z_weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent(
        {qk_grid, dim3(Schedule::kThreads), 0, stream},
        q4_rowsplit_gemm_simt_kernel<Schedule, true, false, 0, Q4SimtStoreEpilogue, false, true>,
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(qk_weight.qdata),
        static_cast<const std::uint8_t*>(qk_weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, qk_ld, 0, kQkRows, kHidden, 4, qk_weight.padded_shape[1], Q4SimtStoreEpilogue{}));
}

} // namespace

void q4_q4_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_qk(x, qk_weight, qk, stream);
    launch_value_z(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
