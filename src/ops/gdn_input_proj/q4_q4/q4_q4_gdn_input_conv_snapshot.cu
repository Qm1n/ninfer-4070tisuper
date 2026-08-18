#include "ops/gdn_input_proj/q4_q4/q4_q4_gdn_input_kernels.h"

// Fork: Q4/Q4 projection-epilogue fusion for GDN snapshot and record entry points.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden      = 5120;
constexpr int kQueryRows   = 2048;
constexpr int kKeyRows     = 2048;
constexpr int kValueRows   = 6144;
constexpr int kZRows       = 6144;
constexpr int kValueZRows  = kValueRows + kZRows;
constexpr int kQkRows      = kQueryRows + kKeyRows;
constexpr int kChannels    = kQkRows + kValueRows;
constexpr int kValueOffset = kQkRows;

using Q4ScheduleC4 = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4ScheduleC8 = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

enum class PdlOrder {
    QkThenValueZ,
    ValueZThenQk,
};

template <class Publish>
GdnConvEpilogue<Publish> make_epilogue(const Tensor& conv_weight, const Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& query, Tensor& key, Tensor& value,
                                       int global_row_offset, Publish publish) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        kChannels,
        kQueryRows,
        kKeyRows,
        kValueRows,
        global_row_offset,
        static_cast<std::int32_t>(query.ne[1]),
        0,
        publish,
    };
}

template <class Publish>
struct QkDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float result) const {
        const float projected[1]{result};
        conv.store(row, projected);
    }
};

template <int Tokens, class Publish>
struct QkSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
        if (active_cols == Tokens) { conv.store(row, projected); }
    }
};

template <class Publish>
struct ValueZDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float result) const {
        if (row < kValueRows) {
            const float projected[1]{result};
            conv.store(row, projected);
        } else {
            z[row - kValueRows] = __float2bfloat16_rn(result);
        }
    }
};

template <int Tokens, class Publish>
struct ValueZSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        if (active_cols != Tokens) { return; }
        if (row < kValueRows) {
            float projected[Tokens];
#pragma unroll
            for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
            conv.store(row, projected);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * kZRows + row - kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_qk_t1(const Tensor& x, const Weight& qk_weight,
                  const GdnConvEpilogue<Publish>& epilogue, Tensor& query, cudaStream_t stream) {
    using Schedule       = Q4GemvR1W8DirectSchedule;
    constexpr int blocks = kQkRows / Schedule::kRowsPerCta;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(blocks), dim3(Schedule::kThreads), 0, stream},
            q4_rowsplit_gemv_kernel<Schedule, false, 0, QkDecodeEpilogue<Publish>, TriggerPdl,
                                    JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQkRows, kHidden,
            QkDecodeEpilogue<Publish>{epilogue}));
    } else {
        q4_rowsplit_gemv_kernel<Schedule, false, 0, QkDecodeEpilogue<Publish>, TriggerPdl, JoinPdl>
            <<<blocks, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, kQkRows, kHidden,
                QkDecodeEpilogue<Publish>{epilogue});
    }
}

template <class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_value_z_t1(const Tensor& x, const Weight& value_z_weight,
                       const GdnConvEpilogue<Publish>& epilogue, Tensor& value, Tensor& z,
                       cudaStream_t stream) {
    using Schedule       = Q4GemvR1W8DirectSchedule;
    constexpr int blocks = kValueZRows / Schedule::kRowsPerCta;
    const ValueZDecodeEpilogue<Publish> output{epilogue, static_cast<__nv_bfloat16*>(z.data)};
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(blocks), dim3(Schedule::kThreads), 0, stream},
            q4_rowsplit_gemv_kernel<Schedule, false, 0, ValueZDecodeEpilogue<Publish>, TriggerPdl,
                                    JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, kHidden, output));
    } else {
        q4_rowsplit_gemv_kernel<Schedule, false, 0, ValueZDecodeEpilogue<Publish>, TriggerPdl,
                                JoinPdl><<<blocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, kHidden, output);
    }
}

template <int Tokens, class Schedule, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_qk_small_t(const Tensor& x, const Weight& qk_weight,
                       const GdnConvEpilogue<Publish>& epilogue, Tensor& query,
                       cudaStream_t stream) {
    const dim3 grid(kQkRows / Schedule::kRowsPerCta, 1u, 1u);
    const QkSmallTEpilogue<Tokens, Publish> output{epilogue};
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {grid, dim3(Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Schedule, false, false, 0,
                                         QkSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQueryRows, 0, kQkRows, kHidden,
            Tokens, kHidden, output));
    } else {
        q4_rowsplit_gemm_simt_kernel<Schedule, false, false, 0,
                                     QkSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, kQueryRows, 0, kQkRows, kHidden,
                Tokens, kHidden, output);
    }
}

template <int Tokens, class Schedule, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_value_z_small_t(const Tensor& x, const Weight& value_z_weight,
                            const GdnConvEpilogue<Publish>& epilogue, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    const dim3 grid(kValueZRows / Schedule::kRowsPerCta, 1u, 1u);
    const ValueZSmallTEpilogue<Tokens, Publish> output{epilogue,
                                                       static_cast<__nv_bfloat16*>(z.data)};
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {grid, dim3(Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Schedule, false, false, 0,
                                         ValueZSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueRows, kZRows, kValueZRows, kHidden, Tokens, kHidden, output));
    } else {
        q4_rowsplit_gemm_simt_kernel<Schedule, false, false, 0,
                                     ValueZSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                kValueRows, kZRows, kValueZRows, kHidden, Tokens, kHidden, output);
    }
}

template <PdlOrder Order, class Publish>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::ValueZThenQk) {
        launch_value_z_t1<Publish, true, false, false>(x, value_z_weight, value_epilogue, value, z,
                                                      stream);
        launch_qk_t1<Publish, false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_qk_t1<Publish, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_value_z_t1<Publish, false, true, true>(x, value_z_weight, value_epilogue, value, z,
                                                     stream);
    }
}

template <int Tokens, class Schedule, PdlOrder Order, class Publish>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::ValueZThenQk) {
        launch_value_z_small_t<Tokens, Schedule, Publish, true, false, false>(
            x, value_z_weight, value_epilogue, value, z, stream);
        launch_qk_small_t<Tokens, Schedule, Publish, false, true, true>(
            x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_qk_small_t<Tokens, Schedule, Publish, true, false, false>(
            x, qk_weight, qk_epilogue, query, stream);
        launch_value_z_small_t<Tokens, Schedule, Publish, false, true, true>(
            x, value_z_weight, value_epilogue, value, z, stream);
    }
}

template <int Tokens, PdlOrder Order, class Publish>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvEpilogue<Publish>& qk_epilogue,
                    const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Tokens, Q4ScheduleC4, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Tokens, Q4ScheduleC8, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <PdlOrder Order, class Publish>
void launch_conv(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
                 const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                 Publish publish, cudaStream_t stream) {
    const GdnConvEpilogue<Publish> qk_epilogue = make_epilogue(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, 0, publish);
    const GdnConvEpilogue<Publish> value_epilogue =
        make_epilogue(conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
                      kValueOffset, publish);

    switch (x.ne[1]) {
    case 1:
        launch_t1<Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                  value, z, stream);
        break;
    case 2:
        launch_small_t<2, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    case 3:
        launch_small_t<3, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    case 5:
        launch_small_t<5, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    case 6:
        launch_small_t<6, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    default:
        throw std::invalid_argument("Q4/Q4 projection-epilogue GDN conv requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q4_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    const SnapshotHistoryPublish publish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         kChannels};
    if (x.ne[1] == 2) {
        launch_conv<PdlOrder::QkThenValueZ>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                           valid_columns, initial_slot, query, key, value, z, publish,
                                           stream);
    } else {
        launch_conv<PdlOrder::ValueZThenQk>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                           valid_columns, initial_slot, query, key, value, z, publish,
                                           stream);
    }
}

void q4_q4_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_slot, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream) {
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data), kChannels,
                                      x.ne[1]};
    if (x.ne[1] == 2) {
        launch_conv<PdlOrder::QkThenValueZ>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                           valid_columns, initial_slot, query, key, value, z, publish,
                                           stream);
    } else {
        launch_conv<PdlOrder::ValueZThenQk>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                           valid_columns, initial_slot, query, key, value, z, publish,
                                           stream);
    }
}

} // namespace ninfer::ops::detail
