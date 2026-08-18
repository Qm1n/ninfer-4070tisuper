#pragma once

// Fork: Closed Q4/Q4 GDN input route catalog.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q4GdnInputScheduleId {
    IndependentDirectFixed,
    GroupedQ4MmaR64C128,
};

enum class Q4Q4GdnInputConvScheduleId {
    ProjectionEpilogueFused,
    Materialized,
};

struct Q4Q4GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qk_rows;
    std::int32_t value_z_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q4GdnInputPlan {
    Q4Q4GdnInputScheduleId schedule;
};

struct Q4Q4GdnInputConvPlan {
    Q4Q4GdnInputConvScheduleId schedule;
};

bool q4_q4_gdn_input_admits(const Q4Q4GdnInputProblem& problem) noexcept;
Q4Q4GdnInputPlan q4_q4_gdn_input_resolve_plan(const Q4Q4GdnInputProblem& problem);
Q4Q4GdnInputConvPlan q4_q4_gdn_input_conv_resolve_plan(const Q4Q4GdnInputProblem& problem,
                                                       std::int32_t batch_size);
void q4_q4_gdn_input_execute_plan(const Q4Q4GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream);
void q4_q4_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
