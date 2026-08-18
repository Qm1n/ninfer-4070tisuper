#pragma once

// Fork: Closed Q4/Q4 attention-input route catalog.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4Q4AttnInputScheduleId {
    ParentSplitFixed,
    GroupedHomogeneousPairMmaR16C64S3,
    GroupedHomogeneousPairMmaR32C64S4,
};

struct Q4Q4AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q4AttnInputPlan {
    Q4Q4AttnInputScheduleId schedule;
};

const char* q4_q4_attn_input_schedule_name(Q4Q4AttnInputScheduleId schedule) noexcept;
bool q4_q4_attn_input_admits(const Q4Q4AttnInputProblem& problem) noexcept;
Q4Q4AttnInputPlan q4_q4_attn_input_resolve_plan(const Q4Q4AttnInputProblem& problem);
void q4_q4_attn_input_execute_plan(const Q4Q4AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight,
                                   const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);
void q4_q4_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
