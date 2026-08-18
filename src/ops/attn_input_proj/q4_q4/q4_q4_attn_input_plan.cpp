#include "ops/attn_input_proj/q4_q4/q4_q4_attn_input_plan.h"

// Fork: Q4/Q4 routes mirror the qualified Q4/Q5 token boundaries.

#include "ops/attn_input_proj/q4_q4/q4_q4_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q4Q4AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 16, Q4Q4AttnInputScheduleId::ParentSplitFixed},
    {17, 20, Q4Q4AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3},
    {21, kAnyCols, Q4Q4AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4},
}};

bool supported_shape(const Q4Q4AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q4_attn_input_schedule_name(Q4Q4AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q4AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q4.parent_split_fixed";
    case Q4Q4AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q4.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q4AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q4.grouped_homogeneous_pair.mma.r32.c64.s4";
    }
    return "attn_input_proj.q4_q4.unknown";
}

bool q4_q4_attn_input_admits(const Q4Q4AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q4AttnInputPlan q4_q4_attn_input_resolve_plan(const Q4Q4AttnInputProblem& problem) {
    if (!q4_q4_attn_input_admits(problem)) {
        throw std::invalid_argument("Q4/Q4 attention input: exact problem is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("Q4/Q4 attention input: admitted problem has no route");
}

void q4_q4_attn_input_execute_plan(const Q4Q4AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight,
                                   const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q4Q4AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    if (q4_q4_attn_input_resolve_plan(problem).schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q4 attention input: plan does not match exact problem");
    }
    switch (plan.schedule) {
    case Q4Q4AttnInputScheduleId::ParentSplitFixed:
        q4_q4_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q4AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q4_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q4AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q4_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q4 attention input: unknown schedule");
}

void q4_q4_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream) {
    const Q4Q4AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q4AttnInputPlan plan = q4_q4_attn_input_resolve_plan(problem);
    q4_q4_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream);
}

} // namespace ninfer::ops::detail
