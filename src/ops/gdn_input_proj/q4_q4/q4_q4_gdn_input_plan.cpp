#include "ops/gdn_input_proj/q4_q4/q4_q4_gdn_input_plan.h"

// Fork: Q4/Q4 GDN routes mirror the qualified Q4/Q5 token boundaries.

#include "ops/gdn_input_proj/q4_q4/q4_q4_gdn_input_kernels.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool supported_shape(const Q4Q4GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

bool q4_q4_gdn_input_admits(const Q4Q4GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q4GdnInputPlan q4_q4_gdn_input_resolve_plan(const Q4Q4GdnInputProblem& problem) {
    if (!q4_q4_gdn_input_admits(problem)) {
        throw std::invalid_argument("Q4/Q4 GDN input: exact problem is not admitted");
    }
    return {problem.cols <= 16 ? Q4Q4GdnInputScheduleId::IndependentDirectFixed
                               : Q4Q4GdnInputScheduleId::GroupedQ4MmaR64C128};
}

Q4Q4GdnInputConvPlan q4_q4_gdn_input_conv_resolve_plan(const Q4Q4GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q4_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("Q4/Q4 GDN input conv: exact problem is not admitted");
    }
    if (batch_size > 1) { return {Q4Q4GdnInputConvScheduleId::Materialized}; }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q4GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q4GdnInputConvScheduleId::Materialized};
    }
}

void q4_q4_gdn_input_execute_plan(const Q4Q4GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, cudaStream_t stream) {
    const Q4Q4GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    if (q4_q4_gdn_input_resolve_plan(problem).schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q4 GDN input: plan does not match exact problem");
    }
    switch (plan.schedule) {
    case Q4Q4GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q4_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q4GdnInputScheduleId::GroupedQ4MmaR64C128:
        q4_q4_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("Q4/Q4 GDN input: unknown schedule");
}

void q4_q4_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              cudaStream_t stream) {
    const Q4Q4GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1]};
    const Q4Q4GdnInputPlan plan = q4_q4_gdn_input_resolve_plan(problem);
    q4_q4_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
