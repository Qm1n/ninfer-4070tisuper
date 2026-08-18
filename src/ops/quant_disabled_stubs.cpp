// Fork stubs: NVFP4 and FP8 execution paths are not built for sm_86 (no FP4/FP8
// tensor cores, no cuda_fp4.h in CUDA 12.4). Any artifact that selects these
// weight formats fails loudly at dispatch time. Groupwise-int (q4/w8/bf16)
// artifacts are unaffected.
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
[[noreturn]] void quant_unsupported(const char* profile) {
    throw std::runtime_error(std::string(profile) +
                             " execution is not built for this GPU architecture "
                             "(NVFP4/FP8 require sm_89+/sm_120a)");
}
}  // namespace

// --- NVFP4 ---------------------------------------------------------------

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight&, const char*) {
    quant_unsupported("NVFP4");
}

void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, DeviceArena*, cudaStream_t) {
    quant_unsupported("NVFP4");
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                  std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                               LinearPolicy, DeviceArena*, cudaStream_t) {
    quant_unsupported("NVFP4");
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                              DeviceArena*, cudaStream_t) {
    quant_unsupported("NVFP4");
}

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                 const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                 Tensor&, Tensor&, LinearPolicy, DeviceArena&, cudaStream_t) {
    quant_unsupported("NVFP4");
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, cudaStream_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, DeviceArena&,
                               cudaStream_t) {
    quant_unsupported("NVFP4");
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                      std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, DeviceArena&,
                                  cudaStream_t) {
    quant_unsupported("NVFP4");
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("NVFP4");
}

// --- FP8 ------------------------------------------------------------------

Fp8WeightGeometry validate_fp8_weight(const Weight&, const char*) {
    quant_unsupported("FP8");
}

void fp8_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, DeviceArena*, cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                std::int32_t, std::int32_t) {
    quant_unsupported("FP8");
}

void fp8_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                             LinearPolicy, DeviceArena*, cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("FP8");
}

void fp8_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&, LinearPolicy,
                            DeviceArena*, cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("FP8");
}

void fp8_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&, Tensor&,
                               const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                               Tensor&, Tensor&, LinearPolicy, DeviceArena&, cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t,
                                                      std::int32_t) {
    quant_unsupported("FP8");
}

void fp8_gdn_record_dispatch(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                             const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                             Tensor&, LinearPolicy, DeviceArena&, cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t,
                                                    std::int32_t) {
    quant_unsupported("FP8");
}

void fp8_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, DeviceArena&,
                             cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t, LinearPolicy,
                                                    std::int32_t, std::int32_t) {
    quant_unsupported("FP8");
}

void fp8_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, DeviceArena&,
                                cudaStream_t) {
    quant_unsupported("FP8");
}

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy, std::int32_t, std::int32_t) {
    quant_unsupported("FP8");
}

}  // namespace ninfer::ops::detail
