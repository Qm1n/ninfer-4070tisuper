// Fork stub: the warp-specialized TMA kernels (setmaxnreg, sm_90+ PTX) are not
// compiled for non-Blackwell architectures. Loading an NVFP4 artifact on this
// build fails loudly here instead of at link time.
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
[[noreturn]] void tma_unsupported() {
    throw std::runtime_error(
        "NVFP4 TMA kernels are not built for this GPU architecture (sm_120a only)");
}
}  // namespace

void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    tma_unsupported();
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                     float, cudaStream_t) {
    tma_unsupported();
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                               float, cudaStream_t) {
    tma_unsupported();
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    tma_unsupported();
}

}  // namespace ninfer::ops::detail
