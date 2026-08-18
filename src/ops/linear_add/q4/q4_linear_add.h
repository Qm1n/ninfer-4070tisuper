#pragma once

// Fork: Q4 residual linear_add for 16 GB cards. Mirrors the Q5 linear_add API
// but routes decode through the Q4 GEMV and multi-token steps through the Q4
// SIMT GEMM. The Q4 MMA path has no epilogue hook, so large-T prefill also uses
// SIMT (slower than MMA; ponytail: add a Q4 MMA residual epilogue if prefill
// throughput matters).

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void q4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream);

std::size_t q4_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols);

}  // namespace ninfer::ops::detail
