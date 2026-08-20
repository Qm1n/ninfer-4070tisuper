#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/** Qwen3.6's target-local mapping from a stable request lane to its hot device state. */
struct LinearStateSlots {
    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency) {
        if (max_concurrency == 0 ||
            max_concurrency > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Qwen3.6 Linear Attention concurrency is invalid");
        }
        // Fork: rewrite checkpoints are pinned-host images, so only hot lane slots live on device.
        return static_cast<std::int32_t>(max_concurrency);
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3.6 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }

};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
