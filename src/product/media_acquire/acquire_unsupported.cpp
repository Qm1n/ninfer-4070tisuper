#include "product/media_acquire/acquire.h"

#include <stdexcept>

// Fork: text-only builds cannot decode any acquired media, so reject it before reading files or
// expanding inline data. This keeps disabled media from consuming its configured byte budget.

namespace ninfer::product::media_acquire {
std::vector<std::uint8_t> acquire_bytes(const Source&, const Policy&) {
    throw std::invalid_argument(
        "media input is unavailable in this build: it was configured without FFmpeg and libcurl "
        "(NINFER_ENABLE_MEDIA=OFF)");
}

} // namespace ninfer::product::media_acquire
