#include "media/decode/decode.h"

namespace ninfer::media::decode {
namespace {

[[noreturn]] void unavailable() {
    throw Error(ErrorKind::BudgetExceeded,
                "image and video decoding are unavailable in this build: it was configured "
                "without FFmpeg (NINFER_ENABLE_MEDIA=OFF)");
}

} // namespace

Image decode_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

} // namespace ninfer::media::decode
