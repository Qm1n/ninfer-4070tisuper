#pragma once

// Fork: no-op NVTX surface for toolkits that do not ship NVTX. CUDA 12.8 for Windows provides
// neither nvtx3/nvToolsExt.h nor the NVTX import library, so this header stands in for it and
// keeps every instrumentation call site compiling while recording nothing. Toolkits that do
// ship NVTX resolve the real header instead and this directory stays off the include path.

#include <cstdint>

using nvtxDomainHandle_t = void*;
using nvtxStringHandle_t = const char*;

inline constexpr int NVTX_VERSION                    = 0;
inline constexpr int NVTX_EVENT_ATTRIB_STRUCT_SIZE   = 0;
inline constexpr int NVTX_COLOR_ARGB                 = 0;
inline constexpr int NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 0;
inline constexpr int NVTX_MESSAGE_TYPE_REGISTERED    = 0;

struct nvtxEventAttributes_t {
    struct Payload {
        unsigned long long ullValue = 0;
    };
    struct Message {
        nvtxStringHandle_t registered = nullptr;
    };

    int version             = 0;
    int size                = 0;
    std::uint32_t category  = 0;
    int colorType           = 0;
    std::uint32_t color     = 0;
    int payloadType         = 0;
    int messageType         = 0;
    Payload payload{};
    Message message{};
};

inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) noexcept { return nullptr; }

inline void nvtxDomainNameCategoryA(nvtxDomainHandle_t, std::uint32_t, const char*) noexcept {}

inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char*) noexcept {
    return nullptr;
}

inline void nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) noexcept {}

inline void nvtxDomainRangePop(nvtxDomainHandle_t) noexcept {}

inline void nvtxRangePushA(const char*) noexcept {}

inline void nvtxRangePop() noexcept {}
