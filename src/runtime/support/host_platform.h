#pragma once

// Small host-platform shims shared by the product, serving, and CLI layers. NInfer is developed
// on Linux; the Windows port routes terminal, process, and calendar queries through this header
// so the call sites stay free of platform branches.

#include <cstdio>
#include <ctime>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer::runtime::support {

inline int process_id() noexcept {
#ifdef _WIN32
    return ::_getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

inline bool stderr_is_terminal() noexcept {
#ifdef _WIN32
    return ::_isatty(::_fileno(stderr)) == 1;
#else
    return ::isatty(STDERR_FILENO) == 1;
#endif
}

inline void local_time(std::time_t value, std::tm& out) noexcept {
#ifdef _WIN32
    ::localtime_s(&out, &value);
#else
    ::localtime_r(&value, &out);
#endif
}

} // namespace ninfer::runtime::support
