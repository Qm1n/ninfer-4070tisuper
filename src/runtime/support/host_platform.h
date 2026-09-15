#pragma once

// Small host-platform shims shared by the product, serving, and CLI layers. NInfer is developed
// on Linux; the Windows port routes terminal, process, and calendar queries through this header
// so the call sites stay free of platform branches.

#include <cstddef>
#include <cstdio>
#include <ctime>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <process.h>
#include <windows.h>
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

// Terminal width for interactive progress rendering, or a stable fallback when the width cannot
// be queried.
inline std::size_t terminal_columns() noexcept {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_ERROR_HANDLE), &info) != 0) {
        const int columns = static_cast<int>(info.srWindow.Right) -
                            static_cast<int>(info.srWindow.Left) + 1;
        if (columns > 0) { return static_cast<std::size_t>(columns); }
    }
    return 120;
#else
    return 120;
#endif
}

inline void local_time(std::time_t value, std::tm& out) noexcept {
#ifdef _WIN32
    ::localtime_s(&out, &value);
#else
    ::localtime_r(&value, &out);
#endif
}

inline void utc_time(std::time_t value, std::tm& out) noexcept {
#ifdef _WIN32
    ::gmtime_s(&out, &value);
#else
    ::gmtime_r(&value, &out);
#endif
}

} // namespace ninfer::runtime::support
