#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q4_a16_conformance() {
    // Fork routes: T=1 GEMV, T=2..8 SIMT C4, T>=9 SIMT C8 (chunked for large T).
    constexpr std::array<std::int32_t, 2> kK6144RouteStarts{2, 9};
    constexpr std::array<std::int32_t, 5> kK6144RouteInteriors{1, 4, 16, 96, 256};
    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q4_A16 LinearAdd", WeightFormat::Q4G64F16S,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kK6144RouteInteriors});
    constexpr std::array<std::int32_t, 2> kK17408RouteStarts{2, 9};
    constexpr std::array<std::int32_t, 5> kK17408RouteInteriors{1, 4, 16, 96, 256};
    failures += ninfer::test::linear_add::run_shape(
        "Q4_A16 LinearAdd", WeightFormat::Q4G64F16S,
        ShapeCase{5120, 17408, 409U, kK17408RouteStarts, kK17408RouteInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q4_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
