#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <functional>
#include <stdexcept>

namespace ninfer {

// Fork: a successful capture that exceeds calibration is replanned instead of retained silently.
class CudaGraphAllowanceExceeded final : public std::runtime_error {
public:
    explicit CudaGraphAllowanceExceeded(std::size_t observed_bytes)
        : std::runtime_error("CUDA Graph observed bytes exceed the calibrated allowance"),
          observed_bytes_(observed_bytes) {}

    [[nodiscard]] std::size_t observed_bytes() const noexcept { return observed_bytes_; }

private:
    std::size_t observed_bytes_ = 0;
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
