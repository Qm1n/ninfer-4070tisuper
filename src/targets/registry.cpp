#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

// Fork: calibrated allowance is aggregate observed graph memory plus a fixed startup margin.
constexpr std::size_t kGraphAllowanceFloor = 16ULL * 1024ULL * 1024ULL;
constexpr std::size_t kGraphSafetyMargin   = 24ULL * 1024ULL * 1024ULL;

std::size_t graph_allowance(std::size_t observed_bytes, std::size_t margin_bytes) {
    if (observed_bytes > std::numeric_limits<std::size_t>::max() - margin_bytes) {
        throw std::overflow_error("calibrated CUDA Graph allowance exceeds size_t");
    }
    return std::max(kGraphAllowanceFloor, observed_bytes + margin_bytes);
}

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Fork: startup-only exact decomposition for capacity-curve qualification.
const char* memory_kv_encoding(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::Int4Group64:
        return "i4-g64";
    case KvCacheStorage::Int4Group128:
        return "i4-g128";
    }
    return "unknown";
}

// Fork: one environment contract controls both failed-plan and completed-startup summaries.
bool memory_verbose_requested() {
    const char* verbose = std::getenv("NINFER_MEMORY_VERBOSE");
    return verbose != nullptr && std::string_view(verbose) == "1";
}

void print_memory_summary(const MemorySummary& memory, const char* kind,
                          std::size_t runtime_shortfall_bytes) {
    std::fprintf(
        stderr,
        // Fork: keep labels stable and byte-exact for downstream qualification capture.
        "MemorySummary kind=%s device=%d max_context=%u kv_capacity=%u kv_encoding=%s\n"
        "  sequence_capacity_bytes=%zu sequence_used_bytes=%zu\n"
        "  workspace_capacity_bytes=%zu workspace_logical_peak_bytes=%zu "
        "request_transient_capacity_bytes=%zu\n"
        "  kv_payload_bytes=%zu\n"
        "  gdn_state_device_hot_bytes=%zu gdn_state_host_checkpoint_bytes=%zu\n"
        "  cuda_graph_allowance_bytes=%zu cuda_graph_observed_bytes=%zu\n"
        "  minimum_runtime_reservation_bytes=%zu kv_capacity_increment_bytes=%zu "
        "runtime_reservation_bytes=%zu planned_slack_bytes=%zu "
        "runtime_shortfall_bytes=%zu\n"
        "  free_after_weights_bytes=%zu free_after_startup_bytes=%zu\n",
        kind, memory.device, memory.max_context, memory.kv_capacity,
        memory_kv_encoding(memory.kv_cache), memory.sequence.capacity_bytes,
        memory.sequence.used_bytes, memory.workspace.capacity_bytes,
        memory.workspace_logical_peak_bytes, memory.request_transient.capacity_bytes,
        memory.kv_payload_bytes, memory.gdn_state_device_hot_bytes,
        memory.gdn_state_host_checkpoint_bytes, memory.cuda_graph_allowance_bytes,
        memory.cuda_graph_observed_bytes, memory.minimum_runtime_reservation_bytes,
        memory.kv_capacity_increment_bytes, memory.runtime_reservation_bytes,
        memory.planned_slack_bytes, runtime_shortfall_bytes,
        memory.available_after_weights_bytes, memory.available_after_startup_bytes);
}

template <class Instance>
void print_memory_summary_if_requested(const Instance& instance) {
    if (!memory_verbose_requested()) { return; }

    MemorySummary memory             = instance.program->memory_summary();
    memory.request_transient         = instance.request_memory.summary();
    const auto& resolution           = instance.kv_capacity_resolution;
    memory.kv_capacity_mode          = resolution.mode;
    memory.kv_capacity               = resolution.resolved_tokens;
    memory.kv_capacity_page_groups     = resolution.main_page_groups;
    memory.kv_capacity_max_page_groups = resolution.maximum_main_page_groups;
    memory.minimum_runtime_reservation_bytes = resolution.minimum_runtime_reservation_bytes;
    memory.kv_capacity_increment_bytes       = resolution.bytes_per_additional_main_page_group;
    memory.runtime_reservation_bytes         = resolution.runtime_reservation_bytes;
    memory.available_after_weights_bytes     = resolution.available_after_weights_bytes;
    memory.available_after_startup_bytes     = resolution.available_after_startup_bytes;
    memory.kv_capacity_headroom_bytes        = resolution.automatic_headroom_bytes;
    memory.planned_slack_bytes               = resolution.planned_slack_bytes;

    print_memory_summary(memory, "startup", 0);
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve provisional_curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes);
    (void)runtime::resolve_kv_capacity(options.kv_capacity, provisional_curve,
                                       preflight_runtime_bytes);

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();

    std::size_t calibrated_graph_observed  = 0;
    std::size_t calibrated_graph_allowance = 0;
    if (options.use_cuda_graph) {
        // Fork: capture every reachable topology against one physical KV page per lane.
        // Fork: MTP code-warm also materializes the lazy W8 module before final free-VRAM sizing.
        auto calibration_plan    = sequence_planner.graph_calibration_plan();
        auto calibration_program =
            Target::create_program(*model, std::move(calibration_plan), device);
        device.synchronize();
        calibrated_graph_observed = calibration_program->memory_summary().cuda_graph_observed_bytes;
        calibration_program.reset();
        device.synchronize();
        calibrated_graph_allowance =
            graph_allowance(calibrated_graph_observed, kGraphSafetyMargin);
        sequence_planner.set_graph_allowance(calibrated_graph_allowance);
    }

    if (memory_verbose_requested() && options.kv_capacity.mode == KvCapacityMode::Explicit) {
        // Fork: materialize a CPU-only target plan so rejected explicit capacities still explain
        // every byte without weakening the authoritative reservation check below.
        auto diagnostic_planner =
            Target::make_sequence_planner(device, options, weights_profile);
        if (options.use_cuda_graph) {
            diagnostic_planner.set_graph_allowance(calibrated_graph_allowance);
        }
        const runtime::SequenceCapacityCurve diagnostic_curve =
            diagnostic_planner.capacity_curve();
        const std::uint64_t requested = options.kv_capacity.explicit_tokens;
        const std::uint32_t requested_pages = static_cast<std::uint32_t>(
            (requested + diagnostic_curve.main_page_tokens - 1U) /
            diagnostic_curve.main_page_tokens);
        auto diagnostic_plan = std::move(diagnostic_planner).finalize(requested_pages);
        MemorySummary memory = diagnostic_plan.planned_memory_summary();
        const std::size_t available = current_free_device_bytes();
        const std::size_t reservation = diagnostic_plan.device_reservation_bytes();
        memory.kv_capacity_mode                 = KvCapacityMode::Explicit;
        memory.kv_capacity_page_groups          = requested_pages;
        memory.kv_capacity_max_page_groups      = diagnostic_curve.maximum_main_page_groups;
        memory.minimum_runtime_reservation_bytes =
            diagnostic_curve.minimum_device_reservation_bytes;
        memory.kv_capacity_increment_bytes =
            diagnostic_curve.bytes_per_additional_main_page_group;
        memory.runtime_reservation_bytes     = reservation;
        memory.available_after_weights_bytes = available;
        memory.planned_slack_bytes = available > reservation ? available - reservation : 0;
        memory.cuda_graph_observed_bytes = calibrated_graph_observed;
        const std::size_t shortfall = reservation > available ? reservation - available : 0;
        print_memory_summary(memory, "planned", shortfall);
    }

    runtime::KvCapacityResolution capacity_resolution;
    std::unique_ptr<typename Target::Program> program;
    std::uint32_t sequence_capacity              = 0;
    std::size_t request_transient_capacity_bytes = 0;
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (attempt != 0) {
            // Fork: a final-capture OOM gets one larger calibrated reservation and full replan.
            sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
            calibrated_graph_allowance =
                graph_allowance(calibrated_graph_observed, 2ULL * kGraphSafetyMargin);
            sequence_planner.set_graph_allowance(calibrated_graph_allowance);
        }
        const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
        capacity_resolution =
            runtime::resolve_kv_capacity(options.kv_capacity, curve, current_free_device_bytes());
        auto sequence_plan =
            std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
        if (sequence_plan.device_reservation_bytes() !=
                capacity_resolution.runtime_reservation_bytes ||
            sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
            throw std::logic_error("resolved KV capacity does not match the finalized target plan");
        }
        sequence_capacity                = sequence_plan.capacity();
        request_transient_capacity_bytes = sequence_plan.request_transient_capacity_bytes();
        try {
            program = Target::create_program(*model, std::move(sequence_plan), device);
            break;
        } catch (const CudaGraphAllowanceExceeded& error) {
            if (!options.use_cuda_graph || attempt != 0) { throw; }
            calibrated_graph_observed =
                std::max(calibrated_graph_observed, error.observed_bytes());
        } catch (const CudaOutOfMemory&) {
            if (!options.use_cuda_graph || attempt != 0) { throw; }
        }
    }
    if (!program) { throw std::logic_error("final CUDA Graph Program retry did not complete"); }

    auto loaded = std::make_unique<Loaded>(std::move(model), options);
    auto instance = std::make_unique<Instance>(
        std::move(loaded), capacity_resolution, sequence_capacity,
        request_transient_capacity_bytes, std::move(program), device);
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    // Fork: print only after all Program and per-request device allocations are resident.
    print_memory_summary_if_requested(*instance);

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         std::uint32_t sequence_capacity,
                                         std::size_t request_transient_capacity_bytes,
                                         std::unique_ptr<Qwen3_6_27B::Program> stable_program,
                                         DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, request_transient_capacity_bytes), capacity(sequence_capacity),
      // Fork: Program construction precedes model ownership transfer so graph OOM can retry.
      program(std::move(stable_program)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               std::uint32_t sequence_capacity,
                                               std::size_t request_transient_capacity_bytes,
                                               std::unique_ptr<Qwen3_6_35BA3B::Program> stable_program,
                                               DeviceContext& device)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, request_transient_capacity_bytes), capacity(sequence_capacity),
      // Fork: keep final graph Program bytes stable across the instance handoff.
      program(std::move(stable_program)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
