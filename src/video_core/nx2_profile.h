// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define NX2_PROFILE_RDTSC 1
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define NX2_PROFILE_RDTSC 1
#endif
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/cpu_features.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h), other sites
    SplitWait,    ///< Split GPU thread: the back waits for work
    // Where the front waits for the back (green-hat-build-5; these were "split drain" before):
    DrainDmaRead, ///< A macro's MME DMA read (Maxwell3D::FrontDmaRead)
    DrainAcquire, ///< A semaphore acquire before it reads its word (Puller)
    DrainPoll,    ///< A suspended acquire's poll (Puller::CanResumeAcquire)
    DrainList,    ///< A command list on GPU-written pages (DmaPusher::FlushGpuWrittenCommands)
    DrainRun,     ///< Work run on the back and waited for (GpuSplit::Run)
    DrainRing,    ///< Space in the full ring (GpuSplit::WaitForSpace)
    WriterWait,   ///< The compute and copy work before a command list fetch (WaitForWriters)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "split drain", "split wait", "drain dma read", "drain acquire",
    "drain poll", "drain list", "drain run", "drain ring", "list writer wait",
};

/// The zone GpuSplit::Drain charges its wait to on this thread (the caller's site).
inline thread_local Zone t_drain_zone = Zone::SplitDrain;

/// Charges the drains in its lifetime to `zone` unless an outer site already chose one.
class DrainSite {
public:
    explicit DrainSite(Zone zone) : previous{t_drain_zone} {
        if (previous == Zone::SplitDrain) {
            t_drain_zone = zone;
        }
    }
    ~DrainSite() {
        t_drain_zone = previous;
    }
    DrainSite(const DrainSite&) = delete;
    DrainSite& operator=(const DrainSite&) = delete;

private:
    Zone previous;
};

// Set at startup by EDEN_NX2_GPU_PROFILE, or for a title by EnableForTitle (ChannelState::Init).
// Enabled() runs several times per method write: a relaxed load, no guard to check.
inline std::atomic<bool> g_enabled{std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr};

inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

/// Profiles without EDEN_NX2_GPU_PROFILE: fix/green-hat measures Ocarina of Time's GPU threads
/// this way while its gameplay performance is investigated (the zones show whether the GPU
/// threads or the host GPU hold the frame back, and how long suspended channels are polled).
inline void EnableForTitle() {
    g_enabled.store(true, std::memory_order_relaxed);
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

// What the zones measure in: the time stamp counter without the fences the wall clock reads it
// with. Ocarina of Time's split back thread enters a zone for each of ~2.4 million register writes
// a second; with the fenced reads the profile slowed its gameplay by ~10-15% (green-hat builds
// 4-7: ~34-36 ms per standing present against ~29-35 ms without the profile in build 3; Lichii
// measured the fences at ~10% of the back thread, 5f6066b2). Report converts with the window.
inline u64 Ticks() {
#ifdef NX2_PROFILE_RDTSC
    return __rdtsc();
#else
    return Now();
#endif
}

struct ThreadData {
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{}; ///< In Ticks
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 15> counters{}; ///< See Counter
    u64 child_ns{};    ///< Time of the zones nested in the current one (Ticks)
    u64 window_start{};
    u64 window_start_ticks{};
};

inline thread_local ThreadData t_data;

class Scope {
public:
    explicit Scope(Zone zone_) : zone{zone_}, enabled{Enabled()} {
        if (!enabled) {
            return;
        }
        parent_child_ns = t_data.child_ns;
        t_data.child_ns = 0;
        start = Ticks();
    }

    ~Scope() {
        if (!enabled) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        data.child_ns = parent_child_ns + elapsed;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    bool enabled;
    u64 start{};
    u64 parent_child_ns{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    UniformBinds,   ///< Graphics uniform buffer binds (BufferCache::BindHostGraphicsUniformBuffer)
    UniformBank64K, ///< Of UniformBinds: 64 KiB native banks (the bank re-upload candidates)
    UniformStreamNew,    ///< Of UniformBinds: streamed with a new allocation and copy
    UniformStreamReused, ///< Of UniformBinds: same bytes as before, stream allocation bound again
    UniformCached,       ///< Of UniformBinds: classic path (SynchronizeBuffer)
    UniformTrackerQueries, ///< Memory tracker IsRegionGpuModified calls made by those binds
    UniformStreamBytes,    ///< Guest bytes read for streamed binds (new or reused)
};

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.window_start_ticks = Ticks();
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    // Zone times are in Ticks: shares of the window's ticks, shown per second of wall time.
    const u64 window_ticks = (std::max<u64>)(Ticks() - data.window_start_ticks, 1);
    // Time in no zone at all is the window minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window_ticks > zoned ? window_ticks - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window_ticks;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    if (data.counters[8] != 0) {
        line += fmt::format(" | uniform binds {:.0f} (64 KiB banks {:.0f}, streamed {:.0f}, reused "
                            "{:.0f}, cached {:.0f}; tracker queries {:.0f}; streamed {:.0f} KiB read)",
                            static_cast<double>(data.counters[8]) * 1e9 / window,
                            static_cast<double>(data.counters[9]) * 1e9 / window,
                            static_cast<double>(data.counters[10]) * 1e9 / window,
                            static_cast<double>(data.counters[11]) * 1e9 / window,
                            static_cast<double>(data.counters[12]) * 1e9 / window,
                            static_cast<double>(data.counters[13]) * 1e9 / window,
                            static_cast<double>(data.counters[14]) * 1e9 / 1024.0 / window);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    data = {};
    data.window_start = now;
    data.window_start_ticks = Ticks();
}

} // namespace VideoCommon::Nx2Profile
