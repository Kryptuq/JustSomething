// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Ocarina of Time's conditional rendering macro on the split GPU thread's back (green-hat-build-16).
//
// With macros on the split's front (Maxwell3D::FrontRunMacro), Ocarina's macro 0xe2c reads a
// 24-byte report compare (MME_DMA_READ_FIFOED) and drains the whole split for it ~16,000 times a
// second. Its code (decoded in green-hat-build-15) writes SET_MME_MEM_ADDRESS_A/B (0x154/0x155)
// from its parameters, MME_DMA_READ_FIFOED (0x158) = 6, then render_enable.mode (0x556) = p0, or
// p0 ^ 1 when the two 64-bit values read differ. Only this exact code (kCode) is ever handled here:
// any other code, macro slot or parameter count runs the original way.
//
// EDEN_NX2_BACK_MACROS:
//   unset, "2" or "verify": verify (default). The original front run (full drain, front read) stays
//       authoritative. The back additionally runs the same code at the macro's queue position without
//       side effects and compares its writes, the words it read and render_enable.mode afterwards
//       with the original's; mismatches are logged and counted, never used.
//   "1", "on" or "run": run on the back. The front queues the call instead of draining; the back
//       runs it at that queue position with the engine's normal method path. The front sets its copy
//       of 0x154/0x155/0x158 (known from the parameters) and leaves render_enable.mode (register and
//       shadow RAM entry) to the back until it needs it (Maxwell3D::FrontSyncBackOwned: drain, copy).
//   "0" or "off": nothing of this; the original path only.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>

#include "common/common_types.h"
#include "common/logging.h"

namespace Tegra::Engines::Nx2BackMacro {

enum class Mode : u8 { Off, Run, Verify };

[[nodiscard]] inline Mode ParseMode(const char* value) {
    if (value == nullptr) {
        return Mode::Verify;
    }
    const std::string_view v{value};
    if (v == "0" || v == "off") {
        return Mode::Off;
    }
    if (v == "1" || v == "on" || v == "run") {
        return Mode::Run;
    }
    return Mode::Verify; // "2", "verify" and anything unrecognized
}

[[nodiscard]] inline Mode GetMode() {
    static const Mode mode = ParseMode(std::getenv("EDEN_NX2_BACK_MACROS"));
    return mode;
}

inline constexpr u32 kMacro = 0xE2C;
inline constexpr u32 kInstructions = 9;
inline constexpr u32 kParameters = 3;
inline constexpr u32 kReadWords = 6;
inline constexpr u32 kMemAddressA = 0x154;
inline constexpr u32 kMemAddressB = 0x155;
inline constexpr u32 kDmaReadFifoed = 0x158;
inline constexpr u32 kRenderEnableMode = 0x556;

/// The macro's code in macro RAM order (three words per instruction), as logged by build 15.
inline constexpr std::array<u32, kInstructions * 3> kCode{
    0x00940003, 0x18c00455, 0x31c10300, 0x8a600003, 0x18c01856, 0x318c0300,
    0x00000007, 0x18c00400, 0x359c1f00, 0x00000003, 0x1d000000, 0x31c08300,
    0x00000003, 0x18c00000, 0x3bcc0300, 0x00000003, 0xa003c000, 0x3810bf00,
    0x00000003, 0x18c00000, 0x0010bb00, 0x0018155b, 0x18c00000, 0x72213c29,
    0x00200003, 0x18c00000, 0x302c0300,
};

/// Whether the macro code at instruction `start` is exactly kCode.
[[nodiscard]] inline bool CodeMatches(std::span<const u32> code, u32 start) {
    const size_t first = size_t{start} * 3;
    return first + kCode.size() <= code.size() &&
           std::memcmp(code.data() + first, kCode.data(), sizeof(kCode)) == 0;
}

enum class Fallback : u8 { Code, Parameters, Yoshi, Count };

/// Counters (front and back threads: relaxed atomics), logged every 10 s while profiling.
struct Counters {
    std::atomic<u64> calls{};           ///< Calls of the macro slot with the matching code
    std::array<std::atomic<u64>, static_cast<size_t>(Fallback::Count)> fallbacks{};
    std::atomic<u64> back_runs{};       ///< Run mode: calls the back ran
    std::atomic<u64> back_failures{};   ///< Run mode: the back's interpreter failed
    std::atomic<u64> back_flushes{};    ///< Back runs whose words the buffer cache had to flush
    std::atomic<u64> back_ns{};         ///< Back time in those runs (wall clock)
    std::atomic<u64> syncs{};           ///< Front drains to take render_enable.mode back
    std::atomic<u64> sync_ns{};
    std::atomic<u64> verified{};        ///< Verify mode: comparisons made
    std::atomic<u64> mismatches{};      ///< Of verified: writes or register differ
    std::atomic<u64> word_differences{}; ///< Of verified: the words read differ (report timing)
    std::atomic<u64> verify_dirty{};    ///< Of verified: back read GPU-modified words (not flushed)
    std::atomic<u64> verify_failures{}; ///< Verify mode: the back's dry run failed
    std::atomic<u64> verify_ns{};       ///< Back time in dry runs and comparisons
};

inline Counters g_counters;

inline void Add(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

/// Front thread, while profiling: the counters per second every 10 s.
inline void MaybeReport() {
    static u64 window_start{};
    static u64 checks{};
    static std::array<u64, 21> last{};
    if ((++checks & 1023) != 0) {
        return;
    }
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    if (window_start == 0) {
        window_start = now;
        return;
    }
    if (now - window_start < 10'000'000'000ULL) {
        return;
    }
    Counters& c = g_counters;
    const std::array<u64, 21> values{
        c.calls.load(), c.fallbacks[0].load(), c.fallbacks[1].load(), c.fallbacks[2].load(),
        c.back_runs.load(), c.back_failures.load(), c.back_flushes.load(), c.back_ns.load(),
        c.syncs.load(), c.sync_ns.load(), c.verified.load(), c.mismatches.load(),
        c.word_differences.load(), c.verify_dirty.load(), c.verify_failures.load(),
        c.verify_ns.load(), 0, 0, 0, 0, 0};
    const double seconds = static_cast<double>(now - window_start) / 1e9;
    const auto rate = [&](size_t i) { return static_cast<double>(values[i] - last[i]) / seconds; };
    const auto ms = [&](size_t i) { return static_cast<double>(values[i] - last[i]) / 1e6 / seconds; };
    constexpr std::array<const char*, 3> mode_names{"off", "run", "verify"};
    LOG_INFO(HW_GPU,
             "nx2: back macro {:#x} ({}) per second: {:.0f} calls; fallbacks code {:.0f}, "
             "parameters {:.0f}, Yoshi {:.0f}; back runs {:.0f} ({:.0f} failed, {:.0f} flushed), "
             "back {:.2f} ms; front takebacks {:.0f} ({:.2f} ms); verified {:.0f}, mismatches "
             "{:.0f}, word differences {:.0f}, GPU-modified {:.0f}, dry run failures {:.0f}, "
             "verify back {:.2f} ms; totals: verified {}, mismatches {}",
             kMacro, mode_names[static_cast<size_t>(GetMode())], rate(0), rate(1), rate(2), rate(3),
             rate(4), rate(5), rate(6), ms(7), rate(8), ms(9), rate(10), rate(11), rate(12),
             rate(13), rate(14), ms(15), values[10], values[11]);
    last = values;
    window_start = now;
}

} // namespace Tegra::Engines::Nx2BackMacro
