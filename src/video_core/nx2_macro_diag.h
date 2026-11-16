// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Diagnostics for one macro run on the split GPU thread's front (Maxwell3D::FrontRunMacro), by
// default 0xe2c (EDEN_NX2_MACRO_DIAG=<method in hex> picks another): the macro that makes all of
// Ocarina of Time's macro DMA reads. Active only while the GPU thread profile runs; it changes
// nothing. Front thread only: no atomics.
//
// - Its code is logged once per macro code upload (raw words, three per instruction).
// - Per call: front time, the part of it spent in its DMA read drains, register reads (MME_READ),
//   register writes, DMA reads, words its DMA reads put in the FIFO, draws it leaves pending.
// - The registers it reads and writes (logged when the sets grow), with write counts.
// - Later front work that depends on the front's copy of a register this macro wrote last:
//   other macros reading it, and the front's same-value check (FrontWrite) comparing a later
//   write against it. Run on the back, the macro would leave those copies to the back.
// - Every macro's front time, net of DMA read drains (top six), to set the macro in proportion.

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <span>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::Nx2MacroDiag {

inline constexpr u32 NumRegs = 0xE00;

struct MacroTime {
    u32 macro; ///< Macro method, 0 when free
    u64 calls;
    u64 ticks;
    u64 drain_ticks;
};

struct RegCount {
    u32 reg; ///< Register + 1, 0 when free
    u64 count;
    u32 macro; ///< Reader macro (0: the front's same-value check)
};

struct State {
    u32 target{};
    // Kept across windows.
    std::array<u16, NumRegs> last_writer{}; ///< Macro index + 1 of the last macro write, 0 else
    std::bitset<NumRegs> read_set{};
    std::bitset<NumRegs> write_set{};
    std::array<u32, NumRegs> write_counts{};
    size_t logged_reads{};
    size_t logged_writes{};
    u64 dumped_generation{~u64{0}};
    // Current call.
    u32 current{};
    bool current_is_target{};
    u64 call_start{};
    u64 call_drain{};
    // Window.
    u64 calls{}, ticks{}, drain_ticks{}, reads{}, writes{}, dma_reads{}, fifo_words{}, draws{};
    u64 cross_reads{}, front_compares{};
    std::array<RegCount, 32> dependents{};
    std::array<MacroTime, 64> macros{};
    u64 total_calls{};
    u64 window_start_ns{};
    u64 window_start_ticks{};
};

inline State g_state;

[[nodiscard]] inline u64 WallNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

[[nodiscard]] inline u32 Target() {
    static const u32 target = [] {
        const char* value = std::getenv("EDEN_NX2_MACRO_DIAG");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 16)) : 0xE2Cu;
    }();
    return target;
}

[[nodiscard]] inline u16 MacroIndex(u32 macro) {
    return static_cast<u16>(((macro - 0xE00) >> 1) + 1);
}

inline void CountDependent(u32 reg, u32 reader) {
    State& s = g_state;
    const size_t start = reg % s.dependents.size();
    for (size_t i = 0; i < s.dependents.size(); ++i) {
        RegCount& r = s.dependents[(start + i) % s.dependents.size()];
        if (r.reg == reg + 1 && r.macro == reader) {
            ++r.count;
            return;
        }
        if (r.reg == 0) {
            r = {reg + 1, 1, reader};
            return;
        }
    }
}

/// A macro starts: `code` the macro RAM (three words per instruction), `start` its first
/// instruction, `positions` every macro's start (to bound the dump), `generation` the code's.
inline void BeginMacro(u32 macro, std::span<const u32> code, std::span<const u32> positions,
                       u32 start, u64 generation) {
    State& s = g_state;
    s.target = Target();
    s.current = macro;
    s.current_is_target = macro == s.target;
    s.call_drain = 0;
    if (s.current_is_target && generation != s.dumped_generation) {
        s.dumped_generation = generation;
        u32 end = static_cast<u32>(code.size() / 3);
        for (const u32 position : positions) {
            if (position > start && position < end) {
                end = position;
            }
        }
        end = (std::min)(end, start + 256);
        LOG_INFO(HW_GPU, "nx2: macro {:#x} code (generation {}): instructions {:#x}..{:#x}, "
                         "three words each in RAM order",
                 macro, generation, start, end);
        for (u32 pc = start; pc < end; pc += 8) {
            std::string line;
            for (u32 i = pc; i < (std::min)(pc + 8, end); ++i) {
                line += fmt::format(" {:08x}{:08x}{:08x}", code[i * 3], code[i * 3 + 1],
                                    code[i * 3 + 2]);
            }
            LOG_INFO(HW_GPU, "nx2: macro {:#x} @{:#x}:{}", macro, pc, line);
        }
    }
    s.call_start = Nx2Profile::Ticks();
}

/// The running macro read a register (MME_READ).
inline void OnRegisterRead(u32 reg) {
    State& s = g_state;
    if (reg >= NumRegs) {
        return;
    }
    if (s.current_is_target) {
        ++s.reads;
        s.read_set.set(reg);
    } else if (s.last_writer[reg] == MacroIndex(s.target)) {
        ++s.cross_reads;
        CountDependent(reg, s.current);
    }
}

/// The running macro wrote a register (every emitted write, same value or not).
inline void OnRegisterWrite(u32 reg) {
    State& s = g_state;
    if (reg >= NumRegs) {
        return;
    }
    s.last_writer[reg] = MacroIndex(s.current);
    if (s.current_is_target) {
        ++s.writes;
        s.write_set.set(reg);
        ++s.write_counts[reg];
    }
}

/// A command list wrote a register on the front (not a macro): the front's copy is its own again.
inline void OnCommandListWrite(u32 reg) {
    if (reg < NumRegs) {
        g_state.last_writer[reg] = 0;
    }
}

/// FrontWrite's same-value check is about to compare a write with the front's copy of `reg`.
inline void OnFrontCompare(u32 reg, bool from_target) {
    State& s = g_state;
    if (reg < NumRegs && !from_target && s.last_writer[reg] == MacroIndex(s.target)) {
        ++s.front_compares;
        CountDependent(reg, 0);
    }
}

/// A DMA read of the running macro: its drain took `ticks`, it queued `words` FIFO words.
inline void OnDmaRead(u64 ticks, u32 words, bool fifo) {
    State& s = g_state;
    s.call_drain += ticks;
    if (s.current_is_target) {
        ++s.dma_reads;
        if (fifo) {
            s.fifo_words += words;
        }
    }
}

[[nodiscard]] inline std::string RegisterList(const std::bitset<NumRegs>& set,
                                              const std::array<u32, NumRegs>* counts) {
    std::string list;
    for (u32 reg = 0; reg < NumRegs; ++reg) {
        if (set.test(reg)) {
            list += counts != nullptr ? fmt::format(" {:#x}x{}", reg, (*counts)[reg])
                                      : fmt::format(" {:#x}", reg);
        }
    }
    return list;
}

inline void Report() {
    State& s = g_state;
    const u64 now_ns = WallNs();
    const u64 now_ticks = Nx2Profile::Ticks();
    const double window_ns = static_cast<double>((std::max<u64>)(now_ns - s.window_start_ns, 1));
    const double ticks_per_ns =
        static_cast<double>((std::max<u64>)(now_ticks - s.window_start_ticks, 1)) / window_ns;
    const double per_second = 1e9 / window_ns;
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) / ticks_per_ns / 1e6 * per_second;
    };
    const auto us_per = [&](u64 ticks, u64 n) {
        return n == 0 ? 0.0 : static_cast<double>(ticks) / ticks_per_ns / 1e3 / n;
    };
    const auto per_call = [&](u64 value) {
        return s.calls == 0 ? 0.0 : static_cast<double>(value) / s.calls;
    };
    std::string line = fmt::format(
        "nx2: macro {:#x} per second: {:.0f} calls, front {:.1f} ms of which DMA read drains {:.1f} "
        "ms (net {:.1f} ms, {:.2f} us per call); per call {:.2f} register reads, {:.2f} writes, "
        "{:.2f} DMA reads, {:.2f} FIFO words, {:.2f} draws left pending; later front work on its "
        "registers: {:.0f} reads by other macros, {:.0f} same-value checks",
        s.target, s.calls * per_second, ms(s.ticks), ms(s.drain_ticks),
        ms(s.ticks - (std::min)(s.drain_ticks, s.ticks)),
        us_per(s.ticks - (std::min)(s.drain_ticks, s.ticks), s.calls), per_call(s.reads),
        per_call(s.writes), per_call(s.dma_reads), per_call(s.fifo_words), per_call(s.draws),
        s.cross_reads * per_second, s.front_compares * per_second);
    std::array<RegCount, 32> dependents = s.dependents;
    std::sort(dependents.begin(), dependents.end(),
              [](const RegCount& a, const RegCount& b) { return a.count > b.count; });
    for (size_t i = 0; i < 6 && dependents[i].reg != 0; ++i) {
        line += fmt::format(" | reg {:#x} by {}: {:.0f}/s", dependents[i].reg - 1,
                            dependents[i].macro == 0 ? std::string{"same-value check"}
                                                     : fmt::format("{:#x}", dependents[i].macro),
                            dependents[i].count * per_second);
    }
    LOG_INFO(HW_GPU, "{}", line);
    std::array<MacroTime, 64> macros = s.macros;
    std::sort(macros.begin(), macros.end(), [](const MacroTime& a, const MacroTime& b) {
        return a.ticks - (std::min)(a.drain_ticks, a.ticks) >
               b.ticks - (std::min)(b.drain_ticks, b.ticks);
    });
    line = "nx2: front macros by net time (method: calls/s, net ms/s, drain ms/s)";
    for (size_t i = 0; i < 6 && macros[i].macro != 0; ++i) {
        const MacroTime& m = macros[i];
        line += fmt::format(" | {:#x}: {:.0f}, {:.1f}, {:.1f}", m.macro, m.calls * per_second,
                            ms(m.ticks - (std::min)(m.drain_ticks, m.ticks)), ms(m.drain_ticks));
    }
    LOG_INFO(HW_GPU, "{}", line);
    if (s.read_set.count() != s.logged_reads || s.write_set.count() != s.logged_writes) {
        s.logged_reads = s.read_set.count();
        s.logged_writes = s.write_set.count();
        LOG_INFO(HW_GPU, "nx2: macro {:#x} reads registers:{}", s.target,
                 RegisterList(s.read_set, nullptr));
        LOG_INFO(HW_GPU, "nx2: macro {:#x} writes registers (total writes):{}", s.target,
                 RegisterList(s.write_set, &s.write_counts));
    }
    s.calls = s.ticks = s.drain_ticks = s.reads = s.writes = s.dma_reads = s.fifo_words = 0;
    s.draws = s.cross_reads = s.front_compares = 0;
    s.dependents = {};
    s.macros = {};
    s.window_start_ns = now_ns;
    s.window_start_ticks = now_ticks;
}

/// The macro ended; `draw_pending` whether it left a draw for FinishFrontMacro.
inline void EndMacro(bool draw_pending) {
    State& s = g_state;
    const u64 ticks = Nx2Profile::Ticks() - s.call_start;
    if (s.current_is_target) {
        ++s.calls;
        s.ticks += ticks;
        s.drain_ticks += s.call_drain;
        if (draw_pending) {
            ++s.draws;
        }
    }
    const size_t start = (s.current >> 1) % s.macros.size();
    for (size_t i = 0; i < s.macros.size(); ++i) {
        MacroTime& m = s.macros[(start + i) % s.macros.size()];
        if (m.macro == s.current || m.macro == 0) {
            m.macro = s.current;
            ++m.calls;
            m.ticks += ticks;
            m.drain_ticks += s.call_drain;
            break;
        }
    }
    s.current = 0;
    s.current_is_target = false;
    if (s.window_start_ns == 0) {
        s.window_start_ns = WallNs();
        s.window_start_ticks = Nx2Profile::Ticks();
    } else if ((++s.total_calls & 1023) == 0 && WallNs() - s.window_start_ns >= 10'000'000'000ULL) {
        Report();
    }
}

} // namespace VideoCommon::Nx2MacroDiag
