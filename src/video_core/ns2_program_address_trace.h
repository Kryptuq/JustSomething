// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Diagnostics only (yoshi-integration, yint-29): where the program address of a rejected native
// program came from. Every write of a 3D pipeline's program address (pipelines[i].reserved[0],
// the high byte, and reserved[1], the low word) is remembered twice: as parsed from the command
// buffer (GPU address of the word, and whether the entry ran from the submit-time copy or from
// memory) and as stored into the register (the macro that stored it, if any). Nothing here
// changes behaviour.

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::NS2ProgramAddressTrace {

using Clock = std::chrono::steady_clock;

/// MAXWELL3D_REG_INDEX(pipelines) (checked by a static_assert in maxwell_3d.cpp).
inline constexpr u32 PipelineBase = 0x800;
inline constexpr u32 PipelineStride = 0x10; // 0x40 bytes per pipeline
inline constexpr u32 NumStages = 6;
inline constexpr u32 HighWord = 5;          // reserved[0]
inline constexpr u32 LowWord = 6;           // reserved[1]

inline bool IsProgramAddressMethod(u32 method) {
    if (method < PipelineBase || method >= PipelineBase + NumStages * PipelineStride) [[likely]] {
        return false;
    }
    const u32 word = (method - PipelineBase) % PipelineStride;
    return word == HighWord || word == LowWord;
}

/// Slot of a program address register: stage * 2 + (0 high, 1 low).
inline size_t Slot(u32 method) {
    const u32 relative = method - PipelineBase;
    return (relative / PipelineStride) * 2 + (relative % PipelineStride - HighWord);
}

/// yint-33: one macro register read (method, value).
struct RegisterRead {
    u32 method = 0;
    u32 value = 0;
};

struct Write {
    bool valid = false;
    u32 value = 0;
    u64 address = 0;              ///< command word GPU address (parsed) / engine segment (stored)
    const char* source = nullptr; ///< which copy of the entry was parsed (parsed writes)
    s32 channel = -1;
    u32 macro = 0;                ///< macro method that stored it (stored writes), 0 = none
    Clock::time_point time{};
    // yint-33, stored writes made by a macro: the call's parameters (first 32), the GPU address
    // they were read from, which copy of the command list entry, and the registers the macro
    // read before this store (first 16).
    std::array<u32, 32> params{};
    u32 param_count = 0;
    u64 param_address = 0;
    const char* param_source = nullptr;
    std::array<RegisterRead, 16> reads{};
    u32 read_count = 0;
};

struct Ring {
    std::array<Write, 16> writes{};
    u32 next = 0;
    void Add(const Write& write) {
        writes[next++ % writes.size()] = write;
    }
    /// Newest write of this value.
    std::optional<Write> Find(u32 value) const {
        for (u32 i = 0; i < writes.size(); ++i) {
            const Write& write = writes[(next + writes.size() - 1 - i) % writes.size()];
            if (write.valid && write.value == value) {
                return write;
            }
        }
        return std::nullopt;
    }
    u32 Count() const {
        u32 count = 0;
        for (const Write& write : writes) {
            count += write.valid ? 1 : 0;
        }
        return count;
    }
};

/// yint-33: an MME DMA read (a macro loading memory into its data RAM or load FIFO).
struct MmeDmaRead {
    bool valid = false;
    u32 macro = 0;
    u64 source = 0;
    u32 ram = 0;
    u32 count = 0;
    bool fifoed = false;
    bool flushed = false; ///< the words were GPU-written and flushed first
    std::array<u32, 4> words{};
    Clock::time_point time{};
};

/// yint-33: a copy-engine copy or inline upload into GPU memory.
struct MemoryWrite {
    bool valid = false;
    const char* kind = "";
    u64 destination = 0;
    u64 size = 0;
    u64 source = 0;          ///< copy source (0 for inline uploads)
    bool source_gpu = false; ///< the copy source lies on pages the GPU has written
    Clock::time_point time{};
};

struct Built {
    Clock::time_point time{};
    u64 hash = 0;
    u32 count = 0;
};

struct State {
    std::mutex mutex;
    std::array<Ring, NumStages * 2> parsed;
    std::array<Ring, NumStages * 2> stored;
    // yint-33
    std::array<std::array<u32, 0xc00>, NumStages * 2> data_ram{}; ///< at each slot's newest store
    std::array<MmeDmaRead, 64> dma_reads{};
    u32 next_dma_read = 0;
    std::array<MemoryWrite, 1024> memory_writes{};
    u32 next_memory_write = 0;
    std::unordered_map<u64, Built> built;
    std::unordered_map<u64, u32> reports;
};
inline State& GetState() {
    static State state;
    return state;
}

/// Which copy of the current command buffer entry is being parsed on this thread.
inline const char*& EntrySource() {
    thread_local const char* source = "unknown copy";
    return source;
}
/// The macro executing on this thread (its method), 0 = none.
inline u32& CurrentMacro() {
    thread_local u32 macro = 0;
    return macro;
}
/// yint-33: the parameters and register reads of the macro call executing on this thread.
struct MacroCall {
    std::span<const u32> params;
    u64 param_address = 0;
    const char* param_source = nullptr;
    std::array<RegisterRead, 16> reads{};
    u32 read_count = 0;
};
inline MacroCall*& CurrentCall() {
    thread_local MacroCall* call = nullptr;
    return call;
}
struct MacroScope {
    u32 previous;
    MacroCall* previous_call;
    MacroCall call;
    explicit MacroScope(u32 macro, std::span<const u32> params = {}, u64 param_address = 0)
        : previous{CurrentMacro()}, previous_call{CurrentCall()},
          call{params, param_address, EntrySource(), {}, 0} {
        CurrentMacro() = macro;
        CurrentCall() = &call;
    }
    ~MacroScope() {
        CurrentMacro() = previous;
        CurrentCall() = previous_call;
    }
    MacroScope(const MacroScope&) = delete;
    MacroScope& operator=(const MacroScope&) = delete;
};

/// yint-33: a register the executing macro read.
inline void NoteMacroRead(u32 method, u32 value) {
    if (MacroCall* const call = CurrentCall(); call != nullptr && call->read_count < call->reads.size())
        [[unlikely]] {
        call->reads[call->read_count++] = RegisterRead{method, value};
    }
}

/// yint-33: an MME DMA read (first words as read, after any flush).
inline void NoteMmeDmaRead(u64 source, u32 ram, u32 count, bool fifoed, bool flushed,
                           std::span<const u32> words) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    MmeDmaRead& read = state.dma_reads[state.next_dma_read++ % state.dma_reads.size()];
    read = MmeDmaRead{true, CurrentMacro(), source, ram, count, fifoed, flushed, {}, Clock::now()};
    std::copy_n(words.begin(), std::min<size_t>(words.size(), read.words.size()), read.words.begin());
}

/// yint-33: a copy-engine copy or inline upload.
inline void NoteMemoryWrite(const char* kind, u64 destination, u64 size, u64 source, bool source_gpu) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.memory_writes[state.next_memory_write++ % state.memory_writes.size()] =
        MemoryWrite{true, kind, destination, size, source, source_gpu, Clock::now()};
}

/// yint-33: a native program built (lowered and translated) at this address.
inline void NoteBuilt(u64 address, u64 hash) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (state.built.size() < 65536 || state.built.contains(address)) {
        Built& built = state.built[address];
        built.time = Clock::now();
        built.hash = hash;
        ++built.count;
    }
}

/// A 3D method parsed from a command buffer word (address 0: an inline header).
inline void NoteParsed(u32 method, u32 value, u64 word_address, s32 channel) {
    if (!IsProgramAddressMethod(method)) [[likely]] {
        return;
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.parsed[Slot(method)].Add(
        Write{true, value, word_address, EntrySource(), channel, 0, Clock::now()});
}

/// A 3D register store (data_ram: the engine's MME data RAM, kept for the slot's newest store).
inline void NoteStored(u32 method, u32 value, u64 segment, const std::array<u32, 0xc00>* data_ram = nullptr) {
    if (!IsProgramAddressMethod(method)) [[likely]] {
        return;
    }
    Write write{true, value, segment, nullptr, -1, CurrentMacro(), Clock::now()};
    if (const MacroCall* const call = CurrentCall(); call != nullptr && write.macro != 0) {
        write.param_count = static_cast<u32>(std::min<size_t>(call->params.size(), write.params.size()));
        std::copy_n(call->params.begin(), write.param_count, write.params.begin());
        write.param_address = call->param_address;
        write.param_source = call->param_source;
        write.reads = call->reads;
        write.read_count = call->read_count;
    }
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    state.stored[Slot(method)].Add(write);
    if (data_ram != nullptr) {
        state.data_ram[Slot(method)] = *data_ram;
    }
}

struct Provenance {
    std::optional<Write> parsed;
    std::optional<Write> stored;
    u32 parsed_count = 0;
};
inline Provenance Find(size_t stage, bool high, u32 value) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const size_t slot = stage * 2 + (high ? 0 : 1);
    return Provenance{state.parsed[slot].Find(value), state.stored[slot].Find(value),
                      state.parsed[slot].Count()};
}

/// yint-33: the newest stores of a slot, newest first.
inline std::vector<Write> StoredHistory(size_t stage, bool high) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const Ring& ring = state.stored[stage * 2 + (high ? 0 : 1)];
    std::vector<Write> writes;
    for (u32 i = 0; i < ring.writes.size(); ++i) {
        const Write& write = ring.writes[(ring.next + ring.writes.size() - 1 - i) % ring.writes.size()];
        if (write.valid) {
            writes.push_back(write);
        }
    }
    return writes;
}

/// yint-33: the data RAM at the slot's newest store.
inline std::array<u32, 0xc00> DataRamAtStore(size_t stage, bool high) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    return state.data_ram[stage * 2 + (high ? 0 : 1)];
}

/// yint-33: copies/uploads overlapping [address, address + size), newest first.
inline std::vector<MemoryWrite> WritesOverlapping(u64 address, u64 size) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    std::vector<MemoryWrite> writes;
    for (u32 i = 0; i < state.memory_writes.size(); ++i) {
        const MemoryWrite& write =
            state.memory_writes[(state.next_memory_write + state.memory_writes.size() - 1 - i) %
                                state.memory_writes.size()];
        if (write.valid && write.destination < address + size && address < write.destination + write.size) {
            writes.push_back(write);
        }
    }
    return writes;
}

/// yint-33: MME DMA reads, newest first.
inline std::vector<MmeDmaRead> DmaReads() {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    std::vector<MmeDmaRead> reads;
    for (u32 i = 0; i < state.dma_reads.size(); ++i) {
        const MmeDmaRead& read =
            state.dma_reads[(state.next_dma_read + state.dma_reads.size() - 1 - i) % state.dma_reads.size()];
        if (read.valid) {
            reads.push_back(read);
        }
    }
    return reads;
}

inline std::optional<Built> FindBuilt(u64 address) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    const auto it = state.built.find(address);
    return it == state.built.end() ? std::nullopt : std::optional<Built>{it->second};
}

/// yint-33: counts reports per address; true while this address may still be reported.
inline bool TakeReport(u64 address, u32 per_address, u32 total) {
    State& state = GetState();
    std::scoped_lock lock{state.mutex};
    static u32 reported = 0;
    if (reported >= total) {
        return false;
    }
    u32& count = state.reports[address];
    if (count >= per_address) {
        return false;
    }
    ++count;
    ++reported;
    return true;
}

} // namespace VideoCommon::NS2ProgramAddressTrace
