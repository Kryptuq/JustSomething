// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Diagnostics only (yoshi-int-build-34, D4): a ring of buffer cache events that change whether a
// range holds GPU-written data the guest has not seen yet (marked GPU-modified, read trap armed,
// downloaded at a fence or by a flush, cleared by a CPU write or a copy). A rejected native
// program's report lists the events overlapping its memory, so it shows how the GPU-written code
// was lost before the shader loader read it. Recording is a few relaxed stores; nothing reads the
// ring except that report.

#include <array>
#include <atomic>
#include <chrono>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::NS2BufferEvents {

enum class Kind : u32 {
    MarkedGpuModifiedFence,     ///< GPU write, downloaded at the next fence
    MarkedGpuModifiedTrap,      ///< GPU write, read trap armed (not downloaded at fences)
    CopyMarkedFence,            ///< copy engine copy of GPU data, downloaded at the next fence
    CopyMarkedTrap,             ///< copy engine copy of GPU data, read trap armed
    CopyDestinationCleared,     ///< copy engine copy: destination's GPU state cleared first
    ClearDestinationCleared,    ///< DMA clear: destination's GPU state cleared
    CpuWriteCleared,            ///< guest/emulator write cleared the GPU state (WriteMemory)
    FlushDownloaded,            ///< DownloadBufferMemory copied this range to guest memory
    FenceDownloaded,            ///< fence download (committed ranges) copied it to guest memory
    OtherCleared,               ///< any other subtraction of GPU state (line names it)
    UploadedToHost,             ///< guest memory uploaded into the host buffer (SynchronizeBuffer)
};

inline const char* Name(Kind kind) {
    switch (kind) {
    case Kind::MarkedGpuModifiedFence: return "GPU write (fence download)";
    case Kind::MarkedGpuModifiedTrap: return "GPU write (read trap, no fence download)";
    case Kind::CopyMarkedFence: return "copy of GPU data (fence download)";
    case Kind::CopyMarkedTrap: return "copy of GPU data (read trap, no fence download)";
    case Kind::CopyDestinationCleared: return "copy destination cleared";
    case Kind::ClearDestinationCleared: return "DMA clear cleared";
    case Kind::CpuWriteCleared: return "CPU write cleared";
    case Kind::FlushDownloaded: return "flush downloaded";
    case Kind::FenceDownloaded: return "fence downloaded";
    case Kind::OtherCleared: return "cleared";
    case Kind::UploadedToHost: return "guest memory uploaded to host buffer";
    }
    return "?";
}

struct Event {
    u64 time_us = 0;
    u64 address = 0;
    u64 size = 0;
    Kind kind{};
    u32 line = 0;
};

using Clock = std::chrono::steady_clock;

struct Ring {
    static constexpr size_t Size = 1 << 16;
    std::array<Event, Size> events{};
    std::atomic<u64> next{0};
    Clock::time_point start = Clock::now();
};

inline Ring& GetRing() {
    static Ring ring;
    return ring;
}

inline u64 NowMicros() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                Clock::now() - GetRing().start)
                                .count());
}

inline void Note(Kind kind, u64 address, u64 size, u32 line) {
    if (size == 0) {
        return;
    }
    Ring& ring = GetRing();
    const u64 index = ring.next.fetch_add(1, std::memory_order_relaxed);
    ring.events[index % Ring::Size] = Event{NowMicros(), address, size, kind, line};
}

/// Events overlapping [address, address + size), newest first, at most `limit`.
inline std::vector<Event> Overlapping(u64 address, u64 size, size_t limit) {
    Ring& ring = GetRing();
    const u64 end = ring.next.load(std::memory_order_relaxed);
    const u64 count = end < Ring::Size ? end : Ring::Size;
    std::vector<Event> found;
    for (u64 i = 0; i < count && found.size() < limit; ++i) {
        const Event event = ring.events[(end - 1 - i) % Ring::Size];
        if (event.address < address + size && address < event.address + event.size) {
            found.push_back(event);
        }
    }
    return found;
}

/// Age of the oldest event still in the ring (how far back "no event" reaches), in microseconds.
inline u64 RingSpanMicros() {
    Ring& ring = GetRing();
    const u64 end = ring.next.load(std::memory_order_relaxed);
    if (end == 0) {
        return 0;
    }
    const u64 oldest = end < Ring::Size ? 0 : end - Ring::Size;
    return NowMicros() - ring.events[oldest % Ring::Size].time_us;
}

/// Bytes the flushes on this thread downloaded (reset by the D4 report around its flush).
inline u64& ThreadDownloadedBytes() {
    thread_local u64 bytes = 0;
    return bytes;
}

} // namespace VideoCommon::NS2BufferEvents
