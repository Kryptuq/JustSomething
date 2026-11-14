// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Diagnostics for the macro DMA reads on the split GPU thread's front (Maxwell3D::FrontDmaRead),
// which drain the whole split before reading. Active only while the GPU thread profile runs
// (EDEN_NX2_GPU_PROFILE / Nx2Profile::EnableForTitle); it changes no synchronization.
//
// The front notes, with its ring position, every entry it sends that can write guest memory: with
// the written range where the front knows it (constant buffer data, 3D report semaphores, puller
// semaphore releases, pitch inline uploads), without where only the back knows it (draws and
// clears, compute, DMA copies, I2M and 2D work, block-linear inline uploads). Before each drain
// the read is put in one bucket:
// - empty:     nothing queued, the drain had nothing to wait for
// - overlap:   a queued entry writes the words read (a narrower wait must wait for it)
// - any:       no queued write to those words, but queued work whose writes only the back knows
//              (a wait for "anything that may write" must wait for the last of it)
// - unrelated: nothing queued can write guest memory (a narrower wait would not wait at all)
// - overflow:  more noted writes are queued than this keeps: unknown, counted apart
// Per bucket: reads, drain time, queued ring words and the words past the last entry a narrower
// wait would need (an estimate of the drain time it would not wait for, assuming the back runs
// the queue at an even pace). Everything here runs on the front (one thread): no atomics.

#include <algorithm>
#include <array>
#include <chrono>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::Nx2DmaReadDiag {

enum class Writer : u8 {
    CbData,       ///< Constant buffer data words (range)
    Report,       ///< 3D report semaphore / query (range, 16 bytes)
    PullerRelease, ///< Puller semaphore release (range, 16 bytes)
    InlineUpload, ///< 3D inline upload (range when pitch, else unranged)
    Draw,         ///< Draw manager methods: draws, clears (unranged)
    Compute,      ///< Compute methods that may write memory (unranged)
    Copy,         ///< DMA engine work (unranged)
    MemoryEngine, ///< I2M (KeplerMemory) work (unranged)
    Fermi2D,      ///< 2D engine work (unranged)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Writer::Count)> writer_names{
    "cb data", "report", "puller release", "inline upload", "draw", "compute", "copy", "i2m", "2d",
};

enum class Bucket : u8 { Empty, Overlap, Any, Unrelated, Overflow, Count };

inline constexpr std::array<const char*, static_cast<size_t>(Bucket::Count)> bucket_names{
    "empty", "overlap", "any", "unrelated", "overflow",
};

struct Range {
    u64 position; ///< Ring position after the entry: run once the back executed this far
    GPUVAddr begin;
    GPUVAddr end;
    Writer writer;
};

struct MacroStats {
    u32 macro; ///< Macro method (0xE00 + 2 * index), 0 when the slot is free
    u64 reads;
    u64 bytes;
    u64 ticks;
    std::array<u64, static_cast<size_t>(Bucket::Count)> buckets;
};

struct PageStats {
    u64 page; ///< 4 KiB page + 1, 0 when free
    u64 reads;
    u32 macro;
};

struct State {
    // Notes (kept across report windows).
    std::array<Range, 1024> ranges{};
    size_t range_next{};
    std::array<u64, static_cast<size_t>(Writer::Count)> last_unranged{};
    std::array<GPUVAddr, 64> report_addresses{}; ///< Recent report / release addresses
    size_t report_next{};
    u32 current_macro{};

    // Window statistics.
    u64 reads{};
    u64 bytes{};
    std::array<u64, static_cast<size_t>(Bucket::Count)> bucket_reads{};
    std::array<u64, static_cast<size_t>(Bucket::Count)> bucket_ticks{};
    std::array<u64, static_cast<size_t>(Bucket::Count)> bucket_depth{};
    std::array<u64, static_cast<size_t>(Bucket::Count)> bucket_spare{};
    std::array<u64, static_cast<size_t>(Bucket::Count)> bucket_spare_ticks{};
    std::array<u64, static_cast<size_t>(Bucket::Count)> bucket_dirty{};
    std::array<u64, static_cast<size_t>(Writer::Count)> overlap_by{};  ///< Latest overlapping
    std::array<u64, static_cast<size_t>(Writer::Count)> limit_by{};    ///< Last needed entry
    std::array<u64, static_cast<size_t>(Writer::Count)> pending_at_read{};
    u64 any_only_draws{}; ///< "any" reads whose unranged queued work is draws only
    u64 source_current_cb{};
    u64 source_report{};
    std::array<MacroStats, 64> macros{};
    std::array<PageStats, 128> pages{};
    u64 window_start_ns{};
    u64 window_start_ticks{};
};

inline State g_state;

[[nodiscard]] inline u64 WallNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// The front sent an entry (ending at ring `position`) that writes [address, address + size).
inline void NoteRange(Writer writer, u64 position, GPUVAddr address, u64 size) {
    State& s = g_state;
    s.ranges[s.range_next++ % s.ranges.size()] = {position, address, address + size, writer};
    if (writer == Writer::Report || writer == Writer::PullerRelease) {
        s.report_addresses[s.report_next++ % s.report_addresses.size()] = address;
    }
}

/// The front sent an entry (ending at ring `position`) whose writes only the back knows.
inline void NoteUnranged(Writer writer, u64 position) {
    g_state.last_unranged[static_cast<size_t>(writer)] = position;
}

struct Sample {
    u64 start_ticks;
    Bucket bucket;
    u64 depth;
    u64 spare; ///< Queued words past the last entry a narrower wait would need
    u32 macro;
    GPUVAddr source;
    u64 bytes;
};

/// Before the drain: what is queued, `executed` the back's position, `written` the front's.
[[nodiscard]] inline Sample BeginRead(u64 executed, u64 written, GPUVAddr source, u64 bytes,
                                      GPUVAddr cb_address, u64 cb_size) {
    State& s = g_state;
    Sample sample{};
    sample.macro = s.current_macro;
    sample.source = source;
    sample.bytes = bytes;
    sample.depth = written > executed ? written - executed : 0;
    const GPUVAddr end = source + bytes;
    if (cb_size != 0 && source < cb_address + cb_size && cb_address < end) {
        ++s.source_current_cb;
    }
    for (const GPUVAddr report : s.report_addresses) {
        if (report != 0 && source < report + 16 && report < end) {
            ++s.source_report;
            break;
        }
    }
    if (sample.depth == 0) {
        sample.bucket = Bucket::Empty;
        sample.start_ticks = Nx2Profile::Ticks();
        return sample;
    }
    u64 need = 0;
    Writer need_by = Writer::Count;
    Writer overlap_by = Writer::Count;
    u64 overlap_position = 0;
    // Notes are in ring order: walk back from the newest while they are still queued (the cost
    // follows what is queued, not the table size). The first overlap found is the latest.
    bool overflow = true;
    const size_t kept = (std::min)(s.range_next, s.ranges.size());
    for (size_t i = 1; i <= kept; ++i) {
        const Range& range = s.ranges[(s.range_next - i) % s.ranges.size()];
        if (range.position <= executed) {
            overflow = false;
            break;
        }
        if (overlap_by == Writer::Count && range.begin < end && source < range.end) {
            overlap_position = range.position;
            overlap_by = range.writer;
        }
    }
    // Every kept note still queued: older queued ones may have been overwritten (unknown), unless
    // fewer notes than the table holds were ever made.
    overflow &= s.range_next >= s.ranges.size();
    if (overlap_by != Writer::Count) {
        need = overlap_position;
        need_by = overlap_by;
        ++s.overlap_by[static_cast<size_t>(overlap_by)];
    }
    bool any = false;
    bool only_draws = true;
    for (size_t i = 0; i < s.last_unranged.size(); ++i) {
        if (s.last_unranged[i] > executed) {
            any = true;
            only_draws &= i == static_cast<size_t>(Writer::Draw);
            ++s.pending_at_read[i];
            if (s.last_unranged[i] > need) {
                need = s.last_unranged[i];
                need_by = static_cast<Writer>(i);
            }
        }
    }
    if (overlap_by != Writer::Count) {
        sample.bucket = Bucket::Overlap;
    } else if (overflow) {
        sample.bucket = Bucket::Overflow;
        need = written;
    } else if (any) {
        sample.bucket = Bucket::Any;
        if (only_draws) {
            ++s.any_only_draws;
        }
    } else {
        sample.bucket = Bucket::Unrelated;
    }
    if (need_by != Writer::Count && sample.bucket != Bucket::Overflow) {
        ++s.limit_by[static_cast<size_t>(need_by)];
    }
    sample.spare = sample.bucket == Bucket::Unrelated ? sample.depth
                   : written > need                 ? written - need
                                                    : 0;
    sample.start_ticks = Nx2Profile::Ticks();
    return sample;
}

inline void Report() {
    State& s = g_state;
    const u64 now_ns = WallNs();
    const u64 now_ticks = Nx2Profile::Ticks();
    const double window_ns = static_cast<double>((std::max<u64>)(now_ns - s.window_start_ns, 1));
    const double ticks_per_ns =
        static_cast<double>((std::max<u64>)(now_ticks - s.window_start_ticks, 1)) / window_ns;
    const double per_second = 1e9 / window_ns;
    const auto ms = [&](u64 ticks) { return static_cast<double>(ticks) / ticks_per_ns / 1e6 * per_second; };
    std::string line = fmt::format("nx2: macro DMA reads per second: {:.0f} reads, {:.0f} bytes; "
                                   "source in the bound constant buffer {:.0f}, at a report address {:.0f}",
                                   s.reads * per_second, s.bytes * per_second,
                                   s.source_current_cb * per_second, s.source_report * per_second);
    for (size_t b = 0; b < s.bucket_reads.size(); ++b) {
        const u64 n = s.bucket_reads[b];
        if (n == 0) {
            continue;
        }
        line += fmt::format(" | {} {:.0f}/s drain {:.1f} ms, queued {:.0f} words, spare {:.0f} words "
                            "(~{:.1f} ms), dirty {:.0f}/s",
                            bucket_names[b], n * per_second, ms(s.bucket_ticks[b]),
                            static_cast<double>(s.bucket_depth[b]) / n,
                            static_cast<double>(s.bucket_spare[b]) / n, ms(s.bucket_spare_ticks[b]),
                            s.bucket_dirty[b] * per_second);
    }
    LOG_INFO(HW_GPU, "{}", line);
    line = fmt::format("nx2: macro DMA read writers per second: any-only-draws {:.0f}",
                       s.any_only_draws * per_second);
    for (size_t w = 0; w < writer_names.size(); ++w) {
        if (s.overlap_by[w] == 0 && s.limit_by[w] == 0 && s.pending_at_read[w] == 0) {
            continue;
        }
        line += fmt::format(" | {}: overlaps {:.0f}, last needed {:.0f}, queued (unranged) {:.0f}",
                            writer_names[w], s.overlap_by[w] * per_second,
                            s.limit_by[w] * per_second, s.pending_at_read[w] * per_second);
    }
    LOG_INFO(HW_GPU, "{}", line);
    std::array<MacroStats, 64> macros = s.macros;
    std::sort(macros.begin(), macros.end(),
              [](const MacroStats& a, const MacroStats& b) { return a.reads > b.reads; });
    line = "nx2: macro DMA reads by macro (method: reads/s, bytes/read, drain ms/s, "
           "empty/overlap/any/unrelated/overflow per s)";
    for (size_t i = 0; i < 6 && macros[i].reads != 0; ++i) {
        const MacroStats& m = macros[i];
        line += fmt::format(" | {:#x}: {:.0f}, {:.0f}, {:.1f}, {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}",
                            m.macro, m.reads * per_second,
                            static_cast<double>(m.bytes) / m.reads, ms(m.ticks),
                            m.buckets[0] * per_second, m.buckets[1] * per_second,
                            m.buckets[2] * per_second, m.buckets[3] * per_second,
                            m.buckets[4] * per_second);
    }
    LOG_INFO(HW_GPU, "{}", line);
    std::array<PageStats, 128> pages = s.pages;
    std::sort(pages.begin(), pages.end(),
              [](const PageStats& a, const PageStats& b) { return a.reads > b.reads; });
    line = "nx2: macro DMA read source pages (page: reads/s, macro)";
    for (size_t i = 0; i < 6 && pages[i].reads != 0; ++i) {
        line += fmt::format(" | {:#x}: {:.0f}, {:#x}", (pages[i].page - 1) << 12,
                            pages[i].reads * per_second, pages[i].macro);
    }
    LOG_INFO(HW_GPU, "{}", line);
    // Keep the notes; restart the window.
    const auto ranges = s.ranges;
    const size_t range_next = s.range_next;
    const auto last_unranged = s.last_unranged;
    const auto report_addresses = s.report_addresses;
    const size_t report_next = s.report_next;
    const u32 current_macro = s.current_macro;
    s = {};
    s.ranges = ranges;
    s.range_next = range_next;
    s.last_unranged = last_unranged;
    s.report_addresses = report_addresses;
    s.report_next = report_next;
    s.current_macro = current_macro;
    s.window_start_ns = now_ns;
    s.window_start_ticks = now_ticks;
}

/// After the drain (and the dirty check the read makes anyway).
inline void EndRead(const Sample& sample, bool dirty) {
    State& s = g_state;
    const u64 ticks = Nx2Profile::Ticks() - sample.start_ticks;
    const size_t b = static_cast<size_t>(sample.bucket);
    ++s.reads;
    s.bytes += sample.bytes;
    ++s.bucket_reads[b];
    s.bucket_ticks[b] += ticks;
    s.bucket_depth[b] += sample.depth;
    s.bucket_spare[b] += sample.spare;
    if (sample.depth != 0) {
        s.bucket_spare_ticks[b] += ticks * sample.spare / sample.depth;
    }
    if (dirty) {
        ++s.bucket_dirty[b];
    }
    // Per macro (small open-addressed table; a full table drops new macros).
    const size_t start = (sample.macro >> 1) % s.macros.size();
    for (size_t i = 0; i < s.macros.size(); ++i) {
        MacroStats& m = s.macros[(start + i) % s.macros.size()];
        if (m.macro == sample.macro || m.macro == 0) {
            m.macro = sample.macro;
            ++m.reads;
            m.bytes += sample.bytes;
            m.ticks += ticks;
            ++m.buckets[b];
            break;
        }
    }
    const u64 page = (sample.source >> 12) + 1;
    const size_t page_start = (page * 0x9E3779B97F4A7C15ULL >> 57) % s.pages.size();
    for (size_t i = 0; i < s.pages.size(); ++i) {
        PageStats& p = s.pages[(page_start + i) % s.pages.size()];
        if (p.page == page || p.page == 0) {
            p.page = page;
            ++p.reads;
            p.macro = sample.macro;
            break;
        }
    }
    if (s.window_start_ns == 0) {
        s.window_start_ns = WallNs();
        s.window_start_ticks = Nx2Profile::Ticks();
    } else if ((s.reads & 255) == 0 && WallNs() - s.window_start_ns >= 10'000'000'000ULL) {
        Report();
    }
}

} // namespace VideoCommon::Nx2DmaReadDiag
