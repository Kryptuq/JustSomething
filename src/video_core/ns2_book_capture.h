// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Diagnostics only: one-frame capture for Yoshi's near-black Book page.
// Eden's screenshot command arms it; the frame rendered after the screenshot frame is captured
// (each screenshot captures again, into capture1, capture2, ...). capture.txt lists every draw and
// dispatch of that frame with its render targets, texture descriptors (raw TIC words) and resolved
// images (text only; yoshi-int-build-36 also saved every image, which exhausted its budget).
// yoshi-int-build-37: the Book page composition draws (fragment programs 60c4a9ac1580d2bf and
// 0c82e37b9766ef61, plus up to 16 draws before the frame's first dispatch) are captured in full
// in composition/: the Maxwell register file, the host pipeline key, constant buffers, vertex and
// index data, native and lowered programs, and the render targets before the first and after
// every such draw. Nothing changes rendering; saving waits for the host GPU (that frame is slow).
// Files: <log dir>/ns2_book_capture_yoshi-integration/captureN/.

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <string_view>

#include "common/common_types.h"
#include "common/fs/path_util.h"

namespace VideoCommon::NS2BookCapture {

enum : u32 { Idle = 0, Armed = 1, Capturing = 2, Done = 3 };

inline std::atomic<u32>& State() {
    static std::atomic<u32> state{Idle};
    return state;
}

inline bool Active() {
    return State().load(std::memory_order_relaxed) == Capturing;
}

inline std::atomic<u32>& CaptureNumber() {
    static std::atomic<u32> number{0};
    return number;
}

/// The screenshot command was used: capture the next frame (again after a finished capture).
inline void Arm() {
    u32 expected = Idle;
    if (!State().compare_exchange_strong(expected, Armed)) {
        expected = Done;
        if (!State().compare_exchange_strong(expected, Armed)) {
            return;
        }
    }
    CaptureNumber().fetch_add(1, std::memory_order_relaxed);
}

/// Called once per presented frame. Returns 1 when the capture frame starts, 2 when it ends.
inline int OnFramePresented() {
    u32 expected = Capturing;
    if (State().compare_exchange_strong(expected, Done)) {
        return 2;
    }
    expected = Armed;
    if (State().compare_exchange_strong(expected, Capturing)) {
        return 1;
    }
    return 0;
}

inline std::filesystem::path BaseDirectory() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "ns2_book_capture_yoshi-integration";
}

inline std::filesystem::path Directory() {
    return BaseDirectory() / ("capture" + std::to_string(CaptureNumber().load(std::memory_order_relaxed)));
}

/// yoshi-int-build-38: the Book composition program that draws nothing in the page image. Its
/// pipelines are recorded when they are built (runtime_programs/: the SPIR-V the driver gets, the
/// runtime info and the texture metadata the translation read), and its draws in the capture frame
/// log the bound texture views and samplers.
constexpr u64 FailingCompositionProgram = 0x60c4a9ac1580d2bfULL;

inline std::filesystem::path RuntimeProgramDirectory() {
    return BaseDirectory() / "runtime_programs";
}

inline std::filesystem::path CompositionDirectory() {
    return Directory() / "composition";
}

/// Budget for saved image data (the user uploads the folder). Only composition render targets are
/// saved: at most 17 captures of a 1920x1080 RGBA8 target (~140 MiB), far below the budget.
constexpr u64 MaxImageBytes = 64ULL << 20;
constexpr u64 MaxTotalBytes = 512ULL << 20;

/// Fragment programs of the Book page composition (yoshi-int-build-36 capture): the illustration,
/// paper and icon draws, and the text draws.
constexpr u64 CompositionFragmentHashes[] = {0x60c4a9ac1580d2bfULL, 0x0c82e37b9766ef61ULL};
/// Draws before the frame's first dispatch also captured in full (the composition runs there).
constexpr u32 MaxEarlyDraws = 16;

struct Composition {
    bool dispatch_seen = false;
    u32 early_draws = 0;
    u32 hash_draws = 0;
    u32 index = 0;
    u64 last_rt0 = 0;
    std::set<u64> programs;
};

inline Composition& GetComposition() {
    static Composition composition;
    return composition;
}

struct Log {
    std::mutex mutex;
    std::ofstream file;
    u32 event = 0;
};

inline Log& GetLog() {
    static Log log;
    return log;
}

inline void Open() {
    Log& log = GetLog();
    std::scoped_lock lock{log.mutex};
    std::error_code ec;
    std::filesystem::create_directories(Directory(), ec);
    std::filesystem::create_directories(CompositionDirectory(), ec);
    log.file.open(Directory() / "capture.txt", std::ios::trunc);
    log.event = 0;
    GetComposition() = Composition{};
}

inline void Line(std::string_view text) {
    Log& log = GetLog();
    std::scoped_lock lock{log.mutex};
    if (log.file.is_open()) {
        log.file << text << '\n';
    }
}

inline u32 NextEvent() {
    Log& log = GetLog();
    std::scoped_lock lock{log.mutex};
    return log.event++;
}

inline void Close() {
    Log& log = GetLog();
    std::scoped_lock lock{log.mutex};
    log.file.close();
}

} // namespace VideoCommon::NS2BookCapture
