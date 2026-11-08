// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Diagnostics only (yoshi-int-build-36): one-frame capture for Yoshi's near-black Book page.
// Eden's screenshot command arms it; the frame rendered after the screenshot frame is captured,
// once per session. For every draw and compute dispatch of that frame the texture cache writes
// the render targets, the texture descriptors (raw TIC words as read from game memory) and the
// image each one resolved to; every distinct image is saved once, before its first use, both as
// the host GPU copy (.ns2img) and as the raw bytes in game memory (.guest); images written by the
// frame (render targets, storage images, blit/copy destinations) are saved again at its end.
// Skipped draws, uploads from game memory and blits are logged too. Nothing changes rendering;
// saving waits for the host GPU, so that one frame is slow.
// Files: <log dir>/ns2_book_capture_yoshi-integration/ (capture.txt + images).

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
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

/// The screenshot command was used: capture the next frame (first time per session only).
inline void Arm() {
    u32 expected = Idle;
    State().compare_exchange_strong(expected, Armed);
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

inline std::filesystem::path Directory() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "ns2_book_capture_yoshi-integration";
}

/// Budget for saved image data (the user uploads the folder).
constexpr u64 MaxImageBytes = 48ULL << 20;
constexpr u64 MaxTotalBytes = 768ULL << 20;

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
    log.file.open(Directory() / "capture.txt", std::ios::trunc);
    log.event = 0;
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
