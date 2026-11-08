// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Buffer cache event ring (diagnostics, yoshi-int-build-34): events are found by overlap, newest
// first.

#include <catch2/catch_test_macros.hpp>

#include "video_core/ns2_buffer_events.h"

namespace Events = VideoCommon::NS2BufferEvents;

TEST_CASE("Buffer cache events are found by overlap, newest first", "[video_core][ns2-events]") {
    Events::Note(Events::Kind::MarkedGpuModifiedTrap, 0x10000, 0x1000, 1);
    Events::Note(Events::Kind::CpuWriteCleared, 0x10800, 0x10, 2);
    Events::Note(Events::Kind::FlushDownloaded, 0x20000, 0x100, 3);
    const auto found = Events::Overlapping(0x10000, 0x1000, 8);
    REQUIRE(found.size() == 2);
    REQUIRE(found[0].kind == Events::Kind::CpuWriteCleared);
    REQUIRE(found[0].line == 2);
    REQUIRE(found[1].kind == Events::Kind::MarkedGpuModifiedTrap);
    REQUIRE(Events::Overlapping(0x30000, 0x100, 8).empty());
    Events::Note(Events::Kind::OtherCleared, 0x40000, 0, 4); // empty ranges are not recorded
    REQUIRE(Events::Overlapping(0x40000, 0x10, 8).empty());
}
