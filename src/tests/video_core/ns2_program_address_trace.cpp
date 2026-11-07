// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Program address trace (diagnostics, yoshi-int-build-33): a macro's store of a program address
// keeps the call's parameters, their source and the registers the macro read; copies into memory
// are found by overlap.

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "video_core/ns2_program_address_trace.h"

namespace Trace = VideoCommon::NS2ProgramAddressTrace;

TEST_CASE("A macro's program address store keeps its call", "[video_core][ns2-trace]") {
    // Stage 1 low word: pipelines[1].reserved[1] = method 0x800 + 0x10 + 6.
    constexpr u32 low_method = Trace::PipelineBase + Trace::PipelineStride + Trace::LowWord;
    const std::array<u32, 3> params{0x11, 0x22, 0x33};
    std::array<u32, 0xc00> data_ram{};
    data_ram[5] = 0xabcd;
    {
        const Trace::MacroScope scope{0xe08, params, 0x1000};
        Trace::NoteMacroRead(0x5f0, 7);
        Trace::NoteStored(low_method, 0x147ced00, 0x2000, &data_ram);
    }
    Trace::NoteStored(low_method, 0x55, 0x3000); // a direct store afterwards
    const auto history = Trace::StoredHistory(1, false);
    REQUIRE(history.size() >= 2);
    REQUIRE(history[0].value == 0x55);
    REQUIRE(history[0].macro == 0);
    REQUIRE(history[0].param_count == 0);
    const auto& stored = history[1];
    REQUIRE(stored.value == 0x147ced00);
    REQUIRE(stored.macro == 0xe08);
    REQUIRE(stored.param_count == 3);
    REQUIRE(stored.params[2] == 0x33);
    REQUIRE(stored.param_address == 0x1000);
    REQUIRE(stored.read_count == 1);
    REQUIRE(stored.reads[0].method == 0x5f0);
    REQUIRE(stored.reads[0].value == 7);
    REQUIRE(Trace::DataRamAtStore(1, false)[5] == 0xabcd);
    REQUIRE(Trace::CurrentCall() == nullptr);
}

TEST_CASE("Copies into a program's memory are found by overlap", "[video_core][ns2-trace]") {
    Trace::NoteMemoryWrite("copy engine", 0x50000, 0x100, 0x90000, true);
    Trace::NoteMemoryWrite("inline upload", 0x70000, 0x10, 0, false);
    const auto writes = Trace::WritesOverlapping(0x500f0, 0x20);
    REQUIRE(writes.size() == 1);
    REQUIRE(writes[0].source == 0x90000);
    REQUIRE(writes[0].source_gpu);
    REQUIRE(Trace::WritesOverlapping(0x60000, 0x1000).empty());
    REQUIRE(Trace::TakeReport(0x1234, 1, 64));
    REQUIRE_FALSE(Trace::TakeReport(0x1234, 1, 64));
}
