// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// LDGSTS (cp.async) lowering. Adapted from Lichii's src/tests/video_core/ampere_async_copy.cpp
// (https://github.com/zortos293/Lichii, commit 281477e, GPL-3.0-or-later): the lowered Maxwell
// code is executed by a small machine and compared with the hardware definition of the copy. The
// predicate at bits 87-89 (negated by bit 90) enables the copy: when it holds, the 16 bytes at the
// global address are stored to shared memory; when it does not, the source is not read and zeros
// are stored. Encodings are those of title 4's network kernel 557f7bdab63f0cb8 (+0x760..0x7f0:
// LDGSTS.E.BYPASS.LTC128B.128 [R11 + 0x1000 * n], desc[UR10][R16.64], Pn).

#include <array>
#include <cstring>
#include <map>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/build124/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

// enable: predicate index (7 = PT) plus 8 when negated.
constexpr Instruction Copy(u32 shared, u32 global, u32 shared_offset, u32 enable) {
    return {{0x00007faeU | (shared << 16) | (global << 24), shared_offset << 12,
             0x08101d4aU | ((enable & 7) << 23) | ((enable >> 3) << 26), 0x0003e200U}};
}

struct Machine {
    std::array<u32, 256> r{};
    std::array<bool, 7> p{};
    std::map<u64, u32> global;      // only these words exist; reading anything else fails
    std::vector<u8> shared = std::vector<u8>(0x4000);
    u32 global_reads = 0;

    u32 R(u64 index) const { return index == 255 ? 0U : r[index]; }
    bool Pred(u64 guard) const { return (((guard & 7) == 7) || p[guard & 7]) != ((guard & 8) != 0); }

    bool Run(const std::vector<u64>& code) {
        bool carry = false;
        for (const u64 insn : code) {
            const auto bits = [insn](u32 offset, u32 count) { return (insn >> offset) & ((1ULL << count) - 1); };
            const u64 top = insn >> 48;
            if ((top & 0xfff0) == 0xe300) return true;                                   // EXIT
            if (!Pred(bits(16, 4))) continue;
            const u64 d = bits(0, 8), a = bits(8, 8);
            const auto set = [&](u64 index, u32 value) { if (index != 255) r[index] = value; };
            if ((top & 0xfff8) == 0x5c10) {                                              // IADD
                if (bits(48, 2) || bits(44, 3) || bits(39, 4)) return false;
                const u64 sum = u64{R(a)} + R(bits(20, 8)) + (bits(43, 1) && carry ? 1 : 0);
                if (bits(47, 1)) carry = (sum >> 32) != 0;
                set(d, static_cast<u32>(sum));
            } else if ((top & 0xfff8) == 0x5c98) {                                       // MOV
                if (bits(39, 4) != 15) return false;
                set(d, R(bits(20, 8)));
            } else if ((top & 0xfff8) == 0xeed0) {                                       // LDG.E
                const u32 words = bits(48, 3) == 6 ? 4 : bits(48, 3) == 5 ? 2 : bits(48, 3) == 4 ? 1 : 0;
                if (!words || !bits(45, 1) || a == 255) return false;
                const u64 address = (u64{R(a)} | (u64{R(a + 1)} << 32)) + static_cast<u64>(static_cast<s64>(static_cast<s32>(bits(20, 24) << 8) >> 8));
                ++global_reads;
                for (u32 k = 0; k < words; ++k) {
                    const auto it = global.find(address + 4 * k);
                    if (it == global.end()) return false;
                    set(d + k, it->second);
                }
            } else if ((top & 0xfff8) == 0xef58) {                                       // STS
                const u32 words = bits(48, 3) == 6 ? 4 : bits(48, 3) == 5 ? 2 : bits(48, 3) == 4 ? 1 : 0;
                const u32 address = R(a) + static_cast<u32>(static_cast<s32>(bits(20, 24) << 8) >> 8);
                if (!words || address + 4 * words > shared.size()) return false;
                for (u32 k = 0; k < words; ++k) {
                    const u32 value = R(d + k);
                    std::memcpy(&shared[address + 4 * k], &value, 4);
                }
            } else {
                return false;
            }
        }
        return true;
    }
};

// Runs Copy(R11 + 0x1000, R16.64, enable) with predicate P3 = p3 and returns the machine.
Machine RunCopy(u32 enable, bool p3) {
    const std::array<Instruction, 2> code{{Copy(11, 16, 0x1000, enable), Exit}};
    const auto lowered = Shader::Ampere::Build124::LowerToMaxwell(code);
    REQUIRE(static_cast<bool>(lowered));
    Machine machine;
    for (u32 i = 0; i < 246; ++i) machine.r[i] = 0x9e3779b9U * (i + 1);
    machine.r[11] = 0x200;                       // shared base
    machine.r[16] = 0xa267b100;                  // global address (557f's weights, sb1)
    machine.r[17] = 0x1a;
    machine.r[216 + 10] = 0;                     // UR10:UR11 = c[0x118], the memory descriptor
    machine.r[216 + 11] = 0;
    machine.p[3] = p3;
    for (u32 k = 0; k < 4; ++k) machine.global[0x1aa267b100ULL + 4 * k] = 0x3c003c00U + k;
    std::memset(machine.shared.data(), 0xcd, machine.shared.size());
    REQUIRE(machine.Run(lowered.instructions));
    return machine;
}

u32 SharedWord(const Machine& machine, u32 address) {
    u32 value;
    std::memcpy(&value, &machine.shared[address], 4);
    return value;
}

void RequireCopied(const Machine& machine) {
    REQUIRE(machine.global_reads == 1);
    for (u32 k = 0; k < 4; ++k) REQUIRE(SharedWord(machine, 0x1200 + 4 * k) == 0x3c003c00U + k);
}

void RequireZeroFilled(const Machine& machine) {
    REQUIRE(machine.global_reads == 0);          // a disabled copy never reads its source
    for (u32 k = 0; k < 4; ++k) REQUIRE(SharedWord(machine, 0x1200 + 4 * k) == 0);
}

TEST_CASE("[Build124 pinned] LDGSTS copies when its predicate holds and zero-fills when it does not", "[video_core][ampere][ldgsts]") {
    RequireCopied(RunCopy(3, true));             // P3 true: enabled
    RequireZeroFilled(RunCopy(3, false));        // P3 false: zeros, no read
    RequireZeroFilled(RunCopy(3 | 8, true));     // !P3 with P3 true
    RequireCopied(RunCopy(3 | 8, false));        // !P3 with P3 false
}

TEST_CASE("[Build124 pinned] LDGSTS with a constant predicate", "[video_core][ampere][ldgsts]") {
    RequireCopied(RunCopy(7, false));            // PT: always copies
    RequireZeroFilled(RunCopy(7 | 8, true));     // !PT: always zeros
}

TEST_CASE("[Build124 pinned] LDGSTS only writes its 16 shared bytes", "[video_core][ampere][ldgsts]") {
    const Machine machine = RunCopy(3, true);
    for (u32 address = 0; address < machine.shared.size(); ++address) {
        if (address >= 0x1200 && address < 0x1210) continue;
        REQUIRE(machine.shared[address] == 0xcd);
    }
}

} // namespace
