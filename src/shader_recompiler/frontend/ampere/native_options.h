// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

namespace Shader::Ampere {

/// Whether HMMA is lowered to NS2_HMMA (warp-shuffle emulation) without EDEN_NX2_TENSOR. Set for
/// the titles that need it (Yoshi and the Mysterious Book, whose upscaling passes use HMMA);
/// other titles keep rejecting HMMA by default (title 4's 372-HMMA network takes the host driver
/// minutes to compile).
inline std::atomic<bool>& NativeTensorByDefault() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

/// Yoshi and the Mysterious Book (full version and demo; set in ChannelState::Init): its native
/// programs are captured and lowered with Main Build 124's code (frontend/ampere/build124/) and
/// the upstream Eglantine runtime heuristics Build 124 did not have stay off for it (the DLSS
/// replacement chain, the network skip window and the matrix multiply stand-in in KeplerCompute,
/// inline CPU writes onto GPU-written pages in the buffer cache). The Eglantine v0.1.0-rc3 merge
/// without this (yoshi-int-build-42) brought back stutter and ghosting in Yoshi.
inline std::atomic<bool>& YoshiCompatibility() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

} // namespace Shader::Ampere
