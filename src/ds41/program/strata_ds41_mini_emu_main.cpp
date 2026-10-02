// src/ds41/program/strata_ds41_mini_emu_main.cpp - DS1-E: `strata-ds41-mini-emu`, the SAME engine (src/ds41/program/cli.hpp, src/ds41/session) instantiated for MiniGeom on the emulated
// device (HostDev) with every kernel compiled for the host (-DDS41_EMU, the thread-model emulation of src/ds41/cuda/ds41_emu.hpp).  It runs the mini GGUF of
// tools/ds41/make_mini_gguf.py end to end with the same options as strata-ds41; the end-to-end ctest compares its trace with the NumPy oracle's.
// DS41_EMU_ORDER=forward|reverse|shuffle[:SEED] picks the emulator's scheduling order.
#include <malloc.h>

#include <cstdio>

#include "cli.hpp"
#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"

int main(int argc, char** argv) {
    using namespace strata::ds41;
    // the emulator allocates and frees big blocks all the time: glibc's default mmap threshold / trim make that 17x slower (as the DS1-C emulator tests found)
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
    mallopt(M_TOP_PAD, 256 << 20);
    try {
        cuda::HostDev dev;
        return program::run_cli<MiniGeom>(dev, argc, argv, "strata-ds41-mini-emu");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-ds41-mini-emu: error: %s\n", e.what());
        return 1;
    }
}
