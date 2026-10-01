// src/ds41/model/model_halves_test.cpp - DS1-A: the loader's half packer against DS-C's `pack_cpu_half` at RealGeom (one random expert, 18.8 MB).
//
// The arena is built by `pack_half` (any shape); the CPU expert kernels were written against `strata::ds41::cpu::pack_cpu_half` (RealGeom's constants).
// They must agree byte for byte, and `unpack_halves` must be the exact inverse (what a cache promotion uploads).
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/ds41/cpu/mxfp4_expert.hpp"
#include "strata/ds41/model/model.hpp"

using namespace strata::ds41;
using namespace strata::ds41::model;

int main() {
    const ExpertDims d = expert_dims<RealGeom>();
    int fail = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) {
            ++fail;
            std::fprintf(stderr, "FAIL %s\n", what);
        }
    };
    check(d.blob_bytes() == kBlobBytes && d.half_bytes() == kHalfBytes && d.half_gate_bytes() == kHalfGateBytes && d.half_down_bytes() == kHalfDownBytes, "ExpertDims(RealGeom) == geometry.hpp");
    check(d.gate_bytes() == kGateBytes && d.down_bytes() == kDownBytes && d.gate_row_bytes() == kGateRowBytes && d.down_row_bytes() == kDownRowBytes, "row and slice sizes");

    std::vector<uint8_t> blob(kBlobBytes);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint8_t& b : blob) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b = (uint8_t) (x >> 32);
    }
    std::vector<uint8_t> mine[2], theirs[2];
    for (int h = 0; h < 2; ++h) {
        mine[h].assign(kHalfBytes, 0xAA);
        theirs[h].assign(kHalfBytes, 0xBB);
        pack_half(d, blob.data() + kBlobGate, blob.data() + kBlobUp, blob.data() + kBlobDown, h, mine[h].data());
        cpu::pack_cpu_half(blob.data(), h, theirs[h].data());
        check(mine[h] == theirs[h], h ? "half 1 == DS-C's pack_cpu_half" : "half 0 == DS-C's pack_cpu_half");
    }
    check(mine[0] != mine[1], "the halves differ");
    std::vector<uint8_t> back(kBlobBytes, 0xEE);
    unpack_halves(d, mine[0].data(), mine[1].data(), back.data());
    check(back == blob, "unpack_halves(pack_half(blob)) == blob");
    std::printf("%s: pack_half == cpu::pack_cpu_half at RealGeom (%zu-byte blob, two %zu-byte halves)\n", fail ? "FAIL" : "PASS", (size_t) kBlobBytes, (size_t) kHalfBytes);
    return fail ? 1 : 0;
}
