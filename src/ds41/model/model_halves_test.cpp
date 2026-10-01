// src/ds41/model/model_halves_test.cpp - DS1-A: the loader's half packer against DS-C / DS1-G's `cpu::pack_cpu_half<G>` at RealGeom and MiniGeom (one random expert each).
//
// The arena is built by `pack_half` (a runtime shape, three separate GGUF slices); the CPU expert kernels read the layout `cpu::HalfLayout<G>` describes and
// `cpu::pack_cpu_half<G>` produces from a contiguous blob.  The two must agree byte for byte (the arena's bytes ARE what the kernels' views read), the shape
// arithmetic of ExpertDims must be HalfLayout's, and `unpack_halves` must be the exact inverse (what a cache promotion uploads).
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/ds41/cpu/mxfp4_expert.hpp"
#include "strata/ds41/model/model.hpp"

using namespace strata::ds41;
using namespace strata::ds41::model;

namespace {
int g_fail = 0;
void check(bool ok, const char* what, const char* geom) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL [%s] %s\n", geom, what);
    }
}

template <class G> void run() {
    constexpr ExpertDims d = expert_dims<G>();
    using L = cpu::HalfLayout<G>;
    // the shape arithmetic, at compile time
    static_assert(d.blob_bytes() == L::kBlobBytesW && d.half_bytes() == L::kHalfBytesW && d.half_gate_bytes() == L::kHalfGateBytesW && d.half_down_bytes() == L::kHalfDownBytesW);
    static_assert(d.gate_row_bytes() == L::kGateRowBytesW && d.down_row_bytes() == L::kDownRowBytesW && d.half_down_blocks() == L::kHalfDownRowBlocksW && (size_t) d.half_ff() == (size_t) L::kHalfFFW);
    static_assert(d.gate_bytes() == L::kBlobUpOff && 2 * d.gate_bytes() == L::kBlobDownOff && d.half_gate_bytes() == L::kHalfUpOff && 2 * d.half_gate_bytes() == L::kHalfDownOff);
    const char* name = G::kName;

    std::vector<uint8_t> blob(L::kBlobBytesW);
    uint64_t x = 0x9E3779B97F4A7C15ull ^ (uint64_t) G::kHidden;
    for (uint8_t& b : blob) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b = (uint8_t) (x >> 32);
    }
    std::vector<uint8_t> mine[2], theirs[2];
    for (int h = 0; h < 2; ++h) {
        mine[h].assign(L::kHalfBytesW, 0xAA);
        theirs[h].assign(L::kHalfBytesW, 0xBB);
        pack_half(d, blob.data() + L::kBlobGateOff, blob.data() + L::kBlobUpOff, blob.data() + L::kBlobDownOff, h, mine[h].data());
        cpu::pack_cpu_half<G>(blob.data(), h, theirs[h].data());
        check(mine[h] == theirs[h], h ? "half 1 == cpu::pack_cpu_half<G>" : "half 0 == cpu::pack_cpu_half<G>", name);
        // and the view the CPU kernels use reads the same rows from the packed half as from the blob in place
        const cpu::ExpertView vh = cpu::view_cpu_half<G>(mine[h].data()), vb = cpu::view_blob_half<G>(blob.data(), h);
        bool same = vh.hidden == vb.hidden && vh.ff == vb.ff;
        for (int r = 0; same && r < vh.ff; ++r)
            same = std::memcmp(vh.gate + (size_t) r * vh.gate_row_bytes(), vb.gate + (size_t) r * vb.gate_row_bytes(), vh.gate_row_bytes()) == 0 &&
                   std::memcmp(vh.up + (size_t) r * vh.gate_row_bytes(), vb.up + (size_t) r * vb.gate_row_bytes(), vh.gate_row_bytes()) == 0;
        for (int r = 0; same && r < vh.hidden; ++r)
            same = std::memcmp(vh.down + (size_t) r * vh.down_row_stride, vb.down + (size_t) r * vb.down_row_stride, (size_t) vh.down_row_blocks() * kBlockBytes) == 0;
        check(same, "view_cpu_half<G>(packed) reads the rows of view_blob_half<G>(blob)", name);
    }
    check(mine[0] != mine[1], "the halves differ", name);
    std::vector<uint8_t> back(L::kBlobBytesW, 0xEE);
    unpack_halves(d, mine[0].data(), mine[1].data(), back.data());
    check(back == blob, "unpack_halves(pack_half(blob)) == blob", name);
    std::printf("PASS [%s]: pack_half == cpu::pack_cpu_half<G> (%zu-byte blob, two %zu-byte halves)\n", name, (size_t) L::kBlobBytesW, (size_t) L::kHalfBytesW);
}
}  // namespace

int main() {
    run<RealGeom>();
    run<MiniGeom>();
    static_assert(expert_dims<RealGeom>().blob_bytes() == kBlobBytes && expert_dims<RealGeom>().half_bytes() == kHalfBytes);
    return g_fail ? 1 : 0;
}
