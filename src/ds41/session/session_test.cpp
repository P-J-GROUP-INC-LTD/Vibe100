// src/ds41/session/session_test.cpp - DS1-E: unit tests of the engine's own parts, no GGUF and no GPU needed.
//
//   --pool      the CPU expert pool against DS-C's reference arrangement (cpu::expert_run on each half, the two partials added): BIT-IDENTICAL for 1 .. 5 workers per group, a verify
//               window whose tokens share experts (the multi-token groups), an empty job, repeated jobs on different layers, a REAL-shaped expert (hidden 5120, ff 2304) and a
//               mini-shaped one; the pool laid out for a faked two-node topology (pinned workers) and for no topology (unpinned groups).
//   --combine   the MoE sum kernel (moe_combine_emu_impl.cpp, the emulated build of the kernel nvcc compiles for the V100) against the oracle's order (ascending expert id, then the shared
//               expert), bit for bit, at both geometries, in forward / reverse / shuffled scheduling.
//   --trace     the trace writer's .npy files and trace.json (read back by hand) and the logits dump.
// No option = all.  Exit status 0 when every check passes ("ALL PASS" is the last line).
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "strata/ds41/cpu/mxfp4_expert.hpp"
#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"
#include "strata/ds41/model/model.hpp"
#include "strata/ds41/session/cpu_pool.hpp"
#include "strata/ds41/session/moe_combine.hpp"
#include "strata/ds41/session/trace.hpp"
#include "ds41_emu.hpp"

using namespace strata::ds41;

namespace {
int g_fail = 0, g_pass = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    if (ok) ++g_pass;
    else ++g_fail;
    std::printf("%s  %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : ": ", detail.c_str());
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the CPU pool
// ---------------------------------------------------------------------------------------------------------------------------------
struct ArenaFixture {
    model::ExpertDims d;
    std::vector<std::vector<uint8_t>> gate, up, down;      // per layer: n_expert slices
    std::vector<model::ExpertSlices> slices;
    model::ExpertArena arena;
    ArenaFixture(const model::ExpertDims& dims, uint64_t seed) : d(dims) {
        std::mt19937_64 rng(seed);
        auto fill = [&](std::vector<uint8_t>& v, size_t n) {
            v.resize(n);
            for (size_t i = 0; i < n; ++i) v[i] = (uint8_t) rng();
            // MXFP4 blocks are [e8m0][16 bytes]: keep the exponents in a sane range (2^-28 .. 2^2) so no sum overflows
            for (size_t i = 0; i + 17 <= n; i += 17) v[i] = (uint8_t) (100 + rng() % 30);
        };
        gate.resize((size_t) d.n_layer);
        up.resize((size_t) d.n_layer);
        down.resize((size_t) d.n_layer);
        for (int l = 0; l < d.n_layer; ++l) {
            fill(gate[(size_t) l], (size_t) d.n_expert * d.gate_bytes());
            fill(up[(size_t) l], (size_t) d.n_expert * d.gate_bytes());
            fill(down[(size_t) l], (size_t) d.n_expert * d.down_bytes());
            model::ExpertSlices s;
            s.gate = gate[(size_t) l].data();
            s.up = up[(size_t) l].data();
            s.down = down[(size_t) l].data();
            s.d = d;
            slices.push_back(s);
        }
        model::ArenaOptions ao;
        ao.threads = 2;
        ao.numa = false;
        ao.hugepages = false;
        ao.log = [](const std::string&) {};
        arena = model::ExpertArena::build(d, slices, ao);
    }
};

cpu::ExpertView half_view(const model::ExpertDims& d, const model::ExpertArena& a, int h, int layer, int expert) {
    const uint8_t* base = a.half(h, layer, expert);
    cpu::ExpertView v;
    v.gate = base;
    v.up = base + d.half_gate_bytes();
    v.down = base + 2 * d.half_gate_bytes();
    v.down_row_stride = (size_t) d.half_down_row_bytes();
    v.hidden = d.hidden;
    v.ff = d.half_ff();
    return v;
}

/// The oracle's arrangement of the pool's work: per miss, each half on its own through DS-C's expert_run (T = 1), the partials added h0 + h1.
std::vector<float> reference_rows(const ArenaFixture& fx, int layer, const std::vector<cpu::ActQ>& x, const std::vector<session::CpuMiss>& miss) {
    const int hidden = fx.d.hidden;
    std::vector<float> out(miss.size() * (size_t) hidden);
    std::unique_ptr<cpu::ExpertScratch> sc(new cpu::ExpertScratch);
    for (size_t i = 0; i < miss.size(); ++i) {
        std::vector<float> y0((size_t) hidden, 0.0f), y1((size_t) hidden, 0.0f);
        const float w = miss[i].weight;
        cpu::expert_run(cpu::Isa::kAuto, half_view(fx.d, fx.arena, 0, layer, miss[i].expert), &x[(size_t) miss[i].token], 1, &w, *sc, y0.data());
        cpu::expert_run(cpu::Isa::kAuto, half_view(fx.d, fx.arena, 1, layer, miss[i].expert), &x[(size_t) miss[i].token], 1, &w, *sc, y1.data());
        for (int c = 0; c < hidden; ++c) out[i * (size_t) hidden + (size_t) c] = y0[(size_t) c] + y1[(size_t) c];
    }
    return out;
}

void suite_pool_shape(const char* tag, const model::ExpertDims& dims, int n_layers_used, std::vector<int> thread_counts) {
    ArenaFixture fx(dims, 20261002);
    const int hidden = dims.hidden;
    std::mt19937_64 rng(7);
    const int Tmax = 8, topk = 6;
    std::vector<float> xf((size_t) Tmax * hidden);
    for (float& v : xf) v = (float) (std::normal_distribution<double>(0.0, 1.0)(rng));
    std::vector<cpu::ActQ> actq((size_t) Tmax);
    cpu::quantize_acts(xf.data(), hidden, Tmax, actq.data());

    struct Case {
        const char* name;
        int T;
        std::vector<session::CpuMiss> miss;
    };
    std::vector<Case> cases;
    {   // decode: six distinct experts, one token
        Case c{"decode T=1, 6 distinct experts", 1, {}};
        for (int k = 0; k < 6; ++k) c.miss.push_back({0, k, (5 * k + 1) % dims.n_expert, 0.1f + 0.2f * (float) k});
        cases.push_back(c);
    }
    {   // a verify window of 4 tokens: tokens 0 / 2 / 3 route to expert 1, tokens 1 / 2 to expert 2 (multi-token groups)
        Case c{"window T=4, experts shared between tokens", 4, {}};
        auto add = [&](int t, int k, int e, float w) { c.miss.push_back({t, k, e % dims.n_expert, w}); };
        add(0, 0, 1, 0.3f);
        add(0, 1, 0, 0.7f);
        add(1, 0, 2, 0.5f);
        add(1, 1, 3, 0.25f);
        add(2, 0, 1, 1.0f);
        add(2, 1, 2, 0.125f);
        add(3, 0, 1, 0.9f);
        add(3, 1, 3, 0.4f);
        cases.push_back(c);
    }
    {   // a full window of 8 tokens, all six slots, many shared experts (the most the pool is asked to hold)
        Case c{"window T=8, 48 misses", 8, {}};
        for (int t = 0; t < 8; ++t)
            for (int k = 0; k < topk; ++k) c.miss.push_back({t, k, (t + 3 * k) % dims.n_expert, 0.05f * (float) (k + 1) + 0.01f * (float) t});
        cases.push_back(c);
    }
    cases.push_back(Case{"empty job (all hits)", 1, {}});

    for (int tpg : thread_counts) {
        session::CpuPoolOptions po;
        po.threads_per_group = tpg;
        po.pin = false;
        po.max_tokens = Tmax;
        po.top_k = topk;
        session::CpuExpertPool pool(fx.arena, po);
        bool all_ok = true;
        std::string bad;
        for (int rep = 0; rep < 2; ++rep)
            for (int layer = 0; layer < n_layers_used; ++layer)
                for (const Case& c : cases) {
                    // a case that names an expert twice for one token (possible by construction in tiny shapes) is not a router output: skip it
                    bool dup = false;
                    for (size_t i = 0; i < c.miss.size(); ++i)
                        for (size_t j = i + 1; j < c.miss.size(); ++j) dup = dup || (c.miss[i].token == c.miss[j].token && c.miss[i].expert == c.miss[j].expert);
                    if (dup) continue;
                    std::vector<float> got(std::max<size_t>(1, c.miss.size()) * (size_t) hidden, -1.0f);
                    pool.run(layer, actq.data(), c.T, c.miss.data(), (int) c.miss.size(), got.data());
                    const std::vector<float> want = reference_rows(fx, layer, actq, c.miss);
                    if (std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) != 0) {
                        all_ok = false;
                        bad += fmt(" [%s, layer %d, rep %d]", c.name, layer, rep);
                    }
                }
        check(all_ok, fmt("pool %s: %d worker(s) per group: every miss row is bit-identical to DS-C's expert_run on each half + h0 + h1 (%d cases x %d layers x 2 jobs)", tag, tpg,
                          (int) cases.size(), n_layers_used),
              bad);
    }
    // the pool refuses what it cannot do
    session::CpuPoolOptions po;
    po.threads_per_group = 1;
    po.pin = false;
    po.max_tokens = 2;
    po.top_k = 2;
    session::CpuExpertPool small(fx.arena, po);
    std::vector<session::CpuMiss> five(5, session::CpuMiss{0, 0, 1, 1.0f});
    std::vector<float> o(5 * (size_t) hidden);
    bool threw = false;
    try {
        small.start(0, actq.data(), 2, five.data(), 5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, fmt("pool %s: more misses than max_tokens * top_k is refused", tag));
    threw = false;
    try {
        session::CpuMiss bad{3, 0, 0, 1.0f};
        small.start(0, actq.data(), 2, &bad, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, fmt("pool %s: a miss naming a token outside the window is refused", tag));
}

void suite_pool() {
    // mini shape (hidden 256, ff 256: halves of 128) and a real-shaped expert (hidden 5120, ff 2304: halves of 1152, 36 chunks, 5120 down rows)
    suite_pool_shape("mini shape", model::expert_dims<MiniGeom>(), 3, {1, 2, 3, 5});
    model::ExpertDims real{2, 8, RealGeom::kHidden, RealGeom::kFF};
    suite_pool_shape("real shape", real, 2, {1, 3, 7});

    // the pool's layout from a topology: two nodes -> pinned groups, one node -> unpinned, off -> unpinned
    strata::platform::NumaTopology topo;
    topo.available = true;
    topo.nodes.resize(2);
    topo.nodes[0].id = 0;
    topo.nodes[0].cpus = {0, 1, 2, 3};
    topo.nodes[1].id = 1;
    topo.nodes[1].cpus = {4, 5, 6, 7};
    {
        const session::CpuPoolOptions o = session::plan_cpu_pool(topo, true, 0, {});
        check(o.pin && o.cpus[0] == std::vector<int>({0, 1, 2, 3}) && o.cpus[1] == std::vector<int>({4, 5, 6, 7}) && o.threads_per_group == 4,
              "plan: two NUMA nodes -> one pinned group per node, a worker per CPU", o.note);
    }
    {
        const session::CpuPoolOptions o = session::plan_cpu_pool(topo, true, 3, {0, 1, 2, 3, 4, 5, 6, 7});
        check(o.pin && o.threads_per_group == 3, "plan: --threads-per-socket 3");
    }
    {
        const session::CpuPoolOptions o = session::plan_cpu_pool(topo, false, 0, {0, 1, 2, 3, 4, 5, 6, 7});
        check(!o.pin && o.cpus[0].empty() && o.threads_per_group == 4, "plan: --numa off -> two unpinned groups, half the CPUs each", o.note);
    }
    {
        const session::CpuPoolOptions o = session::plan_cpu_pool(topo, true, 0, {0, 1, 2, 3});          // the process may only run on node 0
        check(!o.pin, "plan: a process confined to one node's CPUs is not pinned across nodes", o.note);
    }
    {
        strata::platform::NumaTopology one;
        one.available = true;
        one.nodes.resize(1);
        one.nodes[0].cpus = {0, 1, 2, 3};
        const session::CpuPoolOptions o = session::plan_cpu_pool(one, true, 0, {});
        check(!o.pin && o.threads_per_group == 2, "plan: one node -> two unpinned groups of half the CPUs", o.note);
    }
    // a pool pinned to a (faked) topology that fits this machine: the workers pin themselves and the numbers do not change
    {
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw >= 2) {
            ArenaFixture fx(model::expert_dims<MiniGeom>(), 5);
            session::CpuPoolOptions po;
            po.cpus[0] = {0};
            po.cpus[1] = {(int) (hw - 1)};
            po.pin = true;
            po.threads_per_group = 1;
            po.max_tokens = 1;
            po.top_k = 2;
            session::CpuExpertPool pool(fx.arena, po);
            std::vector<float> xf(256);
            std::mt19937_64 rng(1);
            for (float& v : xf) v = (float) std::normal_distribution<double>(0, 1)(rng);
            cpu::ActQ aq;
            cpu::quantize_acts(xf.data(), 256, 1, &aq);
            std::vector<session::CpuMiss> miss = {{0, 0, 3, 0.5f}, {0, 1, 9, 1.0f}};
            std::vector<float> got(2 * 256);
            pool.run(1, &aq, 1, miss.data(), 2, got.data());
            const std::vector<float> want = reference_rows(fx, 1, {aq}, miss);
            check(pool.pinned_workers() == 2 && std::memcmp(got.data(), want.data(), want.size() * 4) == 0, "pool pinned to CPUs {0} / {last}: both workers pinned, rows unchanged", pool.describe());
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the MoE sum kernel
// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
void combine_case(cuda::HostDev& dev, const char* tag, int T, uint64_t seed) {
    constexpr int K = G::kTopK, H = G::kHidden, E = G::kExperts;
    std::mt19937_64 rng(seed);
    std::vector<int32_t> ids((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<int> perm((size_t) E);
        for (int e = 0; e < E; ++e) perm[(size_t) e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int k = 0; k < K; ++k) ids[(size_t) t * K + k] = perm[(size_t) k];
    }
    const int n_cpu = (T * K) / 2;                            // half of the activations (every other one) come from the CPU rows
    std::vector<int32_t> cpu_row((size_t) T * K, -1);
    int used = 0;
    for (int i = 0; i < T * K; i += 2) cpu_row[(size_t) i] = used++;
    auto rnd = [&](size_t n) {
        std::vector<float> v(n);
        for (float& x : v) x = (float) std::normal_distribution<double>(0, 1)(rng);
        return v;
    };
    const std::vector<float> parts = rnd((size_t) T * K * H), cpu_rows = rnd((size_t) std::max(1, n_cpu) * H), shared = rnd((size_t) T * H);
    cuda::DevBuf<int32_t> d_ids(dev, ids.size()), d_row(dev, cpu_row.size());
    cuda::DevBuf<float> d_parts(dev, parts.size()), d_cpu(dev, cpu_rows.size()), d_shared(dev, shared.size()), d_out(dev, (size_t) T * H);
    d_ids.up(ids);
    d_row.up(cpu_row);
    d_parts.up(parts);
    d_cpu.up(cpu_rows);
    d_shared.up(shared);
    cuda::ds41_moe_combine<G>(dev, d_ids.p, d_parts.p, d_row.p, d_cpu.p, d_shared.p, T, d_out.p);
    const std::vector<float> got = d_out.down();
    // the oracle: y = 0; for e in ascending expert id: y += expert output; y += shared
    std::vector<float> want((size_t) T * H);
    for (int t = 0; t < T; ++t) {
        std::vector<int> order((size_t) K);
        for (int k = 0; k < K; ++k) order[(size_t) k] = k;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ids[(size_t) t * K + a] < ids[(size_t) t * K + b]; });
        for (int d = 0; d < H; ++d) {
            volatile float acc = 0.0f;
            for (int k : order) {
                const int row = t * K + k;
                const float v = cpu_row[(size_t) row] >= 0 ? cpu_rows[(size_t) cpu_row[(size_t) row] * H + d] : parts[(size_t) row * H + d];
                acc = acc + v;
            }
            volatile float r = acc + shared[(size_t) t * H + d];
            want[(size_t) t * H + d] = r;
        }
    }
    check(std::memcmp(got.data(), want.data(), got.size() * 4) == 0, fmt("moe_combine %s T=%d: bit-identical to the oracle's order (ascending expert id, then the shared expert), %d GPU + %d CPU rows", tag, T, T * K - n_cpu, n_cpu));
}

void suite_combine() {
    cuda::HostDev dev;
    const char* orders[] = {"forward", "reverse", "shuffle:3"};
    for (const char* o : orders) {
        ds41_emu::set_order_from_string(o);
        std::printf("-- scheduling order %s\n", o);
        for (int T : {1, 3, 8}) {
            combine_case<MiniGeom>(dev, "mini", T, 100 + (uint64_t) T);
            combine_case<RealGeom>(dev, "real", T, 200 + (uint64_t) T);
        }
    }
    ds41_emu::set_order_from_string("forward");
    // a token whose router RANK order differs from its expert-id order in a way FP32 notices: ids {30, 20, 10} carry {1, -1e8, +1e8}.  By ascending id: ((1e8 - 1e8) + 1) = 1; by rank
    // ((1 - 1e8) + 1e8) = 0.  (Two experts cannot tell the orders apart - addition commutes - so this is a RealGeom (six experts) check.)
    {
        using G = RealGeom;
        constexpr int K = G::kTopK, H = G::kHidden;
        std::vector<int32_t> ids = {30, 20, 10, 100, 101, 102};
        std::vector<float> parts((size_t) K * H, 0.0f), shared((size_t) H, 0.0f);
        for (int d = 0; d < H; ++d) {
            parts[(size_t) 0 * H + d] = 1.0f;
            parts[(size_t) 1 * H + d] = -1e8f;
            parts[(size_t) 2 * H + d] = 1e8f;
        }
        cuda::DevBuf<int32_t> d_ids(dev, ids.size()), d_row(dev, ids.size());
        cuda::DevBuf<float> d_parts(dev, parts.size()), d_cpu(dev, (size_t) H), d_shared(dev, shared.size()), d_out(dev, (size_t) H);
        d_ids.up(ids);
        d_row.up(std::vector<int32_t>(ids.size(), -1));
        d_parts.up(parts);
        d_shared.up(shared);
        cuda::ds41_moe_combine<G>(dev, d_ids.p, d_parts.p, d_row.p, d_cpu.p, d_shared.p, 1, d_out.p);
        const std::vector<float> got = d_out.down();
        bool ok = true;
        for (int d = 0; d < H; ++d) ok = ok && got[(size_t) d] == 1.0f;
        check(ok, "moe_combine: the experts are added in ascending order of expert id, not of router rank (the one order FP32 sees as 1, the other as 0)", fmt("got %g", got[0]));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the trace writer
// ---------------------------------------------------------------------------------------------------------------------------------
std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void suite_trace() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("ds41_session_test_trace_" + std::to_string((long) getpid()));
    fs::remove_all(dir);
    {
        session::TraceQuant q;
        q.compressed_kv = false;
        session::TraceWriter w(dir.string(), "MiniGeom", q, 3);
        const float a[6] = {1, 2, 3, 4, 5, 6};
        w.put_f32("q", 7, 2, a, {2, 3});
        w.put_f32("embed", 0, -1, a, {6});
        const int32_t ids[3] = {5, -1, 9};
        w.put_i32("topk", 12, 3, ids, 3);
        w.set_tokens({10, 11, 12, 13}, 3);
        w.flush();
        check(fs::exists(dir / "q.L02.p00007.npy") && fs::exists(dir / "embed.p00000.npy") && fs::exists(dir / "topk.L03.p00012.npy"), "trace: file names <stage>.L<LL>.p<POS>.npy / <stage>.p<POS>.npy");
        const std::string f = slurp(dir / "q.L02.p00007.npy");
        const size_t hl = (unsigned char) f[8] | ((unsigned char) f[9] << 8);
        const std::string hdr = f.substr(10, hl);
        check(f.size() == 10 + hl + 24 && (10 + hl) % 64 == 0 && f.compare(0, 6, "\x93NUMPY") == 0 && f[6] == 1 && f[7] == 0 && hdr.find("'<f4'") != std::string::npos &&
                  hdr.find("'shape': (2, 3)") != std::string::npos && hdr.find("'fortran_order': False") != std::string::npos && hdr.back() == '\n',
              "trace: .npy v1.0 header (magic, padded to 64, '<f4', shape (2, 3), C order) and the raw floats", hdr);
        float back[6];
        std::memcpy(back, f.data() + 10 + hl, 24);
        check(std::equal(back, back + 6, a), "trace: the payload is the data");
        const std::string e = slurp(dir / "embed.p00000.npy");
        check(e.find("'shape': (6,)") != std::string::npos, "trace: a 1-D shape is written as (n,)");
        const std::string t = slurp(dir / "topk.L03.p00012.npy");
        check(t.find("'<i4'") != std::string::npos && t.size() % 64 == 12, "trace: an int stage is '<i4'");
        const std::string j = slurp(dir / "trace.json");
        check(j.find("\"format\": \"ds41-trace\"") != std::string::npos && j.find("\"tokens\": [10, 11, 12, 13]") != std::string::npos && j.find("\"n_prompt\": 3") != std::string::npos &&
                  j.find("\"compressed_kv\": false") != std::string::npos && j.find("\"window_kv\": true") != std::string::npos && j.find("\"positions\": [0, 7, 12]") != std::string::npos &&
                  j.find("\"producer\": \"engine\"") != std::string::npos && j.find("\"geometry\": \"MiniGeom\"") != std::string::npos,
              "trace: trace.json carries format, producer, geometry, the full token list, n_prompt, the quant flags and the positions", j);
    }
    {
        const fs::path f = dir / "logits.bin";
        {
            session::LogitsDump d(f.string(), 4);
            const float r0[4] = {1, 2, 3, 4}, r1[4] = {5, 6, 7, 8};
            d.add_row(r0);
            d.add_row(r1);
        }
        const std::string s = slurp(f);
        int32_t h[2];
        std::memcpy(h, s.data(), 8);
        check(s.size() == 8 + 32 && h[0] == 4 && h[1] == 2, "logits dump: int32 n_vocab, int32 n_rows, then the float32 rows", fmt("%zu bytes, header %d %d", s.size(), h[0], h[1]));
    }
    fs::remove_all(dir);
}

}  // namespace

int main(int argc, char** argv) {
    bool pool = false, combine = false, trace = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pool") pool = true;
        else if (a == "--combine") combine = true;
        else if (a == "--trace") trace = true;
        else if (a == "--order" && i + 1 < argc) {
            if (!ds41_emu::set_order_from_string(argv[++i])) {
                std::fprintf(stderr, "bad --order\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--pool] [--combine] [--trace]\n", argv[0]);
            return 2;
        }
    }
    if (!pool && !combine && !trace) pool = combine = trace = true;
    if (pool) suite_pool();
    if (combine) suite_combine();
    if (trace) suite_trace();
    std::printf("%d passed, %d failed\n%s\n", g_pass, g_fail, g_fail == 0 ? "ALL PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
