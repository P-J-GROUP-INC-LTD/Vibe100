// src/ds41/program/cli.hpp - DS1-E: the command line of `strata-ds41` (RealGeom on the V100) and `strata-ds41-mini-emu` (MiniGeom on the emulated device): ONE source, so the
// mini program the end-to-end test runs has exactly the options and the code path of the real one.
//
//   --gguf FILE          any shard of the model (the whole split set is opened)                                              (required)
//   --tokens 1,2,3      the prompt as token ids (text: tools/ds41/strata_ds41.py tokenises with DS1-F's tokenizer and calls this)   (required)
//   --max-new N          tokens to generate after the prompt (default 16; stops early at the model's EOS)
//   --max-context N      positions the attention caches hold (default: prompt + max-new)
//   --n-slots N|-1       GPU expert-cache slots (-1: as many as the memory plan says fit; default -1);  --fill LIST  which experts go in ("L:E,L:E0-E1,L:*"; default: DS1-A's static fill by index)
//   --kv-quant on|off    the three KV fake-quantisations (window / compressed / index) together; --window-kv / --compressed-kv / --index-quant on|off one by one   (default on)
//   --threads-per-socket N   workers of each CPU expert group (default: the node's CPUs);  --numa auto|off   bind the expert halves and pin the workers (default auto)
//   --trace DIR          write the DS1-F trace (every stage of DS1.md section 6) of every position into DIR
//   --dump-logits FILE   the logits of every processed position, in tools/volta/golden_compare.py's format
//   --stats              tokens/s, per-stage times (synchronised between stages), hit rate, CPU pool; --stats-async the same without the synchronisation
//   --window N           feed the prompt in windows of N tokens (1..8; default 1: the oracle-equivalent path; the numbers do not depend on it)
//   --temperature T --top-k K --seed S   sampling instead of greedy (T > 0)
//   --no-cpu-pool        refuse routed experts outside the GPU cache instead of computing them on the CPU;  --arena-threads N  threads that copy the experts at load
//   --vram-mib N         pretend the device has N MiB (the memory plan; default: what the device says);  -q  quiet
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "strata/ds41/model/model.hpp"
#include "strata/ds41/session/session.hpp"
#include "strata/ds41/session/trace.hpp"
#include "strata/platform/numa.hpp"

namespace strata::ds41::program {

struct CliArgs {
    std::string gguf, trace_dir, dump_logits, fill;
    std::vector<int32_t> tokens;
    int max_new = 16, max_context = 0, n_slots = -1, window = 1, threads_per_socket = 0, arena_threads = 0, top_k = 0;
    int stats = 0;                      // 0 off, 1 synchronised stage times, 2 asynchronous
    bool window_kv = true, compressed_kv = true, index = true;
    bool numa = true, cpu_pool = true, quiet = false;
    double temperature = 0, vram_mib = 0;
    uint64_t seed = 1;
};

inline const char* usage_text() {
    return "usage: %s --gguf FILE --tokens 1,2,3 [--max-new N] [--max-context N] [--n-slots N|-1] [--fill LIST] [--kv-quant on|off]\n"
           "       [--threads-per-socket N] [--numa auto|off] [--trace DIR] [--dump-logits FILE] [--stats] [--window N]\n"
           "       [--temperature T] [--top-k K] [--seed S] [--no-cpu-pool] [--arena-threads N] [--vram-mib N] [-q]\n";
}

inline bool parse_onoff(const std::string& v, bool& out) {
    if (v == "on" || v == "1" || v == "true" || v == "yes") out = true;
    else if (v == "off" || v == "0" || v == "false" || v == "no") out = false;
    else return false;
    return true;
}

/// Returns "" on success, else the problem (the caller prints it with the usage text and exits 2).
inline std::string parse_cli(int argc, char** argv, CliArgs& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string o = argv[i];
        auto need = [&](std::string& v) -> bool {
            if (i + 1 >= argc) return false;
            v = argv[++i];
            return true;
        };
        std::string v;
        try {
            if (o == "--gguf") { if (!need(a.gguf)) return o + " needs a value"; }
            else if (o == "--tokens") {
                if (!need(v)) return o + " needs a value";
                std::stringstream ss(v);
                std::string t;
                a.tokens.clear();
                while (std::getline(ss, t, ',')) {
                    if (t.empty()) continue;
                    size_t used = 0;
                    const long long id = std::stoll(t, &used);
                    if (used != t.size() || id < 0 || id > 0x7fffffff) return "--tokens: '" + t + "' is not a token id";
                    a.tokens.push_back((int32_t) id);
                }
            } else if (o == "--max-new") { if (!need(v)) return o + " needs a value"; a.max_new = std::stoi(v); }
            else if (o == "--max-context") { if (!need(v)) return o + " needs a value"; a.max_context = std::stoi(v); }
            else if (o == "--n-slots") { if (!need(v)) return o + " needs a value"; a.n_slots = std::stoi(v); }
            else if (o == "--fill") { if (!need(a.fill)) return o + " needs a value"; }
            else if (o == "--kv-quant") {
                if (!need(v) || !parse_onoff(v, a.window_kv)) return "--kv-quant takes on or off";
                a.compressed_kv = a.index = a.window_kv;
            } else if (o == "--window-kv") { if (!need(v) || !parse_onoff(v, a.window_kv)) return o + " takes on or off"; }
            else if (o == "--compressed-kv") { if (!need(v) || !parse_onoff(v, a.compressed_kv)) return o + " takes on or off"; }
            else if (o == "--index-quant") { if (!need(v) || !parse_onoff(v, a.index)) return o + " takes on or off"; }
            else if (o == "--threads-per-socket") { if (!need(v)) return o + " needs a value"; a.threads_per_socket = std::stoi(v); }
            else if (o == "--numa") {
                if (!need(v)) return o + " needs a value";
                if (v == "auto" || v == "on") a.numa = true;
                else if (v == "off") a.numa = false;
                else return "--numa takes auto or off";
            } else if (o == "--trace") { if (!need(a.trace_dir)) return o + " needs a value"; }
            else if (o == "--dump-logits") { if (!need(a.dump_logits)) return o + " needs a value"; }
            else if (o == "--stats") a.stats = 1;
            else if (o == "--stats-async") a.stats = 2;
            else if (o == "--window") { if (!need(v)) return o + " needs a value"; a.window = std::stoi(v); }
            else if (o == "--temperature") { if (!need(v)) return o + " needs a value"; a.temperature = std::stod(v); }
            else if (o == "--top-k") { if (!need(v)) return o + " needs a value"; a.top_k = std::stoi(v); }
            else if (o == "--seed") { if (!need(v)) return o + " needs a value"; a.seed = std::stoull(v); }
            else if (o == "--no-cpu-pool") a.cpu_pool = false;
            else if (o == "--arena-threads") { if (!need(v)) return o + " needs a value"; a.arena_threads = std::stoi(v); }
            else if (o == "--vram-mib") { if (!need(v)) return o + " needs a value"; a.vram_mib = std::stod(v); }
            else if (o == "-q" || o == "--quiet") a.quiet = true;
            else return "unknown option " + o;
        } catch (const std::exception&) {
            return o + ": '" + v + "' is not a number";
        }
    }
    if (a.gguf.empty()) return "--gguf is required";
    if (a.tokens.empty()) return "--tokens is required (a comma-separated list of token ids)";
    if (a.max_new < 0) return "--max-new must be >= 0";
    if (a.window < 1 || a.window > 8) return "--window must be 1..8";
    if (a.temperature < 0) return "--temperature must be >= 0";
    return "";
}

/// Host sampling from one row of logits: top-k (0 = all) then temperature softmax.
inline int32_t sample_token(const float* logits, int n, double temperature, int top_k, std::mt19937_64& rng) {
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    const int k = top_k > 0 ? std::min(top_k, n) : n;
    if (k < n) std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int p, int q) { return logits[p] > logits[q] || (logits[p] == logits[q] && p < q); });
    double mx = -1e300;
    for (int i = 0; i < k; ++i) mx = std::max(mx, (double) logits[idx[(size_t) i]]);
    std::vector<double> p((size_t) k);
    double sum = 0;
    for (int i = 0; i < k; ++i) sum += (p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temperature));
    double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
    for (int i = 0; i < k; ++i) {
        r -= p[(size_t) i];
        if (r <= 0) return idx[(size_t) i];
    }
    return idx[(size_t) (k - 1)];
}

inline uint64_t host_ram_bytes() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0, total = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) total = kb;
    std::fclose(f);
    return (uint64_t) total * 1024;
}

inline void print_stats(const session::SessionStats& s, double prefill_s, int prefill_tokens, double decode_s, int decode_tokens, const session::CpuExpertPool* pool, bool synced) {
    std::printf("\n---- stats ----\n");
    if (prefill_tokens > 0) std::printf("prefill: %d tokens in %.3f s = %.2f tok/s\n", prefill_tokens, prefill_s, prefill_s > 0 ? prefill_tokens / prefill_s : 0.0);
    if (decode_tokens > 0) std::printf("decode:  %d tokens in %.3f s = %.2f tok/s\n", decode_tokens, decode_s, decode_s > 0 ? decode_tokens / decode_s : 0.0);
    std::printf("per-stage time over %llu token(s), %llu layer-steps (%s):\n", (unsigned long long) s.tokens, (unsigned long long) s.layer_steps,
                synced ? "device synchronised after every stage" : "host time: device work overlaps, waits land in the stage that waits");
    const double tot = std::max(1e-12, s.total_s - s.trace_s);
    for (int i = 0; i < session::SessionStats::kStageCount; ++i)
        std::printf("  %-44s %9.3f s  %5.1f %%  %9.1f us/layer-step\n", session::SessionStats::stage_name(i), s.seconds[i], 100.0 * s.seconds[i] / tot,
                    s.layer_steps ? 1e6 * s.seconds[i] / (double) s.layer_steps : 0.0);
    std::printf("  waiting for the split's doorbell: %.3f s; CPU pool wall (start..join): %.3f s; trace writing %.3f s (not in the stages)\n", s.doorbell_wait_s, s.cpu_pool_s, s.trace_s);
    std::printf("routed experts: %llu activations, %llu served by the GPU cache (%.1f %% hit rate), %llu by the CPU pool (%llu layers started it)\n", (unsigned long long) (s.hits + s.misses),
                (unsigned long long) s.hits, 100.0 * s.hit_rate(), (unsigned long long) s.misses, (unsigned long long) s.layers_with_misses);
    if (pool) {
        const session::CpuPoolStats& ps = pool->stats();
        std::printf("CPU pool: %llu jobs, %llu distinct experts, %llu entries, %.3f s wall", (unsigned long long) ps.jobs, (unsigned long long) ps.experts, (unsigned long long) ps.entries, ps.seconds);
        if (ps.experts) std::printf(", %.2f ms per expert (both halves)", 1e3 * ps.seconds / (double) ps.experts);
        std::printf("\n");
    }
}

/// The whole program for geometry G on device `dev` (cuda::CudaDev / cuda::HostDev).  Returns the exit status: 0 done, 1 a runtime failure, 2 a usage error.
template <class G, class DevT>
int run_cli(DevT& dev, int argc, char** argv, const char* prog) {
    using Clock = std::chrono::steady_clock;
    CliArgs a;
    const std::string perr = parse_cli(argc, argv, a);
    if (!perr.empty()) {
        std::fprintf(stderr, "%s: %s\n", prog, perr.c_str());
        std::fprintf(stderr, usage_text(), prog);
        return 2;
    }
    try {
        const int need_ctx = (int) a.tokens.size() + a.max_new;
        if (a.max_context == 0) a.max_context = std::max(need_ctx, 1);
        if (a.max_context < need_ctx) {
            std::fprintf(stderr, "%s: --max-context %d is shorter than the prompt (%zu) plus --max-new (%d)\n", prog, a.max_context, a.tokens.size(), a.max_new);
            return 2;
        }
        const model::LogFn log = a.quiet ? model::LogFn([](const std::string&) {}) : model::LogFn([](const std::string& m) { std::fprintf(stderr, "ds41: %s\n", m.c_str()); });

        // ---- the box: device memory, host memory, the NUMA layout
        size_t vfree = 0, vtotal = 0;
        const bool vknown = dev.mem_info(vfree, vtotal);
        model::LoadOptions lo;
        lo.n_slots = a.n_slots;
        lo.log = log;
        lo.arena.numa = a.numa;
        lo.arena.threads = a.arena_threads;
        lo.plan.ctx_tokens = (uint64_t) a.max_context;
        if (const uint64_t ram = host_ram_bytes()) lo.plan.ram_total = ram;
        if (a.vram_mib > 0) lo.plan.vram_total = (uint64_t) (a.vram_mib * 1048576.0);
        else if (vknown && vtotal > 0) lo.plan.vram_total = std::min<uint64_t>(vtotal, (uint64_t) vfree + (400ull << 20));    // the CUDA context is already taken from `free`
        if (!a.quiet && vknown)
            std::fprintf(stderr, "ds41: device memory: %.0f MiB free of %.0f MiB; the memory plan uses %.0f MiB\n", vfree / 1048576.0, vtotal / 1048576.0, lo.plan.vram_total / 1048576.0);
        const platform::NumaTopology topo = platform::numa_discover();
        session::SessionOptions so;
        so.pool = session::plan_cpu_pool(topo, a.numa, a.threads_per_socket, platform::numa_allowed_cpus());
        if (!a.quiet) std::fprintf(stderr, "ds41: %s\n", so.pool.note.c_str());

        if (!a.fill.empty()) {
            // the model's layer / expert counts are G's
            lo.initial_fill = model::parse_expert_list(a.fill, G::kLayers, G::kExperts);
            if (lo.n_slots >= 0 && (int) lo.initial_fill.size() > lo.n_slots)
                throw model::ModelError("--fill names " + std::to_string(lo.initial_fill.size()) + " experts but --n-slots is " + std::to_string(lo.n_slots));
            if (lo.n_slots < 0) lo.n_slots = (int) lo.initial_fill.size();       // a fill list without a slot count: exactly those experts
        }

        // ---- load (prints the memory plan), then the session
        const auto t_load0 = Clock::now();
        std::unique_ptr<model::Ds41Model<G>> model = model::Ds41Model<G>::load(dev, a.gguf, lo);
        const double load_s = std::chrono::duration<double>(Clock::now() - t_load0).count();
        for (const std::string& w : model->warnings()) std::fprintf(stderr, "ds41: warning: %s\n", w.c_str());
        so.max_context = a.max_context;
        so.max_window = a.window;
        so.window_kv = a.window_kv;
        so.compressed_kv = a.compressed_kv;
        so.index = a.index;
        so.use_cpu_pool = a.cpu_pool;
        so.stage_timing = a.stats == 1;
        session::Ds41Session<G> sess(dev, *model, so);
        if (!a.quiet) std::fprintf(stderr, "ds41: loaded in %.1f s\n%s\n", load_s, sess.describe().c_str());

        int eos = -1;
        if (model->gguf().meta().has("tokenizer.ggml.eos_token_id")) eos = (int) model->gguf().meta().get_int("tokenizer.ggml.eos_token_id");

        std::unique_ptr<session::TraceWriter> trace;
        session::TraceQuant tq;
        tq.window_kv = a.window_kv;
        tq.compressed_kv = a.compressed_kv;
        tq.index = a.index;
        if (!a.trace_dir.empty()) {
            trace.reset(new session::TraceWriter(a.trace_dir, G::kName == std::string("mini") ? "MiniGeom" : "RealGeom", tq, (int) a.tokens.size()));
            trace->set_tokens(a.tokens, (int) a.tokens.size());
            trace->flush();
            sess.set_trace(trace.get());
        }
        std::unique_ptr<session::LogitsDump> dump;
        if (!a.dump_logits.empty()) dump.reset(new session::LogitsDump(a.dump_logits, G::kVocab));
        std::mt19937_64 rng(a.seed);
        const bool sampling = a.temperature > 0;

        // ---- the sequence: the prompt one window at a time, then the generated tokens one at a time
        std::vector<int32_t> toks = a.tokens;
        const int n_prompt = (int) a.tokens.size();
        std::vector<int32_t> generated;
        const auto flush_trace = [&] {
            if (trace) {
                trace->set_tokens(toks, n_prompt);
                trace->flush();
            }
        };
        const auto feed = [&](const int32_t* t, int T, bool last_of_prompt_or_generated) {
            // the head is needed after the last prompt token and after every generated one; for every row when the logits are dumped or traced
            const bool all_rows = dump != nullptr;
            session::Want want = session::Want::kNone;
            if (all_rows || last_of_prompt_or_generated) want = (all_rows || sampling) ? session::Want::kLogits : session::Want::kArgmax;
            sess.step(t, T, want);
            if (dump && want == session::Want::kLogits)
                for (int r = 0; r < T; ++r) dump->add_row(sess.logits(r));
            flush_trace();
            return want;
        };
        const auto t0 = Clock::now();
        session::Want last_want = session::Want::kNone;
        int row = 0;
        for (int i = 0; i < n_prompt; i += a.window) {
            const int T = std::min(a.window, n_prompt - i);
            const bool last = i + T >= n_prompt;
            if (last && a.max_new == 0 && !dump && !trace) { sess.step(&toks[(size_t) i], T, session::Want::kNone); break; }
            last_want = feed(&toks[(size_t) i], T, last);
            row = T - 1;
        }
        const double prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
        const auto t1 = Clock::now();
        int decoded = 0;
        for (int g = 0; g < a.max_new; ++g) {
            int32_t next;
            if (sampling) next = sample_token(sess.logits(row), G::kVocab, a.temperature, a.top_k, rng);
            else next = sess.argmax(row);
            toks.push_back(next);
            generated.push_back(next);
            flush_trace();
            if (next == eos) break;
            if (g + 1 == a.max_new) break;                              // the last generated token is an output only: no forward pass
            last_want = feed(&next, 1, true);
            row = 0;
            ++decoded;
        }
        (void) last_want;
        const double decode_s = std::chrono::duration<double>(Clock::now() - t1).count();

        std::printf("prompt (%d tokens):", n_prompt);
        for (int32_t t : a.tokens) std::printf(" %d", t);
        std::printf("\ngenerated (%zu tokens, %s):", generated.size(), sampling ? "sampled" : "greedy");
        for (int32_t t : generated) std::printf(" %d", t);
        std::printf("\n");
        if (trace) std::printf("trace: %llu files, %.1f MB in %s\n", (unsigned long long) trace->files(), (double) trace->bytes() / 1e6, trace->dir().c_str());
        if (dump) std::printf("logits: %d rows of %d in %s\n", dump->rows(), G::kVocab, a.dump_logits.c_str());
        if (a.stats) print_stats(sess.stats(), prefill_s, n_prompt, decode_s, decoded, sess.pool(), a.stats == 1);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: error: %s\n", prog, e.what());
        return 1;
    }
}

}  // namespace strata::ds41::program
