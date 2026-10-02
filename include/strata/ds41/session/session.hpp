// include/strata/ds41/session/session.hpp - DS1-E: the DeepSeek-V4.1-Flash decode session: one loaded model, its caches, and the decode step that turns tokens into logits.
//
//   auto model = Ds41Model<G>::load(dev, "/models/...-00001-of-00012.gguf", load_options);        (DS1-A)
//   Ds41Session<G> sess(dev, *model, options);                                                  allocates the scratch and the attention caches (max_context), starts the CPU pool
//   sess.step(&token, 1, Want::kArgmax);  sess.argmax(0)  /  sess.logits(0)                      one token at position()  (T = 1; a window of T <= max_window consecutive tokens too)
//   sess.reset();                                                                               a new sequence
//
// THE DECODE STEP, in the oracle's order (ref/ds41/model.py; every line of `Model.forward` / `Model.block` is a numbered step of Ds41Session<G>::run_window in session_impl.hpp):
//     hasher push (the token) -> embedding row -> expand to kHc copies (pre_mix = [1, 0, 0, 0])                                 model.py 150-154
//     for each layer l:
//         [Engram layers] gather the hashed rows, dequantise, wkv (int8 act), combine into the stream                            model.py 156-164
//         attention sub-layer:  hc_mixes(x) -> hc_pre with the LAGGED pre -> attn_norm -> attention -> hc_post                     model.py 102-111
//         FFN sub-layer:        hc_mixes(x) -> hc_pre with the attention's own pre -> ffn_norm -> router -> split hits / misses
//                               -> GPU hits (DS-D experts) || CPU misses (two sockets) + shared expert -> FP32 sum -> hc_post     model.py 115-131
//     final hc_pre (the last FFN's pre) -> output_norm -> head -> logits -> argmax                                              model.py 170-177
// Prefill is the prompt fed one token at a time through the same path (equal to the oracle's prefill by construction: CONTRACTS.md, index-K decision); a window of T <= 8
// tokens gives bit-identical rows (every kernel's reduction order is independent of T).
//
// WHERE THE WORK RUNS: everything except the routed experts that missed the GPU cache and the Engram row gather runs on the device, in ONE stream (dev.stream()), asynchronously;
// the host blocks only where it must: the Engram rows (a gather from the mapped tables), the split's doorbell (to learn the misses), and the logits.  A layer with misses starts the
// CPU pool the moment the doorbell rings and enqueues the GPU's hits and the shared expert while the CPU works.
//
// Not thread-safe: one session = one stream = one caller.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/model/model.hpp"
#include "strata/ds41/session/cpu_pool.hpp"
#include "strata/ds41/session/trace.hpp"

namespace strata::ds41::session {

struct SessionOptions {
    int max_context = 4096;              ///< positions the attention caches and the RoPE tables hold (positions 0 .. max_context - 1)
    int max_window = 1;                  ///< tokens per step() (T), 1 .. 8; the scratch is sized for it
    bool window_kv = true;               ///< the three KV fake-quantisations (QuantConfig.window_kv / compressed_kv / index), default all on (DS1.md section 2)
    bool compressed_kv = true;
    bool index = true;
    bool use_cpu_pool = true;            ///< false: a layer with misses is an error (no pool; the arena need not be built)
    CpuPoolOptions pool;                 ///< the pool's layout (plan_cpu_pool); max_tokens / top_k are filled in by the session
    bool stage_timing = false;           ///< synchronise after every stage and time it (--stats): exact per-stage times, no host / device overlap inside a layer boundary
    double doorbell_timeout_s = 120.0;   ///< how long the host waits for a split's doorbell before it reports a hung device
};

/// What step() leaves for the caller.
enum class Want { kNone, kArgmax, kLogits };

/// Per-stage wall time (seconds; with SessionOptions::stage_timing every stage is synchronised, otherwise the host-side issue time plus the waits) and counters.
struct SessionStats {
    enum Stage { kEmbed, kEngram, kAttnIn, kAttention, kAttnOut, kFfnIn, kRouterSplit, kExperts, kCombine, kHead, kStageCount };
    static const char* stage_name(int s) {
        static const char* const names[kStageCount] = {"embed",     "engram",         "attn_in (mHC + norm)", "attention",
                                                       "attn_out (mHC)", "ffn_in (mHC + norm)", "router + split", "experts (GPU hits || CPU misses + shared)",
                                                       "combine + ffn mHC", "head + argmax"};
        return s >= 0 && s < kStageCount ? names[s] : "?";
    }
    double seconds[kStageCount] = {};
    double doorbell_wait_s = 0;          ///< host time spent waiting for the split's doorbell
    double cpu_pool_s = 0;               ///< wall time of the CPU pool's jobs (start .. join)
    double trace_s = 0;                  ///< host time spent writing the trace
    double total_s = 0;                  ///< wall time of all step() calls
    uint64_t tokens = 0;                 ///< tokens run through the model
    uint64_t layer_steps = 0;            ///< (token, layer) pairs
    uint64_t hits = 0, misses = 0;       ///< (token, routed expert) activations served by the GPU cache / the CPU pool
    uint64_t layers_with_misses = 0;     ///< layers (per step) that started the CPU pool
    double hit_rate() const { return hits + misses ? (double) hits / (double) (hits + misses) : 0.0; }
};

template <class G>
class Ds41Session {
public:
    /// `dev` must outlive the session (and the model).  Throws std::invalid_argument for inconsistent options, std::runtime_error for an exhausted device.
    Ds41Session(cuda::Dev& dev, model::Ds41Model<G>& model, const SessionOptions& opt);
    ~Ds41Session();
    Ds41Session(const Ds41Session&) = delete;
    Ds41Session& operator=(const Ds41Session&) = delete;

    const SessionOptions& options() const;
    /// Positions consumed so far (the position of the next token).
    int position() const;
    /// A new sequence: position 0, the attention caches' shared state and the n-gram history forgotten.
    void reset();

    /// Runs `T` (1 .. max_window) consecutive tokens at positions position() .. position() + T - 1 through the model.  `want` != kNone also runs the head: argmax(t) is the
    /// greedy next token after row t (kArgmax: a 4-byte copy per row; kLogits: every row's logits are copied to the host too).  Throws std::out_of_range past max_context.
    void step(const int32_t* tokens, int T, Want want = Want::kArgmax);
    int32_t argmax(int t = 0) const;
    /// Host copy of row t's logits (kVocab floats), valid until the next step(); step() must have been called with Want::kLogits.
    const float* logits(int t = 0) const;
    /// Row t's logits are `vocab()` floats.
    int vocab() const { return G::kVocab; }

    /// Writes every stage DS1.md section 6 lists (and the optional ones the attention exposes) of the following steps into `trace` (nullptr: off).  Tracing synchronises the
    /// device after every stage: a debugging mode.  The positions written are those of the tokens run; the writer's token list is the caller's to keep (set_tokens / flush).
    void set_trace(TraceWriter* trace);

    const SessionStats& stats() const;
    void reset_stats();
    /// A multi-line description of the session: device memory, pool, cache.
    std::string describe() const;
    /// The device bytes the session itself holds (scratch + attention caches + RoPE tables), for the startup log.
    uint64_t device_bytes() const;
    const CpuExpertPool* pool() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Explicit instantiation helper: `DS41_INSTANTIATE_SESSION(::strata::ds41::RealGeom)` after including src/ds41/session/session_impl.hpp.
#define DS41_INSTANTIATE_SESSION(G) template class ::strata::ds41::session::Ds41Session<G>;

}  // namespace strata::ds41::session
