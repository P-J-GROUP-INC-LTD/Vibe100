// include/strata/ds41/cuda/ds41_cuda.hpp - DS-D / DS1-G: the GPU half of the DeepSeek-V4.1-Flash MoE layer, for the V100 (sm_70), as templates
// over the geometry policy G (include/strata/ds41/geom.hpp: RealGeom, MiniGeom), and the activation quantiser every dp4a GEMV shares.
//
// EVERY FUNCTION HERE IS `template <class G>`: the shapes come from G (G::kHidden, G::kFF, G::kExperts, G::kTopK, G::kLayers) and Derived<G>
// (kExpertBlobBytes ...); the numbers below are RealGeom's (5120 / 2304 / 384 / 6) and are examples, not constants.  The nvcc build (src/ds41/cuda/
// ds41_{router,split,experts}.cu, library strata_ds41_cuda) instantiates RealGeom only; the CPU emulation (ds41_emu_impl.cpp, target
// ds41_cuda_emu_test, and any test that is built with -DDS41_EMU against the emulator) instantiates RealGeom and MiniGeom.  Calling a G that the
// build did not instantiate is a LINK error.  What G must satisfy is checked by static_asserts in the impl headers (see "CONSTRAINTS ON G").
//
// THREE JOBS, IN THE ORDER A LAYER NEEDS THEM (math: docs/deepseek/CONTRACTS.md):
//
//   1. ROUTER        router_forward<G>():  x fp32 [T][5120], Wg bf16 [384][5120], bias fp32 [384]
//                    -> ids int32 [T][6], weights fp32 [T][6]
//                    logits in FP32, s = sqrt(softplus(logit)), top-6 of (s + bias) with the LOWEST index winning ties,
//                    w = s[ids], w = w / (sum(w) + 1e-20) * 1.5.   Any T from 1 to a few thousand.
//                    ONE REDUCTION ORDER FOR EVERY T: a token's logits are bit-identical whether it is routed alone (decode), in a
//                    verify window or in a prompt chunk (CONTRACTS.md: otherwise a near-tie routes differently in prefill and decode).
//   2. SPLIT         split_hits_misses<G>():  the router's ids against the residency table of the layer ->
//                    HIT list (GPU computes), MISS list (the CPU pool computes), counts, and the hits grouped by
//                    cache slot.  Entirely on the device: the router's output never visits the host unless a host record is asked for.
//   3. HIT EXPERTS   quantize -> experts_gate_up<G>() -> experts_down<G>()   (experts_hits<G>() runs all three):
//                    the MXFP4 blobs sitting in VRAM cache slots, T = 1..8 tokens, ALL hits of the layer in one launch
//                    per phase (two for T >= 5: a lean and a wide specialisation over the same list), device-side lists only
//                    (no host synchronisation between the router and the experts).
// and, for everybody:  ds41_quantize_acts<G>()  the activation quantiser (any width that is a multiple of 32), see "ACTIVATION AND h LAYOUT".
//
// ---------------------------------------------------------------------------------------------------------------------
// CONSTRAINTS ON G (what the kernels assume beyond geom_ok<G>(); every one is a static_assert at the point of use)
// ---------------------------------------------------------------------------------------------------------------------
//   router    kHidden % 256 == 0 (a row is a whole number of 32-lane x 16-byte chunk groups; 20 for RealGeom: 2 per warp, 10 warps; 1 for a row of 256: one
//             warp), kExperts % 4 == 0 (the prefill kernel's 4 experts per block), kExperts <= 1024, kTopK <= kExperts and kTopK <= 32.  kExperts need NOT be a
//             multiple of 32 (MiniGeom: 16): the selection kernel pads the lanes past the last expert.
//   split     kTopK * 8 <= 64 (the grouped-hit arrays), kLayers >= 1.
//   experts   kHidden % 256 == 0 and kFF % 256 == 0 (blocks of 32: 8 per gate/up row at least, so that a pair of rows is a whole number of 16-byte
//             chunks; kHidden % 512 == 0 additionally stages half rows with 16-byte loads like RealGeom, otherwise a whole pair of rows is staged
//             contiguously - MiniGeom's 8-block rows of 136 B), (kFF / 32) % 4 == 0, kHidden % 64 == 0 (the down tile), kHidden / 32 % 4 == 0.
//
// ---------------------------------------------------------------------------------------------------------------------
// FOR THE OTHER DS-1 PACKAGES (what to include; nothing here pulls a kernel)
// ---------------------------------------------------------------------------------------------------------------------
//   strata/ds41/cuda/ds41_dev.hpp          Dev (device memory, mapped host memory, copies, streams, events, a stopwatch), DevBuf<T>, MappedBuf<T>, HostDev
//                                          (the emulation's Dev), Stream, stream_or_default().  Plain C++: host code compiles unchanged for the V100 and the emulator.
//   strata/ds41/cuda/ds41_cuda_runtime.hpp CudaDev, the Dev for a real GPU (includes cuda_runtime.h; only the engine's and the V100 programs' host code needs it).
//   strata/ds41/cuda/ds41_cuda.hpp         this file: ds41_quantize_acts<G>() (any width; the layout above), the router / split / hot-expert host API.
//   src/ds41/cuda/ds41_dev.cuh             (kernel side, the .cu / emulator impl files only) DS41_KERNEL & co., the intrinsic wrappers, and the launch helper
//                                          dev::launch(kernel<G, ...>, grid, block, smem, stream, args...) + dev::check_launch("what"): the same call under nvcc and in
//                                          the emulator; see ds41_dev.hpp "THE LAUNCH PATTERN".
//   An op of another package is `template <class G> void ds41_<op>(Dev&, args..., Stream = nullptr)` (DS1.md section 5); the shared quantiser's signature is
//       template <class G> void ds41_quantize_acts(Dev& dev, const float* x, int T, int width, int8_t* xq, float* xs, Stream stream = nullptr,
//                                                   ActOrder order = ActOrder::kInterleaved);
//   (xq: T * width bytes, xs: T * width / 32 floats, sizes quant_acts_q_bytes() / quant_acts_s_floats(); explicitly instantiated for RealGeom in ds41_experts.cu and
//   for RealGeom + MiniGeom in the emulator build, ds41_emu_impl.cpp: a G the build did not instantiate is a link error.)

// ---------------------------------------------------------------------------------------------------------------------
// DEVICE-SIDE LIST FORMATS (all plain structs, 16-byte multiples, little endian; the miss list is meant to be read by
// the host)
// ---------------------------------------------------------------------------------------------------------------------
//   HitEntry   { int32 token; int32 k; int32 slot; float weight; }   token = 0..T-1, k = 0..5 (rank in the router's
//              output, i.e. ids[token*6 + k]), slot = the cache slot holding that expert's blob, weight = the routing
//              weight (already normalised and scaled).  Entries are in (token, k) order: deterministic.
//   MissEntry  { int32 token; int32 k; int32 expert; float weight; } the same, but with the expert id (0..383) instead of
//              a slot; this is what the host's CPU expert pool reads.  Also in (token, k) order.
//   SplitCounts{ int32 n_hits; int32 n_misses; int32 n_groups; int32 n_bad_slots; }   DEVICE memory: the expert kernels read n_groups
//              from it (once per block, from L2 - never over PCIe), the caller may copy it out.  n_bad_slots = how many residency
//              entries were corrupt (see "Cache slots" below); a caller asserts it is 0.
//   HitGroup   { int32 slot; int32 n; int32 hit[8]; }   (T <= 8 only)  the hits that share a cache slot, i.e. the same
//              expert for several tokens: n = how many (1..8), hit[j] = index into the HIT list (members in hit order).  The
//              expert kernels stream the blob ONCE per group and use it for all n tokens.  Groups are numbered by DECREASING n
//              (ties: order of first appearance): the expert kernels run the groups with 5..8 tokens in a wide specialisation.
//   Capacity: hits, misses and groups each hold up to kTopK*T entries.  Nothing is zeroed beyond the counts.
//
//   Cache slots.  The residency table is int32 [kLayers][kExperts] (40 x 384): the slot (0 .. n_slots - 1) holding that expert's blob, or -1.  INITIALISE
//   IT TO -1 (every byte 0xFF, cudaMemset(table, 0xFF, bytes)), NEVER to 0: 0 is a valid slot.  A value r is a HIT only if 0 <= r < n_slots (the
//   `n_slots` argument = the number of blobs the cache holds); any other value is a MISS (the CPU computes it), and if r is not the "not
//   resident" marker -1 it is also COUNTED in counts->n_bad_slots / host->n_bad_slots, so a stale or corrupt table shows up as a number
//   instead of as the expert kernels reading `cache + slot * kBlobBytes` outside the cache.
//
// ---------------------------------------------------------------------------------------------------------------------
// THE SPLIT'S RESULT ON THE HOST: ONE HOST-VISIBLE RECORD PER LAYER IN FLIGHT, A SEQUENCE NUMBER AS DOORBELL
// ---------------------------------------------------------------------------------------------------------------------
//   struct SplitHostRecord { uint32 seq; int32 n_hits, n_misses, n_groups, n_bad_slots; int32 reserved[3]; }   32 bytes, MAPPED pinned host memory
//
//   What the caller owns, per LAYER IN FLIGHT (the host may still be reading layer L's result while the stream is already running layer L+1's
//   split, and a graph or a queue of 40 layers launches them all before the first one finishes):
//       * one SplitHostRecord (Dev::alloc_mapped / cudaHostAlloc(..., cudaHostAllocMapped), zero-initialised: seq = 0),
//       * one miss list `misses` of kTopK*T MissEntry in mapped host memory (any pinned buffer the kernel can address).
//     The device-side lists (hits, groups, counts) are scratch for the expert kernels of the SAME layer: stream order protects them, one set
//     serves all layers.
//   The protocol (a sequence number instead of the old "reserved = 0 means done", which could not tell layer L's result from layer L-1's):
//       caller   picks seq = 1, 2, 3, ... for this record (0 is the cleared state; skip it on wrap, split_next_seq()), launches
//                split_hits_misses<G>(..., misses_of_layer, counts, &record, seq, stream)
//       kernel   writes the miss list; __syncthreads; thread 0 writes the counts into the device `counts` AND the host record, issues ONE
//                fence.acq_rel.sys, then stores `seq` into record.seq with st.release.sys.global.u32 - the LAST store; no other thread fences
//       host     split_host_wait(record, seq)  (an ACQUIRE load of record.seq until it equals the expected number), then reads the record's
//                counts and misses[0 .. n_misses): they are the complete result of THIS launch, never of an earlier use of the record
//   With `host == nullptr` nothing is written to the host: copy `counts` (device) and the miss list out after the stream's work
//   (cudaMemcpyAsync + event).  The expert kernels never look at the host record.
//
// ---------------------------------------------------------------------------------------------------------------------
// ACTIVATION AND h LAYOUT (what ds41_quantize_acts<G>() writes and the dp4a kernels read; the other DS-1 packages read it too)
// ---------------------------------------------------------------------------------------------------------------------
//   An activation vector is quantised per 32 values (CONTRACTS.md "Activations"; bit-identical to the CPU's quantize_act() on EVERY input):
//   amax = the largest |x| (an integer maximum of the magnitude bits); d = amax/127 (fp32), id = 127/amax, q = rint(x*id) (ties to EVEN)
//   clamped to [-127,127]; a block with amax < 2^-100 (zero included) has d = 0, q = 0; a block holding an Inf or a NaN has d = NaN
//   (0x7FC00000) and q = 0, so the NaN reaches y.  NaN also propagates through the SwiGLU clamps (min(g,10), clamp(u,-10,10)).
//   The value a block stands for is  x[32 b + j] ~= q[b][j] * d[b].
//
//   A quantised matrix of T rows (tokens) of W values (W a multiple of 32, NB = W / 32 blocks per row) is TWO arrays:
//       xq  int8  [T][NB][32]   token t, block b starts at byte (t * NB + b) * 32; inside the block the byte of natural element j (0..31 = column
//                               32 b + j) is
//                                 ActOrder::kInterleaved (the default)   byte 8*((j&15)>>2) + 2*(j&3) + (j>>4)   (act_perm_pos(j))
//                                 ActOrder::kNatural                     byte j
//       xs  fp32  [T][NB]       the block scales d, token-major (token t, block b at index t * NB + b)
//   INTERLEAVED is what dp4a wants after the MXFP4 decode.  The block is four 8-byte groups, k = 0..3; group k is
//       [x[4k], x[4k+16], x[4k+1], x[4k+17], x[4k+2], x[4k+18], x[4k+3], x[4k+19]]
//   i.e. as 32-bit little-endian words (w = 0..7 = bytes 4w..4w+3 of the block): word 2k = [x[4k], x[4k+16], x[4k+1], x[4k+17]] and word 2k+1 = [x[4k+2],
//   x[4k+18], x[4k+3], x[4k+19]] (x = the natural elements j = 0..31 of the block, byte 0 first).  A GEMV whose weights are stored as Q8_0 (32 int8 in natural
//   order per block) wants NATURAL: word w = x[4w .. 4w+3], 8 dp4a per block with no permute.  One quantise call per (input, order); every GEMV that reads the same
//   input in the same order shares it.  Both orders carry the same scales xs (the scale of block b is independent of the byte order).
//   Alignment: xq 16 bytes, xs 16 bytes when NB % 4 == 0 (else 4).  Sizes: quant_acts_q_bytes(T, W) = T*W bytes, quant_acts_s_floats(T, W) = T*W/32 floats.
//   (act_perm_pos() below is the single definition of the interleaved order.)
//
//   The hot experts' own buffers (ExpertScratch): xq / xs = the quantised x with W = kHidden, INTERLEAVED; hq: int8 [kTopK*T][kFF/32][32] (h, indexed by
//   HIT index, interleaved), hs: fp32 [kTopK*T][kFF/32].
//
// ---------------------------------------------------------------------------------------------------------------------
// A DECODE LAYER (one stream; T = 1..8), and what the caller guarantees
// ---------------------------------------------------------------------------------------------------------------------
//   router_forward<G>(x, wg, bias, T, logits_ws, ids, weights, s);
//   split_hits_misses<G>(ids, weights, T, residency, n_slots, layer, hits, misses, groups, counts, &record[layer], seq[layer], s);
//   ... the host CPU pool: split_host_wait(record[layer], seq[layer]), then reads the record's counts and `misses` and computes the misses ...
//   experts_hits<G>(cache_base, x, hits, groups, counts, T, scratch, parts, s);     // parts rows of the misses stay untouched
//   ... out[t] = shared expert + sum over k of parts[t*6+k] (hits) + the CPU's partials (misses), FP32 ...
//   * every pointer is a device pointer, except the optional host record and the miss list the caller wants the host to read (mapped pinned
//     host memory is addressable by the kernels); `counts`, `hits` and `groups` are DEVICE memory; pass `dev.stream()` as the stream;
//   * alignment: x, wg, logits, xq, xs, hq, hs, parts: 16 bytes; cache_base: 256 bytes (kBlobBytes is a multiple of 256, so every slot is);
//   * the groups given to experts_gate_up / experts_down must be the ones split_hits_misses produced (sorted by decreasing size);
//   * the cache slots must be complete and ordered before the experts kernels on the stream (fills on the same stream, or an event); the
//     residency table names a slot only after its blob is complete, and is initialised to -1 (see "Cache slots");
//   * the blobs are the GGUF layout of CONTRACTS.md: [gate][up][down], MXFP4 blocks of 17 bytes, rows 2720 / 1224 bytes (kHidden/32 and kFF/32 blocks);
//   * experts_* work for T = 1..8 only; router_* and split_hits_misses (without groups) for any T (prefill).
// Errors (bad T, a failed launch) throw std::invalid_argument / std::runtime_error.
//
// Everything here is plain C++ (void* streams) so that host code needs no CUDA headers.
#pragma once

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"
#include "strata/ds41/geometry.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <version>

namespace strata::ds41::cuda {

inline constexpr int kMaxExpertTokens = 8;               // expert kernels: T = 1..8 (decode / a verify window)
inline constexpr int kRouterSmallT = 32;                 // router: T <= 32 -> GEMV kernel, above -> the token-looping kernel (same reduction order, same bits)

/// Byte position of natural element j (0..31) of a 32-block inside the kernel's INTERLEAVED int8 layout (see above).
constexpr int act_perm_pos(int j) { return ((j & 15) >> 2) * 8 + (j & 3) * 2 + (j >> 4); }

/// How the 32 int8 of a quantised block are ordered (see "ACTIVATION AND h LAYOUT").
enum class ActOrder { kInterleaved, kNatural };

/// The shapes of the hot-expert path at geometry G (RealGeom: 160 blocks of x, 72 of h, the 18,800,640-byte blob).
template <class G> struct ExpertDims {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    static constexpr int kHidden = G::kHidden, kFF = G::kFF, kTopK = G::kTopK;
    static constexpr int kActBlocks = G::kHidden / kQK;                       // 32-blocks of x            (160)
    static constexpr int kHBlocks = G::kFF / kQK;                             // 32-blocks of h            (72)
    static constexpr size_t kGateRowBytes = (size_t) kActBlocks * kBlockBytes;   // a gate / up row         (2720)
    static constexpr size_t kDownRowBytes = (size_t) kHBlocks * kBlockBytes;     // a down row              (1224)
    static constexpr size_t kGateBytes = Derived<G>::kExpertGateBytes;           // gate (and up) matrix    (6,266,880)
    static constexpr size_t kDownBytes = Derived<G>::kExpertDownBytes;
    static constexpr size_t kBlobGate = 0;                                    // blob = [gate][up][down]
    static constexpr size_t kBlobUp = kGateBytes;
    static constexpr size_t kBlobDown = 2 * kGateBytes;
    static constexpr size_t kBlobBytes = Derived<G>::kExpertBlobBytes;        // 18,800,640
    static_assert(kGateBytes == (size_t) kFF * kGateRowBytes && kDownBytes == (size_t) kHidden * kDownRowBytes && kBlobBytes % 256 == 0,
                  "one cache slot is a whole number of 256-byte units");
};

struct HitEntry {
    int32_t token;
    int32_t k;
    int32_t slot;
    float weight;
};
struct MissEntry {
    int32_t token;
    int32_t k;
    int32_t expert;
    float weight;
};
struct SplitCounts {            // DEVICE memory
    int32_t n_hits;
    int32_t n_misses;
    int32_t n_groups;
    int32_t n_bad_slots;        // residency entries that were neither -1 nor in [0, n_slots): treated as misses, counted here
};
/// The host-visible result of one split: MAPPED pinned memory, one per layer in flight, zero-initialised.  `seq` is the doorbell.
struct SplitHostRecord {
    uint32_t seq;               // the sequence number of the launch whose result this record holds; written LAST with release semantics
    int32_t n_hits;
    int32_t n_misses;
    int32_t n_groups;
    int32_t n_bad_slots;
    int32_t reserved[3];
};
struct HitGroup {
    int32_t slot;
    int32_t n;
    int32_t hit[kMaxExpertTokens];
};
static_assert(sizeof(HitEntry) == 16 && sizeof(MissEntry) == 16 && sizeof(SplitCounts) == 16 && sizeof(HitGroup) == 40 && sizeof(SplitHostRecord) == 32);

// ---- router -------------------------------------------------------------------------------------------------------
/// Bytes of the logits workspace for T tokens (fp32 [T][kExperts]).
template <class G> inline constexpr size_t router_workspace_bytes(int T) { return (size_t) T * G::kExperts * sizeof(float); }

/// logits[t][e] = sum_k x[t][k] * bf16_to_f32(wg[e][k]), FP32 accumulation in a fixed order (deterministic run to run) that does NOT
/// depend on T: every (token, expert) logit is 10 warp slices of 2 x 8 chained FMAs per lane (RealGeom; a row of 256 is 1 slice of 1 x 8), a 32-lane
/// butterfly per slice, then the slice totals added in order - the same bits whether the token is alone, in a verify window (T <= 32, the GEMV kernel)
/// or in a prompt chunk (T > 32, the token-looping kernel that reads Wg from L2 once per token group).
/// x [T][kHidden] fp32, wg [kExperts][kHidden] raw bf16 bits, logits [T][kExperts].
template <class G> void router_logits(const float* x, const uint16_t* wg_bf16, int T, float* logits, void* stream);

/// ids/weights from logits (see the header).  NaN logits are treated as -inf for selection.
template <class G> void router_select(const float* logits, const float* bias, int T, int32_t* ids, float* weights, void* stream);

/// router_logits + router_select.  `logits_ws` = router_workspace_bytes<G>(T) of device memory (kept: the parity program
/// reads it to report near-ties).
template <class G>
void router_forward(const float* x, const uint16_t* wg_bf16, const float* bias, int T, float* logits_ws, int32_t* ids, float* weights, void* stream);

// ---- hit / miss split ---------------------------------------------------------------------------------------------
/// ids int32 [T][kTopK], weights fp32 [T][kTopK] (router output), residency_table int32 [kLayers][kExperts] (slot or -1; device; INITIALISED TO -1),
/// `n_slots` = the number of blobs in the cache (a residency value outside [0, n_slots) is a miss, and counted in n_bad_slots),
/// `layer` 0..kLayers-1.  Fills hits, misses, counts (DEVICE) and (only when `groups` != nullptr, requires T <= 8) groups.
/// An id outside 0..kExperts-1 is counted as a miss (and passed on unchanged).
/// `host` (optional, mapped host memory, one per layer in flight, see above) receives the counts and then `seq` (!= 0) with release semantics;
/// pass nullptr when the host does not need the result before the stream is synchronised.
template <class G>
void split_hits_misses(const int32_t* ids, const float* weights, int T, const int32_t* residency_table, int n_slots, int layer, HitEntry* hits,
                       MissEntry* misses, HitGroup* groups, SplitCounts* counts, SplitHostRecord* host, uint32_t seq, void* stream);

/// The next sequence number for a host record: 1, 2, ... and back to 1 after 0xFFFFFFFF (0 is the cleared state).
inline uint32_t split_next_seq(uint32_t seq) { return seq + 1u == 0u ? 1u : seq + 1u; }
/// Host side of the doorbell: an ACQUIRE load of the record's `seq`; true once the launch with sequence number `seq` has completed its result
/// (then the record's counts and the miss list the launch wrote are visible).
inline bool split_host_ready(const SplitHostRecord& rec, uint32_t seq) {
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
    return std::atomic_ref<uint32_t>(const_cast<uint32_t&>(rec.seq)).load(std::memory_order_acquire) == seq;
#else
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free, "a lock-free 32-bit atomic");
    return reinterpret_cast<const std::atomic<uint32_t>&>(rec.seq).load(std::memory_order_acquire) == seq;
#endif
}
/// Spin (then yield) until split_host_ready(rec, seq); false if `timeout_seconds` passed first (a launch that never ran, or a wrong `seq`).
inline bool split_host_wait(const SplitHostRecord& rec, uint32_t seq, double timeout_seconds = 10.0) {
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned spins = 0;; ++spins) {
        if (split_host_ready(rec, seq)) return true;
        if ((spins & 1023u) == 1023u) {
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_seconds) return false;
            std::this_thread::yield();
        }
    }
}

// ---- the activation quantiser (all widths) --------------------------------------------------------------------------
/// Bytes of xq and floats of xs for T rows of `width` values (a multiple of 32).
constexpr size_t quant_acts_q_bytes(int T, int width) { return (size_t) T * width; }
constexpr size_t quant_acts_s_floats(int T, int width) { return (size_t) T * (width / 32); }

/// x fp32 [T][width] (row-major, contiguous; 4-byte aligned) -> xq int8 [T][width/32][32], xs fp32 [T][width/32] in the layout of "ACTIVATION AND h
/// LAYOUT" (`order`: INTERLEAVED for the MXFP4 expert kernels, NATURAL for a Q8_0 GEMV).  `width` any multiple of 32 (the dense GEMVs: 1280, 5120,
/// 8192, 32768 ...); T >= 1 (any); the rule is CONTRACTS.md's, bit-identical to the CPU's on every input.  Launched on `stream` (nullptr = dev.stream());
/// stream-ordered, allocation-free.  With width == G::kHidden and the INTERLEAVED order this is the expert path's own kernel (compile-time width).
/// Throws std::invalid_argument for width % 32 != 0, T < 1 or a misaligned pointer.
template <class G>
void ds41_quantize_acts(Dev& dev, const float* x, int T, int width, int8_t* xq, float* xs, Stream stream = nullptr, ActOrder order = ActOrder::kInterleaved);

// ---- hit experts --------------------------------------------------------------------------------------------------
struct ExpertScratch {
    int8_t* xq;   // [T][kHidden/32][32]
    float* xs;    // [T][kHidden/32]
    int8_t* hq;   // [kTopK*T][kFF/32][32]
    float* hs;    // [kTopK*T][kFF/32]
};
namespace detail {
constexpr size_t up256(size_t v) { return (v + 255) & ~(size_t) 255; }
}
/// Device bytes of scratch for T tokens (T = 1..8), and the carving of one allocation (256-byte aligned parts).
template <class G> constexpr size_t expert_scratch_bytes(int T) {
    using D = ExpertDims<G>;
    return detail::up256((size_t) T * D::kActBlocks * 32) + detail::up256((size_t) T * D::kActBlocks * 4) + detail::up256((size_t) G::kTopK * T * G::kFF) +
           detail::up256((size_t) G::kTopK * T * D::kHBlocks * 4);
}
template <class G> inline ExpertScratch expert_scratch_carve(void* device_base, int T) {
    using D = ExpertDims<G>;
    auto* p = static_cast<unsigned char*>(device_base);
    ExpertScratch s;
    s.xq = reinterpret_cast<int8_t*>(p);
    p += detail::up256((size_t) T * D::kActBlocks * 32);
    s.xs = reinterpret_cast<float*>(p);
    p += detail::up256((size_t) T * D::kActBlocks * 4);
    s.hq = reinterpret_cast<int8_t*>(p);
    p += detail::up256((size_t) G::kTopK * T * G::kFF);
    s.hs = reinterpret_cast<float*>(p);
    return s;
}

/// Optional: sets the shared-memory attributes of all expert kernel specialisations (what the first launch does lazily), so that nothing
/// but kernel launches happens inside a CUDA graph capture.  Idempotent.
template <class G> void experts_prepare();

/// Phase 1.  For every group in `groups` (counts->n_groups, read from DEVICE memory): h = silu(min(W1.x,10)) * clamp(W3.x,+-10) * w
/// for every token of the group (NaN propagates through the clamps), quantised into hq/hs at the HIT index.  `cache_base`: slot s is the
/// blob at cache_base + s * kBlobBytes (256-byte aligned; split_hits_misses only emits slots in [0, n_slots)).  `T` only sizes the grid and
/// picks the specialisation (the maximum number of tokens per group).
template <class G>
void experts_gate_up(const uint8_t* cache_base, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int T, const int8_t* xq,
                     const float* xs, int8_t* hq, float* hs, void* stream);

/// Phase 2.  parts[(token*kTopK + k)][kHidden] = W2 . h for every hit; miss entries of `parts` are not touched.
template <class G>
void experts_down(const uint8_t* cache_base, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int T, const int8_t* hq, const float* hs,
                  float* parts, void* stream);

/// quantize x (INTERLEAVED, width kHidden) + experts_gate_up + experts_down.
template <class G>
void experts_hits(const uint8_t* cache_base, const float* x, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int T,
                  const ExpertScratch& scratch, float* parts, void* stream);

// ---- test hooks ---------------------------------------------------------------------------------------------------
/// What the compiler made of one expert kernel specialisation (NT = 1, 2, 4 or 8 tokens per group): registers per thread, static and
/// dynamic shared memory per block and how many blocks the SM can hold.  Zeros in the CPU emulation.
struct ExpertKernelInfo {
    int regs = 0;
    int static_smem = 0;
    int dyn_smem = 0;
    int blocks_per_sm = 0;
};
template <class G> ExpertKernelInfo expert_kernel_info(int NT, bool down);

/// out[e] = the kernels' decode of E8M0 byte e (0..255), as the float 2^(e-128) the way ggml_e8m0_to_fp32_half has it.
void test_e8m0_table(float* out256, void* stream);

}  // namespace strata::ds41::cuda
