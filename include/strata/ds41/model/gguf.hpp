// include/strata/ds41/model/gguf.hpp - DS1-A: a small multi-shard GGUF reader for the deepseek41 loader.
//
// WHY NOT strata/artifact/gguf_reader.hpp.  It was written for the Qwen model and does not fit this one: it does not know ggml
// type 39 (MXFP4), so `tensor_payload_bytes` is 0 for every routed expert and every Engram table, and it keeps only a 64-entry
// sample of every metadata array (token_map has 129,280 entries, the Engram primes / offsets / multipliers 48 / 48 / 8).  Shard
// discovery IS reused (`strata::gguf_split_paths`, gguf_split.hpp); everything else is here.
//
// What it does:
//   * mmaps every shard read-only (the real model is 12 shards, 411 GB: nothing is ever read except through the mapping),
//   * parses the header (metadata + tensor table) with a bounds-checked cursor, keeping every NUMERIC array whole (int64 / double)
//     and the STRING arrays (tokens, merges) as a count plus the file offset of their payload - `read_string_array` re-reads them
//     on demand, so the 129,280 token strings are not held in memory,
//   * checks the split metadata (split.no / split.count / split.tensors.count), the alignment of every tensor offset, that no two
//     tensors overlap and that every tensor lies inside its file (a download that is still running is "truncated", not a crash),
//   * gives the tensor directory as a name -> location table (a TensorDir), which does not need a file at all (tests build the
//     real model's from its saved headers).
//
// Errors are `ModelError` (a std::runtime_error) with a message that names the file / key / tensor and says what was expected.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds41::model {

class ModelError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Several complaints at once (a refused file is easier to fix when everything wrong with it is on the screen).
struct Findings {
    std::vector<std::string> errors, warnings;
    void error(std::string s) { errors.push_back(std::move(s)); }
    void warn(std::string s) { warnings.push_back(std::move(s)); }
    bool ok() const { return errors.empty(); }
    /// "<what>: n problem(s):\n  - ...\n  - ..." (at most `max_shown` lines), thrown as a ModelError when there is any.
    void throw_if_errors(const std::string& what, size_t max_shown = 14) const;
};

// ---- ggml types -------------------------------------------------------------------------------------------------------
/// The ggml type ids of the tensors this model stores, plus every other id as an opaque value (a tensor of another type is
/// carried in the directory so it can be REFUSED by name instead of failing to parse).
enum class GgmlType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, BF16 = 30, MXFP4 = 39 };

/// "F32", "Q8_0", "MXFP4", ... ; "type<id>" for an id this table does not know.
std::string type_name(GgmlType t);
/// Values per block and bytes per block for the types whose size is known (every ggml type of the published GGUFs of this
/// family); false for an unknown id.
bool type_block(GgmlType t, int& block_elems, int& block_bytes);
/// Bytes of a tensor of `ne` (ne[0] fastest) in type `t`; 0 when the type is unknown, ne[0] is not a whole number of blocks, a
/// dimension is 0, or the product overflows.
uint64_t tensor_nbytes(GgmlType t, const uint64_t* ne, int n_dims);

// ---- the tensor directory ---------------------------------------------------------------------------------------------
struct TensorLoc {
    std::string name;
    GgmlType type = GgmlType::F32;
    int n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};     ///< ggml order: ne[0] varies fastest (a row is ne[0] elements)
    int shard = 0;                     ///< index into the model's shards (split order, the metadata shard is 0)
    uint64_t offset = 0;               ///< relative to the shard's data section (as the tensor table stores it)
    uint64_t abs_offset = 0;           ///< from the start of the shard file (data_start + offset); 0 when unknown
    uint64_t nbytes = 0;               ///< 0 when the type is unknown
    uint64_t nrows() const { return ne[1] * ne[2] * ne[3]; }
    uint64_t row_bytes() const { return ne[0] ? nbytes / (ne[1] * ne[2] * ne[3]) : 0; }
    std::string dims_text() const;     ///< "[5120, 2304, 384]"
};

class TensorDir {
public:
    /// Throws a ModelError naming both shards on a duplicate name.
    void add(TensorLoc t);
    const TensorLoc* find(const std::string& name) const;
    /// Throws a ModelError ("tensor `name` is missing") when absent.
    const TensorLoc& at(const std::string& name) const;
    size_t size() const { return by_name_.size(); }
    const std::map<std::string, TensorLoc>& all() const { return by_name_; }

private:
    std::map<std::string, TensorLoc> by_name_;
};

// ---- metadata ---------------------------------------------------------------------------------------------------------
enum class MetaType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, Bool, String, Array, U64, I64, F64 };

struct MetaValue {
    MetaType type = MetaType::U32;     ///< Array for an array
    MetaType elem = MetaType::U32;     ///< element type of an array
    int64_t i = 0;                     ///< integer / bool scalar (U64 above INT64_MAX wraps)
    double f = 0;                      ///< float scalar
    std::string s;                     ///< string scalar
    uint64_t count = 0;                ///< array length
    std::vector<int64_t> ia;           ///< integer / bool array, whole
    std::vector<double> fa;            ///< float array, whole
    uint64_t payload_off = 0;          ///< string array: absolute file offset of its first element (the u64 length)
    bool is_array() const { return type == MetaType::Array; }
    bool is_string() const { return type == MetaType::String; }
    static bool int_type(MetaType t) { return t != MetaType::F32 && t != MetaType::F64 && t != MetaType::String && t != MetaType::Array; }
    static bool float_type(MetaType t) { return t == MetaType::F32 || t == MetaType::F64; }
};

/// Typed access to one shard's key/value pairs; every getter throws a ModelError naming the key (and the file) when the key is
/// missing or of the wrong kind.
class MetaStore {
public:
    MetaStore() = default;
    explicit MetaStore(std::string source) : source_(std::move(source)) {}
    void set(const std::string& key, MetaValue v) { kv_[key] = std::move(v); }
    bool has(const std::string& key) const { return kv_.count(key) != 0; }
    const MetaValue* find(const std::string& key) const;
    int64_t get_int(const std::string& key) const;                  ///< any integer type or bool
    double get_float(const std::string& key) const;                 ///< a float, or an integer (widened)
    bool get_bool(const std::string& key) const;                    ///< bool, or an integer 0 / 1
    std::string get_string(const std::string& key) const;
    std::vector<int64_t> get_ints(const std::string& key) const;    ///< an array of any integer type
    std::vector<double> get_floats(const std::string& key) const;   ///< an array of floats (integers widened)
    uint64_t array_count(const std::string& key) const;             ///< length of any array (strings included)
    const std::map<std::string, MetaValue>& all() const { return kv_; }
    const std::string& source() const { return source_; }

private:
    const MetaValue& need(const std::string& key) const;
    std::map<std::string, MetaValue> kv_;
    std::string source_ = "<metadata>";
};

// ---- the files --------------------------------------------------------------------------------------------------------
struct ShardInfo {
    std::string path;
    uint64_t size = 0;
    uint64_t alignment = 32;
    uint64_t data_start = 0;           ///< absolute offset of the data section
    uint32_t version = 0;
    uint64_t n_tensors = 0;
};

/// One model's shards, opened together.  Immovable (it owns the mappings); keep it in a unique_ptr.
class GgufSet {
public:
    /// Opens every shard of the split model that `any_shard` belongs to (strata::gguf_split_paths: `-0000K-of-0000N.gguf`), or
    /// the single file.  Throws a ModelError for a missing shard, a bad header, inconsistent split metadata or a truncated file.
    static std::unique_ptr<GgufSet> open(const std::string& any_shard);
    explicit GgufSet(const std::vector<std::string>& paths);
    ~GgufSet();
    GgufSet(const GgufSet&) = delete;
    GgufSet& operator=(const GgufSet&) = delete;

    size_t n_shards() const { return shards_.size(); }
    const ShardInfo& shard(size_t i) const { return shards_[i].info; }
    /// The metadata shard (split.no 0): the model's keys.
    const MetaStore& meta() const { return shards_[0].meta; }
    const MetaStore& shard_meta(size_t i) const { return shards_[i].meta; }
    /// Every tensor of every shard, in file order, shard after shard.
    const std::vector<TensorLoc>& tensors() const { return tensors_; }
    TensorDir directory() const;

    /// The tensor's bytes in the mapping (valid for the life of this object; the pages are read from the file on first touch).
    const uint8_t* data(const TensorLoc& t) const { return shards_[(size_t) t.shard].base + t.abs_offset; }
    const uint8_t* shard_base(size_t i) const { return shards_[i].base; }
    /// madvise(MADV_RANDOM) over the tensor: a table read by 136-byte rows at random (the Engram tables, 52 GB each) must not be
    /// read ahead.  Returns false where the platform has no madvise (a no-op there).
    bool advise_random(const TensorLoc& t) const;
    /// Drops the pages of [off, off + len) of the tensor from this process and from the page cache (clean file pages only):
    /// the experts were read once into the arena and must not push the Engram tables out.  A no-op where unsupported.
    void drop_cache(const TensorLoc& t, uint64_t off = 0, uint64_t len = ~0ull) const;
    /// The strings of a string-array key of the metadata shard (tokenizer.ggml.tokens, ...), parsed from the file now.
    std::vector<std::string> read_string_array(const std::string& key) const;

private:
    struct Shard {
        ShardInfo info;
        MetaStore meta;
        const uint8_t* base = nullptr;
        int fd = -1;
        void* map_handle = nullptr;    // Windows
        void* file_handle = nullptr;   // Windows
        Shard() = default;
        Shard(Shard&& o) noexcept { *this = std::move(o); }
        Shard& operator=(Shard&& o) noexcept;
        Shard(const Shard&) = delete;
        Shard& operator=(const Shard&) = delete;
        ~Shard() { release(); }
        void release();                // unmaps and closes (idempotent)
    };
    void open_shard(const std::string& path, Shard& s, int index, Findings& f);
    void validate_split(Findings& f) const;
    void validate_tensor_tables(Findings& f) const;
    std::vector<Shard> shards_;
    std::vector<TensorLoc> tensors_;
};

}  // namespace strata::ds41::model
