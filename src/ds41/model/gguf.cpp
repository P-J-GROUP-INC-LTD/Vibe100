// src/ds41/model/gguf.cpp - DS1-A: the multi-shard GGUF reader (include/strata/ds41/model/gguf.hpp).
#include "strata/ds41/model/gguf.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>

#include "strata/artifact/gguf_split.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::ds41::model {

// ================================================================================================ small helpers
void Findings::throw_if_errors(const std::string& what, size_t max_shown) const {
    if (errors.empty()) return;
    std::string m = what + ": " + std::to_string(errors.size()) + " problem(s):";
    for (size_t i = 0; i < errors.size() && i < max_shown; ++i) m += "\n  - " + errors[i];
    if (errors.size() > max_shown) m += "\n  - ... and " + std::to_string(errors.size() - max_shown) + " more";
    throw ModelError(m);
}

namespace {

struct TypeRow {
    uint32_t id;
    const char* name;
    int elems, bytes;      // 0 / 0 = size not known to this reader
};
// The ggml ids of the published GGUFs of this family (mxxm-t MXFP4, vcruz305 Q2_K..Q8_0).  Sizes are those of gguf_reader.hpp's
// validated table; ids without a size are named only, so that a tensor of another type is refused BY NAME.
constexpr TypeRow kTypes[] = {
    {0, "F32", 1, 4},       {1, "F16", 1, 2},       {2, "Q4_0", 32, 18},    {3, "Q4_1", 32, 20},    {6, "Q5_0", 32, 22},
    {7, "Q5_1", 32, 24},    {8, "Q8_0", 32, 34},    {9, "Q8_1", 32, 36},    {10, "Q2_K", 256, 84},  {11, "Q3_K", 256, 110},
    {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210}, {15, "Q8_K", 0, 0},      {16, "IQ2_XXS", 0, 0},
    {17, "IQ2_XS", 0, 0},   {18, "IQ3_XXS", 0, 0},  {19, "IQ1_S", 0, 0},    {20, "IQ4_NL", 32, 18}, {21, "IQ3_S", 0, 0},
    {22, "IQ2_S", 0, 0},    {23, "IQ4_XS", 0, 0},   {29, "IQ1_M", 0, 0},    {30, "BF16", 1, 2},     {34, "TQ1_0", 0, 0},
    {35, "TQ2_0", 0, 0},    {39, "MXFP4", 32, 17},  {40, "NVFP4", 0, 0},    {42, "Q2_0", 64, 18},
};

const TypeRow* find_type(GgmlType t) {
    for (const TypeRow& r : kTypes)
        if (r.id == (uint32_t) t) return &r;
    return nullptr;
}

uint64_t align_up(uint64_t n, uint64_t a) { return a ? (n + a - 1) / a * a : n; }

/// A bounds-checked cursor over a mapped file: a truncated or corrupt header is a ModelError, not a crash.
class Cursor {
public:
    Cursor(const uint8_t* base, uint64_t size, const std::string& what) : base_(base), size_(size), what_(what) {}
    void need(uint64_t n) const {
        if (n > size_ || pos_ > size_ - n)
            throw ModelError(what_ + ": unexpected end of file in the header (at byte " + std::to_string(pos_) + " of " +
                             std::to_string(size_) + "): truncated or incomplete download?");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = read<uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), (size_t) n);
        pos_ += n;
        return s;
    }
    void skip_str() {
        const uint64_t n = read<uint64_t>();
        need(n);
        pos_ += n;
    }
    uint64_t pos() const { return pos_; }
    void seek(uint64_t p) { pos_ = p; }

private:
    const uint8_t* base_;
    uint64_t size_, pos_ = 0;
    const std::string& what_;
};

constexpr uint64_t kMaxNumericArray = 1ull << 28;   // 268M entries: far above any key of this model (the largest is 129,280)

int64_t read_int(Cursor& c, MetaType t) {
    switch (t) {
    case MetaType::U8: return c.read<uint8_t>();
    case MetaType::I8: return c.read<int8_t>();
    case MetaType::U16: return c.read<uint16_t>();
    case MetaType::I16: return c.read<int16_t>();
    case MetaType::U32: return c.read<uint32_t>();
    case MetaType::I32: return c.read<int32_t>();
    case MetaType::Bool: return c.read<uint8_t>() ? 1 : 0;
    case MetaType::U64: return (int64_t) c.read<uint64_t>();
    case MetaType::I64: return c.read<int64_t>();
    default: throw ModelError("internal: read_int of a non-integer type");
    }
}

double read_float(Cursor& c, MetaType t) { return t == MetaType::F32 ? (double) c.read<float>() : c.read<double>(); }

MetaValue read_value(Cursor& c, MetaType t, const std::string& key, const std::string& what) {
    MetaValue v;
    v.type = t;
    if (t == MetaType::String) {
        v.s = c.str();
    } else if (MetaValue::int_type(t) && t != MetaType::Array) {
        v.i = read_int(c, t);
    } else if (MetaValue::float_type(t)) {
        v.f = read_float(c, t);
    } else if (t == MetaType::Array) {
        v.elem = (MetaType) c.read<uint32_t>();
        v.count = c.read<uint64_t>();
        if ((uint32_t) v.elem > (uint32_t) MetaType::F64 || v.elem == MetaType::Array)
            throw ModelError(what + ": metadata `" + key + "` is an array of an unsupported element type " +
                             std::to_string((uint32_t) v.elem));
        if (v.elem == MetaType::String) {
            v.payload_off = c.pos();
            for (uint64_t i = 0; i < v.count; ++i) c.skip_str();        // counted and skipped: tokens / merges are not held
        } else {
            if (v.count > kMaxNumericArray) throw ModelError(what + ": metadata `" + key + "` has an implausible length " + std::to_string(v.count));
            if (MetaValue::float_type(v.elem)) {
                v.fa.reserve((size_t) v.count);
                for (uint64_t i = 0; i < v.count; ++i) v.fa.push_back(read_float(c, v.elem));
            } else {
                v.ia.reserve((size_t) v.count);
                for (uint64_t i = 0; i < v.count; ++i) v.ia.push_back(read_int(c, v.elem));
            }
        }
    } else {
        throw ModelError(what + ": metadata `" + key + "` has an unknown value type " + std::to_string((uint32_t) t));
    }
    return v;
}

}  // namespace

// ================================================================================================ types
std::string type_name(GgmlType t) {
    if (const TypeRow* r = find_type(t)) return r->name;
    return "type" + std::to_string((uint32_t) t);
}

bool type_block(GgmlType t, int& block_elems, int& block_bytes) {
    const TypeRow* r = find_type(t);
    if (!r || r->elems == 0) return false;
    block_elems = r->elems;
    block_bytes = r->bytes;
    return true;
}

uint64_t tensor_nbytes(GgmlType t, const uint64_t* ne, int n_dims) {
    int be = 0, bb = 0;
    if (n_dims < 1 || n_dims > 4 || !type_block(t, be, bb) || ne[0] % (uint64_t) be) return 0;
    uint64_t rows = 1;
    for (int d = 1; d < n_dims; ++d) {
        if (ne[d] == 0 || rows > std::numeric_limits<uint64_t>::max() / ne[d]) return 0;
        rows *= ne[d];
    }
    if (ne[0] == 0) return 0;
    const uint64_t row = ne[0] / (uint64_t) be * (uint64_t) bb;
    if (rows > std::numeric_limits<uint64_t>::max() / row) return 0;
    return row * rows;
}

std::string TensorLoc::dims_text() const {
    std::string s = "[";
    for (int d = 0; d < n_dims; ++d) s += (d ? ", " : "") + std::to_string(ne[d]);
    return s + "]";
}

// ================================================================================================ directory
void TensorDir::add(TensorLoc t) {
    const auto it = by_name_.find(t.name);
    if (it != by_name_.end())
        throw ModelError("tensor `" + t.name + "` appears twice (shard " + std::to_string(it->second.shard + 1) + " and shard " +
                         std::to_string(t.shard + 1) + "): the GGUF has no way to say which one is meant");
    const std::string name = t.name;
    by_name_.emplace(name, std::move(t));
}

const TensorLoc* TensorDir::find(const std::string& name) const {
    const auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &it->second;
}

const TensorLoc& TensorDir::at(const std::string& name) const {
    if (const TensorLoc* t = find(name)) return *t;
    throw ModelError("tensor `" + name + "` is missing");
}

// ================================================================================================ metadata
const MetaValue* MetaStore::find(const std::string& key) const {
    const auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}

const MetaValue& MetaStore::need(const std::string& key) const {
    const MetaValue* v = find(key);
    if (!v) throw ModelError(source_ + ": required metadata key `" + key + "` is missing");
    return *v;
}

int64_t MetaStore::get_int(const std::string& key) const {
    const MetaValue& v = need(key);
    if (v.is_array() || v.is_string() || MetaValue::float_type(v.type))
        throw ModelError(source_ + ": metadata `" + key + "` must be an integer");
    return v.i;
}

double MetaStore::get_float(const std::string& key) const {
    const MetaValue& v = need(key);
    if (v.is_array() || v.is_string()) throw ModelError(source_ + ": metadata `" + key + "` must be a number");
    return MetaValue::float_type(v.type) ? v.f : (double) v.i;
}

bool MetaStore::get_bool(const std::string& key) const {
    const MetaValue& v = need(key);
    if (v.is_array() || v.is_string() || MetaValue::float_type(v.type)) throw ModelError(source_ + ": metadata `" + key + "` must be a bool");
    return v.i != 0;
}

std::string MetaStore::get_string(const std::string& key) const {
    const MetaValue& v = need(key);
    if (!v.is_string()) throw ModelError(source_ + ": metadata `" + key + "` must be a string");
    return v.s;
}

std::vector<int64_t> MetaStore::get_ints(const std::string& key) const {
    const MetaValue& v = need(key);
    if (!v.is_array() || v.elem == MetaType::String || MetaValue::float_type(v.elem))
        throw ModelError(source_ + ": metadata `" + key + "` must be an array of integers");
    return v.ia;
}

std::vector<double> MetaStore::get_floats(const std::string& key) const {
    const MetaValue& v = need(key);
    if (!v.is_array() || v.elem == MetaType::String) throw ModelError(source_ + ": metadata `" + key + "` must be an array of numbers");
    if (MetaValue::float_type(v.elem)) return v.fa;
    return std::vector<double>(v.ia.begin(), v.ia.end());
}

uint64_t MetaStore::array_count(const std::string& key) const {
    const MetaValue& v = need(key);
    if (!v.is_array()) throw ModelError(source_ + ": metadata `" + key + "` must be an array");
    return v.count;
}

// ================================================================================================ the files
std::unique_ptr<GgufSet> GgufSet::open(const std::string& any_shard) {
    std::vector<std::string> paths;
    try {
        paths = strata::gguf_split_paths(any_shard);
    } catch (const std::exception& e) {
        throw ModelError(e.what());
    }
    return std::make_unique<GgufSet>(paths);
}


GgufSet::Shard& GgufSet::Shard::operator=(Shard&& o) noexcept {
    if (this != &o) {
        release();
        info = std::move(o.info);
        meta = std::move(o.meta);
        base = o.base;
        fd = o.fd;
        map_handle = o.map_handle;
        file_handle = o.file_handle;
        o.base = nullptr;
        o.fd = -1;
        o.map_handle = o.file_handle = nullptr;
    }
    return *this;
}

void GgufSet::Shard::release() {
#ifdef _WIN32
    if (base) UnmapViewOfFile(base);
    if (map_handle) CloseHandle((HANDLE) map_handle);
    if (file_handle) CloseHandle((HANDLE) file_handle);
    map_handle = file_handle = nullptr;
#else
    if (base) munmap((void*) base, (size_t) info.size);
    if (fd >= 0) ::close(fd);
    fd = -1;
#endif
    base = nullptr;
}

GgufSet::~GgufSet() = default;

GgufSet::GgufSet(const std::vector<std::string>& paths) {
    if (paths.empty()) throw ModelError("GGUF: a model needs at least one shard");
    shards_.resize(paths.size());
    Findings f;
    for (size_t i = 0; i < paths.size(); ++i) open_shard(paths[i], shards_[i], (int) i, f);
    validate_split(f);
    f.throw_if_errors("GGUF " + paths[0]);
    validate_tensor_tables(f);
    f.throw_if_errors("GGUF " + paths[0]);
}

void GgufSet::open_shard(const std::string& path, Shard& s, int index, Findings& /*f*/) {
    s.info.path = path;
    s.meta = MetaStore(path);
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw ModelError("cannot open " + path);
    s.file_handle = h;
    LARGE_INTEGER li{};
    GetFileSizeEx(h, &li);
    s.info.size = (uint64_t) li.QuadPart;
    if (s.info.size == 0) throw ModelError(path + " is empty");
    HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) throw ModelError("CreateFileMapping failed for " + path);
    s.map_handle = m;
    s.base = (const uint8_t*) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!s.base) throw ModelError("MapViewOfFile failed for " + path);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw ModelError("cannot open " + path);
    s.fd = fd;
    struct stat st {};
    if (fstat(fd, &st) != 0) throw ModelError("fstat failed for " + path);
    s.info.size = (uint64_t) st.st_size;
    if (s.info.size < 24) throw ModelError(path + " is too small to be a GGUF file (" + std::to_string(s.info.size) + " bytes)");
    void* p = mmap(nullptr, (size_t) s.info.size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) throw ModelError("mmap of " + path + " failed");
    s.base = (const uint8_t*) p;
#endif

    Cursor c(s.base, s.info.size, path);
    if (c.read<uint32_t>() != 0x46554747u) throw ModelError(path + " is not a GGUF file (bad magic)");
    s.info.version = c.read<uint32_t>();
    if (s.info.version != 3) throw ModelError(path + ": GGUF version " + std::to_string(s.info.version) + ", this reader handles version 3");
    s.info.n_tensors = c.read<uint64_t>();
    const uint64_t n_kv = c.read<uint64_t>();
    if (s.info.n_tensors > (1u << 24) || n_kv > (1u << 24))
        throw ModelError(path + ": implausible header (" + std::to_string(s.info.n_tensors) + " tensors, " + std::to_string(n_kv) + " keys)");
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        const MetaType t = (MetaType) c.read<uint32_t>();
        s.meta.set(key, read_value(c, t, key, path));
    }
    s.info.alignment = 32;
    if (const MetaValue* a = s.meta.find("general.alignment"))
        if (!a->is_array() && !a->is_string() && a->i > 0) s.info.alignment = (uint64_t) a->i;

    std::vector<TensorLoc> tl;
    tl.reserve((size_t) s.info.n_tensors);
    for (uint64_t i = 0; i < s.info.n_tensors; ++i) {
        TensorLoc t;
        t.name = c.str();
        const uint32_t nd = c.read<uint32_t>();
        if (nd == 0 || nd > 4) throw ModelError(path + ": tensor `" + t.name + "` has " + std::to_string(nd) + " dimensions (1..4 are valid)");
        t.n_dims = (int) nd;
        for (uint32_t d = 0; d < nd; ++d) t.ne[d] = c.read<uint64_t>();
        t.type = (GgmlType) c.read<uint32_t>();
        t.offset = c.read<uint64_t>();
        t.shard = index;
        t.nbytes = tensor_nbytes(t.type, t.ne, t.n_dims);
        tl.push_back(std::move(t));
    }
    s.info.data_start = align_up(c.pos(), s.info.alignment);
    if (s.info.data_start > s.info.size)
        throw ModelError(path + ": the data section would start at " + std::to_string(s.info.data_start) + ", past the end of the file (" +
                         std::to_string(s.info.size) + " bytes): truncated?");
    for (TensorLoc& t : tl) {
        t.abs_offset = s.info.data_start + t.offset;
        tensors_.push_back(std::move(t));
    }
}

void GgufSet::validate_split(Findings& f) const {
    const size_t n = shards_.size();
    const MetaStore& m0 = shards_[0].meta;
    const MetaValue* count0 = m0.find("split.count");
    const auto as_int = [](const MetaValue* v) -> int64_t { return (v && !v->is_array() && !v->is_string()) ? v->i : -1; };
    if (n == 1) {
        if (count0 && as_int(count0) > 1)
            f.error(shards_[0].info.path + " is shard 1 of " + std::to_string(as_int(count0)) +
                    " (split.count), but it was opened as a whole model: pass the first shard's name so that every shard is found");
        return;
    }
    const int64_t total = as_int(m0.find("split.tensors.count"));
    uint64_t tensors = 0;
    for (size_t i = 0; i < n; ++i) {
        const MetaStore& m = shards_[i].meta;
        const int64_t count = as_int(m.find("split.count")), no = as_int(m.find("split.no")), tc = as_int(m.find("split.tensors.count"));
        if (count != (int64_t) n || no != (int64_t) i || (total >= 0 && tc != total))
            f.error(shards_[i].info.path + " does not declare itself shard " + std::to_string(i + 1) + " of " + std::to_string(n) +
                    " of this model (split.count = " + std::to_string(count) + ", split.no = " + std::to_string(no) +
                    ", split.tensors.count = " + std::to_string(tc) + "): a shard of another model, or a renamed file?");
        tensors += shards_[i].info.n_tensors;
    }
    if (total >= 0 && tensors != (uint64_t) total)
        f.error("the " + std::to_string(n) + " shards hold " + std::to_string(tensors) + " tensors, but split.tensors.count is " + std::to_string(total));
}

void GgufSet::validate_tensor_tables(Findings& f) const {
    // per shard: offsets aligned, tensors inside the file, no two overlapping
    std::vector<std::vector<const TensorLoc*>> by_shard(shards_.size());
    for (const TensorLoc& t : tensors_) by_shard[(size_t) t.shard].push_back(&t);
    for (size_t si = 0; si < shards_.size(); ++si) {
        const ShardInfo& info = shards_[si].info;
        auto& v = by_shard[si];
        std::sort(v.begin(), v.end(), [](const TensorLoc* a, const TensorLoc* b) { return a->offset < b->offset; });
        uint64_t prev_end = 0;
        const TensorLoc* prev = nullptr;
        for (const TensorLoc* t : v) {
            if (t->nbytes == 0) continue;     // an unknown type: refused by name later, only if it is one the model uses
            if (t->offset % info.alignment)
                f.error(info.path + ": tensor `" + t->name + "` has offset " + std::to_string(t->offset) + ", not a multiple of the alignment " +
                        std::to_string(info.alignment));
            if (prev && t->offset < prev_end)
                f.error(info.path + ": tensor `" + t->name + "` starts at " + std::to_string(t->offset) + ", inside `" + prev->name + "` (which ends at " +
                        std::to_string(prev_end) + ")");
            if (t->abs_offset > info.size || t->nbytes > info.size - t->abs_offset)
                f.error(info.path + " is " + std::to_string(info.size) + " bytes but tensor `" + t->name + "` ends at " +
                        std::to_string(t->abs_offset + t->nbytes) + ": truncated or still downloading?");
            prev_end = std::max(prev_end, t->offset + t->nbytes);
            prev = t;
        }
    }
}

TensorDir GgufSet::directory() const {
    TensorDir d;
    for (const TensorLoc& t : tensors_) d.add(t);
    return d;
}

bool GgufSet::advise_random(const TensorLoc& t) const {
#if defined(__linux__)
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t a = align_up(t.abs_offset, page), b = (t.abs_offset + t.nbytes) / page * page;
    if (b <= a) return false;
    return madvise((void*) (shards_[(size_t) t.shard].base + a), (size_t) (b - a), MADV_RANDOM) == 0;
#else
    (void) t;
    return false;
#endif
}

bool GgufSet::prefetch(const TensorLoc& t) const {
#if defined(__linux__)
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t a = t.abs_offset / page * page, b = align_up(t.abs_offset + t.nbytes, page);
    const Shard& s = shards_[(size_t) t.shard];
    return madvise((void*) (s.base + a), (size_t) std::min<uint64_t>(b, s.info.size) - (size_t) a, MADV_WILLNEED) == 0;
#else
    (void) t;
    return false;
#endif
}

void GgufSet::drop_cache(const TensorLoc& t, uint64_t off, uint64_t len) const {
#if defined(__linux__)
    if (off >= t.nbytes) return;
    len = std::min(len, t.nbytes - off);
    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t start = t.abs_offset + off, end = start + len;
    const uint64_t a = align_up(start, page), b = end / page * page;
    if (b <= a) return;
    const Shard& s = shards_[(size_t) t.shard];
    madvise((void*) (s.base + a), (size_t) (b - a), MADV_DONTNEED);          // this process's PTEs of a clean private file mapping
    if (s.fd >= 0) posix_fadvise(s.fd, (off_t) a, (off_t) (b - a), POSIX_FADV_DONTNEED);   // then the page cache (pages nobody maps any more)
#else
    (void) t; (void) off; (void) len;
#endif
}

std::vector<std::string> GgufSet::read_string_array(const std::string& key) const {
    const MetaValue* v = shards_[0].meta.find(key);
    if (!v || !v->is_array() || v->elem != MetaType::String)
        throw ModelError(shards_[0].info.path + ": metadata `" + key + "` is not a string array");
    const std::string what = shards_[0].info.path;
    Cursor c(shards_[0].base, shards_[0].info.size, what);
    c.seek(v->payload_off);
    std::vector<std::string> out;
    out.reserve((size_t) v->count);
    for (uint64_t i = 0; i < v->count; ++i) out.push_back(c.str());
    return out;
}

}  // namespace strata::ds41::model
