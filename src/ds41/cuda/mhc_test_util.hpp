// src/ds41/cuda/mhc_test_util.hpp - DS1-D: what the mHC and Engram emulator tests share: a reader for the golden .npy files that
// src/ds41/engram/golden/gen_golden.py writes (the NumPy oracle's inputs and outputs), PASS / FAIL reporting in the format ctest greps for,
// error statistics, a small deterministic random generator.  Header only; plain C++17, no CUDA.
#pragma once

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace ds1d {

// ---------------------------------------------------------------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------------------------------------------------------------
struct Npy {
    std::string descr;               // "<f4", "<f8", "<i8", "<i4", "|u1", "<u2", ...
    std::vector<size_t> shape;
    std::vector<unsigned char> raw;
    size_t count() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }
    size_t dim(size_t i) const { return i < shape.size() ? shape[i] : 1; }
};

inline Npy read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open golden file " + path + " (run src/ds41/engram/golden/gen_golden.py first)");
    char magic[6];
    f.read(magic, 6);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not an .npy file");
    unsigned char ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        hlen = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        hlen = b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t) b[3] << 24);
    }
    std::string h(hlen, ' ');
    f.read(&h[0], hlen);
    Npy a;
    auto field = [&](const std::string& key) {
        const size_t k = h.find("'" + key + "'");
        if (k == std::string::npos) throw std::runtime_error(path + ": header has no " + key);
        return k + key.size() + 2;
    };
    {
        size_t k = h.find('\'', field("descr"));
        const size_t e = h.find('\'', k + 1);
        a.descr = h.substr(k + 1, e - k - 1);
    }
    {
        const size_t fo = field("fortran_order");
        if (h.substr(fo, h.find(',', fo) - fo).find("True") != std::string::npos) throw std::runtime_error(path + ": fortran order not supported");
    }
    {
        size_t k = h.find('(', field("shape"));
        const size_t e = h.find(')', k);
        size_t i = k + 1;
        while (i < e) {
            while (i < e && (h[i] == ' ' || h[i] == ',')) ++i;
            if (i >= e) break;
            size_t j = i;
            while (j < e && h[j] >= '0' && h[j] <= '9') ++j;
            if (j == i) break;
            a.shape.push_back((size_t) std::strtoull(h.substr(i, j - i).c_str(), nullptr, 10));
            i = j;
        }
    }
    a.raw.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return a;
}

inline size_t descr_size(const std::string& d) { return (size_t) (d.back() - '0'); }

/// Typed access with a dtype check (the golden writer's dtypes are part of the contract).
template <class T> struct NpyType;
template <> struct NpyType<float> { static constexpr const char* d = "<f4"; };
template <> struct NpyType<double> { static constexpr const char* d = "<f8"; };
template <> struct NpyType<int64_t> { static constexpr const char* d = "<i8"; };
template <> struct NpyType<int32_t> { static constexpr const char* d = "<i4"; };
template <> struct NpyType<uint16_t> { static constexpr const char* d = "<u2"; };
template <> struct NpyType<uint8_t> { static constexpr const char* d = "|u1"; };

template <class T>
struct Arr {
    std::vector<T> v;
    std::vector<size_t> shape;
    const T* data() const { return v.data(); }
    size_t size() const { return v.size(); }
    size_t dim(size_t i) const { return i < shape.size() ? shape[i] : 1; }
    const T& operator[](size_t i) const { return v[i]; }
    T& operator[](size_t i) { return v[i]; }
};

class Golden {
   public:
    explicit Golden(std::string dir) : dir_(std::move(dir)) {
        std::ifstream done(dir_ + "/DONE");
        if (!done) throw std::runtime_error("golden directory " + dir_ + " is incomplete (no DONE file): run gen_golden.py");
    }
    template <class T>
    Arr<T> get(const std::string& rel) const {
        Npy n = read_npy(dir_ + "/" + rel + ".npy");
        if (n.descr != NpyType<T>::d) throw std::runtime_error(rel + ": dtype " + n.descr + ", expected " + NpyType<T>::d);
        Arr<T> a;
        a.shape = n.shape;
        a.v.resize(n.count());
        if (n.raw.size() < a.v.size() * sizeof(T)) throw std::runtime_error(rel + ": truncated");
        std::memcpy(a.v.data(), n.raw.data(), a.v.size() * sizeof(T));
        return a;
    }
    const std::string& dir() const { return dir_; }

   private:
    std::string dir_;
};

// ---------------------------------------------------------------------------------------------------------------------------------
// reporting (the ctest properties of cmake/ds41_mhc.cmake: PASS_REGULAR_EXPRESSION "ALL PASS", FAIL_REGULAR_EXPRESSION "FAIL ")
// ---------------------------------------------------------------------------------------------------------------------------------
struct Report {
    int fails = 0, checks = 0;
    void check(bool ok, const std::string& what, const std::string& detail = "") {
        ++checks;
        if (!ok) ++fails;
        std::printf("%s  %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
        std::fflush(stdout);
    }
    void info(const std::string& what) { std::printf("INFO  %s\n", what.c_str()); std::fflush(stdout); }     // not a check: measurements, skipped suites
    int finish(const char* suite) const {
        if (fails == 0) std::printf("%s: ALL PASS (%d checks)\n", suite, checks);
        else std::printf("%s: %d of %d checks FAILED\n", suite, fails, checks);
        return fails == 0 ? 0 : 1;
    }
};

inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// numbers
// ---------------------------------------------------------------------------------------------------------------------------------
/// The tolerance floors of the checks that go through expf (sigmoid, softmax: the coefficients, the Engram gate) are for the host's libm (< 1 ulp, the emulation); CUDA's expf is
/// documented at 2 ulp (and its division / sqrt are IEEE), so the GPU programs (-DDS1D_ON_GPU) widen exactly those floors by this factor.  Everything else is the same.
#ifdef DS1D_ON_GPU
inline constexpr double kLibmFactor = 4.0;
inline constexpr bool kOnGpu = true;
#else
inline constexpr double kLibmFactor = 1.0;
inline constexpr bool kOnGpu = false;
#endif

inline uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float f_of_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// Error of `got` against `ref` (any pair of float / double arrays): max |got - ref|, max |ref|, the index of the worst element, NaN mismatches (a NaN
/// where the reference has none, or none where it has one) and non-finite counts.  A NaN on both sides is a match; so is an equal infinity.
struct Err {
    double max_abs = 0, max_ref = 0;
    size_t worst = 0, nan_mismatch = 0, n = 0;
    double ref_at_worst = 0, got_at_worst = 0;
};
template <class A, class B>
Err compare(const A* got, const B* ref, size_t n) {
    Err e;
    e.n = n;
    for (size_t i = 0; i < n; ++i) {
        const double g = (double) got[i], r = (double) ref[i];
        if (std::isnan(g) || std::isnan(r)) {
            if (std::isnan(g) != std::isnan(r)) ++e.nan_mismatch;
            continue;
        }
        if (std::isinf(g) || std::isinf(r)) {
            if (g != r) ++e.nan_mismatch;
            continue;
        }
        const double d = std::fabs(g - r);
        if (d > e.max_abs) {
            e.max_abs = d;
            e.worst = i;
            e.ref_at_worst = r;
            e.got_at_worst = g;
        }
        e.max_ref = std::fmax(e.max_ref, std::fabs(r));
    }
    return e;
}
inline std::string describe(const Err& e) {
    return fmt("max|err| %.3e (|ref| max %.3e) at [%zu]: got %.9g ref %.9g%s", e.max_abs, e.max_ref, e.worst, e.got_at_worst, e.ref_at_worst,
               e.nan_mismatch ? fmt(", %zu NaN/Inf mismatches", e.nan_mismatch).c_str() : "");
}

/// Bitwise equality of two float arrays (NaNs compare by payload too).  Returns the index of the first difference, or n.
inline size_t first_bit_diff(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (bits_of(a[i]) != bits_of(b[i])) return i;
    return n;
}

/// xorshift64*: deterministic across platforms.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) { next(); next(); }
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }       // [0, 1)
    double normal() {
        const double u1 = 1.0 - uni(), u2 = uni();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
    int range(int lo, int hi) { return lo + (int) (next() % (uint64_t) (hi - lo + 1)); }     // inclusive
};

}  // namespace ds1d
