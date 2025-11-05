#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <simd_stl/algorithm/remove/RemoveCopy.h>

template <typename T>
std::vector<T> make_aligned_buffer(size_t count, size_t align) {
    size_t bytes = count * sizeof(T) + align;
    auto raw = new uint8_t[bytes];
    uint8_t* base = raw;
    uint8_t* aligned = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(base) + (align - 1)) & ~(align - 1)
        );
    T* ptr = reinterpret_cast<T*>(aligned);
    return std::vector<T>(ptr, ptr + count);
}

template <typename T>
void fill_sequential(std::vector<T>& v, T start = T(1)) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<T>(start + T(i));
}

template <typename T>
void fill_random(std::vector<T>& v, uint64_t seed = 1234567) {
    std::mt19937_64 rng(seed);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }
    else {
        std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }
}

template <typename T>
bool equal_range(const std::vector<T>& v, size_t n, const std::vector<T>& expected) {
    return n == expected.size() && std::equal(v.begin(), v.begin() + n, expected.begin());
}

template <typename T>
bool equal_full(const std::vector<T>& v, const std::vector<T>& expected) {
    return v.size() == expected.size() && std::equal(v.begin(), v.end(), expected.begin());
}

template <typename T, typename Pred>
size_t ref_remove_if(std::vector<T>& v, Pred pred) {
    size_t write = 0;
    for (size_t read = 0; read < v.size(); ++read) {
        if (!pred(v[read])) v[write++] = v[read];
    }
    return write;
}

template <typename T>
size_t ref_remove_copy(const std::vector<T>& in, std::vector<T>& out, const T& value) {
    size_t write = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        if (!(in[i] == value)) {
            out[write++] = in[i];
        }
    }
    return write;
}

template <typename T, typename Pred>
size_t ref_remove_copy_if(const std::vector<T>& in, std::vector<T>& out, Pred pred) {
    size_t write = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        if (!pred(in[i])) {
            out[write++] = in[i];
        }
    }
    return write;
}


template <typename T>
void test_remove_if_large() {
    const size_t total_bytes = 4000;
    const size_t N = std::max<size_t>(1, total_bytes / sizeof(T));

    std::vector<T> v(N);
    <T>fill_random(v);

    auto pred_even_index = [&](const T& x) {
        size_t idx = &x - v.data();
        return (idx % 2) == 0;
        };
    auto pred_threshold = [&](const T& x) {
        if constexpr (std::is_integral_v<T>)
            return (x & T(1)) == T(1);
        else
            return x > T(0);
        };
    auto pred_mask = [&](const T& x) {
        if constexpr (std::is_integral_v<T>)
            return (x & T(0xA5)) == T(0xA5);
        else
            return std::fpclassify(x) == FP_NAN; 
        };

    {
        auto v_copy = v;
        auto new_end = simd_stl::algorithm::remove_if(v_copy.begin(), v_copy.end(), pred_even_index);
        size_t simd_kept = std::distance(v_copy.begin(), new_end);

        auto v_ref = v;
        size_t kept_ref = ref_remove_if<T>(v_ref, pred_even_index);
        assert(simd_kept == kept_ref);
        assert(std::equal(v_copy.begin(), new_end, v_ref.begin()));
    }

    {
        auto v_copy = v;
        auto new_end = simd_stl::algorithm::remove_if(v_copy.begin(), v_copy.end(), pred_threshold);
        size_t simd_kept = std::distance(v_copy.begin(), new_end);

        auto v_ref = v;
        size_t kept_ref = ref_remove_if<T>(v_ref, pred_threshold);
        assert(simd_kept == kept_ref);
        assert(std::equal(v_copy.begin(), new_end, v_ref.begin()));
    }

    {
        auto v_copy = v;
        auto new_end = simd_stl::algorithm::remove_if(v_copy.begin(), v_copy.end(), pred_mask);
        size_t simd_kept = std::distance(v_copy.begin(), new_end);

        auto v_ref = v;
        size_t kept_ref = ref_remove_if<T>(v_ref, pred_mask);
        assert(simd_kept == kept_ref);
        assert(std::equal(v_copy.begin(), new_end, v_ref.begin()));
    }

    {
        auto v_copy = v;
        auto new_end = simd_stl::algorithm::remove_if(v_copy.begin(), v_copy.end(), [](const T&) { return false; });
        assert(new_end == v_copy.end());
        assert(std::equal(v_copy.begin(), v_copy.end(), v.begin()));
    }

    {
        auto v_copy = v;
        auto new_end = simd_stl::algorithm::remove_if(v_copy.begin(), v_copy.end(), [](const T&) { return true; });
        assert(new_end == v_copy.begin());
    }
}

template <typename T>
void test_remove_copy_large() {
    const size_t total_bytes = 4000;
    const size_t N = std::max<size_t>(1, total_bytes / sizeof(T));

    std::vector<T> in(N);
    fill_random<T>(in);

    T value_to_remove = in[N / 3];

    std::vector<T> out_simd(N, T{});
    std::vector<T> out_ref(N, T{});

    auto out_it = simd_stl::algorithm::remove_copy(in.begin(), in.end(), out_simd.begin(), value_to_remove);
    size_t simd_written = std::distance(out_simd.begin(), out_it);

    size_t ref_written = ref_remove_copy<T>(in, out_ref, value_to_remove);

    assert(simd_written == ref_written);
    assert(std::equal(out_simd.begin(), out_simd.begin() + simd_written, out_ref.begin()));
}

template <typename T>
void test_remove_copy_if_large() {
    const size_t total_bytes = 4000;
    const size_t N = std::max<size_t>(1, total_bytes / sizeof(T));

    std::vector<T> in(N);
    fill_random<T>(in);

    auto pred = [&](const T& x) {
        size_t idx = &x - in.data();
        if constexpr (std::is_integral_v<T>)
            return ((idx & 3) == 0) || ((x & T(7)) == T(3));
        else
            return (idx % 5 == 0) || (x < T(-100.0));
        };

    std::vector<T> out_simd(N, T{});
    std::vector<T> out_ref(N, T{});

    auto out_it = simd_stl::algorithm::remove_copy_if(in.begin(), in.end(), out_simd.begin(), pred);
    size_t simd_written = std::distance(out_simd.begin(), out_it);

    size_t ref_written = ref_remove_copy_if<T>(in, out_ref, pred);

    assert(simd_written == ref_written);
    assert(std::equal(out_simd.begin(), out_simd.begin() + simd_written, out_ref.begin()));
}

template <typename T>
void test_alignment_variants_remove_copy() {
    const size_t total_bytes = 4000;
    const size_t N = std::max<size_t>(1, total_bytes / sizeof(T));

    std::vector<T> in(N);
    fill_random<T>(in);

    std::vector<T> out_aligned(N, T{});
    std::vector<T> out_unaligned(N + 8, T{});

    T value_to_remove = in[N / 2];

    auto itA = simd_stl::algorithm::remove_copy(in.begin(), in.end(), out_aligned.begin(), value_to_remove);
    size_t wA = std::distance(out_aligned.begin(), itA);

    T* out_unaligned_ptr = out_unaligned.data() + 1;
    auto itU = simd_stl::algorithm::remove_copy(in.begin(), in.end(), out_unaligned_ptr, value_to_remove);
    size_t wU = static_cast<size_t>(itU - out_unaligned_ptr);

    std::vector<T> ref(N, T{});
    size_t wR = ref_remove_copy<T>(in, ref, value_to_remove);

    assert(wA == wR);
    assert(wU == wR);
    assert(std::equal(out_aligned.begin(), out_aligned.begin() + wA, ref.begin()));
    assert(std::equal(out_unaligned_ptr, out_unaligned_ptr + wU, ref.begin()));
}

template <typename T>
void test_alignment_variants_remove_copy_if() {
    const size_t total_bytes = 4000;
    const size_t N = std::max<size_t>(1, total_bytes / sizeof(T));

    std::vector<T> in(N);
    fill_random<T>(in);

    auto pred = [&](const T& x) {
        size_t idx = &x - in.data();
        if constexpr (std::is_integral_v<T>)
            return ((idx & 1) == 1) && ((x & T(0xFF)) > T(127));
        else
            return (idx % 3 == 2) || (x * x > T(10000));
        };

    std::vector<T> out_aligned(N, T{});
    std::vector<T> out_unaligned(N + 8, T{});

    auto itA = simd_stl::algorithm::remove_copy_if(in.begin(), in.end(), out_aligned.begin(), pred);
    size_t wA = std::distance(out_aligned.begin(), itA);

    T* out_unaligned_ptr = out_unaligned.data() + 1;
    auto itU = simd_stl::algorithm::remove_copy_if(in.begin(), in.end(), out_unaligned_ptr, pred);
    size_t wU = static_cast<size_t>(itU - out_unaligned_ptr);

    std::vector<T> ref(N, T{});
    size_t wR = ref_remove_copy_if<T>(in, ref, pred);

    assert(wA == wR);
    assert(wU == wR);
    assert(std::equal(out_aligned.begin(), out_aligned.begin() + wA, ref.begin()));
    assert(std::equal(out_unaligned_ptr, out_unaligned_ptr + wU, ref.begin()));
}

int main() {
    test_remove_if_large<uint8_t>();
    test_remove_if_large<int8_t>();
    test_remove_if_large<uint16_t>();
    test_remove_if_large<int16_t>();
    test_remove_if_large<uint32_t>();
    test_remove_if_large<int32_t>();
    test_remove_if_large<uint64_t>();
    test_remove_if_large<int64_t>();
    test_remove_if_large<float>();
    test_remove_if_large<double>();

    test_remove_copy_large<uint8_t>();
    test_remove_copy_large<int32_t>();
    test_remove_copy_large<double>();

    test_remove_copy_if_large<uint16_t>();
    test_remove_copy_if_large<int64_t>();
    test_remove_copy_if_large<float>();

    test_alignment_variants_remove_copy<int32_t>();
    test_alignment_variants_remove_copy_if<double>();

    return 0;
}
