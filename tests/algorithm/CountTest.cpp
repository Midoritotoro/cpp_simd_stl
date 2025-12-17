#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <random>
#include <simd_stl/algorithm/find/Count.h>

#define Assert(x) assert(x)

template <typename T>
void test_basic() {
    {
        std::vector<T> v;
        Assert(simd_stl::algorithm::count(v.begin(), v.end(), T{}) == std::count(v.begin(), v.end(), T{}));
    }

    {
        std::vector<T> v1 = { T(42) };
        std::vector<T> v2 = { T(13) };
        Assert(simd_stl::algorithm::count(v1.begin(), v1.end(), T(42)) == std::count(v1.begin(), v1.end(), T(42)));
        Assert(simd_stl::algorithm::count(v2.begin(), v2.end(), T(42)) == std::count(v2.begin(), v2.end(), T(42)));
    }

    {
        std::vector<T> v = { T(1), T(2), T(3), T(2), T(4), T(2) };
        for (T val : { T(2), T(4), T(5) }) {
            auto first = simd_stl::algorithm::count(v.begin(), v.end(), val);
            auto second = std::count(v.begin(), v.end(), val);
            Assert(first == second);
        }
    }

    {
        std::array<T, 5> arr = { T(7), T(8), T(7), T(9), T(7) };
        for (T val : { T(7), T(8), T(10) }) {
            Assert(simd_stl::algorithm::count(arr.begin(), arr.end(), val) == std::count(arr.begin(), arr.end(), val));
        }
    }
}

template <typename T>
void test_large(size_t bytes) {
    size_t n = bytes / sizeof(T);
    std::vector<T> v(n);

    std::mt19937 rng(123);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int> dist(0, 100);
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }
    else {
        std::uniform_real_distribution<double> dist(0.0, 10.0);
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }

    for (int i = 0; i < 5; ++i) {
        T val = v[i];
        auto first = simd_stl::algorithm::count(v.begin(), v.end(), val);
        auto second = std::count(v.begin(), v.end(), val);
        Assert(first == second);
    }
}

int main() {
    test_basic<char>();
    test_basic<short>();
    test_basic<int>();
    test_basic<long long>();
    test_basic<float>();
    test_basic<double>();

    test_large<char>(4000);
    test_large<short>(4000);
    test_large<int>(4000);
    test_large<long long>(4000);
    test_large<float>(4000);
    test_large<double>(4000);

    return 0;
}
