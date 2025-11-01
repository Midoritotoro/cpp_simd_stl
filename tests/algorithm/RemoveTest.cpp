#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <simd_stl/algorithm/remove/Remove.h>

struct Custom {
    int x;
    Custom(int v = 0) : x(v) {}
    bool operator==(const Custom& other) const { return x == other.x; }
};

int main() {
    {
        std::vector<int> v = { 1,2,3,2,4 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 2);
        std::vector<int> expected = { 1,3,4 };
        assert(std::equal(v.begin(), it, expected.begin()));
    }

    {
        std::vector<Custom> v = { Custom(1), Custom(2), Custom(3), Custom(2) };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), Custom(2));
        assert((it - v.begin()) == 2);
        assert(v[0].x == 1 && v[1].x == 3);
    }

    {
        std::vector<int> v;
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 42);
        assert(it == v.begin());
    }

    {
        std::vector<int> v = { 1,2,3 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 99);
        assert(it == v.end());
        assert((v == std::vector<int>{1, 2, 3}));
    }

    {
        std::vector<int> v = { 7,7,7 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 7);
        assert(it == v.begin());
    }

    {
        std::vector<int> v = { 1,2,3,2,4 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 2);
        assert(it == v.begin() + 3);
    }

    {
        std::vector<int> v1 = { 1,2,3,2,4 };
        std::vector<int> v2 = v1;

        auto it1 = simd_stl::algorithm::remove(v1.begin(), v1.end(), 2);
        auto it2 = std::remove(v2.begin(), v2.end(), 2);

        assert(std::equal(v1.begin(), it1, v2.begin()));
    }

    {
        constexpr size_t N = 1'000'000;
        std::vector<int> v(N, 1);
        for (size_t i = 0; i < N; i += 10) v[i] = 0;

        std::vector<int> v_std = v;

        auto it1 = simd_stl::algorithm::remove(v.begin(), v.end(), 0);
        auto it2 = std::remove(v_std.begin(), v_std.end(), 0);

        assert(std::equal(v.begin(), it1, v_std.begin()));
    }


 
    {
        std::vector<int> v = { 1,2,3,2,4 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 2);
        v.erase(it, v.end());

        assert((v == std::vector<int>{1, 3, 4}));
        assert(std::none_of(v.begin(), v.end(), [](int x) { return x == 2; }));
    }

    {
        std::vector<int> v = { 7,7,7 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 7);
        v.erase(it, v.end());

        assert(v.empty());
    }

    {
        std::vector<int> v = { 1,2,3 };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 99);
        v.erase(it, v.end());

        assert((v == std::vector<int>{1, 2, 3}));
    }

    {
        std::vector<Custom> v = { Custom(1), Custom(2), Custom(3), Custom(2) };
        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), Custom(2));
        v.erase(it, v.end());

        assert(v.size() == 2);
        assert(v[0].x == 1 && v[1].x == 3);
    }

    {
        std::vector<int> v1 = { 1,2,3,2,4 };
        std::vector<int> v2 = v1;

        auto it1 = simd_stl::algorithm::remove(v1.begin(), v1.end(), 2);
        v1.erase(it1, v1.end());

        auto it2 = std::remove(v2.begin(), v2.end(), 2);
        v2.erase(it2, v2.end());

        assert(v1 == v2);
    }

    {
        constexpr size_t N = 1'000'000;
        std::vector<int> v(N, 1);
        for (size_t i = 0; i < N; i += 10) v[i] = 0;

        auto it = simd_stl::algorithm::remove(v.begin(), v.end(), 0);
        v.erase(it, v.end());

        assert(std::none_of(v.begin(), v.end(), [](int x) { return x == 0; }));
        assert(v.size() == N - N / 10);
    }

    return 0;
}