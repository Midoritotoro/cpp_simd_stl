#include <vector>
#include <array>

#include <simd_stl/algorithm/find/Count.h>

int main() {
    {
        std::vector<int> v;
        Assert(simd_stl::algorithm::count(v.begin(), v.end(), 42) == std::count(v.begin(), v.end(), 42));
    }

    {
        std::vector<int> v1 = { 42 };
        std::vector<int> v2 = { 13 };
        Assert(simd_stl::algorithm::count(v1.begin(), v1.end(), 42) == std::count(v1.begin(), v1.end(), 42));
        Assert(simd_stl::algorithm::count(v2.begin(), v2.end(), 42) == std::count(v2.begin(), v2.end(), 42));
    }

    {
        std::vector<int> v = { 1, 2, 3, 2, 4, 2 };
        for (int val : {2, 4, 5}) {
            auto first = simd_stl::algorithm::count(v.begin(), v.end(), val);
            auto second = std::count(v.begin(), v.end(), val);
            Assert(first == second);
        }
    }

    {
        int arr[] = { 7, 8, 7, 9, 7 };
        for (int val : {7, 8, 10}) {
            Assert(simd_stl::algorithm::count(std::begin(arr), std::end(arr), val) == std::count(std::begin(arr), std::end(arr), val));
        }
    }

    {
        const std::array<char, 5> arr = { 'a', 'b', 'a', 'c', 'a' };
        for (char ch : {'a', 'z'}) {
            Assert(simd_stl::algorithm::count(arr.begin(), arr.end(), ch) == std::count(arr.begin(), arr.end(), ch));
        }
    }

    {
        std::array<double, 4> arr = { 1.0, 2.5, 2.5, 3.14 };
        for (double val : {2.5, 1.0, 0.0}) {
            Assert(simd_stl::algorithm::count(arr.begin(), arr.end(), val) == std::count(arr.begin(), arr.end(), val));
        }
    }

    {
       /* const char* arr[] = { "a", "b", "a", "c" };
        for (const char* val : { arr[0], arr[1], "x" }) {
            Assert(simd_stl::algorithm::count(std::begin(arr), std::end(arr), val) == std::count(std::begin(arr), std::end(arr), val));
        }*/
    }

    {
       /* const void* arr[] = { nullptr, (void*)0x1, nullptr };
        Assert(simd_stl::algorithm::count(std::begin(arr), std::end(arr), nullptr) == std::count(std::begin(arr), std::end(arr), nullptr));*/
    }

    return 0;
}