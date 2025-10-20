#include <cassert>
#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <iostream>
#include <cctype>
#include <random>
#include <simd_stl/algorithm/find/FindLast.h>



int main() {
    {
        std::vector<int> v;
        Assert(simd_stl::algorithm::find_last(v.begin(), v.end(), 42) == v.end());
        Assert(simd_stl::algorithm::find_last_if(v.begin(), v.end(), [](int x) { return x == 42; }) == v.end());
        Assert(simd_stl::algorithm::find_last_if_not(v.begin(), v.end(), [](int x) { return x == 42; }) == v.end());
    }

    {
        std::vector<int> v(64, 1);
        Assert(simd_stl::algorithm::find_last(v.begin(), v.end(), 42) == v.end());
        Assert(simd_stl::algorithm::find_last_if(v.begin(), v.end(), [](int x) { return x == 42; }) == v.end());
        Assert(simd_stl::algorithm::find_last_if_not(v.begin(), v.end(), [](int x) { return x == 42; }) == v.end() - 1);
    }

    {
        std::vector<int> v(64, 0);
        v[10] = 42;
        Assert(simd_stl::algorithm::find_last(v.begin(), v.end(), 42) == (v.begin() + 10));
        Assert(simd_stl::algorithm::find_last_if(v.begin(), v.end(), [](int x) { return x == 42; }) == (v.begin() + 10));
        Assert(simd_stl::algorithm::find_last_if_not(v.begin(), v.end(), [](int x) { return x == 42; }) == v.end() - 1);
    }

    {
        std::vector<int> v(128, 0);
        v[5] = v[50] = v[100] = 42;
        Assert(simd_stl::algorithm::find_last(v.begin(), v.end(), 42) == (v.begin() + 100));
        Assert(simd_stl::algorithm::find_last_if(v.begin(), v.end(), [](int x) { return x == 42; }) == (v.begin() + 100));
        Assert(simd_stl::algorithm::find_last_if_not(v.begin(), v.end(), [](int x) { return x == 42; }) == v.end() - 1);
    }

    {
        std::string s = "HelloHELLO";
        Assert(simd_stl::algorithm::find_last(s.begin(), s.end(), 'L') == (s.begin() + 8));
        Assert(simd_stl::algorithm::find_last_if(s.begin(), s.end(), [](char c) { return std::isupper(c); }) == (s.end() + - 1));
        Assert(simd_stl::algorithm::find_last_if_not(s.begin(), s.end(), [](char c) { return std::isupper(c); }) == s.end() - 6);

    }

    {
        std::vector<uint64_t> v(16, 0xDEADBEEF);
        v[15] = 0xCAFEBABE;
        Assert(simd_stl::algorithm::find_last(v.begin(), v.end(), 0xCAFEBABEull) == v.end() - 1);
    }
    return 0;
}
