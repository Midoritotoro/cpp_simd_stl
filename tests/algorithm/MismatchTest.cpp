#include <cAssert>
#include <cctype>
#include <iostream>

#include <vector>
#include <string>

#include <simd_stl/algorithm/find/Mismatch.h>

template <typename It1, typename It2>
void check_equal(It1 simd_first, It1 std_first, It2 simd_second, It2 std_second) {
    Assert(simd_first == std_first);
    Assert(simd_second == std_second);
}

int main() {
    {
        std::vector<int> a(63, 1);
        std::vector<int> b(63, 1);
        auto simd = simd_stl::algorithm::mismatch(a.begin(), a.end(), b.begin());
        auto stdm = std::mismatch(a.begin(), a.end(), b.begin());
        check_equal(simd.first, stdm.first, simd.second, stdm.second);
    }

    {
        std::vector<int> a(152, 1);
        std::vector<int> b(152, 1);
        b[32] = 9;
        auto simd = simd_stl::algorithm::mismatch(a.begin(), a.end(), b.begin());
        auto stdm = std::mismatch(a.begin(), a.end(), b.begin());
        check_equal(simd.first, stdm.first, simd.second, stdm.second);
        Assert(*simd.first == 1);
        Assert(*simd.second == 9);
    }

    {
        std::vector<int> a(115, 1);
        std::vector<int> b(115, 1);
        b[63] = 7;
        auto simd = simd_stl::algorithm::mismatch(a.begin(), a.end(), b.begin());
        auto stdm = std::mismatch(a.begin(), a.end(), b.begin());
        check_equal(simd.first, stdm.first, simd.second, stdm.second);
        Assert(*simd.first == 1);
        Assert(*simd.second == 7);
    }

    {
        std::vector<int> a(24, 1);
        std::vector<int> b(24, 1);
        b[0] = 0;
        auto simd = simd_stl::algorithm::mismatch(a.begin(), a.end(), b.begin());
        auto stdm = std::mismatch(a.begin(), a.end(), b.begin());
        check_equal(simd.first, stdm.first, simd.second, stdm.second);
        Assert(*simd.first == 1);
        Assert(*simd.second == 0);
    }

    {
        std::vector<int> a(55, 1);
        std::vector<int> b(32, 1);
        auto simd = simd_stl::algorithm::mismatch(a.begin(), a.end(), b.begin(), b.end());
        auto stdm = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
        check_equal(simd.first, stdm.first, simd.second, stdm.second);
        Assert(simd.first == a.begin() + 32);
        Assert(simd.second == b.end());
    }

    {
        std::vector<std::string> a(32, "Hello");
        std::vector<std::string> b(32, "HELLO");
        auto pred = [](const std::string& x, const std::string& y) {
            return std::equal(x.begin(), x.end(), y.begin(), [](char a, char b) {
                return std::tolower(a) == std::tolower(b);
                });
            };
        auto simd = simd_stl::algorithm::mismatch(a.begin(), a.end(), b.begin(), pred);
        auto stdm = std::mismatch(a.begin(), a.end(), b.begin(), pred);
        check_equal(simd.first, stdm.first, simd.second, stdm.second);
    }
}
