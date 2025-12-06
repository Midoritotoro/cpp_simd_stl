#define NOMINMAX

#include <simd_stl/algorithm/copy/CopyN.h>
#include <random> 
#include <deque>
#include <chrono>

template <typename It1, typename It2>
void Assert_equal(It1 first1, It1 last1, It2 first2) {
    auto n = std::distance(first1, last1);
    for (decltype(n) i = 0; i < n; ++i) {
        Assert(*(first1 + i) == *(first2 + i));
    }
}

template <typename T>
void fill_random(std::vector<T>& v, uint64_t seed = 12345) {
    std::mt19937_64 rng(seed);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<long long> dist(
            std::numeric_limits<long long>::min(),
            std::numeric_limits<long long>::max()
        );
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }
    else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<double> dist(-1e6, 1e6);
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }
    else {
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& x : v) x = static_cast<T>(dist(rng));
    }
}

template <>
void fill_random<char>(std::vector<char>& v, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> dist(0, 127);
    for (auto& x : v) x = static_cast<char>(dist(rng));
}

int main() {
    {
        std::vector<int> src;
        std::vector<int> dst(0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.begin());
    }

    {
        std::vector<int> src = { 42 };
        std::vector<int> dst(1, 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        Assert_equal(src.begin(), src.end(), dst.begin());
    }
    
    {
        std::vector<int> src = { 1,2,3,4,5,6,7,8 };
        std::vector<int> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        Assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        std::string src = "simd copy test!";
        std::string dst(src.size(), '\0');
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        Assert(src == dst);
    }

    {
        std::array<int, 16> srcarr{};
        for (int i = 0; i < 16; ++i) srcarr[i] = i * i;
        int dstarr[16] = {};
        auto out = simd_stl::algorithm::copy(srcarr.data(), srcarr.data() + srcarr.size(), dstarr);
        Assert(out == dstarr + 16);
        for (int i = 0; i < 16; ++i) Assert(srcarr[i] == dstarr[i]);
    }

    {
        std::list<int> src = { 10,20,30,40,50 };
        std::list<int> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());

        auto itsrc = src.begin();
        auto itdst = dst.begin();
        for (; itsrc != src.end(); ++itsrc, ++itdst) Assert(*itsrc == *itdst);
    }

   
   {
        std::deque<int> src = { 3,1,4,1,5,9,2,6,5 };
        std::deque<int> dst(src.size());
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        for (size_t i = 0; i < src.size(); ++i) Assert(src[i] == dst[i]);
    }

    {
        std::vector<char> src(1024);
        fill_random(src, 1);
        std::vector<char> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        Assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        std::vector<int> src(4096);
        fill_random(src, 2);
        std::vector<int> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        Assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        std::vector<uint64_t> src(8192);
        fill_random(src, 3);
        std::vector<uint64_t> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        Assert(out == dst.end());
        Assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        const size_t n = 10'000'000;
        std::vector<int> src(n);
        fill_random(src, 4);
        std::vector<int> dst(n, 0);

        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());

        Assert(out == dst.end());
  
        for (size_t i = 0; i < n; i += n / 10) Assert(src[i] == dst[i]);
        Assert(src.front() == dst.front());
        Assert(src.back() == dst.back());
    }


    {
        std::vector<int> src(1024);
        for (int i = 0; i < 1024; ++i) src[i] = i;
        std::vector<int> dst(1024, -1);

        auto out = std::copy(src.begin() + 100, src.begin() + 900, dst.begin() + 50);
        Assert(out == dst.begin() + 50 + (900 - 100));
        for (int i = 0; i < 50; ++i) Assert(dst[i] == -1);
        for (int i = 0; i < 800; ++i) Assert(dst[50 + i] == src[100 + i]);
        for (int i = 850; i < 1024; ++i) Assert(dst[i] == -1);
    }

    {
        std::vector<int> src(100);
        for (int i = 0; i < 100; ++i) src[i] = i * 3;
        std::vector<int> dst;
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), std::back_inserter(dst));
        Assert(dst.size() == src.size());
        for (size_t i = 0; i < src.size(); ++i) Assert(dst[i] == src[i]);
    }

    {
        std::list<int> src = { 1,2,3,4,5 };
        std::list<int> dst;
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), std::inserter(dst, dst.begin()));
        auto its = src.begin();
        auto itd = dst.begin();
        for (; its != src.end(); ++its, ++itd) Assert(*its == *itd);
    }

    {
        int src[5] = { 1, 2, 3, 4, 5 };
        int dst[5] = {};
        simd_stl::algorithm::copy_n(src, 5, dst);
        for (int i = 0; i < 5; ++i) {
            Assert(dst[i] == src[i]);
        }
    }

    // --- частичное копирование ---
    {
        int src[5] = { 10, 20, 30, 40, 50 };
        int dst[5] = {};
        simd_stl::algorithm::copy_n(src, 3, dst);
        Assert(dst[0] == 10);
        Assert(dst[1] == 20);
        Assert(dst[2] == 30);
        // хвост не тронут
        Assert(dst[3] == 0);
        Assert(dst[4] == 0);
    }

    // --- копирование в std::vector ---
    {
        std::vector<int> src{ 7, 8, 9, 10 };
        std::vector<int> dst(4);
        simd_stl::algorithm::copy_n(src.data(), src.size(), dst.data());
        for (size_t i = 0; i < src.size(); ++i) {
            Assert(dst[i] == src[i]);
        }
    }

    // --- разные типы ---
    {
        char src[3] = { 'a', 'b', 'c' };
        char dst[3] = {};
        simd_stl::algorithm::copy_n(src, 3, dst);
        Assert(dst[0] == 'a');
        Assert(dst[1] == 'b');
        Assert(dst[2] == 'c');

        float srcf[2] = { 1.5f, -2.5f };
        float dstf[2] = {};
        simd_stl::algorithm::copy_n(srcf, 2, dstf);
        Assert(dstf[0] == 1.5f);
        Assert(dstf[1] == -2.5f);
    }

    // --- пограничные случаи ---
    {
        int src[3] = { 42, 43, 44 };
        int dst[3] = { 0, 0, 0 };
        simd_stl::algorithm::copy_n(src, 0, dst); // ничего не копируем
        Assert(dst[0] == 0 && dst[1] == 0 && dst[2] == 0);

        simd_stl::algorithm::copy_n(src, 1, dst);
        Assert(dst[0] == 42);
        Assert(dst[1] == 0);
        Assert(dst[2] == 0);
    }

    return 0;
}
