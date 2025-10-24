#define NOMINMAX

#include <simd_stl/algorithm/copy/Copy.h>
#include <random> 

template <typename It1, typename It2>
void assert_equal(It1 first1, It1 last1, It2 first2) {
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

//template <>
//void fill_random<char>(std::vector<char>& v, uint64_t seed) {
//    std::mt19937 rng(static_cast<uint32_t>(seed));
//    std::uniform_int_distribution<int> dist(0, 127);
//    for (auto& x : v) x = static_cast<char>(dist(rng));
//}
//
//struct Counted {
//    int value;
//    static inline size_t copies = 0;
//    Counted() : value(0) {}
//    explicit Counted(int v) : value(v) {}
//    Counted(const Counted& other) : value(other.value) { ++copies; }
//    Counted& operator=(const Counted& other) { value = other.value; ++copies; return *this; }
//    bool operator==(const Counted& rhs) const noexcept { return value == rhs.value; }
//};
//
//struct MightThrow {
//    int value;
//    bool throw_on_copy = false;
//    MightThrow() : value(0), throw_on_copy(false) {}
//    MightThrow(int v, bool t = false) : value(v), throw_on_copy(t) {}
//    MightThrow(const MightThrow& other) {
//        if (other.throw_on_copy) throw std::runtime_error("Copy throw");
//        value = other.value;
//        throw_on_copy = other.throw_on_copy;
//    }
//    MightThrow& operator=(const MightThrow& other) {
//        if (other.throw_on_copy) throw std::runtime_error("Assign throw");
//        value = other.value;
//        throw_on_copy = other.throw_on_copy;
//        return *this;
//    }
//    bool operator==(const MightThrow& rhs) const noexcept { return value == rhs.value; }
//};

int main() {
    {
        std::vector<int> src;
        std::vector<int> dst(0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.begin());
    }

    {
        std::vector<int> src = { 42 };
        std::vector<int> dst(1, 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        assert_equal(src.begin(), src.end(), dst.begin());
    }
    {
        std::vector<int> src = { 1,2,3,4,5,6,7,8 };
        std::vector<int> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        assert_equal(src.begin(), src.end(), dst.begin());
    }

  /*  {
        std::string src = "SIMD copy test!";
        std::string dst(src.size(), '\0');
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        assert(src == dst);
    }

    {
        std::array<int, 16> srcArr{};
        for (int i = 0; i < 16; ++i) srcArr[i] = i * i;
        int dstArr[16] = {};
        auto out = simd_stl::algorithm::copy(srcArr.data(), srcArr.data() + srcArr.size(), dstArr);
        assert(out == dstArr + 16);
        for (int i = 0; i < 16; ++i) assert(srcArr[i] == dstArr[i]);
    }

    {
        std::list<int> src = { 10,20,30,40,50 };
        std::list<int> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());

        auto itSrc = src.begin();
        auto itDst = dst.begin();
        for (; itSrc != src.end(); ++itSrc, ++itDst) assert(*itSrc == *itDst);
    }

   
    {
        std::deque<int> src = { 3,1,4,1,5,9,2,6,5 };
        std::deque<int> dst(src.size());
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        for (size_t i = 0; i < src.size(); ++i) assert(src[i] == dst[i]);
    }

    {
        std::vector<char> src(1024);
        fill_random(src, 1);
        std::vector<char> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        std::vector<int> src(4096);
        fill_random(src, 2);
        std::vector<int> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        std::vector<uint64_t> src(8192);
        fill_random(src, 3);
        std::vector<uint64_t> dst(src.size(), 0);
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        const size_t N = 10'000'000;
        std::vector<int> src(N);
        fill_random(src, 4);
        std::vector<int> dst(N, 0);

        auto start = std::chrono::high_resolution_clock::now();
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        auto stop = std::chrono::high_resolution_clock::now();

        assert(out == dst.end());
  
        for (size_t i = 0; i < N; i += N / 10) assert(src[i] == dst[i]);
        assert(src.front() == dst.front());
        assert(src.back() == dst.back());

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << "Large copy of " << N << " ints took " << ms << " ms\n";
    }

    {
        std::vector<uint8_t> buffer(4096 + 64, 0);
        uint8_t* base = buffer.data();

        uint8_t* src = base + 1;
        uint8_t* dst = base + 33;

        const size_t len = 2048;
        for (size_t i = 0; i < len; ++i) src[i] = static_cast<uint8_t>(i ^ 0x5A);

        auto out = simd_stl::algorithm::copy(src, src + len, dst);
        assert(out == dst + len);
        for (size_t i = 0; i < len; ++i) assert(src[i] == dst[i]);
    }


    {
        std::vector<int> src(1024);
        for (int i = 0; i < 1024; ++i) src[i] = i;
        std::vector<int> dst(1024, -1);

        auto out = simd_stl::algorithm::copy(src.begin() + 100, src.begin() + 900, dst.begin() + 50);
        assert(out == dst.begin() + 50 + (900 - 100));
        for (int i = 0; i < 50; ++i) assert(dst[i] == -1);
        for (int i = 0; i < 800; ++i) assert(dst[50 + i] == src[100 + i]);
        for (int i = 850; i < 1024; ++i) assert(dst[i] == -1);
    }

    {
        Counted::copies = 0;
        std::vector<Counted> src;
        for (int i = 0; i < 1000; ++i) src.emplace_back(i);
        std::vector<Counted> dst(src.size());
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        assert(out == dst.end());
        for (size_t i = 0; i < src.size(); ++i) assert(src[i] == dst[i]);
        assert(Counted::copies > 0);
    }

    {
        std::vector<MightThrow> src;
        for (int i = 0; i < 100; ++i) src.emplace_back(i, i == 50);
        std::vector<MightThrow> dst(src.size());
        bool thrown = false;
        try {
            (void)simd_stl::algorithm::copy(src.begin(), src.end(), dst.begin());
        }
        catch (const std::runtime_error&) {
            thrown = true;
        }
        assert(thrown); 
    }

    {
        std::vector<int> src(100);
        for (int i = 0; i < 100; ++i) src[i] = i * 3;
        std::vector<int> dst;
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), std::back_inserter(dst));
        assert(dst.size() == src.size());
        for (size_t i = 0; i < src.size(); ++i) assert(dst[i] == src[i]);
    }

    {
        std::list<int> src = { 1,2,3,4,5 };
        std::list<int> dst;
        auto out = simd_stl::algorithm::copy(src.begin(), src.end(), std::inserter(dst, dst.begin()));
        auto itS = src.begin();
        auto itD = dst.begin();
        for (; itS != src.end(); ++itS, ++itD) assert(*itS == *itD);
    }*/

    return 0;
}
