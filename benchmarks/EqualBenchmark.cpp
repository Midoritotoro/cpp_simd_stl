#include <simd_stl/algorithm/find/Equal.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h>
#include <cstring>
#include <vector>
#include <algorithm>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdEqualBenchmark {
public:
    static void MismatchInEnd(benchmark::State& state) noexcept {
        std::vector<_Char_> array1(sizeForBenchmark);
        std::vector<_Char_> array2(sizeForBenchmark);

        std::memset(array1.data(), 0, sizeForBenchmark * sizeof(_Char_));
        std::memset(array2.data(), 0, sizeForBenchmark * sizeof(_Char_));

        array2[sizeForBenchmark - 1] = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            auto result = std::equal(array1.begin(), array1.end(), array2.begin());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void MismatchInMiddle(benchmark::State& state) noexcept {
        std::vector<_Char_> array1(sizeForBenchmark);
        std::vector<_Char_> array2(sizeForBenchmark);

        std::memset(array1.data(), 0, sizeForBenchmark * sizeof(_Char_));
        std::memset(array2.data(), 0, sizeForBenchmark * sizeof(_Char_));

        array2[sizeForBenchmark / 2] = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            auto result = std::equal(array1.begin(), array1.end(), array2.begin());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void MismatchInBegin(benchmark::State& state) noexcept {
        std::vector<_Char_> array1(sizeForBenchmark);
        std::vector<_Char_> array2(sizeForBenchmark);

        std::memset(array1.data(), 0, sizeForBenchmark * sizeof(_Char_));
        std::memset(array2.data(), 0, sizeForBenchmark * sizeof(_Char_));

        array2[0] = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            auto result = std::equal(array1.begin(), array1.end(), array2.begin());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlEqualBenchmark {
public:
    static void MismatchInEnd(benchmark::State& state) noexcept {
        std::vector<_Char_> array1(sizeForBenchmark);
        std::vector<_Char_> array2(sizeForBenchmark);

        std::memset(array1.data(), 0, sizeForBenchmark * sizeof(_Char_));
        std::memset(array2.data(), 0, sizeForBenchmark * sizeof(_Char_));

        array2[sizeForBenchmark - 1] = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::equal(array1.begin(), array1.end(), array2.begin());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void MismatchInMiddle(benchmark::State& state) noexcept {
        std::vector<_Char_> array1(sizeForBenchmark);
        std::vector<_Char_> array2(sizeForBenchmark);

        std::memset(array1.data(), 0, sizeForBenchmark * sizeof(_Char_));
        std::memset(array2.data(), 0, sizeForBenchmark * sizeof(_Char_));

        array2[sizeForBenchmark / 2] = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::equal(array1.begin(), array1.end(), array2.begin());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void MismatchInBegin(benchmark::State& state) noexcept {
        std::vector<_Char_> array1(sizeForBenchmark);
        std::vector<_Char_> array2(sizeForBenchmark);

        std::memset(array1.data(), 0, sizeForBenchmark * sizeof(_Char_));
        std::memset(array2.data(), 0, sizeForBenchmark * sizeof(_Char_));

        array2[0] = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::equal(array1.begin(), array1.end(), array2.begin());
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};


SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint8, MismatchInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint16, MismatchInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint32, MismatchInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint64, MismatchInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, float, MismatchInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, double, MismatchInBegin);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint8, MismatchInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint16, MismatchInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint32, MismatchInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint64, MismatchInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, float, MismatchInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, double, MismatchInMiddle);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint8, MismatchInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint16, MismatchInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint32, MismatchInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, simd_stl::uint64, MismatchInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, float, MismatchInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlEqualBenchmark, StdEqualBenchmark, double, MismatchInEnd);

SIMD_STL_BENCHMARK_MAIN();