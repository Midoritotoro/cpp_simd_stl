#include <simd_stl/algorithm/find/FindLast.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h>

#include <ranges>


template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdFindLastBenchmark {
public:
    static void FindInEnd(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark - 1] = 42;

        while (state.KeepRunning()) {
            auto result = std::ranges::find_last(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInMiddle(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark / 2] = 42;

        while (state.KeepRunning()) {
            auto result = std::ranges::find_last(array, array + sizeForBenchmark, array[sizeForBenchmark / 2]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInBegin(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[0] = 42;

        while (state.KeepRunning()) {
            auto result = std::ranges::find_last(array, array + sizeForBenchmark, array[0]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlFindLastBenchmark {
public:
    static void FindInEnd(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark - 1] = 42;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find_last(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInMiddle(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark / 2] = 42;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find_last(array, array + sizeForBenchmark, array[sizeForBenchmark / 2]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInBegin(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[0] = 42;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find_last(array, array + sizeForBenchmark, array[0]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint8, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint16, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint32, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint64, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, float, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, double, FindInBegin);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint8, FindInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint16, FindInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint32, FindInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint64, FindInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, float, FindInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, double, FindInMiddle);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint8, FindInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint16, FindInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint32, FindInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, simd_stl::uint64, FindInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, float, FindInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindLastBenchmark, StdFindLastBenchmark, double, FindInEnd);

SIMD_STL_BENCHMARK_MAIN();