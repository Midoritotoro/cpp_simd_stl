#include <simd_stl/algorithm/find/Contains.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h> 

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdContainsBenchmark {
public:
    static void ContainsInEnd(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark - 1] = 42;

        while (state.KeepRunning()) {
            auto result = std::ranges::contains(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void ContainsInMiddle(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark / 2] = 42;

        while (state.KeepRunning()) {
            auto result = std::ranges::contains(array, array + sizeForBenchmark, array[sizeForBenchmark / 2]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void ContainsInBegin(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[0] = 42;

        while (state.KeepRunning()) {
            auto result = std::ranges::contains(array, array + sizeForBenchmark, array[0]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlContainsBenchmark {
public:
    static void ContainsInEnd(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark - 1] = 42;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::contains(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void ContainsInMiddle(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[sizeForBenchmark / 2] = 42;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::contains(array, array + sizeForBenchmark, array[sizeForBenchmark / 2]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void ContainsInBegin(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        std::memset(array, 0, sizeof(array));

        array[0] = 42;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::contains(array, array + sizeForBenchmark, array[0]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint8, ContainsInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint16, ContainsInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint32, ContainsInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint64, ContainsInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, float, ContainsInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, double, ContainsInBegin);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint8, ContainsInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint16, ContainsInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint32, ContainsInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint64, ContainsInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, float, ContainsInMiddle);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, double, ContainsInMiddle);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint8, ContainsInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint16, ContainsInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint32, ContainsInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, simd_stl::uint64, ContainsInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, float, ContainsInEnd);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlContainsBenchmark, StdContainsBenchmark, double, ContainsInEnd);

SIMD_STL_BENCHMARK_MAIN();