#include <simd_stl/algorithm/find/Count.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdCountBenchmark {
public:
    static void CountAllEqual(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = 42;

        while (state.KeepRunning()) {
            auto result = std::count(array, array + sizeForBenchmark, 42);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void CountEverySecond(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            if (i % 2 == 0)
                array[i] = 42;
            else
                array[i] = 0;

        while (state.KeepRunning()) {
            auto result = std::count(array, array + sizeForBenchmark, 42);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void CountEveryFourth(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            if (i % 4 == 0)
                array[i] = 42;
            else
                array[i] = 0;

        while (state.KeepRunning()) {
            auto result = std::count(array, array + sizeForBenchmark, 42);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlCountBenchmark {
public:
    static void CountAllEqual(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = 42;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::count(array, array + sizeForBenchmark, 42);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void CountEverySecond(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            if (i % 2 == 0)
                array[i] = 42;
            else
                array[i] = 0;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::count(array, array + sizeForBenchmark, 42);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void CountEveryFourth(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            if (i % 4 == 0)
                array[i] = 42;
            else
                array[i] = 0;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::count(array, array + sizeForBenchmark, 42);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint8, CountAllEqual);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint16, CountAllEqual);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint32, CountAllEqual);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint64, CountAllEqual);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, float, CountAllEqual);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, double, CountAllEqual);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint8, CountEverySecond);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint16, CountEverySecond);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint32, CountEverySecond);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint64, CountEverySecond);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, float, CountEverySecond);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, double, CountEverySecond);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint8, CountEveryFourth);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint16, CountEveryFourth);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint32, CountEveryFourth);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, simd_stl::uint64, CountEveryFourth);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, float, CountEveryFourth);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlCountBenchmark, StdCountBenchmark, double, CountEveryFourth);

SIMD_STL_BENCHMARK_MAIN();