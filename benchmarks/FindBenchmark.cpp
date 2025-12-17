#include <simd_stl/algorithm/find/Find.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h> 

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdFindBenchmark {
public:
    static void FindInEnd(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = std::find(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInMiddle(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = std::find(array, array + sizeForBenchmark, array[sizeForBenchmark / 2]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInBegin(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = std::find(array, array + sizeForBenchmark, array[0]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlFindBenchmark {
public:
    static void FindInEnd(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInMiddle(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find(array, array + sizeForBenchmark, array[sizeForBenchmark / 2]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }

    static void FindInBegin(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find(array, array + sizeForBenchmark, array[0]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint8, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint16, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint32, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint64, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, float, FindInBegin);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, double, FindInBegin);

//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint8, FindInMiddle);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint16, FindInMiddle);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint32, FindInMiddle);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint64, FindInMiddle);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, float, FindInMiddle);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, double, FindInMiddle);
//
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint8, FindInEnd);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint16, FindInEnd);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint32, FindInEnd);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint64, FindInEnd);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, float, FindInEnd);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, double, FindInEnd);
//
SIMD_STL_BENCHMARK_MAIN();