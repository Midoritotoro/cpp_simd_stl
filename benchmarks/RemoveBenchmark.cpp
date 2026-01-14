
#include <simd_stl/algorithm/remove/Remove.h>
#include <benchmarks/tools/BenchmarkHelper.h> 

#include <algorithm>
#include <vector>
#include <numeric>
#include <cstddef>


template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdRemoveBenchmark {
public:
    static void RemoveNone(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        _Char_ value_to_remove = static_cast<_Char_>(sizeForBenchmark + 1);

        while (state.KeepRunning()) {
            for (int i = 0; i < sizeForBenchmark; ++i) {
                array[i] = static_cast<_Char_>(i % (sizeForBenchmark > 0 ? sizeForBenchmark : 1));
            }

            auto* new_end = std::remove(array, array + sizeForBenchmark, value_to_remove);
            benchmark::DoNotOptimize(new_end);
            benchmark::ClobberMemory();
        }
    }

    static void RemoveHalf(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        _Char_ value_to_remove = static_cast<_Char_>(0);

        while (state.KeepRunning()) {
            for (int i = 0; i < sizeForBenchmark; ++i) {
                array[i] = static_cast<_Char_>(i % 2);
            }

            auto* new_end = std::remove(array, array + sizeForBenchmark, value_to_remove);
            benchmark::DoNotOptimize(new_end);
            benchmark::ClobberMemory();
        }
    }

    static void RemoveAll(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        _Char_ value_to_remove = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            for (int i = 0; i < sizeForBenchmark; ++i) {
                array[i] = value_to_remove;
            }

            auto* new_end = std::remove(array, array + sizeForBenchmark, value_to_remove);
            benchmark::DoNotOptimize(new_end);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlRemoveBenchmark {
public:
    static void RemoveNone(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        _Char_ value_to_remove = static_cast<_Char_>(sizeForBenchmark + 1);

        while (state.KeepRunning()) {
            for (int i = 0; i < sizeForBenchmark; ++i) {
                array[i] = static_cast<_Char_>(i % (sizeForBenchmark > 0 ? sizeForBenchmark : 1));
            }

            auto* new_end = simd_stl::algorithm::remove(array, array + sizeForBenchmark, value_to_remove);
            benchmark::DoNotOptimize(new_end);
            benchmark::ClobberMemory();
        }
    }

    static void RemoveHalf(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        _Char_ value_to_remove = static_cast<_Char_>(0);

        while (state.KeepRunning()) {
            for (int i = 0; i < sizeForBenchmark; ++i) {
                array[i] = static_cast<_Char_>(i % 2);
            }

            auto* new_end = simd_stl::algorithm::remove(array, array + sizeForBenchmark, value_to_remove);
            benchmark::DoNotOptimize(new_end);
            benchmark::ClobberMemory();
        }
    }

    static void RemoveAll(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];
        _Char_ value_to_remove = static_cast<_Char_>(42);

        while (state.KeepRunning()) {
            for (int i = 0; i < sizeForBenchmark; ++i) {
                array[i] = value_to_remove;
            }

            auto* new_end = simd_stl::algorithm::remove(array, array + sizeForBenchmark, value_to_remove);
            benchmark::DoNotOptimize(new_end);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint8, RemoveNone);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint16, RemoveNone);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint32, RemoveNone);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint64, RemoveNone);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, float, RemoveNone);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, double, RemoveNone);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint8, RemoveHalf);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint16, RemoveHalf);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint32, RemoveHalf);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint64, RemoveHalf);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, float, RemoveHalf);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, double, RemoveHalf);

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint8, RemoveAll);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint16, RemoveAll);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint32, RemoveAll);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, simd_stl::uint64, RemoveAll);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, float, RemoveAll);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlRemoveBenchmark, StdRemoveBenchmark, double, RemoveAll);

SIMD_STL_BENCHMARK_MAIN();