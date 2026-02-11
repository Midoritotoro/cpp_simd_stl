#include <simd_stl/algorithm/order/ReverseCopy.h>
#include <algorithm>
#include <benchmarks/tools/BenchmarkHelper.h>

template <
    typename T,
    SizeForBenchmark sizeForBenchmark>
class StdReverseCopyBenchmark {
public:
    static inline auto src = FixedArray<T, sizeForBenchmark>{};
    static inline auto dest = FixedArray<T, sizeForBenchmark>{};

    static void ReverseCopy(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            std::reverse_copy(src.data, src.data + sizeForBenchmark, dest.data);
            benchmark::DoNotOptimize(dest.data);
        }
    }
};

template <
    typename T,
    SizeForBenchmark sizeForBenchmark>
class SimdStlReverseCopyBenchmark {
public:
    static inline auto src = FixedArray<T, sizeForBenchmark>{};
    static inline auto dest = FixedArray<T, sizeForBenchmark>{};

    static void ReverseCopy(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            simd_stl::algorithm::reverse_copy(src.data, src.data + sizeForBenchmark, dest.data);
            benchmark::DoNotOptimize(dest.data);
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseCopyBenchmark, StdReverseCopyBenchmark, simd_stl::int8, ReverseCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseCopyBenchmark, StdReverseCopyBenchmark, simd_stl::int16, ReverseCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseCopyBenchmark, StdReverseCopyBenchmark, simd_stl::int32, ReverseCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseCopyBenchmark, StdReverseCopyBenchmark, simd_stl::int64, ReverseCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseCopyBenchmark, StdReverseCopyBenchmark, float, ReverseCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseCopyBenchmark, StdReverseCopyBenchmark, double, ReverseCopy);

SIMD_STL_BENCHMARK_MAIN();
