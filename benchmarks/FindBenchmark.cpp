#include <simd_stl/algorithm/find/Find.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdFindBenchmark {
public:
    static void Find(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = std::find(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlFindBenchmark {
public:
    static void Find(benchmark::State& state) noexcept {
        _Char_ array[sizeForBenchmark];

        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto* result = simd_stl::algorithm::find(array, array + sizeForBenchmark, array[sizeForBenchmark - 1]);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::uint64, Find);

SIMD_STL_BENCHMARK_MAIN();