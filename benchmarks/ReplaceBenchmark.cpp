#include <simd_stl/algorithm/replace/Replace.h>
#include <algorithm>

#include <benchmarks/tools/BenchmarkHelper.h>

template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
struct _ReplaceBenchmarkArray {
    _Type_ array[_Size_];
};

template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
_ReplaceBenchmarkArray<_Type_, _Size_> _GenerateArrayForReplaceBenchmark() noexcept {
    _ReplaceBenchmarkArray<_Type_, _Size_> result;

    for (simd_stl::sizetype i = 0; i < _Size_; ++i)
        result.array[i] = i;

    for (simd_stl::sizetype i = 0; i < _Size_; i += 10)
        result.array[i] = simd_stl::math::MaximumIntegralLimit<_Type_>() >> 1;

    return result;
}

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdReplaceBenchmark {
public:
    static inline auto array = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    static void Replace(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            std::replace(array.array, array.array + sizeForBenchmark, static_cast<_Type_>((simd_stl::math::MaximumIntegralLimit<_Type_>() >> 1)),
                simd_stl::math::MaximumIntegralLimit<_Type_>());

            benchmark::DoNotOptimize(array.array);
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlReplaceBenchmark {
public:
    static inline auto array = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    static void Replace(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            simd_stl::algorithm::replace(array.array, array.array + sizeForBenchmark,
                (simd_stl::math::MaximumIntegralLimit<_Type_>() >> 1), simd_stl::math::MaximumIntegralLimit<_Type_>());
            benchmark::DoNotOptimize(array.array);
        }
    }
};

//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, simd_stl::int8, Replace);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, simd_stl::int16, Replace);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, char, Replace);
SIMD_STL_BENCHMARK_MAIN();