#include <simd_stl/algorithm/replace/ReplaceCopy.h>
#include <algorithm>

#include <benchmarks/tools/BenchmarkHelper.h>


template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
struct _ReplaceBenchmarkArray {
    _Type_* array;

    _ReplaceBenchmarkArray() {
        array = new _Type_[_Size_];
    }

    ~_ReplaceBenchmarkArray() {
        delete[] array;
    }
};

template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
_ReplaceBenchmarkArray<_Type_, _Size_> _GenerateArrayForReplaceBenchmark() noexcept {
    _ReplaceBenchmarkArray<_Type_, _Size_> result;

    for (simd_stl::sizetype i = 0; i < _Size_; ++i)
        result.array[i] = i;

    for (simd_stl::sizetype i = 0; i < _Size_; i += 2)
        result.array[i] = simd_stl::math::__maximum_integral_limit<_Type_>() >> 1;

    return result;
}

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdReplaceCopyBenchmark {
public:
    static inline auto array = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();
    static inline auto destination = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();


    static void ReplaceCopy(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            std::replace_copy(array.array, array.array + sizeForBenchmark, destination.array,
                static_cast<_Type_>((simd_stl::math::__maximum_integral_limit<_Type_>() >> 1)),
                simd_stl::math::__maximum_integral_limit<_Type_>());

            benchmark::DoNotOptimize(array.array);
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlReplaceCopyBenchmark {
public:
    static inline auto array = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();
    static inline auto destination = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    static void ReplaceCopy(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            simd_stl::algorithm::replace_copy(array.array, array.array + sizeForBenchmark, destination.array,
                (simd_stl::math::__maximum_integral_limit<_Type_>() >> 1), simd_stl::math::__maximum_integral_limit<_Type_>());
            benchmark::DoNotOptimize(array.array);
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceCopyBenchmark, StdReplaceCopyBenchmark, simd_stl::int8, ReplaceCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceCopyBenchmark, StdReplaceCopyBenchmark, simd_stl::int16, ReplaceCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceCopyBenchmark, StdReplaceCopyBenchmark, simd_stl::int32, ReplaceCopy);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceCopyBenchmark, StdReplaceCopyBenchmark, simd_stl::int64, ReplaceCopy);

SIMD_STL_BENCHMARK_MAIN();