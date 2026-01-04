#include <simd_stl/algorithm/replace/Replace.h>
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
    SizeForBenchmark sizeForBenchmark>
class StdReplaceBenchmark {
public:
    static void Replace(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];

        for (simd_stl::sizetype i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        for (simd_stl::sizetype i = 0; i < sizeForBenchmark; i += 2)
            array[i] = simd_stl::math::__maximum_integral_limit<_Type_>() >> 1;

        while (state.KeepRunning()) {
            std::replace(array, array + sizeForBenchmark, static_cast<_Type_>((simd_stl::math::__maximum_integral_limit<_Type_>() >> 1)),
                simd_stl::math::__maximum_integral_limit<_Type_>());

            benchmark::DoNotOptimize(array);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlReplaceBenchmark {
public:
    static void Replace(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];

        for (simd_stl::sizetype i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        for (simd_stl::sizetype i = 0; i < sizeForBenchmark; i += 2)
            array[i] = simd_stl::math::__maximum_integral_limit<_Type_>() >> 1;

        while (state.KeepRunning()) {
            simd_stl::algorithm::replace(array, array + sizeForBenchmark,
                (simd_stl::math::__maximum_integral_limit<_Type_>() >> 1), simd_stl::math::__maximum_integral_limit<_Type_>());
            benchmark::DoNotOptimize(array);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, simd_stl::int8, Replace);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, simd_stl::int16, Replace);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, simd_stl::int32, Replace);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReplaceBenchmark, StdReplaceBenchmark, simd_stl::int64, Replace);

SIMD_STL_BENCHMARK_MAIN();