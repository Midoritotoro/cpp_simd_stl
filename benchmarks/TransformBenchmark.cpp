#include <simd_stl/algorithm/transform/Transform.h>
#include <algorithm>

#include <benchmarks/tools/BenchmarkHelper.h>

template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
struct _TransformBenchmarkArray {
    _Type_* array;

    _TransformBenchmarkArray() {
        array = new _Type_[_Size_];
    }

    ~_TransformBenchmarkArray() {
        delete[] array;
    }
};

template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
_TransformBenchmarkArray<_Type_, _Size_> _GenerateArrayForReplaceBenchmark() noexcept {
    _TransformBenchmarkArray<_Type_, _Size_> result;

    for (simd_stl::sizetype i = 0; i < _Size_; ++i)
        result.array[i] = i;

    return result;
}

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdTransformBenchmark {
public:
    static inline auto array = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();
    static inline auto array2 = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    static inline auto destination = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    template <typename _Predicate_>
    static void Transform(benchmark::State& state) noexcept {
        constexpr auto pred = _Predicate_{};

        while (state.KeepRunning()) {
            std::transform(array.array, array.array + sizeForBenchmark, array2.array, destination.array, pred);
            benchmark::DoNotOptimize(array.array);
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlTransformBenchmark {
public:
    static inline auto array = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();
    static inline auto array2  = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    static inline auto destination = _GenerateArrayForReplaceBenchmark<_Type_, sizeForBenchmark>();

    template <typename _Predicate_>
    static void Transform(benchmark::State& state) noexcept {
        constexpr auto pred = _Predicate_{};

        while (state.KeepRunning()) {
            simd_stl::algorithm::transform(array.array, array.array + sizeForBenchmark, array2.array, destination.array, pred);

            benchmark::DoNotOptimize(array.array);
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlTransformBenchmark, StdTransformBenchmark, simd_stl::int32, Transform<simd_stl::type_traits::multiplies<>>);


SIMD_STL_BENCHMARK_MAIN();