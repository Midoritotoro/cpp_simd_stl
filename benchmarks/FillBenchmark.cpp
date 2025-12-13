#include <simd_stl/algorithm/fill/Fill.h>
#include <algorithm>

#include <benchmarks/tools/BenchmarkHelper.h>


template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
struct _FillBenchmarkArray {
    _Type_* array;

    _FillBenchmarkArray() {
        array = new _Type_[_Size_];
    }

    ~_FillBenchmarkArray() {
        delete[] array;
    }
};

template <
    typename            _Type_,
    simd_stl::sizetype  _Size_>
_FillBenchmarkArray<_Type_, _Size_> _GenerateArrayForBenchmark() noexcept {
    _FillBenchmarkArray<_Type_, _Size_> result;

    for (simd_stl::sizetype i = 0; i < _Size_; ++i)
        result.array[i] = i;

    return result;
}

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdFillBenchmark {
public:
    static inline auto array = _GenerateArrayForBenchmark<_Type_, sizeForBenchmark>();

    static void Fill(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            std::fill(array.array, array.array + sizeForBenchmark, _Type_(42));
            benchmark::DoNotOptimize(array.array);
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlFillBenchmark {
public:
    static inline auto array = _GenerateArrayForBenchmark<_Type_, sizeForBenchmark>();

    static void Fill(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            simd_stl::algorithm::fill(array.array, array.array + sizeForBenchmark, _Type_(42));
            benchmark::DoNotOptimize(array.array);
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFillBenchmark, StdFillBenchmark, simd_stl::int8, Fill);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFillBenchmark, StdFillBenchmark, simd_stl::int16, Fill);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFillBenchmark, StdFillBenchmark, simd_stl::int32, Fill);
//SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFillBenchmark, StdFillBenchmark, simd_stl::int64, Fill);

SIMD_STL_BENCHMARK_MAIN();