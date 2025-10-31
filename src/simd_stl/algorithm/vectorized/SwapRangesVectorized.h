#pragma once

#include <simd_stl/numeric/BasicSimd.h>
#include <src/simd_stl/algorithm/AdvanceBytes.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Type_>
struct _SwapRangesVectorizedInternal {
	simd_stl_always_inline void operator()(
		_Type_*		first,
		_Type_*		second,
		sizetype	count) noexcept
	{
		using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
		
		const auto bytes		= sizeof(_Type_) * count;
		const auto alignedBytes = bytes & (~(sizeof(_SimdType_) - 1));

		if (alignedBytes != 0) {
			void* stopAt = first;
			AdvanceBytes(stopAt, bytes);

			do {
				const auto loadedFirst = _SimdType_::loadUnaligned(first);
				const auto loadedSecond = _SimdType_::loadUnaligned(second);

				loadedFirst.storeUnaligned(second);
				loadedSecond.storeUnaligned(first);

				AdvanceBytes(first, sizeof(_SimdType_));
				AdvanceBytes(second, sizeof(_SimdType_));
			} while (first != stopAt);

			_SimdType_::zeroUpper();
		}

		auto remainingCount = (bytes - alignedBytes) / sizeof(_Type_);

		if (remainingCount != 0) {
			do {
				_Type_ temp = *first;

				*first = *second;
				*second = *first;

				++first;
				++second;

				--remainingCount;
			} while (remainingCount);
		}
	}
};

template <typename _Type_> 
struct _SwapRangesVectorizedInternal<arch::CpuFeature::None, _Type_> {
	simd_stl_always_inline void operator()(
		_Type_*		first,
		_Type_*		second,
		sizetype	count) noexcept
	{
		if (count != 0) {
			do {
				_Type_ temp = *first;

				*first = *second;
				*second = *first;

				++first;
				++second;

				--count;
			} while (count);
		}
	}
};


template <typename _Type_>
void _SwapRangesVectorized(
	_Type_*		first,
	_Type_*		second,
	sizetype	bytes) noexcept
{
	if (arch::ProcessorFeatures::SSE2())
		return _SwapRangesVectorizedInternal<arch::CpuFeature::SSE2, _Type_>()(first, second, bytes);

	return _SwapRangesVectorizedInternal<arch::CpuFeature::None, _Type_>()(first, second, bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
