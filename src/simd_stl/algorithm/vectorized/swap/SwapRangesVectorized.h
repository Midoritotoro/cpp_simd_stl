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
		using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
		numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

		const auto bytes		= sizeof(_Type_) * count;
		const auto alignedBytes = bytes & (~(sizeof(_SimdType_) - 1));

		if (alignedBytes != 0) {
			void* stopAt = first;
			AdvanceBytes(stopAt, alignedBytes);

			do {
				const auto loadedFirst = _SimdType_::loadUnaligned(first);
				const auto loadedSecond = _SimdType_::loadUnaligned(second);

				loadedFirst.storeUnaligned(second);
				loadedSecond.storeUnaligned(first);

				AdvanceBytes(first, sizeof(_SimdType_));
				AdvanceBytes(second, sizeof(_SimdType_));
			} while (first != stopAt);
		}

		auto remainingCount = (bytes - alignedBytes) / sizeof(_Type_);

		for (sizetype current = 0; current < remainingCount; ++current) {
			_Type_ temp = *first;

			*first++ = *second;
			*second++ = temp;
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
		for (sizetype current = 0; current < count; ++current) {
			_Type_ temp = *first;
			
			*first++ = *second;
			*second++ = temp;
		}
	}
};


template <typename _Type_>
void _SwapRangesVectorized(
	_Type_*		first,
	_Type_*		second,
	sizetype	count) noexcept
{
	if (arch::ProcessorFeatures::SSE2())
		return _SwapRangesVectorizedInternal<arch::CpuFeature::SSE2, _Type_>()(first, second, count);

	return _SwapRangesVectorizedInternal<arch::CpuFeature::None, _Type_>()(first, second, count);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
