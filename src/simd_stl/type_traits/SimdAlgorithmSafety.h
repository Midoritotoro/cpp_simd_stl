#pragma once 

#include <src/simd_stl/type_traits/IteratorCheck.h>

__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <
    class _Type_,
    class _Element_>
struct VectorAlgorithmInFindIsSafeObjectPointers : 
    std::false_type
{};

template <
    class _Type1_,
    class _Type2_>
struct VectorAlgorithmInFindIsSafeObjectPointers<_Type1_*, _Type2_*>:
    std::conjunction<
        std::disjunction<std::is_object<_Type1_>, std::is_void<_Type1_>>,
        std::disjunction<std::is_object<_Type2_>, std::is_void<_Type2_>>,
        std::disjunction<std::is_same<std::remove_cv_t<_Type1_>, std::remove_cv_t<_Type2_>>,
        std::is_void<_Type1_>, std::is_void<_Type2_>>> 
{};

template <
    class _Type_,
    class _Element_>
constexpr bool is_vectorized_algorithm_element_safe_v = std::disjunction_v<
#ifdef __cpp_lib_byte
    std::conjunction<std::is_same<_Type_, std::byte>, std::is_same<_Element_, std::byte>>,
#endif // defined(__cpp_lib_byte)
    std::conjunction<std::is_integral<_Type_>, std::is_integral<_Element_>>,
    std::conjunction<std::is_pointer<_Type_>, std::is_same<_Type_, _Element_>>,
    std::conjunction<std::is_same<_Type_, std::nullptr_t>, std::is_pointer<_Element_>>,
    VectorAlgorithmInFindIsSafeObjectPointers<_Type_, _Element_>>;

template <
    class _Iterator_,
    class _Type_>
constexpr bool is_vectorized_find_algorithm_safe_v =
    is_iterator_contiguous_v<_Iterator_>
    && !is_iterator_volatile_v<_Iterator_>
    && is_vectorized_algorithm_element_safe_v<_Type_, IteratorValueType<_Iterator_>>;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
