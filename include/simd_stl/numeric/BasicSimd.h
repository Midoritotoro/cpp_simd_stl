#pragma once 

#include <simd_stl/numeric/BasicSimdImplementation.h>
#include <simd_stl/numeric/BasicSimdElementReference.h>

#include <src/simd_stl/utility/Assert.h>
#include <xstring> 


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_ = int>
class basic_simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

    friend BasicSimdElementReference;
    using __impl = BasicSimdImplementation<_SimdGeneration_, _Element_>;
public:
    static constexpr auto _Generation = _SimdGeneration_;

    using value_type = typename __impl::value_type;
    using vector_type = typename __impl::vector_type;

    using size_type = typename __impl::size_type;
    using mask_type = typename __impl::mask_type;

    basic_simd() noexcept;

    basic_simd(const value_type value) noexcept;
    basic_simd(const vector_type& other) noexcept;

    ~basic_simd() noexcept;

    /**
        * @brief ���������� �������� �� ������� � ������� 'index' � ��������������� ��������� ������.
        * @param index ������� ��� ����������.
        * @return ����������� ��������.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline value_type extract(const size_type index) const noexcept {
        Assert(index > 0 && index < __impl::vectorElementsCount, "simd_stl::numeric::basic_simd: Index out of range");
        return __impl::extract(_vector, index);
    }

    /**
        * @brief ���������� �������� �� ������� � ������� 'index' � ��������������� ��������� ������.
        * @param index ������� ��� ����������.
        * @return ������ ��� ����������� ���������, ����������� �������� ��������������� ������� �������.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd> extractWrapped(const size_type index) noexcept {
        Assert(index > 0 && index < __impl::vectorElementsCount, "simd_stl::numeric::basic_simd: Index out of range");
        return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
    }

    /**
        * @brief ������� 'value' � ������� 'where' �������
        * @param where ������� ��� �������.
        * @param value �������� ��� �������.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void insert(
        const size_type     where,
        const value_type    value) noexcept
    {
        __impl::insert(_vector, where, value);
    }

    /**
        * @brief ������������ �������� ������� � ����������� �� �������������� ���� � �����
        * @param mask ����� ��� �������������.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void shuffle(basic_simd_mask<_SimdGeneration_, _Element_> mask) noexcept {
        // __impl::shuffle(_vector, mask);
    }

    /**
        * @brief ������� value � ������, ���� ��������������� ��� ����� ������.
        * @param mask �������� �����.
        * @param value �������� ��� �������.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void expand(
        basic_simd_mask<_SimdGeneration_, _Element_>    mask,
        const value_type                                value) noexcept
    {

    }
    
    /**
        * @brief ������������ ������ �� basic_simd<_Element_, _SimdGeneration_> � basic_simd<_OtherElement_, _SimdGeneration_>.
        * ����� ��������� ������ ��� ���������� � �� �������� ����� �� ����� ����������.
        * @return ��������� �����������.
    */
    template <typename _OtherElement_> 
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_OtherElement_, _SimdGeneration_> cast() const noexcept {
        return __impl::cast<
            vector_type,
            type_traits::__deduce_simd_vector_type<_SimdGeneration_, _OtherElement_>
        >(_vector);
    }

    /**
        * @brief ������������ ������ �� basic_simd<_Element_, _SimdGeneration_> � basic_simd<_OtherElement_, _OtherSimdGeneration_>.
        * ����� ��������� ������ ��� ���������� � �� �������� ����� �� ����� ����������. 
        * ������� ����� ���������� �������������� � ����������� ������������.
        * @return ��������� �����������.
    */
    template <
        arch::CpuFeature	_OtherSimdGeneration_,
        typename            _OtherElement_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_OtherElement_, _SimdGeneration_> cast() const noexcept {
        return __impl::cast<
            vector_type,
            type_traits::__deduce_simd_vector_type<_OtherSimdGeneration_, _OtherElement_>
        >(_vector);
    }

    /**
        * @brief ������������ ������ �� basic_simd<_Element_, _SimdGeneration_> � basic_simd<_OtherElement_, _OtherSimdGeneration_>.
        * ���� �� ���������� �������������� � �����������, �� ����� ��������� ������ ��� ���������� � �� �������� ����� �� ����� ����������. 
        * � ��������� ������ ������� ����� ���������� �������������� � ����������� ����������� ������.
        * @return ��������� �����������.
    */
    template <
        arch::CpuFeature	_OtherSimdGeneration_,
        typename            _OtherElement_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_OtherElement_, _SimdGeneration_> safeCast() const noexcept {
        return __impl::cast<
            vector_type,
            type_traits::__deduce_simd_vector_type<_OtherSimdGeneration_, _OtherElement_>
        >(_vector);
    }



    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator+ <>(
        const basic_simd& left, 
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator- <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator* <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator/ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator& <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator| <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator^ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;   
    
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator+() const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator-() const noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline mask_type operator!() const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator~() const noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator+=(const basic_simd& other) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator-=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator*=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator/=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator%=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator=(const basic_simd& left) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ operator[](const size_type index) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd> operator[](const size_type index) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator&=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator|=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator^=(const basic_simd& other) noexcept;
private:
    vector_type _vector;
};

template <
    class _BasicSimdFrom_,
    class _BasicSimdTo_>
constexpr bool is_simd_convertible_v = std::conjunction_v<
        __is_valid_basic_simd_v<_BasicSimdFrom_>,
        __is_valid_basic_simd_v<_BasicSimdTo_>
    >;

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd() noexcept
{
    _vector = __impl::constructZero();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const vector_type& other) noexcept:
    _vector(other)
{}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const value_type value) noexcept {
    _vector = __impl::broadcast(value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::~basic_simd() noexcept
{}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator+=(const basic_simd& other) const noexcept {
    _vector = __impl::add(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator-=(const basic_simd& other) noexcept
{
    _vector = __impl::sub(_vector, other._vector);
    return *this;
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator*=(const basic_simd& other) noexcept {
    _vector = __impl::mul(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator/=(const basic_simd& other) noexcept {
    _vector = __impl::div(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator%=(const basic_simd& other) noexcept {

}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator=(const basic_simd& left) noexcept {
    _vector = left._vector;
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ 
basic_simd<_SimdGeneration_, _Element_>::operator[](const size_type index) const noexcept {
    return __impl::extract(_vector, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>
basic_simd<_SimdGeneration_, _Element_>::operator[](const size_type index) noexcept {
    return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator++(int) noexcept {
    auto& self = *this;
    _vector = __impl::increment(_vector);
    return self;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator++() noexcept {
    _vector = __impl::increment(_vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator--(int) noexcept
{
    auto& self = *this;
    _vector = __impl::decrement(_vector);
    return self;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator--() noexcept 
{
    _vector = __impl::decrement(_vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>::mask_type
basic_simd<_SimdGeneration_, _Element_>::operator!() const noexcept {
    return __impl::convertToMask(__impl::bitwiseNot(_vector));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::operator~() const noexcept {
    return __impl::bitwiseNot(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator&=(const basic_simd& other) noexcept {
    _vector = __impl::bitwiseAnd(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator|=(const basic_simd& other) noexcept {
    _vector = __impl::bitwiseOr(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator^=(const basic_simd& other) noexcept {
    _vector = __impl::bitwiseXor(_vector, other._vector);
    return *this;
}

/**
    * @brief ��������� ������������ ������� ���� ��������.
    * @param left ����� ������-�������.
    * @param right ������ ������-�������.
    * @return ����� ������, ���������� ������� ��������� `left` � `right`.
*/
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator/(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept 
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::div(left._vector, right._vector);
}

/**
    * @brief ��������� ������������ �������� ���� ��������.
    * @param left ����� ������-�������.
    * @param right ������ ������-�������.
    * @return ����� ������, ���������� ����� ��������� `left` � `right`.
*/
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator+(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::add(left._vector, right._vector);
}

/**
    * @brief ��������� ������������ ��������� ���� ��������.
    * @param left ����� ������-�������.
    * @param right ������ ������-�������.
    * @return ����� ������, ���������� �������� ��������� `left` � `right`.
*/
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator-(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::sub(left._vector, right._vector);
}

/**
    * @brief ��������� ������������ ��������� ���� ��������.
    * @param left ����� ������-�������.
    * @param right ������ ������-�������.
    * @return ����� ������, ���������� ������������ ��������� `left` � `right`.
*/
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator*(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept 
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::mul(left._vector, right._vector);
}


/**
    * @brief ��������� ��������� "�" ���� �������� �����������.
    * @param left ����� ������.
    * @param right ������ ������.
    * @return ����� ������ � ����������� ���������� "�" ��������������� ���������.
 */
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator&(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::bitwiseAnd(left._vector, right._vector);
}

/**
    * @brief ��������� ��������� "���" ���� �������� �����������.
    * @param left ����� ������.
    * @param right ������ ������.
    * @return ����� ������ � ����������� ���������� "���" ��������������� ���������.
 */
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator|(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::bitwiseAnd(left._vector, right._vector);
}


/**
    * @brief ��������� ��������� "����������� ���" ���� �������� �����������.
    * @param left ����� ������.
    * @param right ������ ������.
    * @return ����� ������ � ����������� ���������� "����������� ���" ��������������� ���������.
*/
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator^(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::bitwiseXor(left._vector, right._vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator+() const noexcept 
{
    return _vector;
}

/**
    * @brief ������� �����.
    * @return ����� ������ ����� � ��������������� ������.
*/
template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator-() const noexcept 
{
    return __impl::sub(__impl::constructZero(), _vector);
}


__SIMD_STL_NUMERIC_NAMESPACE_END
