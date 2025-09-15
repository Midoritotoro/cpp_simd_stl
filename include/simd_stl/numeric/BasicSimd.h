#pragma once 

#include <simd_stl/numeric/BasicSimdImplementation.h>
#include <simd_stl/numeric/BasicSimdElementReference.h>

#include <src/simd_stl/utility/Assert.h>
#include <xstring> 

#include <tuple>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN


template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_native_mask_load_supported = std::conjunction_v<
    !arch::__is_xmm_v<_SimdGeneration_>,
    arch::__is_ymm_v<_SimdGeneration_>,
    arch::__is_zmm_v<_SimdGeneration_>
>;

template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_native_mask_store_supported = is_native_mask_load_supported<_SimdGeneration_>;



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

    template <typename _OtherType_>
    basic_simd(const basic_simd<_SimdGeneration_, _OtherType_>& other) noexcept {
        using _FromVectorType_ = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _OtherType_>;
        _vector = __impl::template cast<_FromVectorType_, vector_type>(other.unwrap());
    }

    template <
        arch::CpuFeature    _OtherFeature_,
        typename            _OtherType_>
    basic_simd(const basic_simd<_OtherFeature_, _OtherType_>& other) noexcept {
        using _FromVectorType_ = type_traits::__deduce_simd_vector_type<_OtherFeature_, _OtherType_>;
        if constexpr (static_cast<uint8>(_OtherFeature_) > static_cast<uint8>(_SimdGeneration_))
            // Downcast
            _vector = BasicSimdImplementation<_OtherFeature_, _OtherType_>::template cast<_FromVectorType_, vector_type>(other.unwrap());
        else
            _vector = __impl::template cast<_FromVectorType_, vector_type>(other.unwrap());
    }

    ~basic_simd() noexcept;

    /**
        * @brief Извлечение значения из вектора в позиции 'index' с предварительной проверкой границ.
        * @param index Позиция для извлечения.
        * @return Извлечённое значение.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline value_type extract(const size_type index) const noexcept {
        Assert(index > 0 && index < __impl::vectorElementsCount, "simd_stl::numeric::basic_simd: Index out of range");
        return __impl::extract(_vector, index);
    }

    /**
        * @brief Извлечение значения из вектора в позиции 'index' с предварительной проверкой границ.
        * @param index Позиция для извлечения.
        * @return Обёртка над извлеченным значением, позволяющая изменять соответствующий элемент вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd> extractWrapped(const size_type index) noexcept {
        Assert(index > 0 && index < __impl::vectorElementsCount, "simd_stl::numeric::basic_simd: Index out of range");
        return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
    }

    /**
        * @brief Вставка 'value' в позицию 'where' вектора
        * @param where Позиция для вставки.
        * @param value Значение для вставки.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void insert(
        const size_type     where,
        const value_type    value) noexcept
    {
        __impl::insert(_vector, where, value);
    }

    /**
        * @brief Перемешивает элементы вектора в зависимости от установленного бита в маске
        * @param mask Маска для перемешивания.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void shuffle(basic_simd_mask<_SimdGeneration_, _Element_> mask) noexcept {
        // __impl::shuffle(_vector, mask);
    }

    /**
        * @brief Вставка value в вектор, если соответствующий бит маски ложный.
        * @param mask Числовая маска.
        * @param value Значение для вставки.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void expand(
        basic_simd_mask<_SimdGeneration_, _Element_>    mask,
        const value_type                                value) noexcept
    {

    }
    
    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_SimdGeneration_, _OtherElement_>.
        * Метод необходим только для компиляции и не занимает время во время выполнения.
        * @return Результат конвертации.
    */
    template <typename _OtherElement_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _OtherElement_> cast() const noexcept {
        return __impl::template cast<vector_type, type_traits::__deduce_simd_vector_type<_SimdGeneration_, _OtherElement_>>(_vector);
    }

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_OtherSimdGeneration_, _OtherElement_>
        * Метод необходим только для компиляции и не занимает время во время выполнения. 
        * @return Результат конвертации.
    */
    template <
        arch::CpuFeature	_OtherSimdGeneration_,
        typename            _OtherElement_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_OtherSimdGeneration_, _OtherElement_> cast() const noexcept {
        return __impl::template cast<vector_type, type_traits::__deduce_simd_vector_type<_OtherSimdGeneration_, _OtherElement_>>(_vector);
    }

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в _BasicSimdTo_.
        * Если не происходит преобразование с расширением, то метод необходим только для компиляции и не занимает время во время выполнения. 
        * В противном случае старшая часть результата преобразования с расширением заполняется нулями.
        * @return Результат конвертации.
    */
    template <class _BasicSimdTo_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _BasicSimdTo_ safeCast(const basic_simd& from) noexcept {
        static_assert(__is_valid_basic_simd_v<_BasicSimdTo_>,   "_BasicSimdTo_ must be a basic_simd class or a subclass of it");

        using _FromVectorType_   = typename basic_simd::vector_type;
        using _ToVectorType_     = typename _BasicSimdTo_::vector_type;

        if constexpr (static_cast<uint8>(basic_simd::_Generation) > static_cast<uint8>(_BasicSimdTo_::_Generation))
            // Downcast
            return BasicSimdImplementation<
                basic_simd::_Generation,
                typename basic_simd::value_type>::template cast<_FromVectorType_, _ToVectorType_, true>(from._vector);
        else 
            return BasicSimdImplementation<
                _BasicSimdTo_::_Generation,
                typename basic_simd::value_type>::template cast< _FromVectorType_, _ToVectorType_, true>(from._vector);
    }

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_OtherSimdGeneration_, _OtherElement_>
        * Метод необходим только для компиляции и не занимает время во время выполнения. 
        * Старшая часть результата преобразования с расширением неопределена.
        * @return Результат конвертации.
    */
    template <class _BasicSimdTo_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline _BasicSimdTo_ cast(const basic_simd& from) const noexcept {
        static_assert(__is_valid_basic_simd_v<_BasicSimdTo_>,   "_BasicSimdTo_ must be a basic_simd class or a subclass of it");

        using _FromVectorType_   = typename basic_simd::vector_type;
        using _ToVectorType_     = typename _BasicSimdTo_::vector_type;

        if constexpr (static_cast<uint8>(basic_simd::_Generation) > static_cast<uint8>(_BasicSimdTo_::_Generation))
            // Downcast
            return BasicSimdImplementation<
                basic_simd::_Generation,
                typename basic_simd::value_type>::template cast<_FromVectorType_, _ToVectorType_, false>(from._vector);
        else
            return BasicSimdImplementation<
                _BasicSimdTo_::_Generation,
                typename basic_simd::value_type>::template cast< _FromVectorType_, _ToVectorType_, false>(from._vector);
    }


    /**
        * @brief Загружает sizeof(basic_simd<_SimdGeneration_, _Element_>::vector_type) байт из памяти по невыровненному адресу.
        * @param where Указатель на память для загрузки.
        * @return Загруженный вектор.
    */
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd loadUnaligned(const value_type* where) noexcept {
        return __impl::loadUnaligned(where);
    }

    /**
        * @brief Загружает sizeof(basic_simd<_SimdGeneration_, _Element_>::vector_type) байт из памяти по выровненному адресу.
        * @param where Указатель на память для загрузки.
        * @return Загруженный вектор.
    */
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd loadAligned(const value_type* where) noexcept {
        return __impl::loadAligned(where);
    }

    /**
        * @brief Сохраняет вектор в память по невыровненному адресу.
        * @param where Указатель на память для сохранения вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void storeUnaligned(const value_type* where) noexcept {
        __impl::storeUnaligned(where, _vector);
    }

    /**
        * @brief Сохраняет вектор в память по выровненному адресу.
        * @param where Указатель на память для сохранения вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void storeAligned(const value_type* where) noexcept {
        __impl::storeAligned(where, _vector);
    }


    /**
        * @brief Загружает вектор из памяти по невыровненному адресу, используя маску.
        * @param where Указатель на память для загрузки.
        * @param mask Маска.
        * @return Загруженный вектор.
    */
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd maskLoadUnaligned(
        const value_type*   where,
        const mask_type     mask) noexcept 
    {

    }

    /**
        * @brief Загружает вектор из памяти по выровненному адресу, используя маску.
        * @param where Указатель на память для загрузки.
        * @param mask Маска.
        * @return Загруженный вектор.
    */
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd maskLoadAligned(
        const value_type*   where,
        const mask_type     mask) noexcept
    {

    }

    /**
        * @brief Сохраняет элемент вектора в память по невыровненному адресу, если соответствующий бит маски установлен.
        * @param where Указатель на память для сохранения вектора.
        * @param mask Маска.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreUnaligned(
        const mask_type     mask,
        const value_type*   where) noexcept
    {

    }

    /**
        * @brief Сохраняет элемент вектора в память по выровненному адресу, если соответствующий бит маски установлен.
        * @param where Указатель на память для сохранения вектора.
        * @param mask Маска.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreAligned(
        const value_type*   where,
        const mask_type     mask) noexcept
    {

    }

    simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type unwrap() const noexcept {
        return _vector;
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
    * @brief Выполняет поэлементное деление двух векторов.
    * @param left Левый вектор-операнд.
    * @param right Правый вектор-операнд.
    * @return Новый вектор, содержащий частное элементов `left` и `right`.
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
    * @brief Выполняет поэлементное сложение двух векторов.
    * @param left Левый вектор-операнд.
    * @param right Правый вектор-операнд.
    * @return Новый вектор, содержащий сумму элементов `left` и `right`.
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
    * @brief Выполняет поэлементное вычитание двух векторов.
    * @param left Левый вектор-операнд.
    * @param right Правый вектор-операнд.
    * @return Новый вектор, содержащий разность элементов `left` и `right`.
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
    * @brief Выполняет поэлементное умножение двух векторов.
    * @param left Левый вектор-операнд.
    * @param right Правый вектор-операнд.
    * @return Новый вектор, содержащий произведение элементов `left` и `right`.
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
    * @brief Выполняет побитовое "И" двух векторов поэлементно.
    * @param left Левый вектор.
    * @param right Правый вектор.
    * @return Новый вектор с результатом побитового "И" соответствующих элементов.
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
    * @brief Выполняет побитовое "Или" двух векторов поэлементно.
    * @param left Левый вектор.
    * @param right Правый вектор.
    * @return Новый вектор с результатом побитового "Или" соответствующих элементов.
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
    * @brief Выполняет побитовое "Исключающее или" двух векторов поэлементно.
    * @param left Левый вектор.
    * @param right Правый вектор.
    * @return Новый вектор с результатом побитового "Исключающее или" соответствующих элементов.
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
    * @brief Унарный минус.
    * @return Новый вектор чисел с противоположным знаком.
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
