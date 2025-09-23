#pragma once 

#if defined(max) 
#  undef max
#endif

#if defined(min) 
#  undef min
#endif

#include <simd_stl/numeric/BasicSimdImplementation.h>
#include <simd_stl/numeric/BasicSimdElementReference.h>

#include <src/simd_stl/utility/Assert.h>
#include <xstring> 

#include <tuple>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN


template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_native_mask_load_supported_v = std::conjunction_v<
    !arch::__is_xmm_v<_SimdGeneration_>,
    arch::__is_ymm_v<_SimdGeneration_>,
    arch::__is_zmm_v<_SimdGeneration_>
>;

template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_native_mask_store_supported_v = is_native_mask_load_supported_v<_SimdGeneration_>;

template <
    arch::CpuFeature _SimdGenerationFirst_,
    arch::CpuFeature _SimdGenerationSecond_>
constexpr bool is_simd_feature_superior_v = (static_cast<uint8>(_SimdGenerationFirst_) > static_cast<uint8>(_SimdGenerationSecond_));


template <
    class _BasicSimdFrom_,
    class _BasicSimdTo_>
using deduce_superior_basic_simd_type = std::conditional_t<
        is_simd_feature_superior_v<
            _BasicSimdFrom_::_Generation,
            _BasicSimdTo_::_Generation>,
        _BasicSimdFrom_,
        _BasicSimdTo_
    >;
        


/**
    * @class basic_simd
    * @brief Обёртка над SIMD-векторами для различных архитектур CPU.
    * 
    * Предоставляет высокоуровневый интерфейс для векторных вычислений:
    * - арифметика и математика
    * - побитовые операции
    * - перестановки и сдвиги
    * - загрузка/сохранение
    * - утилиты
    * 
    * @tparam _SimdGeneration_ Архитектура SIMD (SSE-SSE4.2, AVX, AVX2, AVX-512 и т.д.).
    * @tparam _Element_ Тип элементов вектора ('int', 'float', 'double' и т.д.). По умолчанию 'int'.
*/
template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_ = int32>
class basic_simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

    friend BasicSimdElementReference;
public:
    using implementation = BasicSimdImplementation<_SimdGeneration_>;
    static constexpr auto _Generation = _SimdGeneration_;

    using value_type    = _Element_;
    using vector_type   = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_>;

    using size_type     = uint8;
    using mask_type     = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_>;

    basic_simd() noexcept;

    
    /**
        * @brief Заполнение вектора значением.
        * @param value Значение, которым будет заполнен вектор.
    */
    basic_simd(const value_type value) noexcept;

    basic_simd(const vector_type& other) noexcept;

    /**
        * @brief Загрузка вектора из памяти по адресу address.
        * @param address Адрес для загрузки.
    */
    basic_simd(const value_type* address) noexcept;

    template <typename _OtherType_>
    basic_simd(const basic_simd<_SimdGeneration_, _OtherType_>& other) noexcept;

    template <
        arch::CpuFeature    _OtherFeature_,
        typename            _OtherType_>
    basic_simd(const basic_simd<_OtherFeature_, _OtherType_>& other) noexcept;

    ~basic_simd() noexcept;

    /**
        * @brief Заполнение вектора значением.
        * @param value Значение, которым будет заполнен вектор.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline void fill(const value_type value) noexcept;

    /**
        * @brief Извлечение значения из вектора в позиции 'index' с предварительной проверкой границ.
        * @param index Позиция для извлечения.
        * @return Извлечённое значение.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline _DesiredType_ extract(const size_type index) const noexcept;

    /**
        * @brief Извлечение значения из вектора в позиции 'index' с предварительной проверкой границ.
        * @param index Позиция для извлечения.
        * @return Обёртка над извлеченным значением, позволяющая изменять соответствующий элемент вектора.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd> extractWrapped(const size_type index) noexcept;

    /**
        * @brief Вставка 'value' в позицию 'where' вектора
        * @param where Позиция для вставки.
        * @param value Значение для вставки.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline void insert(
        const size_type     where,
        const value_type    value) noexcept;

    /**
        * @brief Перемешивает элементы вектора в зависимости от установленных битов в маске
        * @param mask Маска для перемешивания.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline void shuffle(basic_simd_mask<_SimdGeneration_, _Element_> mask) noexcept;

    /**
        * @brief Вставка value в вектор, если соответствующий бит маски установлен.
        * @param mask Числовая маска.
        * @param value Значение для вставки.
    */

    simd_stl_constexpr_cxx20 simd_stl_always_inline void expand(
        basic_simd_mask<_SimdGeneration_, _Element_>    mask,
        const value_type                                value) noexcept;
    
    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_SimdGeneration_, _OtherElement_>.
        * Метод необходим только для компиляции и не занимает время во время выполнения.
        * @return Результат конвертации.
    */
    template <typename _OtherElement_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _OtherElement_> cast() const noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_OtherSimdGeneration_, _OtherElement_>
        * Метод необходим только для компиляции и не занимает время во время выполнения. 
        * @return Результат конвертации.
    */
    template <
        arch::CpuFeature	_OtherSimdGeneration_,
        typename            _OtherElement_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_OtherSimdGeneration_, _OtherElement_> cast() const noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в _BasicSimdTo_.
        * Если не происходит преобразование с расширением, то метод необходим только для компиляции и не занимает время во время выполнения. 
        * В противном случае старшая часть результата преобразования с расширением заполняется нулями.
        * @return Результат конвертации.
    */
    template <class _BasicSimdTo_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _BasicSimdTo_ safeCast(const basic_simd& from) noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_OtherSimdGeneration_, _OtherElement_>
        * Метод необходим только для компиляции и не занимает время во время выполнения. 
        * Старшая часть результата преобразования с расширением неопределена.
        * @return Результат конвертации.
    */
    template <class _BasicSimdTo_>
    simd_stl_constexpr_cxx20 simd_stl_always_inline _BasicSimdTo_ cast(const basic_simd& from) const noexcept;

    /**
        * @brief Загружает sizeof(basic_simd<_SimdGeneration_, _Element_>::vector_type) байт из памяти по невыровненному адресу.
        * @param where Указатель на память для загрузки.
        * @return Загруженный вектор.
    */
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd loadUnaligned(const value_type* where) noexcept;

    /**
        * @brief Загружает sizeof(basic_simd<_SimdGeneration_, _Element_>::vector_type) байт из памяти по выровненному адресу.
        * @param where Указатель на память для загрузки.
        * @return Загруженный вектор.
    */
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd loadAligned(const value_type* where) noexcept;

    /**
        * @brief Сохраняет вектор в память по невыровненному адресу.
        * @param where Указатель на память для сохранения вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void storeUnaligned(value_type* where) noexcept;

    /**
        * @brief Сохраняет вектор в память по выровненному адресу.
        * @param where Указатель на память для сохранения вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline void storeAligned(value_type* where) noexcept;


    /**
        * @brief Загружает вектор из памяти по невыровненному адресу, используя маску.
        * @param where Указатель на память для загрузки.
        * @param mask Маска.
        * @return Загруженный вектор.
    */
    template <typename _DesiredType_ = value_type>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd maskLoadUnaligned(
        const value_type*   where,
        const mask_type     mask) noexcept;

    /**
        * @brief Загружает вектор из памяти по выровненному адресу, используя маску.
        * @param where Указатель на память для загрузки.
        * @param mask Маска.
        * @return Загруженный вектор.
    */
    template <typename _DesiredType_ = value_type>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd maskLoadAligned(
        const value_type*   where,
        const mask_type     mask) noexcept;

    /**
        * @brief Сохраняет элемент вектора в память по невыровненному адресу, если соответствующий бит маски установлен.
        * @param where Указатель на память для сохранения вектора.
        * @param mask Маска.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreUnaligned(
        value_type*     where,
        const mask_type mask) noexcept;

    /**
        * @brief Сохраняет элемент вектора в память по выровненному адресу, если соответствующий бит маски установлен.
        * @param where Указатель на память для сохранения вектора.
        * @param mask Маска.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreAligned(
        value_type*     where,
        const mask_type mask) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type unwrap() const noexcept {
        return _vector;
    }

    /**
        * @brief Выполняет поэлементное сложение двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий сумму элементов `left` и `right`.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator+ <>(
        const basic_simd& left, 
        const basic_simd& right) noexcept;

    /**
        * @brief Выполняет поэлементное вычитание двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий разность элементов `left` и `right`.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator- <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    /**
        * @brief Выполняет поэлементное умножение двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий произведение элементов `left` и `right`.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator* <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    
    /**
        * @brief Выполняет поэлементное деление двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий частное элементов `left` и `right`.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator/ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    /**
        * @brief Выполняет побитовое "И" двух векторов поэлементно.
        * @param left Левый вектор.
        * @param right Правый вектор.
        * @return Новый вектор с результатом побитового "И" соответствующих элементов.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator& <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    /**
        * @brief Выполняет побитовое "Или" двух векторов поэлементно.
        * @param left Левый вектор.
        * @param right Правый вектор.
        * @return Новый вектор с результатом побитового "Или" соответствующих элементов.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator| <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    
    /**
        * @brief Выполняет побитовое "Исключающее или" двух векторов поэлементно.
        * @param left Левый вектор.
        * @param right Правый вектор.
        * @return Новый вектор с результатом побитового "Исключающее или" соответствующих элементов.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator^ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;   
    
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator+() const noexcept;

    /**
        * @brief Унарный минус.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator-() const noexcept;

    /**
        * @brief Инкрементирует каждый элемент вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator++(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++() noexcept;

    /**
        * @brief Декрементирует каждый элемент вектора.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator--(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline mask_type operator!() const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator~() const noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator+=(const basic_simd& other) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator-=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator*=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator/=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator%=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator=(const basic_simd& left) noexcept;

     /**
        * @brief Получает элемент по индексу без проверки границ вектора.
        * @param index Индекс элемента.
        * @return Элемент вектора с типом 'value_type'.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ operator[](const size_type index) const noexcept;

    /**
        * @brief Получает обёртку над элементом вектора по индексу без проверки границ вектора.
        * @param index Индекс элемента.
        * @return Обёртка над элементом вектора с типом 'BasicSimdElementReference<basic_simd>'.
    */
    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd> operator[](const size_type index) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator&=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator|=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator^=(const basic_simd& other) noexcept;

    //    template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd pow(const basic_simd& exp) const noexcept;
    
    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd sqrt() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd rsqrt() const noexcept;
    
    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd exp() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd log() const noexcept;
    
    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd log10() const noexcept;

    //    template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd abs() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd minimum(const basic_simd& other) const noexcept;
    
    //     template <typename _DesiredType_ = value_type>
    //simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd maximum(const basic_simd& other) const noexcept;

    template <typename _ElementType_ = _Element_>
    static constexpr uint8 size() noexcept {
        static_assert(type_traits::__is_vector_type_supported_v<_ElementType_>, "Unsupported element type");
        return (sizeof(vector_type) / sizeof(_ElementType_));
    }
private:
    vector_type _vector;
};

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd() noexcept
{
    _vector = implementation::template constructZero<vector_type>();
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
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const value_type* address) noexcept {
    _vector = loadUnaligned(address).unwrap();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const value_type value) noexcept {
    _vector = fill(value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::~basic_simd() noexcept
{}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void
basic_simd<_SimdGeneration_, _Element_>::fill(const value_type value) noexcept {
    _vector = implementation::template broadcast<vector_type>(value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator+=(const basic_simd& other) const noexcept {
    _vector = implementation::template add<value_type>(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator-=(const basic_simd& other) noexcept
{
    _vector = implementation::template sub<value_type>(_vector, other._vector);
    return *this;
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator*=(const basic_simd& other) noexcept {
    _vector = implementation::template mul<value_type>(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator/=(const basic_simd& other) noexcept {
    _vector = implementation::template div<value_type>(_vector, other._vector);
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
    return implementation::template extract<value_type>(_vector, index);
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
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator++(int) noexcept {
    auto self = *this;
    _vector = implementation::template increment<value_type>(_vector);
    return self;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator++() noexcept {
    _vector = implementation::template increment<value_type>(_vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::operator--(int) noexcept
{
    auto self = *this;
    _vector = implementation::template decrement<value_type>(_vector);
    return self;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator--() noexcept 
{
    _vector = implementation::template decrement<value_type>(_vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>::mask_type
basic_simd<_SimdGeneration_, _Element_>::operator!() const noexcept {
    return implementation::template convertToMask(implementation::template bitwiseNot(_vector));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::operator~() const noexcept {
    return implementation::template bitwiseNot(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator&=(const basic_simd& other) noexcept {
    _vector = implementation::template bitwiseAnd(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator|=(const basic_simd& other) noexcept {
    _vector = implementation::template bitwiseOr(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator^=(const basic_simd& other) noexcept {
    _vector = implementation::template bitwiseXor(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator/(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept 
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template div<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator+(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template add<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator-(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template sub<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator*(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept 
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template mul<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator&(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template bitwiseAnd(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator|(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template bitwiseAnd(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator^(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::implementation
        ::template bitwiseXor(left._vector, right._vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator+() const noexcept 
{
    return _vector;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator-() const noexcept 
{
    return implementation::template unaryMinus<value_type>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _OtherType_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const basic_simd<_SimdGeneration_, _OtherType_>& other) noexcept {
    _vector = safeCast<basic_simd>(other);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    arch::CpuFeature    _OtherFeature_,
    typename            _OtherType_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const basic_simd<_OtherFeature_, _OtherType_>& other) noexcept {
    _vector = safeCast<basic_simd>(other);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _DesiredType_
basic_simd<_SimdGeneration_, _Element_>::extract(const size_type index) const noexcept
{
    Assert(index > 0 && index < implementation::vectorElementsCount, "simd_stl::numeric::basic_simd: Index out of range");
    return implementation::template extract<value_type>(_vector, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>> 
basic_simd<_SimdGeneration_, _Element_>::extractWrapped(const size_type index) noexcept 
{
    Assert(index > 0 && index < implementation::vectorElementsCount, "simd_stl::numeric::basic_simd: Index out of range");
    return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::insert(
    const size_type     where,
    const value_type    value) noexcept
{
    implementation::template insert<value_type>(_vector, where, value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::shuffle(
    basic_simd_mask<_SimdGeneration_, _Element_> mask) noexcept
{
    return implementation::template shuffle<_Element_>(_vector, mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::expand(
    basic_simd_mask<_SimdGeneration_, _Element_>    mask,
    const value_type                                value) noexcept
{

}
    

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _OtherElement_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _OtherElement_> 
basic_simd<_SimdGeneration_, _Element_>::cast() const noexcept
{
    return implementation::template cast<vector_type, 
        type_traits::__deduce_simd_vector_type<_SimdGeneration_, _OtherElement_>>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    arch::CpuFeature	_OtherSimdGeneration_,
    typename            _OtherElement_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_OtherSimdGeneration_, _OtherElement_> 
basic_simd<_SimdGeneration_, _Element_>::cast() const noexcept
{
    return implementation::template cast<vector_type, 
        type_traits::__deduce_simd_vector_type<_OtherSimdGeneration_, _OtherElement_>>(_vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <class _BasicSimdTo_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _BasicSimdTo_ basic_simd<_SimdGeneration_, _Element_>::safeCast(const basic_simd& from) noexcept {
    static_assert(__is_valid_basic_simd_v<_BasicSimdTo_>,   "_BasicSimdTo_ must be a basic_simd class or a subclass of it");
    using _SuperiorBasicSimdType_ = deduce_superior_basic_simd_type<basic_simd, _BasicSimdTo_>;

    return BasicSimdImplementation<_SuperiorBasicSimdType_::_Generation>::template cast<
        typename basic_simd::vector_type,
        typename _BasicSimdTo_::vector_type, true>(from._vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <class _BasicSimdTo_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _BasicSimdTo_
basic_simd<_SimdGeneration_, _Element_>::cast(const basic_simd& from) const noexcept 
{
    static_assert(__is_valid_basic_simd_v<_BasicSimdTo_>,   "_BasicSimdTo_ must be a basic_simd class or a subclass of it");
    using _SuperiorBasicSimdType_ = deduce_superior_basic_simd_type<basic_simd, _BasicSimdTo_>;

    return BasicSimdImplementation<_SuperiorBasicSimdType_::_Generation>::template cast<
        typename basic_simd::vector_type,
        typename _BasicSimdTo_::vector_type, false>(from._vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::loadUnaligned(const value_type* where) noexcept
{
    return implementation::template loadUnaligned<vector_type>(where);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::loadAligned(const value_type* where) noexcept
{
    return implementation::template loadAligned<vector_type>(where);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::storeUnaligned(value_type* where) noexcept {
    implementation::template storeUnaligned(where, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::storeAligned(value_type* where) noexcept {
    implementation::template storeAligned(where, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::maskLoadUnaligned(
    const value_type*   where,
    const mask_type     mask) noexcept 
{
    return implementation::template maskLoadUnaligned(where, mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::maskLoadAligned(
    const value_type*   where,
    const mask_type     mask) noexcept
{
    return implementation::template maskLoadAligned(where, mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::maskStoreUnaligned(
    value_type*     where,
    const mask_type mask) noexcept
{
    implementation::template maskStoreUnaligned(where, mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::maskStoreAligned(
    value_type*     where,
    const mask_type mask) noexcept
{
    implementation::template maskStoreAligned(where, mask, _vector);
}


__SIMD_STL_NUMERIC_NAMESPACE_END
