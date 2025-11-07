#pragma once 

#if defined(max) 
#  undef max
#endif

#if defined(min) 
#  undef min
#endif

#include <simd_stl/numeric/BasicSimdElementReference.h>

#include <src/simd_stl/numeric/SimdArithmetic.h>
#include <src/simd_stl/numeric/SimdCompare.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/utility/Assert.h>

#include <simd_stl/numeric/BasicSimdMask.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN



/**
    * @class basic_simd
    * @brief Обёртка над SIMD-векторами для различных архитектур CPU.
    *
    * Предоставляет высокоуровневый интерфейс для векторных вычислений:
    * - арифметика
    * - побитовые операции
    * - перестановки и сдвиги
    * - загрузка/сохранение
    * - приведения типов
    * - вставка и извлечение
    * - проверка поддержки сета инструкций во времени выполнения
    *
    * @tparam _SimdGeneration_ Поколение SIMD (SSE-SSE4.2, AVX, AVX2, AVX-512F и т.д.).
    * @tparam _Element_ Тип элементов вектора ('int', 'float', 'double' и т.д.). По умолчанию 'int'.
*/

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_ = int32>
class basic_simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<std::decay_t<_Element_>>);

    friend BasicSimdElementReference;
public:
    using simdCast          = SimdCast<_SimdGeneration_>;
    using simdConvert       = SimdConvert<_SimdGeneration_>;

    using simdCompare       = SimdCompare<_SimdGeneration_>;
    using simdElementWise   = SimdElementWise<_SimdGeneration_>;

    using simdElementAccess = SimdElementAccess<_SimdGeneration_>;
    using simdMemoryAccess  = SimdMemoryAccess<_SimdGeneration_>;

    using simdArithmetic    = SimdArithmetic<_SimdGeneration_>;


    static constexpr auto _Generation = _SimdGeneration_;

    using value_type    = _Element_;
    using vector_type   = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_>;

    using size_type     = uint32;
    using mask_type     = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_>;

    template <bool _ZeroMemset_ = true>
    basic_simd() noexcept;

    /**
        * @brief Заполнение вектора значением.
        * @param value Значение, которым будет заполнен вектор.
    */
    basic_simd(const value_type value) noexcept;

    basic_simd(vector_type other) noexcept;

    /**
        * @brief Загрузка вектора из памяти по адресу 'address'.
        * @param address Адрес для загрузки.
    */
    basic_simd(const void* address) noexcept;

    template <typename _OtherType_>
    basic_simd(const basic_simd<_SimdGeneration_, _OtherType_>& other) noexcept;

    template <
        arch::CpuFeature    _OtherFeature_,
        typename            _OtherType_>
    basic_simd(const basic_simd<_OtherFeature_, _OtherType_>& other) noexcept;

    ~basic_simd() noexcept;

    /**
        * @brief Выполняет действие конвертации вектора в '_BasicSimdTo_'
        * Например: 
        * basic_simd<SSE2, int32>(1).convert<basic_simd<SSE2, int8>>() ->
        *     basic_simd<SSE2, int8>(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1);
    */
    template <class _BasicSimdTo_>
    simd_stl_always_inline _BasicSimdTo_ convert() const noexcept;

    /**
        * @brief Поддержан ли сет инструкций _SimdGeneration_ на текущей машине
    */
    static simd_stl_always_inline bool isSupported() noexcept;

    /**
        * @brief Заполнение вектора значением.
        * @param value Значение, которым будет заполнен вектор.
    */
    template <typename _DesiredType_>
    simd_stl_always_inline void fill(const typename std::type_identity<_DesiredType_>::type value) noexcept;

    /**
        * @brief Извлечение значения из вектора в позиции 'index' с предварительной проверкой границ.
        * @param index Позиция для извлечения.
        * @return Извлечённое значение.
    */
    template <typename _DesiredType_>
    simd_stl_always_inline _DesiredType_ extract(const size_type index) const noexcept;

    /**
        * @brief Извлечение значения из вектора в позиции 'index' с предварительной проверкой границ.
        * @param index Позиция для извлечения.
        * @return Обёртка над извлеченным значением, позволяющая изменять соответствующий элемент вектора.
    */
    template <typename _DesiredType_>
    simd_stl_always_inline BasicSimdElementReference<basic_simd> extractWrapped(const size_type index) noexcept;

    /**
        * @brief Вставка 'value' в позицию 'where' вектора
        * @param where Позиция для вставки.
        * @param value Значение для вставки.
    */
    template <typename _DesiredType_>
    simd_stl_always_inline void insert(
        const size_type                                         where,
        const typename std::type_identity<_DesiredType_>::type  value) noexcept;

    /**
        * @brief Перемешивает элементы вектора в зависимости от установленных битов в маске
        * @param mask Маска для перемешивания.
    */
    template <
        typename    _DesiredType_,
        uint8 ...   _Indices_>
    simd_stl_always_inline void permute() noexcept;

    /**
        * @brief Вставка value в вектор, если соответствующий бит маски установлен.
        * @param mask Числовая маска.
        * @param value Значение для вставки.
    */

    simd_stl_always_inline void expand(
        basic_simd_mask<_SimdGeneration_, _Element_>    mask,
        const value_type                                value) noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_SimdGeneration_, _OtherElement_>.
        * Метод необходим только для компиляции и не занимает время во время выполнения.
        * @return Результат конвертации.
    */
    template <typename _OtherElement_>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _OtherElement_> bitcast() const noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_OtherSimdGeneration_, _OtherElement_>
        * Метод необходим только для компиляции и не занимает время во время выполнения.
        * @return Результат конвертации.
    */
    template <
        arch::CpuFeature	_OtherSimdGeneration_,
        typename            _OtherElement_>
    simd_stl_always_inline basic_simd<_OtherSimdGeneration_, _OtherElement_> bitcast() const noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в _BasicSimdTo_.
        * Если не происходит преобразование с расширением, то метод необходим только для компиляции и не занимает время во время выполнения.
        * В противном случае старшая часть результата преобразования с расширением заполняется нулями.
        * @return Результат конвертации.
    */
    template <class _BasicSimdTo_>
    static simd_stl_always_inline _BasicSimdTo_ safeBitcast(const basic_simd& from) noexcept;

    /**
        * @brief Конвертирует вектор из basic_simd<_SimdGeneration_, _Element_> в basic_simd<_OtherSimdGeneration_, _OtherElement_>
        * Метод необходим только для компиляции и не занимает время во время выполнения.
        * Старшая часть результата преобразования с расширением неопределена.
        * @return Результат конвертации.
    */
    template <class _BasicSimdTo_>
    simd_stl_always_inline _BasicSimdTo_ bitcast(const basic_simd& from) const noexcept;

    /**
        * @brief Загружает sizeof(basic_simd<_SimdGeneration_, _Element_>::vector_type) байт из памяти по невыровненному адресу.
        * @param where Указатель на память для загрузки.
        * @return Загруженный вектор.
    */
    static simd_stl_always_inline basic_simd loadUnaligned(const void* where) noexcept;

    /**
        * @brief Загружает sizeof(basic_simd<_SimdGeneration_, _Element_>::vector_type) байт из памяти по выровненному адресу.
        * @param where Указатель на память для загрузки.
        * @return Загруженный вектор.
    */
    static simd_stl_always_inline basic_simd loadAligned(const void* where) noexcept;

    /**
        * @brief Сохраняет вектор в память по невыровненному адресу.
        * @param where Указатель на память для сохранения вектора.
    */
    simd_stl_always_inline void storeUnaligned(void* where) const noexcept;

    /**
        * @brief Сохраняет вектор в память по выровненному адресу.
        * @param where Указатель на память для сохранения вектора.
    */
    simd_stl_always_inline void storeAligned(void* where) const noexcept;

    static simd_stl_always_inline basic_simd loadUpperHalf(const void* where) noexcept;
    static simd_stl_always_inline basic_simd loadLowerHalf(const void* where) noexcept;

    /**
        * @brief Загружает вектор из памяти по невыровненному адресу, используя маску.
        * @param where Указатель на память для загрузки.
        * @param mask Маска.
        * @return Загруженный вектор.
    */
    template <typename _DesiredType_ = value_type>
    static simd_stl_always_inline basic_simd maskLoadUnaligned(
        const void*                                                                     where,
        const type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>     mask) noexcept;

    /**
        * @brief Загружает вектор из памяти по выровненному адресу, используя маску.
        * @param where Указатель на память для загрузки.
        * @param mask Маска.
        * @return Загруженный вектор.
    */
    template <typename _DesiredType_ = value_type>
    static simd_stl_always_inline basic_simd maskLoadAligned(
        const void*                                                                 where,
        const type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_> mask) noexcept;

    /**
        * @brief Сохраняет элемент вектора в память по невыровненному адресу, если соответствующий бит маски установлен.
        * @param where Указатель на память для сохранения вектора.
        * @param mask Маска.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskStoreUnaligned(
        void*     where,
        const mask_type mask) const noexcept;

    /**
        * @brief Сохраняет элемент вектора в память по выровненному адресу, если соответствующий бит маски установлен.
        * @param where Указатель на память для сохранения вектора.
        * @param mask Маска.
    */
    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskStoreAligned(
        void* where,
        const mask_type mask) const noexcept;


    /** 
        * @brief Деление вектора на константу времени компиляции.     
        * @tparam _Divisor_ делитель
    */
    template <_Element_ _Divisor_>
    simd_stl_always_inline void divideByConst() noexcept;

    /** 
        * @brief Умножение вектора на константу времени компиляции.     
        * @tparam _Divisor_ множитель
    */
    template <_Element_ _Divisor_>
    simd_stl_always_inline void multiplyByConst() noexcept;

    simd_stl_always_inline vector_type unwrap() const noexcept {
        return _vector;
    }

    /*
        Операторы с числами
    */

    simd_stl_always_inline friend basic_simd operator+<>(
        const basic_simd&   left,
        const value_type    right) noexcept;

    simd_stl_always_inline friend basic_simd operator-<>(
        const basic_simd&   left,
        const value_type    right) noexcept;

    simd_stl_always_inline friend basic_simd operator*<>(
        const basic_simd&   left,
        const value_type    right) noexcept;
  
    simd_stl_always_inline friend basic_simd operator/<>(
        const basic_simd&   left,
        const value_type    right) noexcept;

    /*
        Операторы с constexpr-числами
    */

    template <_Element_ _Value_>
    simd_stl_always_inline friend basic_simd operator-<>(
        const basic_simd&                           left,
        std::integral_constant<_Element_, _Value_>  right) noexcept
    {
        return basic_simd::simdArithmetic::template sub<_Element_>(left._vector,
            basic_simd::simdElementAccess::template broadcast<typename basic_simd::vector_type>(right));
    }

    template <_Element_ _Divisor_>
    simd_stl_always_inline friend basic_simd operator/<>(
        const basic_simd& left,
        std::integral_constant<_Element_, _Divisor_>  right) noexcept
    {
        return simdArithmetic::template divideByConst<_Element_, _Divisor_, basic_simd::vector_type>(left._vector);
    }


    /*
        Операторы с векторами
    */

    /**
        * @brief Выполняет поэлементное сложение двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий сумму элементов `left` и `right`.
    */

    /**
        * @brief Выполняет поэлементное сложение двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий сумму элементов `left` и `right`.
    */
    simd_stl_always_inline friend basic_simd operator+ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    /**
        * @brief Выполняет поэлементное вычитание двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий разность элементов `left` и `right`.
    */
    simd_stl_always_inline friend basic_simd operator- <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    /**
        * @brief Выполняет поэлементное умножение двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий произведение элементов `left` и `right`.
    */
    simd_stl_always_inline friend basic_simd operator* <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    /**
        * @brief Выполняет поэлементное деление двух векторов.
        * @param left Левый вектор-операнд.
        * @param right Правый вектор-операнд.
        * @return Новый вектор, содержащий частное элементов `left` и `right`.
    */
    simd_stl_always_inline friend basic_simd operator/ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    /**
        * @brief Выполняет побитовое "И" двух векторов поэлементно.
        * @param left Левый вектор.
        * @param right Правый вектор.
        * @return Новый вектор с результатом побитового "И" соответствующих элементов.
    */
    simd_stl_always_inline friend basic_simd operator& <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    /**
        * @brief Выполняет побитовое "Или" двух векторов поэлементно.
        * @param left Левый вектор.
        * @param right Правый вектор.
        * @return Новый вектор с результатом побитового "Или" соответствующих элементов.
    */
    simd_stl_always_inline friend basic_simd operator| <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    /**
        * @brief Выполняет побитовое "Исключающее или" двух векторов поэлементно.
        * @param left Левый вектор.
        * @param right Правый вектор.
        * @return Новый вектор с результатом побитового "Исключающее или" соответствующих элементов.
    */
    simd_stl_always_inline friend basic_simd operator^ <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_always_inline friend basic_simd operator>> <>(
        const basic_simd left,
        const uint32 shift) noexcept;

    simd_stl_always_inline friend basic_simd operator<< <>(
        const basic_simd left,
        const uint32 shift) noexcept;




    simd_stl_always_inline basic_simd operator+() const noexcept;

    /**
        * @brief Унарный минус.
    */
    simd_stl_always_inline basic_simd operator-() const noexcept;

    /**
        * @brief Инкрементирует каждый элемент вектора.
    */
    simd_stl_always_inline basic_simd operator++(int) noexcept;
    simd_stl_always_inline basic_simd& operator++() noexcept;

    simd_stl_always_inline basic_simd& operator>>=(const uint32 shift) noexcept;
    simd_stl_always_inline basic_simd& operator<<=(const uint32 shift) noexcept;
    
    /**
        * @brief Декрементирует каждый элемент вектора.
    */
    simd_stl_always_inline basic_simd operator--(int) noexcept;
    simd_stl_always_inline basic_simd& operator--() noexcept;

    simd_stl_always_inline mask_type operator!() const noexcept;
    simd_stl_always_inline basic_simd operator~() const noexcept;


    simd_stl_always_inline basic_simd& operator+=(const basic_simd& other) const noexcept;
    simd_stl_always_inline basic_simd& operator-=(const basic_simd& other) noexcept;

    simd_stl_always_inline basic_simd& operator*=(const basic_simd& other) noexcept;
    simd_stl_always_inline basic_simd& operator/=(const basic_simd& other) noexcept;


    simd_stl_always_inline basic_simd& operator%=(const basic_simd& other) noexcept;

    simd_stl_always_inline basic_simd& operator=(const basic_simd& left) noexcept;

    /**
       * @brief Получает элемент по индексу без проверки границ вектора.
       * @param index Индекс элемента.
       * @return Элемент вектора с типом 'value_type'.
   */
    simd_stl_always_inline _Element_ operator[](const size_type index) const noexcept;

    /**
        * @brief Получает обёртку над элементом вектора по индексу без проверки границ вектора.
        * @param index Индекс элемента.
        * @return Обёртка над элементом вектора с типом 'BasicSimdElementReference<basic_simd>'.
    */
    simd_stl_always_inline BasicSimdElementReference<basic_simd> operator[](const size_type index) noexcept;


    simd_stl_always_inline basic_simd& operator&=(const basic_simd& other) noexcept;
    simd_stl_always_inline basic_simd& operator|=(const basic_simd& other) noexcept;
    simd_stl_always_inline basic_simd& operator^=(const basic_simd& other) noexcept;

    simd_stl_always_inline friend bool operator== <>(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    

    //    template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd pow(const basic_simd& exp) const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd sqrt() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd rsqrt() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd exp() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd log() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd log10() const noexcept;

    //    template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd abs() const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd minimum(const basic_simd& other) const noexcept;

    //     template <typename _DesiredType_ = value_type>
    //simd_stl_always_inline basic_simd maximum(const basic_simd& other) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline bool isEqual(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> maskNotEqual(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> maskEqual(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> maskGreater(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> maskLess(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> maskGreaterEqual(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> maskLessEqual(const basic_simd& right) const noexcept;

    
    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> notEqual(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> equal(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> greater(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> less(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> greaterEqual(const basic_simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> lessEqual(const basic_simd& right) const noexcept;


    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> toMask() const noexcept;

    template <
        typename _DesiredOutputType_,
        typename _DesiredType_ = value_type>
    simd_stl_always_inline _DesiredOutputType_ reduce() const noexcept;

    template <typename _ElementType_ = _Element_>
    static constexpr int width() noexcept;

    template <typename _ElementType_ = _Element_>
    static constexpr int size() noexcept;

    template <typename _ElementType_ = _Element_>
    static constexpr int length() noexcept;

    static constexpr int registersCount() noexcept;

    simd_stl_always_inline static void streamingFence() noexcept;
    
    // Если _SimdGeneration_ не поддерживает потоковые загрузки/сохранения, то они будут заменены
    // на обычные выровненные загрузки/сохранения

    simd_stl_always_inline static basic_simd nonTemporalLoad(const void* where) noexcept;
    simd_stl_always_inline void nonTemporalStore(void* where) const noexcept;

    static simd_stl_always_inline void zeroUpper() noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_* compressStoreLowerHalf(
        void* where,
        type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_* compressStoreUpperHalf(
        void* where,
        type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept;

    template <typename _DesiredType_ = _Element_> 
    simd_stl_always_inline _DesiredType_* compressStoreUnaligned(
        void*                                                                   where,
        type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept;

    template <typename _DesiredType_ = _Element_> 
    simd_stl_always_inline _DesiredType_* compressStoreAligned(
        void*                                                                   where,
        type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept;

    template <typename _DesiredType_ = _Element_> 
    simd_stl_always_inline _DesiredType_* compressStoreMergeUnaligned(
        void*                                                                   where,
        type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask,
        const basic_simd&                                                       source) const noexcept;

    template <typename _DesiredType_ = _Element_> 
    simd_stl_always_inline _DesiredType_* compressStoreMergeAligned(
        void*                                                                   where,
        type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask,
        const basic_simd&                                                       source) const noexcept;

    template <
        sizetype _Mask,
        typename _DesiredType_ = _Element_>
    simd_stl_always_inline void blend(const basic_simd& vector) noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline void reverse() noexcept;
private:
    vector_type _vector;
};


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <bool _ZeroMemset_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd() noexcept
{
    if constexpr (_ZeroMemset_)
        _vector = simdElementAccess::template constructZero<vector_type>();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(vector_type other) noexcept:
    _vector(other)
{}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const void* address) noexcept {
    _vector = loadUnaligned(address).unwrap();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const value_type value) noexcept {
    fill<value_type>(value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _OtherType_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const basic_simd<_SimdGeneration_, _OtherType_>& other) noexcept {
    _vector = basic_simd<_SimdGeneration_, _OtherType_>::template safeBitcast<basic_simd>(other).unwrap();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    arch::CpuFeature    _OtherFeature_,
    typename            _OtherType_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const basic_simd<_OtherFeature_, _OtherType_>& other) noexcept {
    _vector = basic_simd<_SimdGeneration_, _OtherType_>::template safeBitcast<basic_simd>(other).unwrap();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_>::~basic_simd() noexcept
{}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <class _BasicSimdTo_>
simd_stl_always_inline _BasicSimdTo_ basic_simd<_SimdGeneration_, _Element_>::convert() const noexcept {
    return {};
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline bool basic_simd<_SimdGeneration_, _Element_>::isSupported() noexcept {
    return arch::ProcessorFeatures::isSupported<_SimdGeneration_>();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline void
basic_simd<_SimdGeneration_, _Element_>::fill(const typename std::type_identity<_DesiredType_>::type value) noexcept {
    _vector = simdElementAccess::template broadcast<vector_type, _DesiredType_>(value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator+=(const basic_simd& other) const noexcept {
    _vector = simdArithmetic::template add<value_type>(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator-=(const basic_simd& other) noexcept
{
    _vector = simdArithmetic::template sub<value_type>(_vector, other._vector);
    return *this;
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator*=(const basic_simd& other) noexcept {
    _vector = simdArithmetic::template mul<value_type>(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator/=(const basic_simd& other) noexcept {
    _vector = simdArithmetic::template div<value_type>(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator%=(const basic_simd& other) noexcept {

}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator=(const basic_simd& left) noexcept {
    _vector = left._vector;
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline _Element_ 
basic_simd<_SimdGeneration_, _Element_>::operator[](const size_type index) const noexcept {
    return simdElementAccess::template extract<value_type>(_vector, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>
basic_simd<_SimdGeneration_, _Element_>::operator[](const size_type index) noexcept {
    return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator++(int) noexcept {
    auto self = *this;
    _vector = simdArithmetic::template increment<value_type>(_vector);
    return self;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator++() noexcept {
    _vector = simdArithmetic::template increment<value_type>(_vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::operator--(int) noexcept
{
    auto self = *this;
    _vector = simdArithmetic::template decrement<value_type>(_vector);
    return self;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator--() noexcept 
{
    _vector = simdArithmetic::template decrement<value_type>(_vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>::mask_type
basic_simd<_SimdGeneration_, _Element_>::operator!() const noexcept {
    // return implementation::template convertToMask(simdArithmetic::template bitwiseNot(_vector)); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::operator~() const noexcept {
    return simdArithmetic::template bitwiseNot(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator&=(const basic_simd& other) noexcept {
    _vector = simdArithmetic::template bitwiseAnd(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator|=(const basic_simd& other) noexcept {
    _vector = simdArithmetic::template bitwiseOr(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator^=(const basic_simd& other) noexcept {
    _vector = simdArithmetic::template bitwiseXor(_vector, other._vector);
    return *this;
}

//template <
//    arch::CpuFeature    _SimdGeneration_,
//    typename            _Element_>
//template <_Element_ _Value_>
//simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator-(
//    const basic_simd<_SimdGeneration_, _Element_>&  left,
//    std::integral_constant<_Element_, _Value_>      right) noexcept
//{
//    return imp
//}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator/(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template div<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator+(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template add<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator-(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template sub<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator*(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template mul<_Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator&(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template bitwiseAnd(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator|(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template bitwiseAnd(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator^(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic
        ::template bitwiseXor(left._vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator+(
    const basic_simd<_SimdGeneration_, _Element_>&                      left,
    const typename basic_simd<_SimdGeneration_, _Element_>::value_type  right) noexcept
{
    using _LeftType_ = basic_simd<_SimdGeneration_, _Element_>;
    return _LeftType_::simdArithmetic::template add<_Element_>(
        left._vector, _LeftType_::simdElementAccess::template broadcast<typename _LeftType_::vector_type>(right));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator-(
    const basic_simd<_SimdGeneration_, _Element_>&                      left,
    const typename basic_simd<_SimdGeneration_, _Element_>::value_type  right) noexcept
{
    using _LeftType_ = basic_simd<_SimdGeneration_, _Element_>;
    return _LeftType_::simdArithmetic::template sub<_Element_>(
        left._vector, _LeftType_::simdElementAccess::template broadcast<typename _LeftType_::vector_type>(right));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator*(
    const basic_simd<_SimdGeneration_, _Element_>&                      left,
    const typename basic_simd<_SimdGeneration_, _Element_>::value_type  right) noexcept
{
    using _LeftType_ = basic_simd<_SimdGeneration_, _Element_>;
    return _LeftType_::simdArithmetic::template mul<_Element_>(
        left._vector, _LeftType_::simdElementAccess::template broadcast<typename _LeftType_::vector_type>(right));
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator/(
    const basic_simd<_SimdGeneration_, _Element_>&                      left,
    const typename basic_simd<_SimdGeneration_, _Element_>::value_type  right) noexcept
{
    using _LeftType_ = basic_simd<_SimdGeneration_, _Element_>;
    return _LeftType_::simdArithmetic::template div<_Element_>(
        left._vector, _LeftType_::simdElementAccess::template broadcast<typename _LeftType_::vector_type>(right));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator+() const noexcept 
{
    return _vector;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::operator-() const noexcept 
{
    return simdArithmetic::template unaryMinus<value_type>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_
basic_simd<_SimdGeneration_, _Element_>::extract(const size_type index) const noexcept
{
    DebugAssert(index >= 0 && index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return simdElementAccess::template extract<_DesiredType_>(_vector, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>> 
basic_simd<_SimdGeneration_, _Element_>::extractWrapped(const size_type index) noexcept 
{
    DebugAssert(index >= 0 && index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::insert(
    const size_type                                         where,
    const typename std::type_identity<_DesiredType_>::type  value) noexcept
{
    simdElementAccess::template insert<_DesiredType_>(_vector, where, value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    typename    _DesiredType_,
    uint8 ...   _Indices_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::permute() noexcept
{
    return simdElementWise::template permute<_DesiredType_, _Indices_...>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::expand(
    basic_simd_mask<_SimdGeneration_, _Element_>    mask,
    const value_type                                value) noexcept
{

}   

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _OtherElement_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _OtherElement_> 
basic_simd<_SimdGeneration_, _Element_>::bitcast() const noexcept
{
    return simdCast::template cast<vector_type, 
        type_traits::__deduce_simd_vector_type<_SimdGeneration_, _OtherElement_>>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    arch::CpuFeature	_OtherSimdGeneration_,
    typename            _OtherElement_>
simd_stl_always_inline basic_simd<_OtherSimdGeneration_, _OtherElement_> 
basic_simd<_SimdGeneration_, _Element_>::bitcast() const noexcept
{
    return simdCast::template cast<vector_type, 
        type_traits::__deduce_simd_vector_type<_OtherSimdGeneration_, _OtherElement_>>(_vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <class _BasicSimdTo_>
simd_stl_always_inline _BasicSimdTo_ basic_simd<_SimdGeneration_, _Element_>::safeBitcast(const basic_simd& from) noexcept {
    static_assert(__is_valid_basic_simd_v<_BasicSimdTo_>,   "_BasicSimdTo_ must be a basic_simd class or a subclass of it");
    using _SuperiorBasicSimdType_ = type_traits::deduce_superior_basic_simd_type<basic_simd, _BasicSimdTo_>;

    return SimdCast<_SuperiorBasicSimdType_::_Generation>::template cast<
        typename basic_simd::vector_type,
        typename _BasicSimdTo_::vector_type, true>(from._vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <class _BasicSimdTo_>
simd_stl_always_inline _BasicSimdTo_
basic_simd<_SimdGeneration_, _Element_>::bitcast(const basic_simd& from) const noexcept
{
    static_assert(__is_valid_basic_simd_v<_BasicSimdTo_>,   "_BasicSimdTo_ must be a basic_simd class or a subclass of it");
    using _SuperiorBasicSimdType_ = type_traits::deduce_superior_basic_simd_type<basic_simd, _BasicSimdTo_>;

    return SimdCast<_SuperiorBasicSimdType_::_Generation>::template cast<
        typename basic_simd::vector_type,
        typename _BasicSimdTo_::vector_type, false>(from._vector);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::loadUnaligned(const void* where)  noexcept
{
    return simdMemoryAccess::template loadUnaligned<vector_type>(reinterpret_cast<const value_type*>(where));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::loadAligned(const void* where) noexcept
{
    return simdMemoryAccess::template loadAligned<vector_type>(reinterpret_cast<const value_type*>(where));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::storeUnaligned(void* where) const noexcept {
    simdMemoryAccess::template storeUnaligned(reinterpret_cast<value_type*>(where), _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::storeAligned(void* where) const noexcept {
    simdMemoryAccess::template storeAligned(reinterpret_cast<value_type*>(where), _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_> basic_simd<_SimdGeneration_, _Element_>::loadUpperHalf(const void* where) noexcept {
    return simdMemoryAccess::template loadUpperHalf<vector_type>(where);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_> basic_simd<_SimdGeneration_, _Element_>::loadLowerHalf(const void* where) noexcept {
    return simdMemoryAccess::template loadLowerHalf<vector_type>(where);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::maskLoadUnaligned(
    const void*                                                                 where,
    const type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_> mask) noexcept
{
    return simdMemoryAccess::template maskLoadUnaligned<vector_type>(reinterpret_cast<const _DesiredType_*>(where), mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> 
basic_simd<_SimdGeneration_, _Element_>::maskLoadAligned(
    const void*                                                                 where,
    const type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_> mask) noexcept
{
    return simdMemoryAccess::template maskLoadAligned<vector_type>(reinterpret_cast<const _DesiredType_*>(where), mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::maskStoreUnaligned(
    void*           where,
    const mask_type mask) const noexcept
{
    simdMemoryAccess::template maskStoreUnaligned(reinterpret_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::maskStoreAligned(
    void*           where,
    const mask_type mask) const noexcept
{
    simdMemoryAccess::template maskStoreAligned(reinterpret_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <_Element_ _Divisor_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::divideByConst() noexcept {
    _vector = simdArithmetic::template divideByConst<value_type, _Divisor_, vector_type>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <_Element_ _Divisor_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_>::multiplyByConst() noexcept {
    
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator>>(
    const basic_simd<_SimdGeneration_, _Element_>   left,
    const uint32                                    shift) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic::template shiftRight
        <typename basic_simd<_SimdGeneration_, _Element_>::value_type>(left._vector, shift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator<<(
    const basic_simd<_SimdGeneration_, _Element_>   left,
    const uint32                                    shift) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::simdArithmetic::template shiftLeft
        <typename basic_simd<_SimdGeneration_, _Element_>::value_type>(left._vector, shift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator>>=(const uint32 shift) noexcept {
    _vector = basic_simd<_SimdGeneration_, _Element_>::simdArithmetic::template shiftRight
        <typename basic_simd<_SimdGeneration_, _Element_>::value_type>(_vector, shift);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator<<=(const uint32 shift) noexcept {
    _vector = basic_simd<_SimdGeneration_, _Element_>::simdArithmetic::template shiftLeft
        <typename basic_simd<_SimdGeneration_, _Element_>::value_type>(_vector, shift);
    return *this;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
simd_stl_always_inline bool operator==(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return left.isEqual(right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline bool basic_simd<_SimdGeneration_, _Element_>::isEqual(const basic_simd& right) const noexcept {
    const auto mask = basic_simd::simdCompare::template compare<_DesiredType_, type_traits::equal_to<>, vector_type>(_vector, right._vector);
    return (toMask<_DesiredType_>(mask) == math::MaximumIntegralLimit<decltype(mask)>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_>
    basic_simd<_SimdGeneration_, _Element_>::maskNotEqual(const basic_simd& right) const noexcept
{
    const auto mask = basic_simd::simdCompare::template compare
        <_DesiredType_, type_traits::not_equal_to<>, vector_type>(_vector, right._vector);

    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(basic_simd(mask).toMask<_DesiredType_>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::maskEqual(const basic_simd& right) const noexcept
{
    const auto mask = basic_simd::simdCompare::template compare
        <_DesiredType_, type_traits::equal_to<>, vector_type>(_vector, right._vector);

    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(basic_simd(mask).toMask<_DesiredType_>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::maskGreater(const basic_simd& right) const noexcept
{
    const auto mask = basic_simd::simdCompare::template compare
        <_DesiredType_, type_traits::greater<>, vector_type>(_vector, right._vector);

    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(basic_simd(mask).toMask<_DesiredType_>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::maskLess(const basic_simd& right) const noexcept
{
    const auto mask = basic_simd::simdCompare::template compare
        <_DesiredType_, type_traits::less<>, vector_type>(_vector, right._vector);

    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(basic_simd(mask).toMask<_DesiredType_>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::maskGreaterEqual(const basic_simd& right) const noexcept 
{
    const auto mask = basic_simd::simdCompare::template compare
        <_DesiredType_, type_traits::greater_equal<>, vector_type>(_vector, right._vector);

    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(basic_simd(mask).toMask<_DesiredType_>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::maskLessEqual(const basic_simd& right) const noexcept
{
    const auto mask = basic_simd::simdCompare::template compare
        <_DesiredType_, type_traits::less_equal<>, vector_type>(_vector, right._vector);

    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(basic_simd(mask).toMask<_DesiredType_>());
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::notEqual(const basic_simd& right) const noexcept 
{ 
    return simdCompare::template compare<_DesiredType_, type_traits::not_equal_to<>, vector_type>(_vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::equal(const basic_simd& right) const noexcept
{ 
    return simdCompare::template compare<_DesiredType_, type_traits::equal_to<>, vector_type>(_vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::greater(const basic_simd& right) const noexcept
{ 
    return simdCompare::template compare<_DesiredType_, type_traits::greater<>, vector_type>(_vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::less(const basic_simd& right) const noexcept 
{ 
    return simdCompare::template compare<_DesiredType_, type_traits::less<>, vector_type>(_vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::greaterEqual(const basic_simd& right) const noexcept 
{ 
    return simdCompare::template compare<_DesiredType_, type_traits::greater_equal<>, vector_type>(_vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::lessEqual(const basic_simd& right) const noexcept 
{ 
    return simdCompare::template compare<_DesiredType_, type_traits::less_equal<>, vector_type>(_vector, right._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_> 
    basic_simd<_SimdGeneration_, _Element_>::toMask() const noexcept 
{ 
    const auto mask = simdConvert::template convertToMask<_DesiredType_>(_vector);
    return basic_simd_mask<_SimdGeneration_, _DesiredType_>(mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    typename _DesiredOutputType_,
    typename _DesiredType_>
simd_stl_always_inline _DesiredOutputType_ basic_simd<_SimdGeneration_, _Element_>::reduce() const noexcept {
    return simdArithmetic::template reduce<_DesiredType_, _DesiredOutputType_>(_vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _ElementType_>
constexpr int basic_simd<_SimdGeneration_, _Element_>::width() noexcept {
    static_assert(type_traits::__is_vector_type_supported_v<_ElementType_>, "Unsupported element type");

    constexpr auto width = sizeof(vector_type);
    return width;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _ElementType_>
constexpr int basic_simd<_SimdGeneration_, _Element_>::size() noexcept {
    static_assert(type_traits::__is_vector_type_supported_v<_ElementType_>, "Unsupported element type");

    constexpr auto length = (sizeof(vector_type) / sizeof(_ElementType_));
    return length;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _ElementType_>
constexpr int basic_simd<_SimdGeneration_, _Element_>::length() noexcept {
    return size<_ElementType_>();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
constexpr int basic_simd<_SimdGeneration_, _Element_>::registersCount() noexcept {
    if      constexpr (arch::__is_xmm_v<_SimdGeneration_>)
        return 8;
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>)
        return 16;
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>)
        return 32;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
void basic_simd<_SimdGeneration_, _Element_>::streamingFence() noexcept {
    return simdMemoryAccess::streamingFence();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
basic_simd<_SimdGeneration_, _Element_> basic_simd<_SimdGeneration_, _Element_>::nonTemporalLoad(const void* where) noexcept {
    return simdMemoryAccess::nonTemporalLoad(where);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
void basic_simd<_SimdGeneration_, _Element_>::nonTemporalStore(void* where) const noexcept {
    return simdMemoryAccess::nonTemporalStore(where);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
void basic_simd<_SimdGeneration_, _Element_>::zeroUpper() noexcept {
    if constexpr (type_traits::is_zeroupper_required_v<_SimdGeneration_>)
        _mm256_zeroupper();
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
_DesiredType_* basic_simd<_SimdGeneration_, _Element_>::compressStoreLowerHalf(
    void*                                                                   where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept 
{
    return simdMemoryAccess::template compressStoreLowerHalf<_DesiredType_>(static_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
_DesiredType_* basic_simd<_SimdGeneration_, _Element_>::compressStoreUpperHalf(
    void*                                                                   where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept 
{
    return simdMemoryAccess::template compressStoreUpperHalf<_DesiredType_>(static_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_> 
_DesiredType_* basic_simd<_SimdGeneration_, _Element_>::compressStoreUnaligned(
    void*                                                                   where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept
{
    return simdMemoryAccess::template compressStoreUnaligned<_DesiredType_>(static_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_> 
_DesiredType_* basic_simd<_SimdGeneration_, _Element_>::compressStoreAligned(
    void*                                                                   where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask) const noexcept
{
    return simdMemoryAccess::template compressStoreAligned<_DesiredType_>(static_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_> 
_DesiredType_* basic_simd<_SimdGeneration_, _Element_>::compressStoreMergeUnaligned(
    void*                                                                   where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask,
    const basic_simd&                                                       source) const noexcept
{
    return simdMemoryAccess::template compressStoreMergeAligned<_DesiredType_>(static_cast<_DesiredType_*>(where), mask, _vector, source._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_> 
_DesiredType_* basic_simd<_SimdGeneration_, _Element_>::compressStoreMergeAligned(
    void*                                                                   where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_, _DesiredType_>   mask,
    const basic_simd&                                                       source) const noexcept
{
    return simdMemoryAccess::template compressStoreMergeAligned<_DesiredType_>(static_cast<_DesiredType_*>(where), mask, _vector, source._vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <
    sizetype _Mask,
    typename _DesiredType_>
void basic_simd<_SimdGeneration_, _Element_>::blend(const basic_simd& vector) noexcept {

}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
template <typename _DesiredType_>
void basic_simd<_SimdGeneration_, _Element_>::reverse() noexcept {
    _vector = simdElementWise::template reverse<_DesiredType_>(_vector);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
