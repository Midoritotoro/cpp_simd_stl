#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <src/simd_stl/utility/Assert.h>

#include <simd_stl/math/BitMath.h>


#if defined(simd_stl_os_win)
#  include <Windows.h>
#endif // defined(simd_stl_os_win)

__SIMD_STL_ARCH_NAMESPACE_BEGIN

class ProcessorInformation {
public:
    simd_stl_nodiscard simd_stl_always_inline static uint32 hardwareConcurrency() noexcept {
        return _processorInformationInternal._logicalProcessors;
    }
private:
    class ProcessorInformationInternal
    {
    public:
        uint32 _logicalProcessors = 0;

        ProcessorInformationInternal() noexcept {
#if defined(simd_stl_os_win)
#  if defined(simd_stl_os_win64)
            constexpr auto length = 48;
#  else 
            constexpr auto length = 44;
#  endif

            uint8 buffer[length];
            uint8* informationBuffer = reinterpret_cast<uint8*>(&buffer);
            
            dword_t bufferLength = length;

            Assert(GetLogicalProcessorInformationEx(RelationProcessorPackage, 
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(informationBuffer), &bufferLength));

            while (bufferLength > 0) {
                const auto information  = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(informationBuffer);
                const auto informationSize = information->Size;

                for (int16 i = 0; i != information->Processor.GroupCount; ++i)
                    _logicalProcessors += math::PopulationCount(information->Processor.GroupMask[i].Mask);

                informationBuffer += informationSize;
                bufferLength -= informationSize;
            }
        }
#endif // defined(simd_stl_os_win)
     
    };

    static inline ProcessorInformationInternal _processorInformationInternal;
};

__SIMD_STL_ARCH_NAMESPACE_END
