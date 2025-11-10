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
            dword_t length = 0;
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX processorInformationExtended;

            char* informationBuffer = reinterpret_cast<char*>(&processorInformationExtended);
            Assert(GetLogicalProcessorInformationEx(RelationProcessorPackage, &processorInformationExtended, &length));

            while (length > 0) {
                const auto information  = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(informationBuffer);
                const auto informationSize = information->Size;

                for (int16 i = 0; i != information->Processor.GroupCount; ++i)
                    _logicalProcessors += math::PopulationCount(information->Processor.GroupMask[i].Mask);

                informationBuffer += informationSize;
                length -= informationSize;
            }
        }
#endif // defined(simd_stl_os_win)
     
    };

    static inline ProcessorInformationInternal _processorInformationInternal;
};

__SIMD_STL_ARCH_NAMESPACE_END
