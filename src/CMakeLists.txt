# @project optox - GPU Operators to X
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 09.07.2018

project(optoX)

##-----------------------------------------------------------------------------
## Files
##-----------------------------------------------------------------------------

set(COMMON_HEADER
    optox_api.h
)

set(OPERATOR_HEADER
    operators/ioperator.h
    operators/addoperator.h
)

set(OPERATOR_SOURCE
    operators/addoperator.cu
)

set(HEADERS
    #${COMMON_HEADER}
    ${OPERATOR_HEADER}
)

set(SOURCES
    #${COMMON_SOURCE}
    ${OPERATOR_SOURCE}
)

#add_flags(CUDA_NVCC_FLAGS "--ptxas-options=-v")

##-----------------------------------------------------------------------------
## Core lib
##-----------------------------------------------------------------------------
cuda_add_library(optox SHARED ${HEADERS} ${SOURCES} OPTIONS "-DOPTOX_EXPORTS")
set_target_properties(optox PROPERTIES DEFINE_SYMBOL OPTOX_EXPORTS)
target_link_libraries(optox ${IMAGEUTILITIES_LIBRARIES})
