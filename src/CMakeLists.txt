# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

project(optoX)

##-----------------------------------------------------------------------------
## Operator Files
##-----------------------------------------------------------------------------

set(COMMON_HEADER
    optox_api.h
)

set(OPERATOR_HEADER
    operators/ioperator.h
    operators/activations/act.h
    operators/activations/act_rbf.h
    operators/activations/act_linear.h
    operators/activations/utils.cuh
    operators/nabla_operator.h
    operators/nabla2_operator.h
    operators/demosaicing_operator.h
)

set(OPERATOR_SOURCE
    operators/activations/act_rbf.cu
    operators/activations/act_linear.cu
    operators/nabla_operator.cu
    operators/nabla2_operator.cu
    operators/demosaicing_operator.cu
)

set(HEADERS
    ${COMMON_HEADER}
    ${OPERATOR_HEADER}
)

set(SOURCES
    ${OPERATOR_SOURCE}
)

#add_flags(CUDA_NVCC_FLAGS "--ptxas-options=-v")

##-----------------------------------------------------------------------------
## Core lib
##-----------------------------------------------------------------------------
cuda_add_library(optox SHARED ${HEADERS} ${SOURCES} OPTIONS "-DOPTOX_EXPORTS")
set_target_properties(optox PROPERTIES DEFINE_SYMBOL OPTOX_EXPORTS)
