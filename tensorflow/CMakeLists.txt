# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

project(optoX)

##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
include("../cmake/macros.cmake")


##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/tf)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/tf)

##----------------------------------------------------------------------------
## Tensorflow (for wrappers)
##----------------------------------------------------------------------------
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)

message("tensorflow include dir: ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link dir: ${Tensorflow_LIB_DIRS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
link_directories(${Tensorflow_LIB_DIRS})

##-----------------------------------------------------------------------------
## compiler flags
##-----------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -D GOOGLE_CUDA=1" )

#pass flags to c++ compiler
#SET(CUDA_PROPAGATE_HOST_FLAGS ON)

##----------------------------------------------------------------------------
## add operator
##----------------------------------------------------------------------------
cuda_add_library(TfAddOperator SHARED tf_add_operator.cpp tf_add_operator.cu)

target_link_libraries(TfAddOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TfAddOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TfAddOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)


##----------------------------------------------------------------------------
## gpuNUFFT operator
##----------------------------------------------------------------------------
if(WITH_GPUNUFFT)
    cuda_add_library(TfGpuNufftOperator SHARED tf_gpu_nufft_operator.cpp tf_gpu_nufft_operator.cu)

    target_link_libraries(TfGpuNufftOperator ${GPUNUFFT_LIBRARIES} "tensorflow_framework")
    # remove lib prefix
    set_target_properties(TfGpuNufftOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
    if(WIN32)
        SET_TARGET_PROPERTIES(TfGpuNufftOperator PROPERTIES SUFFIX ".pyd")
    endif(WIN32)
endif(WITH_GPUNUFFT)
