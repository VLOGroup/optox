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
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1" )

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
## nabla operator
##----------------------------------------------------------------------------
cuda_add_library(TfNablaOperator SHARED tf_nabla_operator.cpp tf_nabla_operator.cu)

target_link_libraries(TfNablaOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TfNablaOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TfNablaOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

##----------------------------------------------------------------------------
## activation operators
##----------------------------------------------------------------------------
cuda_add_library(TfActivationOperators SHARED tf_activations.cpp tf_activations.cu)

target_link_libraries(TfActivationOperators optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TfActivationOperators PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TfActivationOperators PROPERTIES SUFFIX ".pyd")
endif(WIN32)

##----------------------------------------------------------------------------
## activation operators
##----------------------------------------------------------------------------
cuda_add_library(TfFftOperators SHARED tf_fftutils.cpp tf_fftutils.cu)

target_link_libraries(TfFftOperators optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TfFftOperators PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TfFftOperators PROPERTIES SUFFIX ".pyd")
endif(WIN32)

##----------------------------------------------------------------------------
## rotate filters operators
##----------------------------------------------------------------------------
cuda_add_library(TfRotateFiltersOperator SHARED tf_filters.cpp tf_filters.cu)

target_link_libraries(TfRotateFiltersOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TfRotateFiltersOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TfRotateFiltersOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

##----------------------------------------------------------------------------
## metamorphosis operators
##----------------------------------------------------------------------------
cuda_add_library(TfMetamorphosisOperator SHARED tf_metamorphosis.cpp tf_metamorphosis.cu)

target_link_libraries(TfMetamorphosisOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TfMetamorphosisOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TfMetamorphosisOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

##----------------------------------------------------------------------------
## demosaicing operators
##----------------------------------------------------------------------------
cuda_add_library(TFDemosaicingOperator SHARED tf_demosaicing.cpp tf_demosaicing.cu)

target_link_libraries(TFDemosaicingOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TFDemosaicingOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TFDemosaicingOperator PROPERTIES SUFFIX ".pyd")
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


#########################################################
# Add python  tests
# add enable_testing() needs to be in toplevel module
enable_testing() 
find_package(PythonInterp)
message ("python path: ${PYTHON_EXECUTABLE}")
##----------------------------------------------------------------------------
## warping operator
##----------------------------------------------------------------------------
include_directories(warpimage)
cuda_add_library(TFWarpimageOperator SHARED warpimage/warpimage.cc warpimage/warpimage.cu)

target_link_libraries(TFWarpimageOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TFWarpimageOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TFWarpimageOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

## Add tests:
add_test(NAME TFWarpimageOperator_grad_test
 COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/warpimage/warpimage__GradientCheck.py
 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
add_test(NAME TFWarpimageOperator_example_test
 COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/warpimage/warpimage_Mini_Example.py
 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
##----------------------------------------------------------------------------


##----------------------------------------------------------------------------
## mapcoordinates operators
##----------------------------------------------------------------------------
include_directories(mapcoordinates)
cuda_add_library(TFMapcoordinatesOperator SHARED mapcoordinates/mapcoordinates.cc mapcoordinates/mapcoordinates.cu)

target_link_libraries(TFMapcoordinatesOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TFMapcoordinatesOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TFMapcoordinatesOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

## Add tests
add_test(NAME TFMapcoordinatesOperator_grad_test
 COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/mapcoordinates/Mapcoordinates__GradientCheck.py
 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
add_test(NAME TFMapcoordinatesOperator_example_test
 COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/mapcoordinates/Mapcoordinates_Mini_Example.py
 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
##----------------------------------------------------------------------------


##----------------------------------------------------------------------------
## median Filter operators
##----------------------------------------------------------------------------
include_directories(medianfilter)
cuda_add_library(TFMedianfilterOperator SHARED medianfilter/median_filter_4D.cc medianfilter/median_filter_4D.cu)

target_link_libraries(TFMedianfilterOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TFMedianfilterOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TFMedianfilterOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

## Add tests
add_test(NAME TFMedianfilterOperator_test
 COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/medianfilter/median_filter_4D_test_local.py 
 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
##----------------------------------------------------------------------------


##----------------------------------------------------------------------------
## pad2d  operators
##----------------------------------------------------------------------------
include_directories(pad2d)
cuda_add_library(TFpad2dOperator SHARED pad2d/pad2d.cc pad2d/pad2d.cu)

target_link_libraries(TFpad2dOperator optox ${IMAGEUTILITIES_LIBRARIES} "tensorflow_framework")
# remove lib prefix
set_target_properties(TFpad2dOperator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(TFpad2dOperator PROPERTIES SUFFIX ".pyd")
endif(WIN32)

## Add tests:
add_test(NAME TFpad2dOperator_test
COMMAND  ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pad2d/pad2d_test_local.py
WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
##-------------------------------------------------------------------------------
