# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

project(optoX)

cmake_minimum_required(VERSION 3.5)

##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

##-----------------------------------------------------------------------------
## ImageUtilities
##-----------------------------------------------------------------------------
set(ImageUtilities_DIR $ENV{IMAGEUTILITIES_ROOT})
find_package(ImageUtilities REQUIRED COMPONENTS iucore iumath)
include_directories(${IMAGEUTILITIES_INCLUDE_DIR})

##-----------------------------------------------------------------------------
## gpuNUFFT
##-----------------------------------------------------------------------------
option(WITH_GPUNUFFT "Use gpuNUFFT for non-uniform MRI operator. Requires GPUNUFFT_ROOT to be set." OFF)

if(WITH_GPUNUFFT)
    message(STATUS "Use gpuNUFFT (required for non-uniform MRI operator)")
    find_package(GPUNUFFT REQUIRED)
    include_directories(${GPUNUFFT_INCLUDE_DIRS})
else(WITH_GPUNUFFT)
    message(STATUS "Disable gpuNUFFT (required for non-uniform MRI operator)")
endif(WITH_GPUNUFFT)

include_directories(src)
add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(tensorflow)
add_subdirectory(matlab)
