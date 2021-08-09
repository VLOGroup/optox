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
# CUDA
##-----------------------------------------------------------------------------
find_package(CUDA 5.0 REQUIRED)

cuda_include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_SDK_ROOT_DIR $ENV{CUDA_SDK_ROOT_DIR}/common/inc)
cuda_include_directories(${CUDA_SDK_ROOT_DIR})
include_directories(${CUDA_SDK_ROOT_DIR})

##-----------------------------------------------------------------------------
# Compiler flags
##-----------------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/settings.cmake)

##-----------------------------------------------------------------------------
## sources
##-----------------------------------------------------------------------------
include_directories(src)
add_subdirectory(src)

##-----------------------------------------------------------------------------
## optox include and library directories
##-----------------------------------------------------------------------------
set(OPTOX_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(OPTOX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
message(STATUS "optox include dir: ${OPTOX_INCLUDE_DIR}")
message(STATUS "optox library dir: ${OPTOX_LIBRARY_DIR}")

##-----------------------------------------------------------------------------
## interfaces
##-----------------------------------------------------------------------------
option(WITH_PYTHON "Use Python - build python interface. Requires numpy, pybind11" OFF)
if(WITH_PYTHON)
    message(STATUS "[+] optox build python interfaces")
    add_subdirectory(python)
else(WITH_PYTHON)
    message(STATUS "[-] optox exclude python interfaces")
endif(WITH_PYTHON)

option(WITH_TENSORFLOW "Use Tensorflow - build tensorflow interface. Requires tensorflow" OFF)
if(WITH_TENSORFLOW)
    message(STATUS "[+] optox build tensorflow interfaces")
    add_subdirectory(tensorflow)
else(WITH_TENSORFLOW)
    message(STATUS "[-] optox exclude tensorflow interfaces")
endif(WITH_TENSORFLOW)

option(WITH_PYTORCH "Use Pytorch - build pytorch interface. Requires pytorch" ON)
if(WITH_PYTORCH)
    message(STATUS "[+] optox build pytorch interfaces")
    add_subdirectory(pytorch)
else(WITH_PYTORCH)
    message(STATUS "[-] optox exclude pytorch interfaces")
endif(WITH_PYTORCH)

enable_testing()
