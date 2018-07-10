# @project pdToolbox - Primal-Dual Toolbox
# @author Kerstin Hammernik <hammernik@icg.tugraz.at>
# @date 01.09.2016

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

include_directories(src)
add_subdirectory(src)
add_subdirectory(tensorflow)
add_subdirectory(python)
#add_subdirectory(tests)
#add_subdirectory(matlab)
