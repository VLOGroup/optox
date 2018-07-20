##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
include("../cmake/macros.cmake")

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/matlab)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/matlab)

##----------------------------------------------------------------------------
## Matlab
##----------------------------------------------------------------------------
FIND_PACKAGE(Matlab_ICG)
if(MATLAB_FOUND)
    INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

    CUDA_ADD_LIBRARY(mex_add_operator SHARED  mex_add_operator.cpp OPTIONS "-DOPTOX_EXPORTS")
    finish_mex_lib(mex_add_operator)

    CUDA_ADD_LIBRARY(mex_add_operator_adjoint SHARED  mex_add_operator_adjoint.cpp OPTIONS "-DOPTOX_EXPORTS")
    finish_mex_lib(mex_add_operator_adjoint)
else(MATLAB_FOUND)
    message("   [-] Omitting matlab module. Matlab not found.")
endif(MATLAB_FOUND)
