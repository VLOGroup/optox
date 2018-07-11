# @project pdToolbox - Primal-Dual Toolbox
# @author Kerstin Hammernik <hammernik@icg.tugraz.at>
# @date 01.09.2016

##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
include("../cmake/macros.cmake")

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/python)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/python)

##----------------------------------------------------------------------------
## Python (for wrappers)
##----------------------------------------------------------------------------
execute_process(COMMAND python -c "import sys; print(sys.executable, end='')" OUTPUT_VARIABLE PYTHON_EXECUTABLE)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS python)
include_directories(${Boost_INCLUDE_DIRS})

#find_package(Eigen3 REQUIRED)
#INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )
##----------------------------------------------------------------------------
## Generate lib for MRI Reconstruction
##----------------------------------------------------------------------------
cuda_add_library(PyAddOperator SHARED  pyaddoperator.cpp)
finish_python_lib(PyAddOperator)
