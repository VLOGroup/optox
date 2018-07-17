# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

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

##----------------------------------------------------------------------------
## Generate lib for Add Operator
##----------------------------------------------------------------------------
cuda_add_library(PyAddOperator SHARED  py_add_operator.cpp)
finish_python_lib(PyAddOperator)
