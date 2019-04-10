# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 04.2019

project(optoX)

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

##----------------------------------------------------------------------------
## get python
##----------------------------------------------------------------------------
find_program(PYTHON "python")
message(STATUS "python: ${PYTHON}")

##----------------------------------------------------------------------------
## Tensorflow (for wrappers)
##----------------------------------------------------------------------------
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)

message("tensorflow include dir: ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link dir: ${Tensorflow_LIB_DIRS}")

include_directories(${Tensorflow_INCLUDE_DIRS})
link_directories(${Tensorflow_LIB_DIRS})

##-----------------------------------------------------------------------------
## compiler flags
##-----------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -fPIC -O2 -D GOOGLE_CUDA=1")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -D GOOGLE_CUDA=1")

##----------------------------------------------------------------------------
## nabla operator
##----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/optotf/nabla)
cuda_add_library(tf_nabla_operator SHARED tf_nabla_operator.cpp)

target_link_libraries(tf_nabla_operator optox "tensorflow_framework")
set_target_properties(tf_nabla_operator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

##----------------------------------------------------------------------------
## demosaicing operator
##----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/optotf/demosaicing)
cuda_add_library(tf_demosaicing_operator SHARED tf_demosaicing.cpp)

target_link_libraries(tf_demosaicing_operator optox "tensorflow_framework")
set_target_properties(tf_demosaicing_operator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

##----------------------------------------------------------------------------
## packageing using python setup
##----------------------------------------------------------------------------
if (PYTHON)
    set(SETUP_TF_IN setup.py.in)
    set(SETUP_TF "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS_TF
        optotf/__init__.py
        tf_utils.h
        optotf/nabla/__init__.py
        tf_nabla_operator.cpp
        optotf/demosaicing/__init__.py
        tf_demosaicing.cpp
    )
    set(OUTPUT_TF "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    configure_file(${SETUP_TF_IN} ${SETUP_TF})

    add_custom_command(OUTPUT ${OUTPUT_TF}
                       COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/optotf ${CMAKE_CURRENT_BINARY_DIR}
                       COMMAND ${PYTHON} ${SETUP_TF} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_TF}
                       DEPENDS ${DEPS_TF})

    add_custom_target(setup_tf ALL DEPENDS ${OUTPUT_TF} ${DEPS_TF} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(setup_tf optox tf_nabla_operator)

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_TF} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
endif()
