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
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; print(tf.sysconfig.get_link_flags()[1], end='')" OUTPUT_VARIABLE Tensorflow_LIB)
execute_process(COMMAND ${PYTHON} -c "import tensorflow as tf; print(tf.sysconfig.get_compile_flags()[1], end='')" OUTPUT_VARIABLE Tensorflow_FLAGS)

message(STATUS "tensorflow include dir: ${Tensorflow_INCLUDE_DIRS}")
message(STATUS "tensorflow flags: ${Tensorflow_FLAGS}")
message(STATUS "tensorflow link dir: ${Tensorflow_LIB_DIRS}")
message(STATUS "tensorflow lib: ${Tensorflow_LIB}")

include_directories(${Tensorflow_INCLUDE_DIRS})
link_directories(${Tensorflow_LIB_DIRS})

##-----------------------------------------------------------------------------
## compiler flags
##-----------------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14  -fPIC -O2 -D GOOGLE_CUDA=1 ${Tensorflow_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -D GOOGLE_CUDA=1 ${Tensorflow_FLAGS}")

##-----------------------------------------------------------------------------
## define macro to finish lib
##-----------------------------------------------------------------------------
macro (finish_tensorflow_lib opname)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/optotf/${opname})
cuda_add_library(tf_${opname}_operator SHARED tf_${opname}_operator.cpp)

target_link_libraries(tf_${opname}_operator optox ${Tensorflow_LIB})
set_target_properties(tf_${opname}_operator PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
set(DEPS_TF
    ${DEPS_TF}
    optotf/${opname}/__init__.py
    tf_${opname}_operator.cpp
    optotf/keras/${opname}.py
)
set(DEPS_BUILD
    ${DEPS_BUILD}
    tf_${opname}_operator
)
endmacro (finish_tensorflow_lib)
##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------
## add new modules
##-----------------------------------------------------------------------------
finish_tensorflow_lib(nabla)
finish_tensorflow_lib(demosaicing)
finish_tensorflow_lib(activations)
finish_tensorflow_lib(pad)
finish_tensorflow_lib(warp)

##----------------------------------------------------------------------------
## packageing using python setup
##----------------------------------------------------------------------------
if (PYTHON)
    set(SETUP_TF_IN setup.py.in)
    set(SETUP_TF "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS_TF
        optotf/__init__.py
        tf_utils.h
        ${DEPS_TF}
    )
    set(OUTPUT_TF "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    configure_file(${SETUP_TF_IN} ${SETUP_TF})

    add_custom_command(OUTPUT ${OUTPUT_TF}
                       COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/optotf ${CMAKE_CURRENT_BINARY_DIR}
                       COMMAND ${PYTHON} ${SETUP_TF} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_TF}
                       DEPENDS ${DEPS_TF})

    add_custom_target(setup_tf ALL DEPENDS ${OUTPUT_TF} ${DEPS_TF} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(setup_tf optox ${DEPS_BUILD})

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_TF} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
endif()
