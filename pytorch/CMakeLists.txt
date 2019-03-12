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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/pytorch)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/pytorch)

##----------------------------------------------------------------------------
## compile using python setup
##----------------------------------------------------------------------------
find_program(PYTHON "python")

message(STATUS "python: ${PYTHON}")

if (PYTHON)
    set(SETUP_TH_IN setup.py.in)
    set(SETUP_TH "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS_TH
        optoth/__init__.py
        optoth/activations/__init__.py
        optoth/activations/act.py
        th_act_operator.cpp
        th_utils.h
        optoth/nabla.py
	    optoth/nabla2.py
        th_nabla_operator.cpp
    )
    set(OUTPUT_TH "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_TH_IN} ${SETUP_TH})

    add_custom_command(OUTPUT ${OUTPUT_TH}
                       COMMAND ${PYTHON} ${SETUP_TH} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_TH}
                       DEPENDS ${DEPS_TH})

    add_custom_target(setup_th ALL DEPENDS ${OUTPUT_TH} ${DEPS_TH})

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_TH} install)")
endif()
