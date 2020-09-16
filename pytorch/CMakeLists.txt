# @project optox - operator to X 
# @author Erich Kobler <erich.kobler@icg.tugraz.at>
# @date 01.07.2018

project(optoX)

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
        optoth/demosaicing.py
        th_demosaicing_operator.cpp
        optoth/rot.py
        th_rot_operator.cpp
        optoth/warp.py
        th_warp_operator.cpp
        optoth/pad2d.py
        th_pad2d_operator.cpp
        optoth/pad3d.py
        th_pad3d_operator.cpp
    )
    set(OUTPUT_TH "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    configure_file(${SETUP_TH_IN} ${SETUP_TH})

    add_custom_command(OUTPUT ${OUTPUT_TH}
                       COMMAND ${PYTHON} ${SETUP_TH} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_TH}
                       DEPENDS ${DEPS_TH})

    add_custom_target(setup_th ALL DEPENDS ${OUTPUT_TH} ${DEPS_TH} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(setup_th optox)

    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_TH} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
endif()
