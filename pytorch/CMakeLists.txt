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

##-----------------------------------------------------------------------------
## define macro to finish lib
##-----------------------------------------------------------------------------
macro (add_deps_th opname)
set(DEPS_TH
    ${DEPS_TH}
    optoth/${opname}.py
    th_${opname}_operator.cpp
)
endmacro (add_deps_th)

##-----------------------------------------------------------------------------
## add new modules
##-----------------------------------------------------------------------------
# activations
set(DEPS_TH
    ${DEPS_TH}
    optoth/activations/__init__.py
    optoth/activations/act.py
    th_act_operator.cpp
)

# nabla & nabla2
add_deps_th(nabla)
set(DEPS_TH
    ${DEPS_TH}
    optoth/nabla2.py
)

# other
add_deps_th(demosaicing)
add_deps_th(rot)
add_deps_th(warp)
add_deps_th(pad2d)
add_deps_th(pad3d)

if (PYTHON)
    set(SETUP_TH_IN setup.py.in)
    set(SETUP_TH "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS_TH
        optoth/__init__.py
        th_utils.h
        ${DEPS_TH}
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
