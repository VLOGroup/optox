macro (finish_python_lib target_lib)
target_link_libraries(${target_lib} optox ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${IMAGEUTILITIES_LIBRARIES})
# remove lib prefix
set_target_properties(${target_lib} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
if(WIN32)
    SET_TARGET_PROPERTIES(${target_lib} PROPERTIES SUFFIX ".pyd")
endif(WIN32)
endmacro (finish_python_lib)