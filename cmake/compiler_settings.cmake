include(${CMAKE_CURRENT_LIST_DIR}/flags_functions.cmake)

option(WARN_LMEM_USAGE "Warnings when kernels use local memory" ON)
option(BUILD_SHARED_LIBS "ON/OFF" ON)
## include_directories(${CMAKE_CURRENT_BINARY_DIR}) # to find generated *.h files

if(NOT COMPILER_FLAGS_ALREADY_SET)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	# using GCC or using Clang
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	#
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -fmax-errors=3 -fpermissive -ggdb -Wall -Wc++11-compat")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -expt-extended-lambda -ftemplate-backtrace-limit=2 -lineinfo -std=c++11 -Xcompiler -fPIC")
        if(BUILD_RELEASE OR CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR NOT CMAKE_BUILD_TYPE) #RELEASE
		add_flags(CMAKE_CXX_FLAGS "-O2 -ffast-math -DNDEBUG")
		add_flags(CUDA_NVCC_FLAGS "-O2 -keep -src-in-ptx -DNDEBUG")
		# warn about local memory and register spills
		add_flags(CUDA_NVCC_FLAGS "-Xptxas -warn-lmem-usage -Xptxas -warn-spills")
	else() # DEBUG
		add_flags(CMAKE_CXX_FLAGS "-ggdb")
		# debug and and device code stack range checking
		add_flags(CUDA_NVCC_FLAGS "-G -g -keep -src-in-ptx -pg -Xptxas -g")
	endif()
	if(CUDA_VERSION VERSION_LESS "8.0")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES")
	endif(CUDA_VERSION VERSION_LESS "8.0")
elseif (MSVC)
	# using Visual Studio C++
	#
	set(CompilerFlags
	        CMAKE_CXX_FLAGS
	        CMAKE_CXX_FLAGS_DEBUG
	        CMAKE_CXX_FLAGS_RELEASE
	        CMAKE_C_FLAGS
	        CMAKE_C_FLAGS_DEBUG
	        CMAKE_C_FLAGS_RELEASE)
	#
	set(CUDA_PROPAGATE_HOST_FLAGS ON) # does not work properly, will have to propagate manually
	#
	# Disable Warning	C4267: conversion from 'size_t' to 'thrust', possible loss of data
	# Disable Warning C4800: forcing value to bool 'true' of 'false'
	# Disable Warning C4996: This function of variable may be unsafe. Consider using strerror_s instead.
	ADD_DEFINITIONS("/wd4267 /wd4800 /wd4996")
	#set(CMAKE_VS_PLATFORM_TOOLSET "CTP_Nov2013")
	add_flags(CMAKE_CXX_FLAGS "/Zi")
	## ADD_DEFINITIONS("/Zi") #Generates complete debugging information
	ADD_DEFINITIONS("/MP") #Multiprocessir Compilation
	#Linker: Enable Debug Information
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -expt-extended-lambda -ftemplate-backtrace-limit=2 -lineinfo")
	add_definitions("/D_ALLOW_KEYWORD_MACROS") # allow defining constexpr to old msvc version
	if(BUILD_SHARED_LIBS)
		add_definitions("/MD")
        add_definitions("/bigobj")
		foreach(CompilerFlag ${CompilerFlags})
		  string(REPLACE "/MT" "/MD" ${CompilerFlag} "${${CompilerFlag}}")
		endforeach()
		add_flags(CUDA_NVCC_FLAGS "-Xcompiler /MD")
	else()
		add_definitions("/MT")
		add_definitions("/bigobj")
		foreach(CompilerFlag ${CompilerFlags})
		  string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
		endforeach()
		add_flags(CUDA_NVCC_FLAGS "-Xcompiler /MT")
	endif()
        if(BUILD_RELEASE OR CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR NOT CMAKE_BUILD_TYPE) #RELEASE
		add_flags(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /bigobj")
		add_flags(CUDA_NVCC_FLAGS_RELEASE "-O2 -keep -src-in-ptx")
		add_flags(CUDA_NVCC_FLAGS_RELEASE "-Xptxas -warn-lmem-usage -Xptxas -warn-spills")
	else() # DEBUG
		# debug and and device code stack range checking
		add_flags(CUDA_NVCC_FLAGS_DEBUG "-G -g -keep -src-in-ptx -Xptxas -g")
	endif()

endif()
set(COMPILER_FLAGS_ALREADY_SET ON)
endif()

if(NOT WARN_LMEM_USAGE)
	clear_flag(CUDA_NVCC_FLAGS "-Xptxas -warn-lmem-usage")
endif()

#SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic" )
#set(CUDA_SEPARABLE_COMPILATION ON)

include(${CMAKE_CURRENT_LIST_DIR}/custom_settings.cmake.txt OPTIONAL)

cleanup_flags(CUDA_NVCC_FLAGS CUDA_NVCC_FLAGS)

message("COMPILER FLAGS:")
#MESSAGE(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
#MESSAGE(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )
MESSAGE( STATUS "BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS} )
MESSAGE(STATUS "CMAKE_C_FLAGS= " ${CMAKE_C_FLAGS} )
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")
#message(STATUS "OTHER_CFLAGS =  ${OTHER_CFLAGS}")
message(STATUS "CUDA_NVCC_FLAGS= ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_PROPAGATE_HOST_FLAGS= ${CUDA_PROPAGATE_HOST_FLAGS}")
get_directory_property(DirDefs COMPILE_DEFINITIONS )
message(STATUS "COMPILE_DEFINITIONS = ${DirDefs}")
#get_directory_property(DirOps COMPILE_OPTIONS )
#message(STATUS "COMPILE_OPTIONS = ${DirOps}")
