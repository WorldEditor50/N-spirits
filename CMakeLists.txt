cmake_minimum_required(VERSION 3.5)

project(N-spirits LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# specify default 3rd-party library path
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# load option
option(ENABLE_SIMCL "ENABLE CL" OFF)
option(ENABLE_SIMD "ENABLE SIMD" OFF)
option(USE_JPEG "USE JPEG" OFF)
# compiler option
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()
# test simd
if (ENABALE_SIMD)
    if (MSVC)
        add_compile_options("$<$<C_COMPILER_ID:MSVC>:/bigobj>")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
        add_compile_options("$<$<C_COMPILER_ID:MSVC>:/arch:AVX2>")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>")
        set(CMAKE_CXX_FLAGS "-O2")
    endif()

    if (UNIX)
        add_compile_options(-march=native)
        set(CMAKE_CXX_FLAGS "-msse2 -mfma -O2")
    endif()
endif()
# basic
add_subdirectory(${PROJECT_SOURCE_DIR}/basic)
# net
include_directories(${PROJECT_SOURCE_DIR}/net)
# utils
include_directories(${PROJECT_SOURCE_DIR}/utils)
# cl
if (ENABLE_SIMCL)
    if (MSVC)
	set(OpenCL_ROOT "D:/home/3rdparty/opencl/opencl")
	set(OpenCL_INCLUDE_DIR "${OpenCL_ROOT}/include")
        set(OpenCL_LIBRARIES "${OpenCL_ROOT}/lib/OpenCL.lib")
    endif()
    message(STATUS "OpenCL_LIBRARIES=${OpenCL_LIBRARIES}")
    include_directories(${OpenCL_INCLUDE_DIR})
    add_subdirectory(${PROJECT_SOURCE_DIR}/simcl)
    # test cl
    add_executable(test_simcl test/test_simcl.cpp)
    target_link_libraries(test_simcl SIMCL ${OpenCL_LIBRARIES})
endif()
# improcess
file(GLOB IMPROCESS_3RDPARTY "")
if (USE_JPEG)
    find_package(JPEG QUIET)
    if (JPEG_FOUND)
        include_directories(${JPEG_INCLUDE_DIR})
        message(STATUS "JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}")
        message(STATUS "JPEG_LIBRARIES=${JPEG_LIBRARIES}")
    else()
        if (MSVC)
            #set(JPEG_ROOT "D:/home/3rdparty/libjpeg")
            set(JPEG_INCLUDE_DIR "D:/home/3rdparty/libjpeg/include")
            set(JPEG_LIBRARIES "D:/home/3rdparty/libjpeg/lib/jpeg-static.lib")
            include_directories(${JPEG_INCLUDE_DIR})
        endif()
    endif()
    list(APPEND IMPROCESS_3RDPARTY ${JPEG_LIBRARIES})
    add_definitions(-DENABLE_JPEG)
    message(STATUS "ENABLE USE JPEG")
endif()
add_subdirectory(${PROJECT_SOURCE_DIR}/improcess)
# test improcess
add_executable(test_improcess test/test_improcess.cpp)
target_link_libraries(test_improcess IMPROCESS BASIC ${IMPROCESS_3RDPARTY})

# copy images
add_custom_command(TARGET test_improcess
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/images" "${CMAKE_CURRENT_BINARY_DIR}/images"
                   COMMENT "COPY IMAGES")

# fluid
add_subdirectory(${PROJECT_SOURCE_DIR}/fluid)
# test
add_executable(test_basic test.cpp)
target_link_libraries(test_basic BASIC)
# test ml
add_executable(test_ml test/test_ml.cpp)
target_link_libraries(test_ml BASIC)
# test dl
add_executable(test_dl test/test_dl.cpp)
target_link_libraries(test_dl BASIC)
# test simd
if (ENABLE_SIMD)
    add_executable(test_simd test/test_simd.cpp)
    target_link_libraries(test_simd BASIC)
endif()
# test fluid
add_executable(test_fluid test/test_lbm.cpp)
target_link_libraries(test_fluid FLUID IMPROCESS BASIC ${IMPROCESS_3RDPARTY})
# create diretory
add_custom_command(TARGET test_fluid
                  POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/cylinder"
                  COMMENT "CREATE DIR")
# test cpp
add_executable(test_cpp test/testcpp.cpp)
target_link_libraries(test_cpp BASIC)
