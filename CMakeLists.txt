cmake_minimum_required(VERSION 3.5)

project(N-spirits LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# load option
set(LOAD_SIMCL True)
set(LOAD_IMPROCESS True)
# compiler option
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/bigobj>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/arch:AVX2>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>")
endif()
# test simd
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")

# basic
add_subdirectory(${PROJECT_SOURCE_DIR}/basic)
# net
include_directories(${PROJECT_SOURCE_DIR}/net)
# utils
include_directories(${PROJECT_SOURCE_DIR}/utils)
# cl
if (LOAD_SIMCL)
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
if (LOAD_IMPROCESS)
    if (MSVC)
        set(JPEG_ROOT "D:/home/3rdparty/libjpeg")
    endif()
    find_package(JPEG REQUIRED)
    if (JPEG_FOUND)
        include_directories(${JPEG_INCLUDE_DIR})
        message(STATUS "JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}")
        message(STATUS "JPEG_LIBRARIES=${JPEG_LIBRARIES}")
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/improcess)
    # test improcess
    add_executable(test_improcess test/test_improcess.cpp)
    target_link_libraries(test_improcess IMPROCESS BASIC ${JPEG_LIBRARIES})
endif()
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
add_executable(test_simd test/test_simd.cpp)
target_link_libraries(test_simd BASIC)
# test fluid
add_executable(test_fluid test/test_lbm.cpp)
#target_link_libraries(test_lbm FLUID BASIC)
target_link_libraries(test_fluid FLUID IMPROCESS BASIC ${JPEG_LIBRARIES})
# test cpp
add_executable(test_cpp test/testcpp.cpp)
target_link_libraries(test_cpp BASIC)
