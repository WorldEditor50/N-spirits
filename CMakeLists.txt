cmake_minimum_required(VERSION 3.5)

project(N-spirits LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
# improcess
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
# fluid
add_subdirectory(${PROJECT_SOURCE_DIR}/fluid)
# test
add_executable(test_basic test.cpp)
target_link_libraries(test_basic BASIC)
# test improcess
add_executable(test_improcess test/test_improcess.cpp)
target_link_libraries(test_improcess IMPROCESS BASIC ${JPEG_LIBRARIES})
# test fluid
add_executable(test_fluid test/test_lbm.cpp)
#target_link_libraries(test_lbm FLUID BASIC)
target_link_libraries(test_fluid FLUID IMPROCESS BASIC ${JPEG_LIBRARIES})
# test cpp
add_executable(test_cpp test/testcpp.cpp)
target_link_libraries(test_cpp BASIC)
