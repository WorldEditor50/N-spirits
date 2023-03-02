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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

# basic
add_subdirectory(${PROJECT_SOURCE_DIR}/basic)
# net
include_directories(${PROJECT_SOURCE_DIR}/net)
# utils
include_directories(${PROJECT_SOURCE_DIR}/utils)
# test
add_executable(test test.cpp)
target_link_libraries(test BASIC)
# test cpp
add_executable(testcpp testcpp.cpp)
target_link_libraries(testcpp BASIC)
