cmake_minimum_required(VERSION 3.5)

project(SimpleML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
# basic
LIST(APPEND BASIC_DIR basic)
foreach(BASIC_CURRENT_DIR ${BASIC_DIR})
        file(GLOB_RECURSE BASIC_SRCS ${BASIC_CURRENT_DIR}/*.cpp)
        file(GLOB_RECURSE BASIC_HDRS ${BASIC_CURRENT_DIR}/*.h)
        file(GLOB_RECURSE BASIC_HPPS ${BASIC_CURRENT_DIR}/*.hpp)
        LIST(APPEND BASIC_SRC_LIST ${BASIC_SRCS})
        LIST(APPEND BASIC_HDR_LIST ${BASIC_HDRS} ${BASIC_HPPS})
endforeach()
add_library(BASIC STATIC ${BASIC_SRC_LIST} ${BASIC_HDR_LIST})
# net
include_directories(${PROJECT_SOURCE_DIR}/net)
# utils
include_directories(${PROJECT_SOURCE_DIR}/utils)
# target
add_executable(MAIN main.cpp)
target_link_libraries(MAIN BASIC)
