cmake_minimum_required(VERSION 3.5)


set(BASIC_DIR "${PROJECT_SOURCE_DIR}/basic")
file(GLOB BASIC_FILES
    ${BASIC_DIR}/*.h
    ${BASIC_DIR}/*.hpp
    ${BASIC_DIR}/*.cpp)
if (ENABLE_SIMD)
    file(GLOB SIMD_FILES
        ${BASIC_DIR}/simd/*.h
        ${BASIC_DIR}/simd/*.hpp
        ${BASIC_DIR}/simd/*.cpp)
    list(APPEND BASIC_FILES ${SIMD_FILES})
endif()
add_library(BASIC STATIC ${BASIC_FILES})
