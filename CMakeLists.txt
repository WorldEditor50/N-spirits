cmake_minimum_required(VERSION 3.5)

project(SimpleML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${PROJECT_SOURCE_DIR}/net")
add_library(Utils
    ${PROJECT_SOURCE_DIR}/largeint.h
    ${PROJECT_SOURCE_DIR}/largeint.cpp)
add_executable(SimpleML main.cpp)

