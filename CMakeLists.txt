cmake_minimum_required(VERSION 3.5)

project(MLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${PROJECT_SOURCE_DIR}")
#    ${PROJECT_SOURCE_DIR}/kernel.h
#    ${PROJECT_SOURCE_DIR}/kmeans.h
#    ${PROJECT_SOURCE_DIR}/linearregression.h
#    ${PROJECT_SOURCE_DIR}/svm.h
#    ${PROJECT_SOURCE_DIR}/vec.h
#    ${PROJECT_SOURCE_DIR}/bayes.h
#    ${PROJECT_SOURCE_DIR}/csv.h
#    ${PROJECT_SOURCE_DIR}/dataset.h
#    ${PROJECT_SOURCE_DIR}/gmm.h
#    ${PROJECT_SOURCE_DIR}/hmm.h)
add_executable(MLib main.cpp)

