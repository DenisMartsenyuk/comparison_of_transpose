cmake_minimum_required(VERSION 3.17)
project(comparison_of_transpose)

set(CMAKE_CXX_STANDARD 14)

add_executable(comparison_of_transpose main.cpp gpu/TransposeMatricesGPU.cpp gpu/TransposeMatricesGPU.h)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
target_include_directories (comparison_of_transpose PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (comparison_of_transpose ${OpenCL_LIBRARY})