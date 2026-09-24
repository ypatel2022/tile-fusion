cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()
project(cuda_graph_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Reuse the repository's SWBench/Stats library without pulling in the sparse
# fusion project's MKL, METIS, or datasets. Our GPU events use an explicit stream,
# so we do not enable SWBench's optional default-stream GPU Timer implementation.
set(ENABLE_CUDA OFF)
add_subdirectory(../../fusion/benchmark swbench EXCLUDE_FROM_ALL)

add_executable(gemm_dag gemm_dag.cu)
add_executable(gemm_dag_benchmark gemm_dag_benchmark.cu)
target_link_libraries(gemm_dag_benchmark PRIVATE swbench::swbench)
