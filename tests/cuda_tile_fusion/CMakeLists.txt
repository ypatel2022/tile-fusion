cmake_minimum_required(VERSION 3.20)
project(cuda_tile_fusion_benchmark LANGUAGES C CXX CUDA)

# SWBench supplies trial storage and CSV output. Timing uses explicit CUDA
# events on the workload stream, so its optional GPU timer is not needed.
set(ENABLE_CUDA OFF)
add_subdirectory(../../fusion/benchmark swbench EXCLUDE_FROM_ALL)
set(TILE_FUSION_BUILD_LEGACY_CUDA_DEMOS OFF)
add_subdirectory(../../fusion/gpu cuda_tile_fusion EXCLUDE_FROM_ALL)

add_executable(spmm_spmm_graph_benchmark benchmark.cpp)
target_compile_features(spmm_spmm_graph_benchmark PRIVATE cxx_std_14)
target_link_libraries(spmm_spmm_graph_benchmark PRIVATE
    swiftware::cuda_tile_fusion swbench::swbench)
