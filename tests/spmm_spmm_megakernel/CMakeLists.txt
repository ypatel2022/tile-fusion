cmake_minimum_required(VERSION 3.20)
project(spmm_spmm_megakernel_evaluation LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()
set(EVALUATION_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
find_package(CUDAToolkit REQUIRED)
set(ENABLE_CUDA ON CACHE BOOL "Build CUDA benchmark dependencies" FORCE)
set(TILE_FUSION_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "Tile-fusion source")
add_subdirectory("${TILE_FUSION_ROOT}/fusion" fusion EXCLUDE_FROM_ALL)

add_executable(spmm_spmm_megakernel_benchmark evaluation.cu)
set_target_properties(spmm_spmm_megakernel_benchmark PROPERTIES
  CUDA_ARCHITECTURES "${EVALUATION_CUDA_ARCHITECTURES}")
target_include_directories(spmm_spmm_megakernel_benchmark PRIVATE
  "${TILE_FUSION_ROOT}/fusion/include"
  "${TILE_FUSION_ROOT}/fusion/aggregation/include")
target_compile_options(spmm_spmm_megakernel_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
target_link_libraries(spmm_spmm_megakernel_benchmark PRIVATE
  fusion aggregation swbench::swbench CUDA::cudart CUDA::cusparse)

# Implementation sources are added only after the baseline evaluation is frozen.
set(SPMM_MEGAKERNEL_SOURCE "" CACHE FILEPATH "Experimental megakernel source")
if(SPMM_MEGAKERNEL_SOURCE)
  target_sources(spmm_spmm_megakernel_benchmark PRIVATE "${SPMM_MEGAKERNEL_SOURCE}")
  target_compile_definitions(spmm_spmm_megakernel_benchmark PRIVATE
    SPMM_MEGAKERNEL_IMPLEMENTATION)
  set_target_properties(spmm_spmm_megakernel_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
endif()
