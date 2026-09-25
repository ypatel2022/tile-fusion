
cmake_minimum_required(VERSION 3.20)

# The graph library can be built on its own, without the CPU research demos'
# MKL/METIS dependencies. The main fusion build retains those demos by default.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(cuda_tile_fusion LANGUAGES CXX CUDA)
    set(legacy_cuda_demos_default OFF)
    include(CTest)
else()
    enable_language(CUDA)
    set(legacy_cuda_demos_default ON)
endif()
find_package(CUDAToolkit REQUIRED)

add_library(cuda_tile_fusion SpMM_SpMM_Graph.cu)
add_library(swiftware::cuda_tile_fusion ALIAS cuda_tile_fusion)
target_include_directories(cuda_tile_fusion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cuda_tile_fusion PUBLIC CUDA::cudart)
target_compile_features(cuda_tile_fusion PUBLIC cxx_std_14)
set_target_properties(cuda_tile_fusion PROPERTIES
    CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)

add_executable(spmm_spmm_graph_demo demo/SpMMSpMMGraphDemo.cu)
target_link_libraries(spmm_spmm_graph_demo PRIVATE swiftware::cuda_tile_fusion)
set_target_properties(spmm_spmm_graph_demo PROPERTIES
    CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
if(BUILD_TESTING)
    add_test(NAME cuda_spmm_spmm_graph COMMAND spmm_spmm_graph_demo)
endif()

option(TILE_FUSION_BUILD_LEGACY_CUDA_DEMOS
    "Build the original CUDA demos with the main fusion project"
    ${legacy_cuda_demos_default})
if(NOT TILE_FUSION_BUILD_LEGACY_CUDA_DEMOS)
    return()
endif()
if(NOT TARGET fusion)
    message(FATAL_ERROR "Legacy CUDA demos require configuring the fusion project")
endif()

add_executable(spmm_demo_gpu
        ../demo/SpMMCudaDemo.cu
        )

set_target_properties(spmm_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}"
        )

target_include_directories(spmm_demo_gpu
        PRIVATE
        ${SYM_AGG}/include
        ${SYM_FUS}/include
        )
target_link_libraries(spmm_demo_gpu
        cusparse
        cublas
        CUDA::cusparse
        fusion
        aggregation
        swbench::swbench
        )

add_executable(spmm_spmm_demo_gpu
        demo/SpMMSpMMCudaDemo.cu
)

set_target_properties(spmm_spmm_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}"
)

target_include_directories(spmm_spmm_demo_gpu
        PRIVATE
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)
target_link_libraries(spmm_spmm_demo_gpu
        cusparse
        cublas
        CUDA::cusparse
        fusion
        aggregation
        swbench::swbench
)

add_executable(spmm_seqred_demo_gpu
        demo/SeqReduceBlockDemo.cu
)

set_target_properties(spmm_seqred_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}"
)

target_include_directories(spmm_seqred_demo_gpu
        PRIVATE
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)
target_link_libraries(spmm_seqred_demo_gpu
        cusparse
        cublas
        CUDA::cusparse
        fusion
        aggregation
        swbench::swbench
)

add_executable(cluster_group_demo
        demo/ClusterGroupTest.cu
)

set_target_properties(cluster_group_demo PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}"
)

target_include_directories(cluster_group_demo
        PRIVATE
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

target_link_libraries(cluster_group_demo
        cusparse
        cublas
        CUDA::cusparse
        fusion
        aggregation
        swbench::swbench
)


#if(OpenMP_CXX_FOUND)
#    target_link_libraries(spmv_demo_gpu OpenMP::OpenMP_CXX)
#endif()
