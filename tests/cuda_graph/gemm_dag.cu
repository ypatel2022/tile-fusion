#include "gemm_dag.cuh"

int main() {
    constexpr int n = 64;  // Try 65 to exercise the kernel's boundary check.
    const size_t elements = n * n;
    const size_t bytes = elements * sizeof(float);

    // 1. Prepare inputs and GPU buffers.
    std::vector<float> A(elements), B(elements);
    initialize_inputs(A, B, n);
    float *d_A, *d_B;
    float* d_C[num_gemms];
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    for (auto& output : d_C) CUDA_CHECK(cudaMalloc(&output, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    // 2. Build the explicit 30-node DAG (see build_graph in gemm_dag.cuh).
    cudaGraph_t graph = build_graph(d_A, d_B, d_C, n);

    // 3. Prepare the executable graph and launch it once.
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 4. Compare every node's output with the CPU reference, outside GPU work.
    std::vector<float> reference[num_gemms];
    reference_dag(A, B, reference, n);
    std::printf("GEMM DAG: %d matrix multiplications of size %d x %d\n", num_gemms, n, n);
    double max_error;
    bool all_ok = verify_outputs(d_C, reference, n, max_error, true);

    // 5. Clean up after GPU work has completed.
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto output : d_C) CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
