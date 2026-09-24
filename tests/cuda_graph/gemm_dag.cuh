#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

// Stop at the first CUDA error, with the failing call and source location.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, \
                         cudaGetErrorString(error));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// C = A * B for square, row-major matrices. Each thread computes one C element.
// This deliberately uses only global memory: no tiling or shared memory yet.
__global__ void gemm(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

// Describe a kernel launch and its prerequisites. This does NOT run the kernel.
cudaGraphNode_t add_gemm(cudaGraph_t graph, const float* A, const float* B,
                         float* C, int n,
                         std::initializer_list<cudaGraphNode_t> dependencies) {
    // kernelParams holds addresses of argument values, in kernel signature order.
    // CUDA copies these values when adding the node; device buffers must stay alive.
    void* args[] = {&A, &B, &C, &n};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(gemm);
    params.blockDim = dim3(16, 16);
    params.gridDim = dim3((n + 15) / 16, (n + 15) / 16);
    params.kernelParams = args;

    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddKernelNode(
        &node, graph, dependencies.size() ? dependencies.begin() : nullptr,
        dependencies.size(), &params));
    return node;
}

// Ordinary CPU matrix multiplication, used only to check correctness.
void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// Shared by the correctness example and performance test.
constexpr int num_gemms = 30;

void initialize_inputs(std::vector<float>& A, std::vector<float>& B, int n) {
    for (int row = 0; row < n; ++row) {
        float sum_A = 0.0f, sum_B = 0.0f;
        for (int col = 0; col < n; ++col) {
            int i = row * n + col;
            A[i] = static_cast<float>(i % 7 + 1);
            B[i] = static_cast<float>(i % 11 + 1);
            sum_A += A[i];
            sum_B += B[i];
        }
        for (int col = 0; col < n; ++col) {
            int i = row * n + col;
            float diagonal = (row == col) ? 0.9f : 0.0f;
            // Positive rows summing to 1 prevent long products from vanishing
            // or exploding. Mixing in the identity preserves more variation.
            A[i] = diagonal + 0.1f * A[i] / sum_A;
            B[i] = diagonal + 0.1f * B[i] / sum_B;
        }
    }

}

cudaGraph_t build_graph(const float* d_A, const float* d_B, float* const* d_C, int n) {
    // 2. Build the DAG. Dependencies in {...} are nodes that must finish first.
    // Buffers alone do not establish ordering: we must declare these edges.
    // G1 and G2 are roots. Branches split and rejoin until the final node G30.
    // The README draws the full graph; every node contributes to G30.
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    auto g1 = add_gemm(graph, d_A,    d_B,    d_C[0], n, {});       // C1 = A * B
    auto g2 = add_gemm(graph, d_B,    d_A,    d_C[1], n, {});       // C2 = B * A
    auto g3 = add_gemm(graph, d_C[0], d_A,    d_C[2], n, {g1});     // C3 = C1 * A
    auto g4 = add_gemm(graph, d_C[0], d_C[1], d_C[3], n, {g1, g2}); // C4 = C1 * C2
    auto g5 = add_gemm(graph, d_C[2], d_C[3], d_C[4], n, {g3, g4}); // C5 = C3 * C4

    // Widen the graph with branches from earlier results.
    auto g6  = add_gemm(graph, d_C[1], d_B,    d_C[5],  n, {g2});     // C6 = C2 * B
    auto g7  = add_gemm(graph, d_C[2], d_B,    d_C[6],  n, {g3});     // C7 = C3 * B
    auto g8  = add_gemm(graph, d_C[3], d_A,    d_C[7],  n, {g4});     // C8 = C4 * A
    auto g9  = add_gemm(graph, d_C[4], d_C[5], d_C[8],  n, {g5, g6}); // C9 = C5 * C6
    auto g10 = add_gemm(graph, d_C[5], d_C[6], d_C[9],  n, {g6, g7}); // C10 = C6 * C7
    auto g11 = add_gemm(graph, d_C[6], d_C[7], d_C[10], n, {g7, g8}); // C11 = C7 * C8
    auto g12 = add_gemm(graph, d_C[7], d_B,    d_C[11], n, {g8});     // C12 = C8 * B

    // Three groups of four: neighboring branches join and feed later branches.
    auto g13 = add_gemm(graph, d_C[8],  d_C[9],  d_C[12], n, {g9, g10});  // C13 = C9 * C10
    auto g14 = add_gemm(graph, d_C[9],  d_C[10], d_C[13], n, {g10, g11}); // C14 = C10 * C11
    auto g15 = add_gemm(graph, d_C[10], d_C[11], d_C[14], n, {g11, g12}); // C15 = C11 * C12
    auto g16 = add_gemm(graph, d_C[11], d_A,     d_C[15], n, {g12});      // C16 = C12 * A

    auto g17 = add_gemm(graph, d_C[12], d_C[13], d_C[16], n, {g13, g14}); // C17 = C13 * C14
    auto g18 = add_gemm(graph, d_C[13], d_C[14], d_C[17], n, {g14, g15}); // C18 = C14 * C15
    auto g19 = add_gemm(graph, d_C[14], d_C[15], d_C[18], n, {g15, g16}); // C19 = C15 * C16
    auto g20 = add_gemm(graph, d_C[15], d_B,     d_C[19], n, {g16});      // C20 = C16 * B

    auto g21 = add_gemm(graph, d_C[16], d_C[17], d_C[20], n, {g17, g18}); // C21 = C17 * C18
    auto g22 = add_gemm(graph, d_C[17], d_C[18], d_C[21], n, {g18, g19}); // C22 = C18 * C19
    auto g23 = add_gemm(graph, d_C[18], d_C[19], d_C[22], n, {g19, g20}); // C23 = C19 * C20
    auto g24 = add_gemm(graph, d_C[19], d_A,     d_C[23], n, {g20});      // C24 = C20 * A

    // Merge the remaining branches into one final output, C30.
    auto g25 = add_gemm(graph, d_C[20], d_C[21], d_C[24], n, {g21, g22}); // C25 = C21 * C22
    auto g26 = add_gemm(graph, d_C[21], d_C[22], d_C[25], n, {g22, g23}); // C26 = C22 * C23
    auto g27 = add_gemm(graph, d_C[22], d_C[23], d_C[26], n, {g23, g24}); // C27 = C23 * C24
    auto g28 = add_gemm(graph, d_C[24], d_C[25], d_C[27], n, {g25, g26}); // C28 = C25 * C26
    auto g29 = add_gemm(graph, d_C[25], d_C[26], d_C[28], n, {g26, g27}); // C29 = C26 * C27
    add_gemm(graph, d_C[27], d_C[28], d_C[29], n, {g28, g29});          // C30 = C28 * C29

    return graph;
}

// Ordinary launches in topological order on one stream: each GEMM waits for
// the previous one. The caller checks launch errors after submitting the batch.
void launch_sequential(const float* d_A, const float* d_B, float* const* d_C,
                       int n, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    gemm<<<grid, block, 0, stream>>>(d_A, d_B, d_C[0], n);
    gemm<<<grid, block, 0, stream>>>(d_B, d_A, d_C[1], n);
    gemm<<<grid, block, 0, stream>>>(d_C[0], d_A, d_C[2], n);
    gemm<<<grid, block, 0, stream>>>(d_C[0], d_C[1], d_C[3], n);
    gemm<<<grid, block, 0, stream>>>(d_C[2], d_C[3], d_C[4], n);
    gemm<<<grid, block, 0, stream>>>(d_C[1], d_B, d_C[5], n);
    gemm<<<grid, block, 0, stream>>>(d_C[2], d_B, d_C[6], n);
    gemm<<<grid, block, 0, stream>>>(d_C[3], d_A, d_C[7], n);
    gemm<<<grid, block, 0, stream>>>(d_C[4], d_C[5], d_C[8], n);
    gemm<<<grid, block, 0, stream>>>(d_C[5], d_C[6], d_C[9], n);
    gemm<<<grid, block, 0, stream>>>(d_C[6], d_C[7], d_C[10], n);
    gemm<<<grid, block, 0, stream>>>(d_C[7], d_B, d_C[11], n);
    gemm<<<grid, block, 0, stream>>>(d_C[8], d_C[9], d_C[12], n);
    gemm<<<grid, block, 0, stream>>>(d_C[9], d_C[10], d_C[13], n);
    gemm<<<grid, block, 0, stream>>>(d_C[10], d_C[11], d_C[14], n);
    gemm<<<grid, block, 0, stream>>>(d_C[11], d_A, d_C[15], n);
    gemm<<<grid, block, 0, stream>>>(d_C[12], d_C[13], d_C[16], n);
    gemm<<<grid, block, 0, stream>>>(d_C[13], d_C[14], d_C[17], n);
    gemm<<<grid, block, 0, stream>>>(d_C[14], d_C[15], d_C[18], n);
    gemm<<<grid, block, 0, stream>>>(d_C[15], d_B, d_C[19], n);
    gemm<<<grid, block, 0, stream>>>(d_C[16], d_C[17], d_C[20], n);
    gemm<<<grid, block, 0, stream>>>(d_C[17], d_C[18], d_C[21], n);
    gemm<<<grid, block, 0, stream>>>(d_C[18], d_C[19], d_C[22], n);
    gemm<<<grid, block, 0, stream>>>(d_C[19], d_A, d_C[23], n);
    gemm<<<grid, block, 0, stream>>>(d_C[20], d_C[21], d_C[24], n);
    gemm<<<grid, block, 0, stream>>>(d_C[21], d_C[22], d_C[25], n);
    gemm<<<grid, block, 0, stream>>>(d_C[22], d_C[23], d_C[26], n);
    gemm<<<grid, block, 0, stream>>>(d_C[24], d_C[25], d_C[27], n);
    gemm<<<grid, block, 0, stream>>>(d_C[25], d_C[26], d_C[28], n);
    gemm<<<grid, block, 0, stream>>>(d_C[27], d_C[28], d_C[29], n);
}

void reference_dag(const std::vector<float>& A, const std::vector<float>& B,
                   std::vector<float>* reference, int n) {
    const size_t elements = static_cast<size_t>(n) * n;
    for (int node = 0; node < num_gemms; ++node) reference[node].resize(elements);
    gemm_cpu(A,            B,            reference[0], n);
    gemm_cpu(B,            A,            reference[1], n);
    gemm_cpu(reference[0], A,            reference[2], n);
    gemm_cpu(reference[0], reference[1], reference[3], n);
    gemm_cpu(reference[2], reference[3], reference[4], n);

    gemm_cpu(reference[1], B,             reference[5],  n);
    gemm_cpu(reference[2], B,             reference[6],  n);
    gemm_cpu(reference[3], A,             reference[7],  n);
    gemm_cpu(reference[4], reference[5],  reference[8],  n);
    gemm_cpu(reference[5], reference[6],  reference[9],  n);
    gemm_cpu(reference[6], reference[7],  reference[10], n);
    gemm_cpu(reference[7], B,             reference[11], n);

    gemm_cpu(reference[8],  reference[9],  reference[12], n);
    gemm_cpu(reference[9],  reference[10], reference[13], n);
    gemm_cpu(reference[10], reference[11], reference[14], n);
    gemm_cpu(reference[11], A,             reference[15], n);

    gemm_cpu(reference[12], reference[13], reference[16], n);
    gemm_cpu(reference[13], reference[14], reference[17], n);
    gemm_cpu(reference[14], reference[15], reference[18], n);
    gemm_cpu(reference[15], B,             reference[19], n);

    gemm_cpu(reference[16], reference[17], reference[20], n);
    gemm_cpu(reference[17], reference[18], reference[21], n);
    gemm_cpu(reference[18], reference[19], reference[22], n);
    gemm_cpu(reference[19], A,             reference[23], n);

    gemm_cpu(reference[20], reference[21], reference[24], n);
    gemm_cpu(reference[21], reference[22], reference[25], n);
    gemm_cpu(reference[22], reference[23], reference[26], n);
    gemm_cpu(reference[24], reference[25], reference[27], n);
    gemm_cpu(reference[25], reference[26], reference[28], n);
    gemm_cpu(reference[27], reference[28], reference[29], n);

}

bool verify_outputs(float* const* d_C, const std::vector<float>* reference,
                    int n, double& max_error, bool verbose = false) {
    const size_t elements = static_cast<size_t>(n) * n;
    std::vector<float> result(elements);
    max_error = 0.0;
    bool all_ok = true;
    for (int node = 0; node < num_gemms; ++node) {
        CUDA_CHECK(cudaMemcpy(result.data(), d_C[node], elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double node_error = 0.0;
        bool ok = true;
        for (size_t i = 0; i < elements; ++i) {
            if (!std::isfinite(result[i]) || !std::isfinite(reference[node][i])) {
                ok = false;
                node_error = INFINITY;
                continue;
            }
            double error = std::fabs(static_cast<double>(result[i]) - reference[node][i]);
            node_error = std::max(node_error, error);
            double tolerance = 1e-6 + 1e-4 * std::fabs(reference[node][i]);
            if (error > tolerance) ok = false;
        }
        if (verbose || !ok) {
            std::fprintf(stderr, "G%d: %s (max absolute error: %.3e)\n",
                         node + 1, ok ? "PASS" : "FAIL", node_error);
        }
        max_error = std::max(max_error, node_error);
        all_ok = all_ok && ok;
    }
    return all_ok;
}
