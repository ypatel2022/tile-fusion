// Small GeMM-SpMM tile fusion example (Fig. 2f) as a CUDA graph, for checking
// the schedule and the result by hand on a small banded matrix:
//   D1 = B C, then D = A D1, with A n x n sparse (CSR), B n x k, C k x k,
//   and the dense matrices row-major.
// Rows are split into tiles of t consecutive rows (-ip).
// First-level node f: GeMM for rows [f t, (f + 1) t) of D1, then SpMM for the
// rows of D in that tile that read only D1 rows of the same tile.
// Second-level node: SpMM for consecutive leftover rows that read the same
// tiles. It waits only for the first-level nodes of those tiles.
// The same graph is built twice: with hand-written GeMM and SpMM kernels
// (runGraphHandWritten), and with cuBLAS GeMM and cuSPARSE SpMM
// (runGraphLibraries).
//
// Names starting with "device" point to GPU memory. The repo's
// CudaTensorInputs holds the GPU copies of the inputs: DBx is B, and DACsrAp,
// DACsrI and DACsrVal are the row pointers, column indices and values of A.
#include "../Cuda_SpMM_Demo_Utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <set>
#include <vector>

// CHECK(call) exits with the call and the library's error message if a CUDA,
// cuBLAS or cuSPARSE call fails. Success is 0 in all three.
const char *statusName(cudaError_t S) { return cudaGetErrorString(S); }
const char *statusName(cublasStatus_t S) { return cublasGetStatusString(S); }
const char *statusName(cusparseStatus_t S) { return cusparseGetErrorString(S); }
template <class Status> void check(Status S, const char *Call) {
  if (S != 0) {
    std::cerr << Call << ": " << statusName(S) << "\n";
    std::exit(1);
  }
}
#define CHECK(call) check((call), #call)

// Hand-written GeMM for one tile: D1 = B C for Rows rows, where B and D1 point
// to the tile's first row. One thread per element of the Rows x K output.
__global__ void gemmKernel(int Rows, int K, const float *B, const float *C,
                           float *D1) {
  int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < Rows * K) {
    int row = element / K, col = element % K;
    float sum = 0;
    for (int l = 0; l < K; ++l) {
      sum += B[row * K + l] * C[l * K + col];
    }
    D1[row * K + col] = sum;
  }
}

// One node of the schedule, which becomes one node of the CUDA graph.
struct Node {
  // Rows [GemmBegin, GemmEnd) of D1 computed by GeMM; empty in level 2.
  int GemmBegin, GemmEnd;
  // Rows of D computed by SpMM, after the GeMM.
  std::vector<int> SpMMRows;
  // Tiles (= first-level nodes) whose D1 rows the SpMM reads; empty in level 1.
  std::vector<int> DependsOn;
};

// Builds the Fig. 2f schedule. The first-level nodes come first, so node f is
// tile f. In CSR, the nonzeros of row r of A are at positions A->p[r] to
// A->p[r + 1] - 1, and A->i[nz] is the column of nonzero nz, which is the row
// of D1 that it reads.
std::vector<Node> buildSchedule(const CSR *A, int RowsPerTile) {
  std::vector<Node> nodes;
  std::vector<int> leftover; // SpMM rows that read D1 rows of another tile

  // First level: one node per tile, with the SpMM rows that read only it.
  for (int begin = 0; begin < A->m; begin += RowsPerTile) {
    int end = MIN(begin + RowsPerTile, A->m);
    Node node{begin, end, {}, {}};
    for (int row = begin; row < end; ++row) {
      bool readsOnlyThisTile = true;
      for (int nz = A->p[row]; nz < A->p[row + 1]; ++nz) {
        readsOnlyThisTile =
            readsOnlyThisTile && A->i[nz] >= begin && A->i[nz] < end;
      }
      (readsOnlyThisTile ? node.SpMMRows : leftover).push_back(row);
    }
    nodes.push_back(node);
  }

  // Second level: consecutive leftover rows (cuSPARSE needs a contiguous
  // block) that read the same tiles share a node, which depends on exactly
  // those tiles.
  auto tilesReadBy = [&](int Row) {
    std::set<int> tiles;
    for (int nz = A->p[Row]; nz < A->p[Row + 1]; ++nz) {
      tiles.insert(A->i[nz] / RowsPerTile);
    }
    return tiles;
  };
  for (size_t groupBegin = 0; groupBegin < leftover.size();) {
    std::set<int> tiles = tilesReadBy(leftover[groupBegin]);
    size_t groupEnd = groupBegin + 1;
    while (groupEnd < leftover.size() &&
           leftover[groupEnd] == leftover[groupEnd - 1] + 1 &&
           tilesReadBy(leftover[groupEnd]) == tiles) {
      ++groupEnd;
    }
    nodes.push_back({0, 0,
                     std::vector<int>(leftover.begin() + groupBegin,
                                      leftover.begin() + groupEnd),
                     std::vector<int>(tiles.begin(), tiles.end())});
    groupBegin = groupEnd;
  }
  return nodes;
}

// Allocates Count floats on the GPU, filled with a large value, so an output
// row no node writes, or an SpMM that runs before the GeMM it reads, cannot
// match the reference.
float *allocOutput(int Count) {
  float *deviceOut;
  CHECK(cudaMalloc(&deviceOut, Count * sizeof(float)));
  CHECK(cudaMemset(deviceOut, 0x7F, Count * sizeof(float)));
  return deviceOut;
}

// Builds the CUDA graph of the schedule and runs it once. LaunchNode(nodeId)
// launches that node's GeMM and SpMM on Stream. cuBLAS and cuSPARSE calls can
// only enter a graph through stream capture, so the launches are captured into
// a child graph (launches on one stream run in order, so the GeMM finishes
// before the SpMM), which is added with an edge from each tile the node
// depends on. Those tiles' graph nodes already exist, since first-level nodes
// come first.
void buildAndLaunchGraph(const std::vector<Node> &Nodes, cudaStream_t Stream,
                         const std::function<void(size_t)> &LaunchNode) {
  cudaGraph_t graph;
  CHECK(cudaGraphCreate(&graph, 0));
  std::vector<cudaGraphNode_t> graphNodes(Nodes.size()); // one per Nodes entry
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    CHECK(cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal));
    LaunchNode(nodeId);
    CHECK(cudaGetLastError()); // catches a kernel launch that failed
    cudaGraph_t nodeGraph;
    CHECK(cudaStreamEndCapture(Stream, &nodeGraph));
    std::vector<cudaGraphNode_t> dependencies;
    for (int tile : Nodes[nodeId].DependsOn) {
      dependencies.push_back(graphNodes[tile]);
    }
    CHECK(cudaGraphAddChildGraphNode(&graphNodes[nodeId], graph,
                                     dependencies.data(), dependencies.size(),
                                     nodeGraph));
    cudaGraphDestroy(nodeGraph); // the graph keeps its own copy
  }
  cudaGraphExec_t graphExec;
  CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
  // The fills and copies before this ran on the default stream, which a
  // non-blocking stream does not wait for.
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGraphLaunch(graphExec, Stream));
  CHECK(cudaStreamSynchronize(Stream));
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
}

// Hand-written version: gemmKernel for the GeMM, and the repo's SpMM kernel
// over a list of rows. Returns D on the host.
std::vector<float> runGraphHandWritten(CudaTensorInputs *In,
                                       const float *DeviceC,
                                       const std::vector<Node> &Nodes) {
  int n = In->M, k = In->N;
  float *deviceD1 = allocOutput(n * k), *deviceD = allocOutput(n * k);
  // The SpMM kernel takes the list of rows to compute, one list per node.
  std::vector<int *> deviceSpMMRows(Nodes.size(), nullptr);
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    const std::vector<int> &rows = Nodes[nodeId].SpMMRows;
    if (!rows.empty()) {
      CHECK(cudaMalloc(&deviceSpMMRows[nodeId], rows.size() * sizeof(int)));
      CHECK(cudaMemcpy(deviceSpMMRows[nodeId], rows.data(),
                       rows.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
  }

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  buildAndLaunchGraph(Nodes, stream, [&](size_t nodeId) {
    const Node &node = Nodes[nodeId];
    int numGemmRows = node.GemmEnd - node.GemmBegin;
    int numSpMMRows = node.SpMMRows.size();
    if (numGemmRows > 0) {
      gemmKernel<<<CEIL(numGemmRows * k, 128), 128, 0, stream>>>(
          numGemmRows, k, In->DBx + node.GemmBegin * k, DeviceC,
          deviceD1 + node.GemmBegin * k);
    }
    if (numSpMMRows > 0) {
      // One thread per element of D, at least 128 threads per block.
      // threadIdx.x picks the column and threadIdx.y the row; columns past the
      // first 128 go to more blocks in blockIdx.y.
      int colsPerBlock = MIN(k, 128);
      dim3 block(colsPerBlock, CEIL(128, colsPerBlock));
      dim3 grid(CEIL(numSpMMRows, block.y), CEIL(k, colsPerBlock));
      csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<grid, block, 0,
                                                             stream>>>(
          numSpMMRows, k, n, In->DACsrAp, In->DACsrI, In->DACsrVal, deviceD1,
          deviceD, deviceSpMMRows[nodeId]);
    }
  });

  std::vector<float> hostD(n * k);
  CHECK(cudaMemcpy(hostD.data(), deviceD, hostD.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  cudaStreamDestroy(stream);
  for (int *rows : deviceSpMMRows) {
    cudaFree(rows);
  }
  cudaFree(deviceD1);
  cudaFree(deviceD);
  return hostD;
}

// Library version: cuBLAS for the GeMM and cuSPARSE for the SpMM. Returns D on
// the host.
std::vector<float> runGraphLibraries(CudaTensorInputs *In, const float *DeviceC,
                                     const std::vector<Node> &Nodes) {
  int n = In->M, k = In->N;
  const int *aRowPtr = In->ACsr->p; // CSR row pointers of A, on the host
  float *deviceD1 = allocOutput(n * k), *deviceD = allocOutput(n * k);
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t cublasHandle;
  CHECK(cublasCreate(&cublasHandle));
  CHECK(cublasSetStream(cublasHandle, stream));
  // Without workspace the captured GeMM has no allocation nodes, which child
  // graphs do not allow (cublasSetStream resets it, so set it after).
  CHECK(cublasSetWorkspace(cublasHandle, nullptr, 0));
  cusparseHandle_t cusparseHandle;
  CHECK(cusparseCreate(&cusparseHandle));
  CHECK(cusparseSetStream(cusparseHandle, stream));
  cusparseDnMatDescr_t d1Descr; // all of D1, as cuSPARSE sees it
  CHECK(cusparseCreateDnMat(&d1Descr, n, k, k, deviceD1, CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));

  // cusparseSpMM takes no row list and computes every row of its CSR input, so
  // each node gets a CSR view of just its rows of A, which must be consecutive
  // (checked below): new row pointers starting at 0, and A's own column and
  // value arrays from the node's first nonzero. It also gets a dense view of
  // the same rows of D.
  std::vector<int *> deviceNodeRowPtr(Nodes.size(), nullptr);
  std::vector<cusparseSpMatDescr_t> nodeADescr(Nodes.size(), nullptr);
  std::vector<cusparseDnMatDescr_t> nodeDDescr(Nodes.size(), nullptr);
  std::vector<void *> deviceSpMMBuffer(Nodes.size(), nullptr);
  // Both libraries compute out = alpha * product + beta * out.
  float alpha = 1, beta = 0;
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    const std::vector<int> &rows = Nodes[nodeId].SpMMRows;
    if (rows.empty()) {
      continue;
    }
    int firstRow = rows.front(), numRows = rows.size();
    if (rows.back() - firstRow + 1 != numRows) {
      std::cerr << "cuSPARSE needs contiguous rows in every node\n";
      std::exit(1);
    }
    std::vector<int> nodeRowPtr;
    for (int row = firstRow; row <= firstRow + numRows; ++row) {
      nodeRowPtr.push_back(aRowPtr[row] - aRowPtr[firstRow]);
    }
    CHECK(
        cudaMalloc(&deviceNodeRowPtr[nodeId], nodeRowPtr.size() * sizeof(int)));
    CHECK(cudaMemcpy(deviceNodeRowPtr[nodeId], nodeRowPtr.data(),
                     nodeRowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cusparseCreateCsr(
        &nodeADescr[nodeId], numRows, n, nodeRowPtr.back(),
        deviceNodeRowPtr[nodeId], In->DACsrI + aRowPtr[firstRow],
        In->DACsrVal + aRowPtr[firstRow], CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK(cusparseCreateDnMat(&nodeDDescr[nodeId], numRows, k, k,
                              deviceD + firstRow * k, CUDA_R_32F,
                              CUSPARSE_ORDER_ROW));
    size_t bufferBytes = 0;
    CHECK(cusparseSpMM_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, nodeADescr[nodeId], d1Descr,
        &beta, nodeDDescr[nodeId], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferBytes));
    CHECK(cudaMalloc(&deviceSpMMBuffer[nodeId], bufferBytes));
  }

  buildAndLaunchGraph(Nodes, stream, [&](size_t nodeId) {
    const Node &node = Nodes[nodeId];
    int numGemmRows = node.GemmEnd - node.GemmBegin;
    if (numGemmRows > 0) {
      // cuBLAS is column-major, and reads a row-major matrix as its transpose.
      // So row-major D1 = B C is column-major D1^T = C^T B^T: C goes first.
      CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, k, numGemmRows,
                        k, &alpha, DeviceC, k, In->DBx + node.GemmBegin * k, k,
                        &beta, deviceD1 + node.GemmBegin * k, k));
    }
    if (!node.SpMMRows.empty()) {
      CHECK(cusparseSpMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                         nodeADescr[nodeId], d1Descr, &beta, nodeDDescr[nodeId],
                         CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                         deviceSpMMBuffer[nodeId]));
    }
  });

  std::vector<float> hostD(n * k);
  CHECK(cudaMemcpy(hostD.data(), deviceD, hostD.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    if (nodeADescr[nodeId]) {
      cusparseDestroySpMat(nodeADescr[nodeId]);
      cusparseDestroyDnMat(nodeDDescr[nodeId]);
    }
    cudaFree(deviceNodeRowPtr[nodeId]);
    cudaFree(deviceSpMMBuffer[nodeId]);
  }
  cusparseDestroyDnMat(d1Descr);
  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
  cudaStreamDestroy(stream);
  cudaFree(deviceD1);
  cudaFree(deviceD);
  return hostD;
}

int main(const int argc, const char *argv[]) {
  // The repo's options: -sm matrix file, -bc k, -ip rows per tile.
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  sp.IterPerPartition = 8; // rows per tile unless -ip is given
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  // A symmetric .mtx file stores one triangle; make_full adds the other.
  CSC *aCSCFull = aCSC->stype == 0 ? copy_sparse(aCSC) : make_full(aCSC);
  delete aCSC;
  // Values cycle through 7 levels, so a row or value pointer that is off by 1
  // to 6 nonzeros changes D.
  for (int j = 0; j < aCSCFull->nnz; ++j) {
    aCSCFull->x[j] = 0.5 + j % 7 / 7.0;
  }
  int n = aCSCFull->m, k = tp._b_cols;
  int rowsPerTile = std::max(1, sp.IterPerPartition);
  // Holds A (as CSR) and B, on the host and on the GPU.
  auto *inputs = new CudaTensorInputs(n, k, n, n, aCSCFull, aCSCFull, 1, 1,
                                      "GeMM_SpMM_Graph_Example");
  // B is n x k with values in [-1, 1].
  for (int i = 0; i < n * k; ++i) {
    inputs->Bx[i] = (i * 37 % 101 - 50) / 50.0f;
  }
  CHECK(cudaMemcpy(inputs->DBx, inputs->Bx, n * k * sizeof(float),
                   cudaMemcpyHostToDevice));
  // C is k x k with values in [-1/k, 1/k], so D1 = B C stays in [-1, 1].
  std::vector<float> hostC(k * k);
  for (int i = 0; i < k * k; ++i) {
    hostC[i] = (i * 13 % 29 - 14) / (14.0f * k);
  }
  float *deviceC;
  CHECK(cudaMalloc(&deviceC, hostC.size() * sizeof(float)));
  CHECK(cudaMemcpy(deviceC, hostC.data(), hostC.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  // Reference in double: D1 = B C on the CPU, then the repo's sequential SpMM.
  std::vector<double> referenceD1(n * k), referenceD(n * k);
  for (int i = 0; i < n; ++i) {
    for (int l = 0; l < k; ++l) {
      for (int j = 0; j < k; ++j) {
        referenceD1[i * k + j] +=
            double(inputs->Bx[i * k + l]) * hostC[l * k + j];
      }
    }
  }
  swiftware::sparse::spmmCsrSequential<double>(
      n, k, n, inputs->ACsr->p, inputs->ACsr->i, inputs->ACsr->x,
      referenceD1.data(), referenceD.data());

  std::vector<Node> nodes = buildSchedule(inputs->ACsr, rowsPerTile);
  std::cout << tp._matrix_name << ": n = " << n
            << ", nnz = " << inputs->ACsr->nnz << ", k = " << k << ", "
            << rowsPerTile << " rows per tile\n";
  for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
    const Node &node = nodes[nodeId];
    if (node.GemmEnd > node.GemmBegin) {
      std::cout << "first-level node " << nodeId << ": GeMM rows "
                << node.GemmBegin << "-" << node.GemmEnd - 1 << ", SpMM rows";
    } else {
      std::cout << "second-level node " << nodeId << ": SpMM rows";
    }
    for (int row : node.SpMMRows) {
      std::cout << " " << row;
    }
    if (!node.DependsOn.empty()) {
      std::cout << ", after nodes";
      for (int tile : node.DependsOn) {
        std::cout << " " << tile;
      }
    }
    std::cout << "\n";
  }

  bool correct = true;
  for (bool useLibraries : {false, true}) {
    std::vector<float> gpuD = useLibraries
                                  ? runGraphLibraries(inputs, deviceC, nodes)
                                  : runGraphHandWritten(inputs, deviceC, nodes);
    double maxError = 0;
    bool ok = true;
    for (size_t i = 0; i < gpuD.size(); ++i) {
      double error = std::abs(gpuD[i] - referenceD[i]);
      maxError = std::max(maxError, error);
      ok = ok && error < 1e-4; // also false for NaN
    }
    correct = correct && ok;
    std::cout << (useLibraries ? "cuBLAS + cuSPARSE" : "hand-written kernels")
              << ": max abs error " << maxError
              << (ok ? " (ok)\n" : " (FAILED)\n");
  }
  cudaFree(deviceC);
  delete inputs;
  delete aCSCFull;
  return correct ? 0 : 1;
}
