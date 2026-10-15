// GeMM-SpMM tile fusion example (Fig. 2f) as a CUDA graph, small enough to
// check the printed schedule by hand:
//   D1 = B C, then D = A D1, with A n x n sparse (CSR), and B n x k and C k x k
//   dense and row-major.
// Rows are split into tiles of -ip rows. First-level node f runs the GeMM for
// tile f, then the SpMM for the rows of tile f that read only D1 rows of that
// tile. Each second-level node runs the SpMM for consecutive leftover rows
// that read the same tiles, and waits only for those tiles.
// The graph is built once with hand-written kernels and once with cuBLAS and
// cuSPARSE.
//
// Names starting with "device" point to GPU memory. In the repo's
// CudaTensorInputs, DBx is B, and DACsrAp, DACsrI and DACsrVal are A's CSR row
// pointers, column indices and values.
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

template <class T> T *copyToDevice(const std::vector<T> &Host) {
  T *device;
  CHECK(cudaMalloc(&device, Host.size() * sizeof(T)));
  CHECK(cudaMemcpy(device, Host.data(), Host.size() * sizeof(T),
                   cudaMemcpyHostToDevice));
  return device;
}

std::vector<float> copyToHost(const float *Device, int Count) {
  std::vector<float> host(Count);
  CHECK(cudaMemcpy(host.data(), Device, Count * sizeof(float),
                   cudaMemcpyDeviceToHost));
  return host;
}

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
  int GemmBegin, GemmEnd;     // D1 rows computed by GeMM; empty in level 2
  std::vector<int> SpMMRows;  // D rows computed by SpMM, after the GeMM
  std::vector<int> DependsOn; // tiles the SpMM waits for; empty in level 1
};

// Builds the Fig. 2f schedule, first-level nodes first, so node f is tile f.
// A->i[nz] is the column of nonzero nz, i.e. the row of D1 that it reads.
std::vector<Node> buildSchedule(const CSR *A, int RowsPerTile) {
  std::vector<Node> nodes;
  std::vector<int> leftover; // SpMM rows that read D1 rows of another tile
  for (int begin = 0; begin < A->m; begin += RowsPerTile) {
    int end = MIN(begin + RowsPerTile, A->m);
    Node node{begin, end, {}, {}};
    for (int row = begin; row < end; ++row) {
      bool readsOnlyThisTile =
          std::all_of(A->i + A->p[row], A->i + A->p[row + 1],
                      [&](int col) { return col >= begin && col < end; });
      (readsOnlyThisTile ? node.SpMMRows : leftover).push_back(row);
    }
    nodes.push_back(node);
  }

  // cuSPARSE needs a contiguous block, so only consecutive rows share a node.
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

void printSchedule(const std::vector<Node> &Nodes) {
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    const Node &node = Nodes[nodeId];
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
}

// Allocates Count floats on the GPU, filled with a large value, so a row that
// no node writes, or that is read before it is written, fails the check.
float *allocOutput(int Count) {
  float *deviceOut;
  CHECK(cudaMalloc(&deviceOut, Count * sizeof(float)));
  CHECK(cudaMemset(deviceOut, 0x7F, Count * sizeof(float)));
  return deviceOut;
}

// Captures each node's launches (LaunchNode) into a child graph, adds it with
// an edge from each tile it depends on, and runs the graph once. Library calls
// can only enter a graph through capture. Within a node the GeMM runs before
// the SpMM, since both go to one stream.
void buildAndLaunchGraph(const std::vector<Node> &Nodes, cudaStream_t Stream,
                         const std::function<void(size_t)> &LaunchNode) {
  cudaGraph_t graph;
  CHECK(cudaGraphCreate(&graph, 0));
  std::vector<cudaGraphNode_t> graphNodes(Nodes.size());
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    CHECK(cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal));
    LaunchNode(nodeId);
    CHECK(cudaGetLastError()); // kernel launches report errors here
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
  // The setup fills and copies ran on the default stream, which a non-blocking
  // stream does not wait for.
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGraphLaunch(graphExec, Stream));
  CHECK(cudaStreamSynchronize(Stream));
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
}

// The hand-written SpMM kernel takes the list of rows to compute.
std::vector<int *> uploadSpMMRows(const std::vector<Node> &Nodes) {
  std::vector<int *> deviceSpMMRows(Nodes.size(), nullptr);
  for (size_t nodeId = 0; nodeId < Nodes.size(); ++nodeId) {
    if (!Nodes[nodeId].SpMMRows.empty()) {
      deviceSpMMRows[nodeId] = copyToDevice(Nodes[nodeId].SpMMRows);
    }
  }
  return deviceSpMMRows;
}

std::vector<float> runGraphHandWritten(CudaTensorInputs *In,
                                       const float *DeviceC,
                                       const std::vector<Node> &Nodes) {
  int n = In->M, k = In->N;
  float *deviceD1 = allocOutput(n * k), *deviceD = allocOutput(n * k);
  std::vector<int *> deviceSpMMRows = uploadSpMMRows(Nodes);
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  auto launchNode = [&](size_t nodeId) {
    const Node &node = Nodes[nodeId];
    int numGemmRows = node.GemmEnd - node.GemmBegin;
    int numSpMMRows = node.SpMMRows.size();
    if (numGemmRows > 0) {
      gemmKernel<<<CEIL(numGemmRows * k, 128), 128, 0, stream>>>(
          numGemmRows, k, In->DBx + node.GemmBegin * k, DeviceC,
          deviceD1 + node.GemmBegin * k);
    }
    if (numSpMMRows > 0) {
      // One thread per element of D: threadIdx.x is the column, threadIdx.y
      // the row.
      int colsPerBlock = MIN(k, 128);
      dim3 block(colsPerBlock, CEIL(128, colsPerBlock));
      dim3 grid(CEIL(numSpMMRows, block.y), CEIL(k, colsPerBlock));
      csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<grid, block, 0,
                                                             stream>>>(
          numSpMMRows, k, n, In->DACsrAp, In->DACsrI, In->DACsrVal, deviceD1,
          deviceD, deviceSpMMRows[nodeId]);
    }
  };
  buildAndLaunchGraph(Nodes, stream, launchNode);

  std::vector<float> hostD = copyToHost(deviceD, n * k);
  cudaStreamDestroy(stream);
  for (int *rows : deviceSpMMRows) {
    cudaFree(rows);
  }
  cudaFree(deviceD1);
  cudaFree(deviceD);
  return hostD;
}

// cuBLAS and cuSPARSE compute out = Alpha * product + Beta * out.
const float Alpha = 1, Beta = 0;

struct LibraryHandles {
  cublasHandle_t Cublas;
  cusparseHandle_t Cusparse;
};

LibraryHandles createLibraryHandles(cudaStream_t Stream) {
  LibraryHandles handles;
  CHECK(cublasCreate(&handles.Cublas));
  CHECK(cublasSetStream(handles.Cublas, Stream));
  // Without workspace the captured GeMM has no allocation nodes, which child
  // graphs do not allow (cublasSetStream resets it, so set it after).
  CHECK(cublasSetWorkspace(handles.Cublas, nullptr, 0));
  CHECK(cusparseCreate(&handles.Cusparse));
  CHECK(cusparseSetStream(handles.Cusparse, Stream));
  return handles;
}

// cuSPARSE inputs for one node. cusparseSpMM has no row list, so each node gets
// a CSR view of its (consecutive) rows of A, which reuses A's column and value
// arrays with new row pointers, and a dense view of the same rows of D.
struct NodeSpMM {
  int *DeviceRowPtr = nullptr;
  cusparseSpMatDescr_t A = nullptr;
  cusparseDnMatDescr_t D = nullptr;
  void *DeviceBuffer = nullptr; // cuSPARSE work space
};

std::vector<NodeSpMM> createNodeSpMMs(CudaTensorInputs *In,
                                      const std::vector<Node> &Nodes,
                                      cusparseHandle_t Cusparse,
                                      cusparseDnMatDescr_t D1Descr,
                                      float *DeviceD) {
  int n = In->M, k = In->N;
  const int *aRowPtr = In->ACsr->p; // CSR row pointers of A, on the host
  std::vector<NodeSpMM> nodeSpMMs(Nodes.size());
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
    std::vector<int> rowPtr;
    for (int row = firstRow; row <= firstRow + numRows; ++row) {
      rowPtr.push_back(aRowPtr[row] - aRowPtr[firstRow]);
    }
    NodeSpMM &spmm = nodeSpMMs[nodeId];
    spmm.DeviceRowPtr = copyToDevice(rowPtr);
    CHECK(cusparseCreateCsr(&spmm.A, numRows, n, rowPtr.back(),
                            spmm.DeviceRowPtr, In->DACsrI + aRowPtr[firstRow],
                            In->DACsrVal + aRowPtr[firstRow],
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK(cusparseCreateDnMat(&spmm.D, numRows, k, k, DeviceD + firstRow * k,
                              CUDA_R_32F, CUSPARSE_ORDER_ROW));
    size_t bufferBytes = 0;
    CHECK(cusparseSpMM_bufferSize(Cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha,
                                  spmm.A, D1Descr, &Beta, spmm.D, CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, &bufferBytes));
    CHECK(cudaMalloc(&spmm.DeviceBuffer, bufferBytes));
  }
  return nodeSpMMs;
}

void destroyNodeSpMMs(const std::vector<NodeSpMM> &NodeSpMMs) {
  for (const NodeSpMM &spmm : NodeSpMMs) {
    if (spmm.A) {
      cusparseDestroySpMat(spmm.A);
      cusparseDestroyDnMat(spmm.D);
    }
    cudaFree(spmm.DeviceRowPtr);
    cudaFree(spmm.DeviceBuffer);
  }
}

std::vector<float> runGraphLibraries(CudaTensorInputs *In, const float *DeviceC,
                                     const std::vector<Node> &Nodes) {
  int n = In->M, k = In->N;
  float *deviceD1 = allocOutput(n * k), *deviceD = allocOutput(n * k);
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  LibraryHandles handles = createLibraryHandles(stream);
  cusparseDnMatDescr_t d1Descr;
  CHECK(cusparseCreateDnMat(&d1Descr, n, k, k, deviceD1, CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));
  std::vector<NodeSpMM> nodeSpMMs =
      createNodeSpMMs(In, Nodes, handles.Cusparse, d1Descr, deviceD);

  auto launchNode = [&](size_t nodeId) {
    const Node &node = Nodes[nodeId];
    int numGemmRows = node.GemmEnd - node.GemmBegin;
    if (numGemmRows > 0) {
      // cuBLAS is column-major, and reads a row-major matrix as its transpose.
      // So row-major D1 = B C is column-major D1^T = C^T B^T: C goes first.
      CHECK(cublasSgemm(handles.Cublas, CUBLAS_OP_N, CUBLAS_OP_N, k,
                        numGemmRows, k, &Alpha, DeviceC, k,
                        In->DBx + node.GemmBegin * k, k, &Beta,
                        deviceD1 + node.GemmBegin * k, k));
    }
    if (!node.SpMMRows.empty()) {
      const NodeSpMM &spmm = nodeSpMMs[nodeId];
      CHECK(cusparseSpMM(handles.Cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, spmm.A,
                         d1Descr, &Beta, spmm.D, CUDA_R_32F,
                         CUSPARSE_SPMM_ALG_DEFAULT, spmm.DeviceBuffer));
    }
  };
  buildAndLaunchGraph(Nodes, stream, launchNode);

  std::vector<float> hostD = copyToHost(deviceD, n * k);
  destroyNodeSpMMs(nodeSpMMs);
  cusparseDestroyDnMat(d1Descr);
  cusparseDestroy(handles.Cusparse);
  cublasDestroy(handles.Cublas);
  cudaStreamDestroy(stream);
  cudaFree(deviceD1);
  cudaFree(deviceD);
  return hostD;
}

// Reads A from the -sm file (both triangles if it is stored as symmetric) and
// gives it values that cycle through 7 levels, so a row or value pointer that
// is off by 1 to 6 nonzeros changes D.
CSC *loadA(TestParameters *Tp) {
  CSC *stored = get_matrix_from_parameter(Tp);
  CSC *a = stored->stype == 0 ? copy_sparse(stored) : make_full(stored);
  delete stored;
  for (int j = 0; j < a->nnz; ++j) {
    a->x[j] = 0.5 + j % 7 / 7.0;
  }
  return a;
}

// B is n x k with values in [-1, 1].
void fillB(CudaTensorInputs *In) {
  int size = In->M * In->N;
  for (int i = 0; i < size; ++i) {
    In->Bx[i] = (i * 37 % 101 - 50) / 50.0f;
  }
  CHECK(cudaMemcpy(In->DBx, In->Bx, size * sizeof(float),
                   cudaMemcpyHostToDevice));
}

// C is k x k with values in [-1/k, 1/k], so D1 = B C stays in [-1, 1].
std::vector<float> makeC(int K) {
  std::vector<float> c(K * K);
  for (int i = 0; i < K * K; ++i) {
    c[i] = (i * 13 % 29 - 14) / (14.0f * K);
  }
  return c;
}

// D = A (B C) in double on the CPU, using the repo's sequential SpMM.
std::vector<double> computeReference(CudaTensorInputs *In,
                                     const std::vector<float> &C) {
  int n = In->M, k = In->N;
  std::vector<double> d1(n * k), d(n * k);
  for (int i = 0; i < n; ++i) {
    for (int l = 0; l < k; ++l) {
      for (int j = 0; j < k; ++j) {
        d1[i * k + j] += double(In->Bx[i * k + l]) * C[l * k + j];
      }
    }
  }
  swiftware::sparse::spmmCsrSequential<double>(
      n, k, n, In->ACsr->p, In->ACsr->i, In->ACsr->x, d1.data(), d.data());
  return d;
}

bool checkResult(const char *Version, const std::vector<float> &D,
                 const std::vector<double> &Reference) {
  double maxError = 0;
  bool ok = true;
  for (size_t i = 0; i < D.size(); ++i) {
    double error = std::abs(D[i] - Reference[i]);
    maxError = std::max(maxError, error);
    ok = ok && error < 1e-4; // also false for NaN
  }
  std::cout << Version << ": max abs error " << maxError
            << (ok ? " (ok)\n" : " (FAILED)\n");
  return ok;
}

int main(const int argc, const char *argv[]) {
  // The repo's options: -sm matrix file, -bc k, -ip rows per tile.
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  sp.IterPerPartition = 8; // rows per tile unless -ip is given
  parse_args(argc, argv, &sp, &tp);
  CSC *a = loadA(&tp);
  int n = a->m, k = tp._b_cols;
  int rowsPerTile = std::max(1, sp.IterPerPartition);
  // Holds A (as CSR) and B, on the host and on the GPU.
  auto *inputs =
      new CudaTensorInputs(n, k, n, n, a, a, 1, 1, "GeMM_SpMM_Graph_Example");
  fillB(inputs);
  std::vector<float> hostC = makeC(k);
  float *deviceC = copyToDevice(hostC);
  std::vector<double> referenceD = computeReference(inputs, hostC);

  std::vector<Node> nodes = buildSchedule(inputs->ACsr, rowsPerTile);
  std::cout << tp._matrix_name << ": n = " << n
            << ", nnz = " << inputs->ACsr->nnz << ", k = " << k << ", "
            << rowsPerTile << " rows per tile\n";
  printSchedule(nodes);

  std::vector<float> handWrittenD = runGraphHandWritten(inputs, deviceC, nodes);
  bool handWrittenOk =
      checkResult("hand-written kernels", handWrittenD, referenceD);
  std::vector<float> librariesD = runGraphLibraries(inputs, deviceC, nodes);
  bool librariesOk = checkResult("cuBLAS + cuSPARSE", librariesD, referenceD);
  cudaFree(deviceC);
  delete inputs;
  delete a;
  return handWrittenOk && librariesOk ? 0 : 1;
}
