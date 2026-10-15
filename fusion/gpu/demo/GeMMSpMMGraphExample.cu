// Small GeMM-SpMM tile fusion example (Fig. 2f) as a CUDA graph, for checking
// the schedule and the result by hand on a small banded matrix:
//   D1 = B C, then D = A D1, with A n x n CSR, B n x k, C k x k, all row-major.
// First-level node f: GeMM for rows [f t, (f + 1) t) of D1, then SpMM for the
// rows of D whose nonzeros all lie in that block. Second-level node: SpMM for
// consecutive remaining rows that read the same tiles, after only those tiles.
// The same graph is built twice: with hand-written GeMM and SpMM kernels, and
// with cuBLAS GeMM and cuSPARSE SpMM.
#include "../Cuda_SpMM_Demo_Utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

const char *statusName(cudaError_t S) { return cudaGetErrorString(S); }
const char *statusName(cublasStatus_t S) { return cublasGetStatusString(S); }
const char *statusName(cusparseStatus_t S) { return cusparseGetErrorString(S); }
// Success is 0 for CUDA, cuBLAS and cuSPARSE.
template <class Status> void check(Status S, const char *Call) {
  if (S != 0) {
    std::cerr << Call << ": " << statusName(S) << "\n";
    std::exit(1);
  }
}
#define CHECK(call) check((call), #call)

// Hand-written GeMM: one thread per element of the Rows x K block of D1.
__global__ void gemmKernel(int Rows, int K, const float *B, const float *C,
                           float *D1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Rows * K) {
    int i = idx / K, j = idx % K;
    float sum = 0;
    for (int l = 0; l < K; ++l) {
      sum += B[i * K + l] * C[l * K + j];
    }
    D1[i * K + j] = sum;
  }
}

struct Node {
  int GemmBegin, GemmEnd;    // rows of D1 computed by GeMM; empty in level 2
  std::vector<int> SpMMRows; // rows of D computed by SpMM
  std::vector<int> Deps;     // first-level nodes whose D1 rows the SpMM reads
};

// Splits rows into tiles of t and builds both levels of the Fig. 2f schedule.
std::vector<Node> buildSchedule(const CSR *A, int t) {
  std::vector<Node> nodes;
  std::vector<int> leftover;
  for (int begin = 0; begin < A->m; begin += t) {
    int end = MIN(begin + t, A->m);
    Node node{begin, end, {}, {}};
    for (int row = begin; row < end; ++row) {
      bool local = true;
      for (int j = A->p[row]; j < A->p[row + 1]; ++j) {
        local = local && A->i[j] >= begin && A->i[j] < end;
      }
      (local ? node.SpMMRows : leftover).push_back(row);
    }
    nodes.push_back(node);
  }
  auto tilesRead = [&](int Row) {
    std::set<int> tiles;
    for (int j = A->p[Row]; j < A->p[Row + 1]; ++j) {
      tiles.insert(A->i[j] / t);
    }
    return tiles;
  };
  for (size_t first = 0; first < leftover.size();) {
    std::set<int> tiles = tilesRead(leftover[first]);
    size_t last = first + 1;
    while (last < leftover.size() && leftover[last] == leftover[last - 1] + 1 &&
           tilesRead(leftover[last]) == tiles) {
      ++last;
    }
    nodes.push_back(
        {0, 0,
         std::vector<int>(leftover.begin() + first, leftover.begin() + last),
         std::vector<int>(tiles.begin(), tiles.end())});
    first = last;
  }
  return nodes;
}

// Runs the graph once with either kernel version and returns D on the host.
std::vector<float> runGraph(CudaTensorInputs *In, const float *DC,
                            const std::vector<Node> &Nodes, bool UseLibraries) {
  int n = In->M, k = In->N;
  const int *ap = In->ACsr->p;
  float *dD1, *dD;
  CHECK(cudaMalloc(&dD1, n * k * sizeof(float)));
  CHECK(cudaMalloc(&dD, n * k * sizeof(float)));
  // A large value in every output, so a row no node writes, or an SpMM that
  // runs before the GeMM it reads, cannot match the reference.
  CHECK(cudaMemset(dD1, 0x7F, n * k * sizeof(float)));
  CHECK(cudaMemset(dD, 0x7F, n * k * sizeof(float)));

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t blas = nullptr;
  cusparseHandle_t sparse = nullptr;
  cusparseDnMatDescr_t dnD1 = nullptr;
  if (UseLibraries) {
    CHECK(cublasCreate(&blas));
    CHECK(cublasSetStream(blas, stream));
    // Without workspace the captured GeMM has no allocation nodes, which child
    // graphs do not allow (cublasSetStream resets it, so set it after).
    CHECK(cublasSetWorkspace(blas, nullptr, 0));
    CHECK(cusparseCreate(&sparse));
    CHECK(cusparseSetStream(sparse, stream));
    CHECK(cusparseCreateDnMat(&dnD1, n, k, k, dD1, CUDA_R_32F,
                              CUSPARSE_ORDER_ROW));
  }

  // Per node: its SpMM rows for the hand-written kernel, or a CSR descriptor
  // of those rows (zero-based row offsets, global columns) for cuSPARSE.
  std::vector<int *> dRows(Nodes.size()), dOffsets(Nodes.size());
  std::vector<cusparseSpMatDescr_t> spA(Nodes.size());
  std::vector<cusparseDnMatDescr_t> dnD(Nodes.size());
  std::vector<void *> buffers(Nodes.size(), nullptr);
  float one = 1, zero = 0;
  for (size_t v = 0; v < Nodes.size(); ++v) {
    const std::vector<int> &rows = Nodes[v].SpMMRows;
    if (rows.empty()) {
      continue;
    }
    CHECK(cudaMalloc(&dRows[v], rows.size() * sizeof(int)));
    CHECK(cudaMemcpy(dRows[v], rows.data(), rows.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    if (!UseLibraries) {
      continue;
    }
    int first = rows.front(), count = rows.size();
    if (rows.back() - first + 1 != count) {
      std::cerr << "cuSPARSE needs contiguous rows in every node\n";
      std::exit(1);
    }
    std::vector<int> offsets;
    for (int row = first; row <= first + count; ++row) {
      offsets.push_back(ap[row] - ap[first]);
    }
    CHECK(cudaMalloc(&dOffsets[v], offsets.size() * sizeof(int)));
    CHECK(cudaMemcpy(dOffsets[v], offsets.data(), offsets.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
    CHECK(cusparseCreateCsr(&spA[v], count, n, offsets.back(), dOffsets[v],
                            In->DACsrI + ap[first], In->DACsrVal + ap[first],
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK(cusparseCreateDnMat(&dnD[v], count, k, k, dD + first * k, CUDA_R_32F,
                              CUSPARSE_ORDER_ROW));
    size_t bytes = 0;
    CHECK(cusparseSpMM_bufferSize(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                  spA[v], dnD1, &zero, dnD[v], CUDA_R_32F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
    CHECK(cudaMalloc(&buffers[v], bytes));
  }

  // Each node's launches are captured into a child graph; edges come from Deps.
  cudaGraph_t graph;
  CHECK(cudaGraphCreate(&graph, 0));
  std::vector<cudaGraphNode_t> graphNodes(Nodes.size());
  for (size_t v = 0; v < Nodes.size(); ++v) {
    const Node &node = Nodes[v];
    int gemmRows = node.GemmEnd - node.GemmBegin, rows = node.SpMMRows.size();
    const float *B = In->DBx + node.GemmBegin * k;
    float *D1 = dD1 + node.GemmBegin * k;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (gemmRows > 0 && UseLibraries) {
      // Row-major D1 = B C is column-major D1^T = C^T B^T.
      CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, k, gemmRows, k, &one,
                        DC, k, B, k, &zero, D1, k));
    } else if (gemmRows > 0) {
      gemmKernel<<<CEIL(gemmRows * k, 128), 128, 0, stream>>>(gemmRows, k, B,
                                                              DC, D1);
    }
    if (rows > 0 && UseLibraries) {
      CHECK(cusparseSpMM(sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &one, spA[v], dnD1,
                         &zero, dnD[v], CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                         buffers[v]));
    } else if (rows > 0) {
      // The repo's SpMM over a list of rows: one thread per output element,
      // 128 threads per block, columns split over blockIdx.y.
      int columns = MIN(k, 128);
      dim3 block(columns, CEIL(128, columns));
      dim3 grid(CEIL(rows, block.y), CEIL(k, columns));
      csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<grid, block, 0,
                                                             stream>>>(
          rows, k, n, In->DACsrAp, In->DACsrI, In->DACsrVal, dD1, dD, dRows[v]);
    }
    CHECK(cudaGetLastError());
    cudaGraph_t child;
    CHECK(cudaStreamEndCapture(stream, &child));
    std::vector<cudaGraphNode_t> deps;
    for (int f : node.Deps) {
      deps.push_back(graphNodes[f]);
    }
    CHECK(cudaGraphAddChildGraphNode(&graphNodes[v], graph, deps.data(),
                                     deps.size(), child));
    cudaGraphDestroy(child);
  }
  cudaGraphExec_t exec;
  CHECK(cudaGraphInstantiate(&exec, graph, 0));
  // The fills and copies above ran on the default stream, which this
  // non-blocking stream does not wait for.
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGraphLaunch(exec, stream));
  CHECK(cudaStreamSynchronize(stream));

  std::vector<float> d(n * k);
  CHECK(cudaMemcpy(d.data(), dD, d.size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  for (size_t v = 0; v < Nodes.size(); ++v) {
    if (spA[v]) {
      cusparseDestroySpMat(spA[v]);
      cusparseDestroyDnMat(dnD[v]);
    }
    cudaFree(dRows[v]);
    cudaFree(dOffsets[v]);
    cudaFree(buffers[v]);
  }
  if (UseLibraries) {
    cusparseDestroyDnMat(dnD1);
    cusparseDestroy(sparse);
    cublasDestroy(blas);
  }
  cudaFree(dD1);
  cudaFree(dD);
  return d;
}

int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  sp.IterPerPartition = 8; // rows per tile unless -ip is given
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  CSC *aCSCFull = aCSC->stype == 0 ? copy_sparse(aCSC) : make_full(aCSC);
  delete aCSC;
  // Distinct values, so a misplaced row or value pointer changes D.
  for (int j = 0; j < aCSCFull->nnz; ++j) {
    aCSCFull->x[j] = 0.5 + j % 7 / 7.0;
  }
  int n = aCSCFull->m, k = tp._b_cols, t = std::max(1, sp.IterPerPartition);
  auto *in = new CudaTensorInputs(n, k, n, n, aCSCFull, aCSCFull, 1, 1,
                                  "GeMM_SpMM_Graph_Example");
  for (int i = 0; i < n * k; ++i) {
    in->Bx[i] = (i * 37 % 101 - 50) / 50.0f;
  }
  CHECK(cudaMemcpy(in->DBx, in->Bx, n * k * sizeof(float),
                   cudaMemcpyHostToDevice));
  std::vector<float> c(k * k);
  for (int i = 0; i < k * k; ++i) {
    c[i] = (i * 13 % 29 - 14) / (14.0f * k);
  }
  float *dC;
  CHECK(cudaMalloc(&dC, c.size() * sizeof(float)));
  CHECK(cudaMemcpy(dC, c.data(), c.size() * sizeof(float),
                   cudaMemcpyHostToDevice));

  // Reference in double: D1 = B C on the CPU, then the repo's sequential SpMM.
  std::vector<double> d1(n * k), reference(n * k);
  for (int i = 0; i < n; ++i) {
    for (int l = 0; l < k; ++l) {
      for (int j = 0; j < k; ++j) {
        d1[i * k + j] += double(in->Bx[i * k + l]) * c[l * k + j];
      }
    }
  }
  swiftware::sparse::spmmCsrSequential<double>(n, k, n, in->ACsr->p,
                                               in->ACsr->i, in->ACsr->x,
                                               d1.data(), reference.data());

  std::vector<Node> nodes = buildSchedule(in->ACsr, t);
  std::cout << tp._matrix_name << ": n = " << n << ", nnz = " << in->ACsr->nnz
            << ", k = " << k << ", " << t << " rows per tile\n";
  for (size_t v = 0; v < nodes.size(); ++v) {
    const Node &node = nodes[v];
    if (node.GemmEnd > node.GemmBegin) {
      std::cout << "first-level node " << v << ": GeMM rows " << node.GemmBegin
                << "-" << node.GemmEnd - 1 << ", SpMM rows";
    } else {
      std::cout << "second-level node " << v << ": SpMM rows";
    }
    for (int row : node.SpMMRows) {
      std::cout << " " << row;
    }
    if (!node.Deps.empty()) {
      std::cout << ", after nodes";
      for (int f : node.Deps) {
        std::cout << " " << f;
      }
    }
    std::cout << "\n";
  }

  bool correct = true;
  for (bool useLibraries : {false, true}) {
    std::vector<float> d = runGraph(in, dC, nodes, useLibraries);
    double maxError = 0;
    bool ok = true;
    for (size_t i = 0; i < d.size(); ++i) {
      double error = std::abs(d[i] - reference[i]);
      maxError = std::max(maxError, error);
      ok = ok && error < 1e-4; // also false for NaN
    }
    correct = correct && ok;
    std::cout << (useLibraries ? "cuBLAS + cuSPARSE" : "hand-written kernels")
              << ": max abs error " << maxError
              << (ok ? " (ok)\n" : " (FAILED)\n");
  }
  cudaFree(dC);
  delete in;
  delete aCSCFull;
  return correct ? 0 : 1;
}
