#ifndef SPARSE_FUSION_CUDA_GEMM_SPMM_GRAPH_H
#define SPARSE_FUSION_CUDA_GEMM_SPMM_GRAPH_H

#include "Cuda_SpMM_SpMM_Demo_Utils.h"

#include <cmath>
#include <iostream>
#include <set>
#include <vector>

// GeMM-SpMM: D1 = B C, then D = A D1.
// A is an n x n CSR matrix; B (n x k), C (k x k), D1 and D (n x k) are dense
// and row-major, so bCol = cCol = k = InTensor->N. B is InTensor->Bx, D1 is
// OutTensor->ACx and D is OutTensor->Xx.

// Sequential CPU reference, accumulated in double, used to check GPU methods.
class SeqGeMMSpMM : public SeqSpMMSpMM {
protected:
  const float *Cx; // k x k

  Timer execute() override {
    OutTensor->reset();
    int n = InTensor->M, k = InTensor->N;
    std::vector<double> d1(size_t(n) * k), d(size_t(n) * k);
    Timer t;
    t.start();
    for (int i = 0; i < n; ++i) {
      for (int l = 0; l < k; ++l) {
        double b = InTensor->Bx[size_t(i) * k + l];
        for (int j = 0; j < k; ++j) {
          d1[size_t(i) * k + j] += b * Cx[l * k + j];
        }
      }
    }
    swiftware::sparse::spmmCsrSequential<double>(
        n, k, n, InTensor->ACsr->p, InTensor->ACsr->i, InTensor->ACsr->x,
        d1.data(), d.data());
    std::copy(d.begin(), d.end(), OutTensor->Xx);
    t.stop("GeMMSpMM");
    return t;
  }

public:
  SeqGeMMSpMM(CudaTensorInputs *In1, Stats *Stat1, const float *C1)
      : SeqSpMMSpMM(In1, Stat1), Cx(C1) {}
};

enum class SpMMKernel { SeqReduceRowBalance, CuSparseAlg2, CuSparseAlg3 };

enum class Schedule {
  // Second-level nodes wait only for the first-level nodes they read from.
  TileFusion,
  // Second-level nodes wait for all first-level nodes, the paper's barrier.
  TileFusionBarrier,
  // Control: the same tiled GeMMs and SpMMs, with every SpMM after all GeMMs as
  // in unfused execution.
  TiledUnfused
};

// Reads Count floats so that a following timed run starts with unrelated, clean
// L2 contents. The impossible store keeps the compiler from removing the loads.
__global__ void evictL2Kernel(const float *Data, size_t Count, float *Sink) {
  float sum = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < Count;
       i += size_t(gridDim.x) * blockDim.x) {
    sum += Data[i];
  }
  if (sum == -1.0f) {
    *Sink = sum;
  }
}

// Tile fusion schedule (Fig. 2f) of the GeMM-SpMM dependence DAG as a CUDA
// graph of black-box kernels: cuBLAS GeMM, and either cuSPARSE or the repo's
// sequential-reduction row-balance SpMM.
//
// First-level node f covers rows [f t, (f + 1) t) for t = TileRows: a GeMM for
// those rows of D1, then the SpMM for the rows of D whose nonzeros all lie in
// that block. The remaining rows form second-level nodes running only SpMM,
// ordered after first-level nodes as Schedule says. With
// TileRows >= n there is one first-level node: the unfused GeMM then SpMM.
class GeMMSpMMTileFusionGraph : public SeqGeMMSpMM {
  struct Task {
    // Rows [GemmBegin, GemmEnd) of D1 computed by GeMM; empty in second level.
    int GemmBegin, GemmEnd;
    // Rows[SpMMBegin, SpMMEnd) lists the rows of D computed by SpMM.
    int SpMMBegin, SpMMEnd;
    // First-level tasks producing the rows of D1 that the SpMM reads.
    std::vector<int> Tiles;
  };

  int TileRows;
  SpMMKernel Kernel;
  Schedule Sched;
  bool UseGraph;
  int ThreadPerBlock;
  int MBlockDim, NBlockDim, NGridDim;

  std::vector<Task> Tasks; // first-level tasks, then second-level tasks
  int NumFirstLevel = 0;
  std::vector<int> Rows;
  int *DRows = nullptr;
  float *DC = nullptr;
  bool Valid = true;

  cublasHandle_t Blas = nullptr;
  cusparseHandle_t Sparse = nullptr;
  cusparseDnMatDescr_t D1Mat = nullptr;
  std::vector<cusparseSpMatDescr_t> TaskA;
  std::vector<cusparseDnMatDescr_t> TaskD;
  std::vector<void *> TaskBuffer;
  int *DTaskAp = nullptr, *DTaskAi = nullptr;
  float *DTaskAx = nullptr;

  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;

  float *DEvict = nullptr, *DSink = nullptr;
  size_t EvictCount = 0;

  // Records the first failed call; verify() then rejects every trial.
  void check(bool Ok, const char *Call) {
    if (!Ok && Valid) {
      std::cerr << "GeMM-SpMM graph: " << Call << " failed\n";
    }
    Valid = Valid && Ok;
  }
  void check(cudaError_t Status, const char *Call) {
    check(Status == cudaSuccess, Call);
  }
  void check(cublasStatus_t Status, const char *Call) {
    check(Status == CUBLAS_STATUS_SUCCESS, Call);
  }
  void check(cusparseStatus_t Status, const char *Call) {
    check(Status == CUSPARSE_STATUS_SUCCESS, Call);
  }

  cusparseSpMMAlg_t cuSparseAlgorithm() const {
    return Kernel == SpMMKernel::CuSparseAlg2 ? CUSPARSE_SPMM_CSR_ALG2
                                              : CUSPARSE_SPMM_CSR_ALG3;
  }

  void buildSchedule() {
    int n = InTensor->M;
    const int *ap = InTensor->ACsr->p, *ai = InTensor->ACsr->i;
    std::vector<Task> tiledSpMM;
    std::vector<int> unfused;
    for (int begin = 0; begin < n; begin += TileRows) {
      int end = MIN(begin + TileRows, n);
      int tile = Tasks.size();
      Task task{begin, end, int(Rows.size()), 0, {}};
      for (int row = begin; row < end; ++row) {
        bool local = true;
        for (int j = ap[row]; j < ap[row + 1]; ++j) {
          local = local && ai[j] >= begin && ai[j] < end;
        }
        if (local) {
          Rows.push_back(row);
        } else {
          unfused.push_back(row);
        }
      }
      task.SpMMEnd = Rows.size();
      if (Sched == Schedule::TiledUnfused) {
        tiledSpMM.push_back({0, 0, task.SpMMBegin, task.SpMMEnd, {tile}});
        task.SpMMEnd = task.SpMMBegin;
      }
      Tasks.push_back(task);
    }
    NumFirstLevel = Tasks.size();
    Tasks.insert(Tasks.end(), tiledSpMM.begin(), tiledSpMM.end());
    // One tile is the unfused baseline, which keeps the default of zero.
    if (TileRows < n) {
      int fusedRows = 0, fusedNnz = 0;
      for (int f = 0; f < NumFirstLevel; ++f) {
        for (int r = Tasks[f].SpMMBegin; r < Tasks[f].SpMMEnd; ++r) {
          ++fusedRows;
          fusedNnz += ap[Rows[r] + 1] - ap[Rows[r]];
        }
      }
      St->OtherStats["Number of Fused Rows"] = {double(fusedRows)};
      St->OtherStats["Number of Fused Nnz"] = {double(fusedNnz)};
    }

    // Consecutive remaining rows that read the same tiles form one second-level
    // node, so no row waits for a tile it does not read.
    auto tilesRead = [&](int Row) {
      std::set<int> tiles;
      for (int j = ap[Row]; j < ap[Row + 1]; ++j) {
        tiles.insert(ai[j] / TileRows);
      }
      return tiles;
    };
    for (size_t first = 0; first < unfused.size();) {
      std::set<int> tiles = tilesRead(unfused[first]);
      size_t last = first + 1;
      while (last < unfused.size() && unfused[last] == unfused[last - 1] + 1 &&
             tilesRead(unfused[last]) == tiles) {
        ++last;
      }
      Task task{0, 0, int(Rows.size()), 0, {}};
      Rows.insert(Rows.end(), unfused.begin() + first, unfused.begin() + last);
      task.SpMMEnd = Rows.size();
      task.Tiles.assign(tiles.begin(), tiles.end());
      Tasks.push_back(task);
      first = last;
    }
    St->OtherStats["First Level Nodes"] = {double(NumFirstLevel)};
    St->OtherStats["Second Level Nodes"] = {
        double(Tasks.size() - NumFirstLevel)};
  }

  // cuSPARSE sees each task as its own CSR matrix: a copy of the task's
  // contiguous rows of A with zero-based row offsets and global column indices,
  // multiplied by all of D1 and written to the same rows of D. Each copy starts
  // on a 128-byte boundary, as a separately allocated matrix would.
  void createCuSparseTasks() {
    int n = InTensor->M, k = InTensor->N;
    const int *ap = InTensor->ACsr->p, *ai = InTensor->ACsr->i;
    const float *ax = InTensor->HACsrVal;
    auto aligned = [](size_t Count) { return (Count + 31) / 32 * 32; };
    std::vector<int> offsets, colInd;
    std::vector<float> values;
    std::vector<size_t> offsetStart, nnzStart;
    for (const auto &task : Tasks) {
      offsets.resize(aligned(offsets.size()));
      colInd.resize(aligned(colInd.size()));
      values.resize(colInd.size());
      offsetStart.push_back(offsets.size());
      nnzStart.push_back(colInd.size());
      if (task.SpMMEnd == task.SpMMBegin) {
        continue;
      }
      int first = Rows[task.SpMMBegin],
          last = first + task.SpMMEnd - task.SpMMBegin;
      for (int r = task.SpMMBegin; r < task.SpMMEnd; ++r) {
        if (Rows[r] != first + r - task.SpMMBegin) {
          check(false, "cuSPARSE task rows are not contiguous");
          return;
        }
      }
      for (int row = first; row <= last; ++row) {
        offsets.push_back(ap[row] - ap[first]);
      }
      colInd.insert(colInd.end(), ai + ap[first], ai + ap[last]);
      values.insert(values.end(), ax + ap[first], ax + ap[last]);
    }
    check(cudaMalloc(&DTaskAp, offsets.size() * sizeof(int)), "cudaMalloc");
    check(cudaMalloc(&DTaskAi, colInd.size() * sizeof(int)), "cudaMalloc");
    check(cudaMalloc(&DTaskAx, values.size() * sizeof(float)), "cudaMalloc");
    check(cudaMemcpy(DTaskAp, offsets.data(), offsets.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
    check(cudaMemcpy(DTaskAi, colInd.data(), colInd.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
    check(cudaMemcpy(DTaskAx, values.data(), values.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
    check(cusparseCreateDnMat(&D1Mat, n, k, k, OutTensor->DACx, CUDA_R_32F,
                              CUSPARSE_ORDER_ROW),
          "cusparseCreateDnMat");
    float one = 1, zero = 0;
    TaskA.assign(Tasks.size(), nullptr);
    TaskD.assign(Tasks.size(), nullptr);
    TaskBuffer.assign(Tasks.size(), nullptr);
    for (size_t i = 0; i < Tasks.size() && Valid; ++i) {
      int rows = Tasks[i].SpMMEnd - Tasks[i].SpMMBegin;
      if (rows == 0) {
        continue;
      }
      int first = Rows[Tasks[i].SpMMBegin];
      check(cusparseCreateCsr(&TaskA[i], rows, n, ap[first + rows] - ap[first],
                              DTaskAp + offsetStart[i], DTaskAi + nnzStart[i],
                              DTaskAx + nnzStart[i], CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_32F),
            "cusparseCreateCsr");
      check(cusparseCreateDnMat(&TaskD[i], rows, k, k,
                                OutTensor->DXx + size_t(first) * k, CUDA_R_32F,
                                CUSPARSE_ORDER_ROW),
            "cusparseCreateDnMat");
      size_t bytes = 0;
      check(cusparseSpMM_bufferSize(Sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    TaskA[i], D1Mat, &zero, TaskD[i],
                                    CUDA_R_32F, cuSparseAlgorithm(), &bytes),
            "cusparseSpMM_bufferSize");
      if (bytes > 0) {
        check(cudaMalloc(&TaskBuffer[i], bytes), "cudaMalloc");
      }
      // Each task keeps its own descriptor and buffer, so concurrent tasks
      // never share preprocessed data or scratch space.
      check(cusparseSpMM_preprocess(
                Sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &one, TaskA[i], D1Mat, &zero,
                TaskD[i], CUDA_R_32F, cuSparseAlgorithm(), TaskBuffer[i]),
            "cusparseSpMM_preprocess");
    }
  }

  void useStream(cudaStream_t Stream) {
    check(cublasSetStream(Blas, Stream), "cublasSetStream");
    // cublasSetStream restores the default workspace pool. Every method runs
    // without workspace: captured GeMMs then contain no allocation nodes, which
    // child graphs reject, and concurrent GeMMs share no scratch memory.
    check(cublasSetWorkspace(Blas, nullptr, 0), "cublasSetWorkspace");
    check(cusparseSetStream(Sparse, Stream), "cusparseSetStream");
  }

  void launchTask(int Id, cudaStream_t Stream) {
    const Task &task = Tasks[Id];
    int k = InTensor->N;
    float one = 1, zero = 0;
    if (task.GemmEnd > task.GemmBegin) {
      // Row-major D1 = B C is column-major D1^T = C^T B^T.
      check(cublasSgemm(Blas, CUBLAS_OP_N, CUBLAS_OP_N, k,
                        task.GemmEnd - task.GemmBegin, k, &one, DC, k,
                        InTensor->DBx + size_t(task.GemmBegin) * k, k, &zero,
                        OutTensor->DACx + size_t(task.GemmBegin) * k, k),
            "cublasSgemm");
    }
    int rows = task.SpMMEnd - task.SpMMBegin;
    if (rows == 0) {
      return;
    }
    if (Kernel == SpMMKernel::SeqReduceRowBalance) {
      // One thread per output element; the row list need not be contiguous.
      dim3 grid(CEIL(rows, MBlockDim), NGridDim, 1);
      dim3 block(NBlockDim, MBlockDim, 1);
      csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<grid, block, 0,
                                                             Stream>>>(
          rows, k, InTensor->M, InTensor->DACsrAp, InTensor->DACsrI,
          InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx,
          DRows + task.SpMMBegin);
      check(cudaGetLastError(), "SpMM kernel launch");
    } else {
      check(cusparseSpMM(Sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &one, TaskA[Id],
                         D1Mat, &zero, TaskD[Id], CUDA_R_32F,
                         cuSparseAlgorithm(), TaskBuffer[Id]),
            "cusparseSpMM");
    }
  }

  // Adds Tasks[Id] as a child graph captured from its library calls.
  cudaGraphNode_t addTaskNode(int Id, cudaStream_t Stream,
                              const std::vector<cudaGraphNode_t> &Deps) {
    cudaGraph_t child = nullptr;
    cudaGraphNode_t node = nullptr;
    check(cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal),
          "cudaStreamBeginCapture");
    launchTask(Id, Stream);
    check(cudaStreamEndCapture(Stream, &child), "cudaStreamEndCapture");
    if (Valid) {
      check(cudaGraphAddChildGraphNode(&node, Graph, Deps.data(), Deps.size(),
                                       child),
            "cudaGraphAddChildGraphNode");
    }
    if (child) {
      cudaGraphDestroy(child); // the parent graph holds a copy
    }
    return node;
  }

  void buildGraph() {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
    check(cudaGraphCreate(&Graph, 0), "cudaGraphCreate");
    useStream(stream);
    std::vector<cudaGraphNode_t> nodes;
    for (int f = 0; f < NumFirstLevel && Valid; ++f) {
      nodes.push_back(addTaskNode(f, stream, {}));
    }
    cudaGraphNode_t barrier = nullptr;
    if (Sched != Schedule::TileFusion && Valid &&
        int(Tasks.size()) > NumFirstLevel) {
      check(cudaGraphAddEmptyNode(&barrier, Graph, nodes.data(), nodes.size()),
            "cudaGraphAddEmptyNode");
    }
    for (int s = NumFirstLevel; s < int(Tasks.size()) && Valid; ++s) {
      std::vector<cudaGraphNode_t> deps;
      if (barrier) {
        deps.push_back(barrier);
      } else {
        for (int tile : Tasks[s].Tiles) {
          deps.push_back(nodes[tile]);
        }
      }
      addTaskNode(s, stream, deps);
    }
    if (Valid) {
      check(cudaGraphInstantiate(&GraphExec, Graph, 0), "cudaGraphInstantiate");
    }
    useStream(0);
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  }

  // Direct launches run every task in order on the default stream.
  void launchAll() {
    if (UseGraph) {
      check(cudaGraphLaunch(GraphExec, 0), "cudaGraphLaunch");
    } else {
      for (int id = 0; id < int(Tasks.size()); ++id) {
        launchTask(id, 0);
      }
    }
  }

protected:
  void setup() override {
    SeqGeMMSpMM::setup();
    int k = InTensor->N;
    NGridDim = CEIL(k, ThreadPerBlock);
    NBlockDim = MIN(k, ThreadPerBlock);
    MBlockDim = CEIL(ThreadPerBlock, NBlockDim);
    check(cudaMalloc(&DC, size_t(k) * k * sizeof(float)), "cudaMalloc");
    check(cudaMemcpy(DC, Cx, size_t(k) * k * sizeof(float),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
    check(cublasCreate(&Blas), "cublasCreate");
    check(cusparseCreate(&Sparse), "cusparseCreate");

    int device = 0, l2Bytes = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device);
    EvictCount = 2 * size_t(l2Bytes) / sizeof(float);
    check(cudaMalloc(&DEvict, EvictCount * sizeof(float)), "cudaMalloc");
    check(cudaMalloc(&DSink, sizeof(float)), "cudaMalloc");
    check(cudaMemset(DEvict, 0, EvictCount * sizeof(float)), "cudaMemset");
  }

  // Inspection, library setup and graph construction; the warm-up is untimed.
  Timer analysis() override {
    Timer t;
    t.start();
    buildSchedule();
    check(cudaMalloc(&DRows, Rows.size() * sizeof(int)), "cudaMalloc");
    check(cudaMemcpy(DRows, Rows.data(), Rows.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy");
    if (Kernel != SpMMKernel::SeqReduceRowBalance) {
      createCuSparseTasks();
    }
    useStream(0);
    if (UseGraph && Valid) {
      buildGraph();
    }
    t.stop("Analysis");
    // One run loads kernels and uploads the graph before the trials.
    if (Valid) {
      launchAll();
      check(cudaDeviceSynchronize(), "warm-up run");
    }
    return t;
  }

  void preExecute() override {
    size_t bytes = size_t(InTensor->M) * InTensor->N * sizeof(float);
    // Fill D1 and D with 3.4e38 so a row no task writes, or a SpMM that runs
    // before the GeMM it depends on, fails the check instead of reusing the
    // previous trial's result. Unlike NaN, the finite value is harmless in the
    // zero-weight padding reads cuSPARSE ALG3 makes of D1 rows it does not use.
    check(cudaMemset(OutTensor->DACx, 0x7F, bytes), "cudaMemset");
    check(cudaMemset(OutTensor->DXx, 0x7F, bytes), "cudaMemset");
    // Every method starts with the same cold L2, so reuse of D1 within a run
    // is the only cache reuse it can get.
    evictL2Kernel<<<1024, 256>>>(DEvict, EvictCount, DSink);
    check(cudaDeviceSynchronize(), "L2 eviction");
  }

  Timer execute() override {
    Timer t;
    t.startGPU();
    if (Valid) {
      launchAll();
    }
    t.stopGPU("GeMMSpMM");
    check(cudaGetLastError(), "GeMM-SpMM run");
    OutTensor->copyDeviceToHost();
    check(cudaGetLastError(), "copyDeviceToHost");
    return t;
  }

  bool verify(double &Error) override {
    // The inherited maximum-error check does not reject NaN or infinity.
    for (size_t i = 0; i < size_t(InTensor->M) * InTensor->N && Valid; ++i) {
      Valid = std::isfinite(OutTensor->Xx[i]);
    }
    if (!Valid) {
      Error = INFINITY;
      return false;
    }
    return SeqGeMMSpMM::verify(Error);
  }

public:
  GeMMSpMMTileFusionGraph(CudaTensorInputs *In1, Stats *Stat1, const float *C1,
                          int TileRows, SpMMKernel Kernel, Schedule Sched,
                          bool UseGraph, int ThreadPerBlock)
      : SeqGeMMSpMM(In1, Stat1, C1), TileRows(TileRows), Kernel(Kernel),
        Sched(Sched), UseGraph(UseGraph), ThreadPerBlock(ThreadPerBlock) {}

  ~GeMMSpMMTileFusionGraph() {
    if (GraphExec) {
      cudaGraphExecDestroy(GraphExec);
    }
    if (Graph) {
      cudaGraphDestroy(Graph);
    }
    for (size_t i = 0; i < TaskA.size(); ++i) {
      if (TaskA[i]) {
        cusparseDestroySpMat(TaskA[i]);
      }
      if (TaskD[i]) {
        cusparseDestroyDnMat(TaskD[i]);
      }
      cudaFree(TaskBuffer[i]);
    }
    if (D1Mat) {
      cusparseDestroyDnMat(D1Mat);
    }
    if (Sparse) {
      cusparseDestroy(Sparse);
    }
    if (Blas) {
      cublasDestroy(Blas);
    }
    cudaFree(DTaskAp);
    cudaFree(DTaskAi);
    cudaFree(DTaskAx);
    cudaFree(DRows);
    cudaFree(DC);
    cudaFree(DEvict);
    cudaFree(DSink);
  }
};

#endif // SPARSE_FUSION_CUDA_GEMM_SPMM_GRAPH_H
