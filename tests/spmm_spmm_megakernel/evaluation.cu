#include "evaluation.h"
#include "../../fusion/gpu/Cuda_SpMM_SpMM_Graph.h"

#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace spmm_megakernel {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int Trials = 7;
constexpr double Tolerance = 1e-6;

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

void check(cusparseStatus_t status) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error("cuSPARSE status " + std::to_string(status));
  }
}

double seconds(Clock::time_point begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

// Native Stats owns Timer objects whose legacy destructor does not free events.
// We never copy these timers; release their original handles locally.
struct NativeStats {
  Stats value;
  NativeStats(const std::string &name, const std::string &matrix)
      : value(name, "SpMMSpMM", Trials, matrix, 1) {
    for (auto *trial : value.ProfilingInfoTrials) {
      trial->NumSubregions = 2;
      trial->NumEvents = 0;
    }
  }
  ~NativeStats() {
    for (auto *trial : value.ProfilingInfoTrials) {
      cudaEventDestroy(trial->AnalysisTime.StartGpuTime);
      cudaEventDestroy(trial->AnalysisTime.StopGpuTime);
      cudaEventDestroy(trial->ExecutorTime.StartGpuTime);
      cudaEventDestroy(trial->ExecutorTime.StopGpuTime);
    }
  }
};

uint64_t mix(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::vector<float> makeInput(int rows, uint64_t variant) {
  std::vector<float> input(static_cast<size_t>(rows) * Features);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = (static_cast<int>(mix(i + variant * 0x100000001b3ULL) % 2049)
                - 1024) / 1024.0f;
  }
  return input;
}

Matrix makeMatrix(int rows, const std::string &family, int structure) {
  Matrix matrix;
  matrix.rows = rows;
  matrix.rowPtr.push_back(0);
  for (int row = 0; row < rows; ++row) {
    int first = row, end = row + 1;
    if (family == "banded") {
      first = std::max(0, row - structure / 2);
      end = std::min(rows, row + structure / 2 + 1);
    } else if (family == "block_diagonal") {
      first = row / structure * structure;
      end = std::min(rows, first + structure);
    } else if (family == "zero" || (family == "empty_row" && row % 3 == 0)) {
      end = first;
    } else if (family == "empty_row" || family == "duplicates") {
      first = std::max(0, row - 1);
      end = std::min(rows, row + 2);
    } else if (family != "identity") {
      throw std::runtime_error("unknown matrix family: " + family);
    }
    for (int col = first; col < end; ++col) {
      float value = col == row ? 0.5f
          : family == "block_diagonal" ? 0.5f / (structure - 1)
                                       : 0.25f / (structure / 2);
      if (family == "identity") {
        value = 1.0f;
      } else if (family == "empty_row" || family == "duplicates") {
        value = col == row ? 0.5f : 0.25f;
      }
      matrix.columns.push_back(col);
      matrix.values.push_back(value);
    }
    if (family == "duplicates") {
      matrix.columns.push_back(row);
      matrix.values.push_back(-0.125f);
      // Unsorted indices and duplicates remain intact in DeviceData.
      int start = matrix.rowPtr.back();
      std::reverse(matrix.columns.begin() + start, matrix.columns.end());
      std::reverse(matrix.values.begin() + start, matrix.values.end());
    }
    matrix.rowPtr.push_back(static_cast<int>(matrix.columns.size()));
  }
  return matrix;
}

void replaceSparseValues(Matrix &matrix, uint64_t variant) {
  for (int row = 0; row < matrix.rows; ++row) {
    double norm = 0;
    for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
      matrix.values[p] = (static_cast<int>(mix(p + variant * 65537) % 2049)
                          - 1024) / 1024.0f;
      norm += std::abs(matrix.values[p]);
    }
    for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
      matrix.values[p] /= static_cast<float>(std::max(1.0, norm));
    }
  }
}

struct Reference {
  std::vector<double> h, y;
};

Reference reference(const Matrix &matrix, const std::vector<float> &input) {
  // Independent scalar oracle: no production kernel or GPU buffer is reused.
  const auto values = matrix.values;
  const auto x = input;
  Reference result;
  size_t elements = static_cast<size_t>(matrix.rows) * Features;
  result.h.assign(elements, 0.0);
  result.y.assign(elements, 0.0);
  for (int row = 0; row < matrix.rows; ++row) {
    for (int feature = 0; feature < Features; ++feature) {
      double sum = 0;
      for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
        sum += static_cast<double>(values[p]) *
               x[static_cast<size_t>(matrix.columns[p]) * Features + feature];
      }
      result.h[static_cast<size_t>(row) * Features + feature] = sum;
    }
  }
  for (int row = 0; row < matrix.rows; ++row) {
    for (int feature = 0; feature < Features; ++feature) {
      double sum = 0;
      for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
        sum += static_cast<double>(values[p]) *
               result.h[static_cast<size_t>(matrix.columns[p]) * Features + feature];
      }
      result.y[static_cast<size_t>(row) * Features + feature] = sum;
    }
  }
  return result;
}

double compare(const std::vector<float> &actual,
               const std::vector<double> &expected) {
  double error = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      return std::numeric_limits<double>::infinity();
    }
    error = std::max(error, std::abs(static_cast<double>(actual[i]) - expected[i]));
  }
  return error;
}

double verify(Method &method, DeviceData &data, int slot,
              const Reference &expected, std::vector<float> &readback) {
  size_t bytes = readback.size() * sizeof(float);
  check(cudaMemcpy(readback.data(), data.slots[slot].y, bytes,
                   cudaMemcpyDeviceToHost));
  double error = compare(readback, expected.y);
  if (method.writesH()) {
    check(cudaMemcpy(readback.data(), data.slots[slot].h, bytes,
                     cudaMemcpyDeviceToHost));
    error = std::max(error, compare(readback, expected.h));
  }
  return error;
}

void poison(DeviceData &data, int slot) {
  size_t bytes = static_cast<size_t>(data.rows) * Features * sizeof(float);
  // 0xff bytes are float NaNs. This is harness preparation, not a required reset.
  check(cudaMemset(data.slots[slot].h, 0xff, bytes));
  check(cudaMemset(data.slots[slot].y, 0xff, bytes));
  check(cudaDeviceSynchronize());
}

struct Schedule {
  int *fPtr = nullptr, *fId = nullptr, *deferred = nullptr;
  int count = 0, tiles = 0;
  size_t bytes = 0;
  ~Schedule() {
    cudaFree(fPtr);
    cudaFree(fId);
    cudaFree(deferred);
  }
};

// Reuse the original inspector, then move only its device schedule out. Its
// temporary tensor allocations are released before steady-state measurements.
class Inspector : public FusedSpMMSpMMSeqReduceRowBalance {
public:
  Inspector(CudaTensorInputs *input, Stats *stats, int tileRows)
      : FusedSpMMSpMMSeqReduceRowBalance(input, stats, Features * tileRows) {
    HFPtr = HFId = HUFPtr = nullptr;
    DFPtr = DFId = DUFPtr = nullptr;
  }
  void prepare(Schedule &schedule) {
    setup();
    analysis();
    check(cudaGetLastError());
    schedule.fPtr = DFPtr;
    schedule.fId = DFId;
    schedule.deferred = DUFPtr;
    schedule.count = UFDim;
    schedule.tiles = MGridDim;
    schedule.bytes = static_cast<size_t>(InTensor->M + MGridDim + 1) * sizeof(int);
    DFPtr = DFId = DUFPtr = nullptr;
  }
};

void inspect(const Matrix &matrix, int tileRows, Schedule &schedule) {
  sym_lib::CSR csr(matrix.rows, matrix.rows, static_cast<int>(matrix.values.size()));
  std::copy(matrix.rowPtr.begin(), matrix.rowPtr.end(), csr.p);
  std::copy(matrix.columns.begin(), matrix.columns.end(), csr.i);
  std::copy(matrix.values.begin(), matrix.values.end(), csr.x);
  std::unique_ptr<sym_lib::CSC> csc(sym_lib::csr_to_csc(&csr));
  CudaTensorInputs inputs(matrix.rows, Features, matrix.rows, matrix.rows,
                          csc.get(), csc.get(), 1, 1, "MegakernelInspector");
  NativeStats stats("inspector", "inspector");
  Inspector inspector(&inputs, &stats.value, tileRows);
  inspector.prepare(schedule);
  check(cudaDeviceSynchronize());
}

class KernelPair : public Method {
  DeviceData &Data;
  bool Fused, Waited, Replay;
  Schedule RowSchedule;
  std::vector<cudaGraphExec_t> Graphs;
  cudaStream_t PreparationStream = nullptr;
  cudaGraph_t CapturedGraph = nullptr;
  bool Capturing = false;

  void cleanup() {
    if (Capturing) cudaStreamEndCapture(PreparationStream, &CapturedGraph);
    Capturing = false;
    cudaDeviceSynchronize();
    for (auto graph : Graphs) cudaGraphExecDestroy(graph);
    Graphs.clear();
    if (CapturedGraph) cudaGraphDestroy(CapturedGraph);
    CapturedGraph = nullptr;
    if (PreparationStream) cudaStreamDestroy(PreparationStream);
    PreparationStream = nullptr;
  }

  void submit(int slot, cudaStream_t stream) {
    const auto &buffers = Data.slots[slot];
    dim3 block(Features, Data.tileRows, 1);
    dim3 grid((Data.rows + Data.tileRows - 1) / Data.tileRows, 1, 1);
    if (Fused) {
      csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<grid, block, 0, stream>>>(
          Data.rows, Features, Data.rows, Data.rowPtr, Data.columns, Data.values,
          buffers.x, buffers.h, buffers.y, RowSchedule.fPtr, RowSchedule.fId);
    } else {
      csrspmm_seqreduce_rowbalance_kernel<<<grid, block, 0, stream>>>(
          Data.rows, Features, Data.rows, Data.rowPtr, Data.columns, Data.values,
          buffers.x, buffers.h);
    }
    if (Waited) {
      check(cudaDeviceSynchronize());
    }
    if (Fused) {
      grid.x = std::max(1, (RowSchedule.count + Data.tileRows - 1) / Data.tileRows);
      csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<grid, block, 0, stream>>>(
          RowSchedule.count, Features, Data.rows, Data.rowPtr, Data.columns,
          Data.values, buffers.h, buffers.y, RowSchedule.deferred);
    } else {
      csrspmm_seqreduce_rowbalance_kernel<<<grid, block, 0, stream>>>(
          Data.rows, Features, Data.rows, Data.rowPtr, Data.columns, Data.values,
          buffers.h, buffers.y);
    }
  }

public:
  KernelPair(const Matrix &matrix, DeviceData &data, bool fused, bool waited,
             bool replay) : Data(data), Fused(fused), Waited(waited), Replay(replay) {
    try {
      hWriteBytes = static_cast<size_t>(Data.rows) * Features * sizeof(float);
      if (Fused) {
        auto begin = Clock::now();
        inspect(matrix, Data.tileRows, RowSchedule);
        inspectionSeconds = seconds(begin);
        scheduleBytes = RowSchedule.bytes;
      }
      if (Replay) {
        auto begin = Clock::now();
        check(cudaStreamCreate(&PreparationStream));
        for (int slot = 0; slot < static_cast<int>(Data.slots.size()); ++slot) {
          cudaGraphExec_t executable = nullptr;
          check(cudaStreamBeginCapture(PreparationStream, cudaStreamCaptureModeGlobal));
          Capturing = true;
          submit(slot, PreparationStream);
          check(cudaStreamEndCapture(PreparationStream, &CapturedGraph));
          Capturing = false;
          check(cudaGraphInstantiate(&executable, CapturedGraph, nullptr, nullptr, 0));
          Graphs.push_back(executable);
          check(cudaGraphUpload(executable, PreparationStream));
          check(cudaGraphDestroy(CapturedGraph));
          CapturedGraph = nullptr;
        }
        check(cudaStreamSynchronize(PreparationStream));
        check(cudaStreamDestroy(PreparationStream));
        PreparationStream = nullptr;
        graphSeconds = seconds(begin);
      }
    } catch (...) { cleanup(); throw; }
  }
  void launch(int slot) override {
    if (Replay) {
      check(cudaGraphLaunch(Graphs[slot], 0));
    } else {
      submit(slot, 0);
      if (Waited) {
        check(cudaDeviceSynchronize());
      }
    }
    check(cudaGetLastError());
  }
  size_t workspaceBytes() const override { return RowSchedule.bytes; }
  ~KernelPair() { cleanup(); }
};

class VendorPair : public Method {
  struct Preparation {
    cusparseSpMatDescr_t firstA = nullptr, secondA = nullptr;
    cusparseDnMatDescr_t x = nullptr, h = nullptr, y = nullptr;
    void *firstWorkspace = nullptr, *secondWorkspace = nullptr;
    cudaGraphExec_t graph = nullptr;
  };
  DeviceData &Data;
  cusparseHandle_t Handle = nullptr;
  cusparseSpMMAlg_t Algorithm;
  bool Replay;
  size_t Bytes = 0;
  std::vector<Preparation> Prepared;
  cudaStream_t PreparationStream = nullptr;
  cudaGraph_t CapturedGraph = nullptr;
  bool Capturing = false;

  void cleanup() {
    if (Capturing) cudaStreamEndCapture(PreparationStream, &CapturedGraph);
    Capturing = false;
    cudaDeviceSynchronize();
    if (CapturedGraph) cudaGraphDestroy(CapturedGraph);
    CapturedGraph = nullptr;
    for (auto &p : Prepared) {
      if (p.graph) cudaGraphExecDestroy(p.graph);
      if (p.x) cusparseDestroyDnMat(p.x);
      if (p.h) cusparseDestroyDnMat(p.h);
      if (p.y) cusparseDestroyDnMat(p.y);
      if (p.firstA) cusparseDestroySpMat(p.firstA);
      if (p.secondA) cusparseDestroySpMat(p.secondA);
      cudaFree(p.firstWorkspace);
      cudaFree(p.secondWorkspace);
      p = Preparation{};
    }
    if (Handle) cusparseDestroy(Handle);
    Handle = nullptr;
    if (PreparationStream) cudaStreamDestroy(PreparationStream);
    PreparationStream = nullptr;
  }

  void submit(int slot) {
    const auto &p = Prepared[slot];
    const float alpha = 1, beta = 0;
    check(cusparseSpMM(Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.firstA, p.x,
                       &beta, p.h, CUDA_R_32F, Algorithm, p.firstWorkspace));
    check(cusparseSpMM(Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.secondA, p.h,
                       &beta, p.y, CUDA_R_32F, Algorithm, p.secondWorkspace));
  }

public:
  VendorPair(const Matrix &matrix, DeviceData &data, cusparseSpMMAlg_t algorithm,
             bool replay) : Data(data), Algorithm(algorithm), Replay(replay),
                            Prepared(data.slots.size()) {
    try {
      hWriteBytes = static_cast<size_t>(Data.rows) * Features * sizeof(float);
      auto begin = Clock::now();
      check(cusparseCreate(&Handle));
      check(cudaStreamCreate(&PreparationStream));
      check(cusparseSetStream(Handle, PreparationStream));
      librarySeconds = seconds(begin);
      const float alpha = 1, beta = 0;
      for (int slot = 0; slot < static_cast<int>(Data.slots.size()); ++slot) {
        begin = Clock::now();
        auto &p = Prepared[slot];
        for (auto *sparse : {&p.firstA, &p.secondA}) {
          check(cusparseCreateCsr(sparse, Data.rows, Data.rows, matrix.values.size(),
                                 Data.rowPtr, Data.columns, Data.values,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        }
        check(cusparseCreateDnMat(&p.x, Data.rows, Features, Features,
                                 Data.slots[slot].x, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        check(cusparseCreateDnMat(&p.h, Data.rows, Features, Features,
                                 Data.slots[slot].h, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        check(cusparseCreateDnMat(&p.y, Data.rows, Features, Features,
                                 Data.slots[slot].y, CUDA_R_32F, CUSPARSE_ORDER_ROW));
        size_t firstBytes = 0, secondBytes = 0;
        check(cusparseSpMM_bufferSize(Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.firstA, p.x, &beta, p.h,
            CUDA_R_32F, Algorithm, &firstBytes));
        check(cusparseSpMM_bufferSize(Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.secondA, p.h, &beta, p.y,
            CUDA_R_32F, Algorithm, &secondBytes));
        if (firstBytes) check(cudaMalloc(&p.firstWorkspace, firstBytes));
        if (secondBytes) check(cudaMalloc(&p.secondWorkspace, secondBytes));
        Bytes += firstBytes + secondBytes;
        check(cusparseSpMM_preprocess(Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.firstA, p.x, &beta, p.h,
            CUDA_R_32F, Algorithm, p.firstWorkspace));
        check(cusparseSpMM_preprocess(Handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p.secondA, p.h, &beta, p.y,
            CUDA_R_32F, Algorithm, p.secondWorkspace));
        submit(slot);
        check(cudaStreamSynchronize(PreparationStream));
        librarySeconds += seconds(begin);
        if (Replay) {
          begin = Clock::now();
          check(cudaStreamBeginCapture(PreparationStream, cudaStreamCaptureModeGlobal));
          Capturing = true;
          submit(slot);
          check(cudaStreamEndCapture(PreparationStream, &CapturedGraph));
          Capturing = false;
          check(cudaGraphInstantiate(&p.graph, CapturedGraph, nullptr, nullptr, 0));
          check(cudaGraphUpload(p.graph, PreparationStream));
          check(cudaGraphDestroy(CapturedGraph));
          CapturedGraph = nullptr;
          check(cudaStreamSynchronize(PreparationStream));
          graphSeconds += seconds(begin);
        }
      }
      check(cudaStreamSynchronize(PreparationStream));
      check(cusparseSetStream(Handle, 0));
      check(cudaStreamDestroy(PreparationStream));
      PreparationStream = nullptr;
      libraryBytes = Bytes;
    } catch (...) { cleanup(); throw; }
  }
  void launch(int slot) override {
    if (Replay) check(cudaGraphLaunch(Prepared[slot].graph, 0));
    else submit(slot);
    check(cudaGetLastError());
  }
  size_t workspaceBytes() const override { return Bytes; }
  ~VendorPair() { cleanup(); }
};

std::vector<std::string> methods() {
  std::vector<std::string> names = {"original_waited", "direct_same_stream", "graph",
      "unfused_direct", "unfused_graph", "cusparse_alg2_direct",
      "cusparse_alg2_graph", "cusparse_alg3_direct", "cusparse_alg3_graph"};
#ifdef SPMM_MEGAKERNEL_IMPLEMENTATION
  names.insert(names.end(), {"megakernel_barrier", "megakernel_events_global",
                            "megakernel_events_shared"});
#endif
  return names;
}

std::unique_ptr<Method> makeMethod(const std::string &name, const Matrix &matrix,
                                  DeviceData &data) {
  if (name == "original_waited" || name == "direct_same_stream" || name == "graph") {
    return std::unique_ptr<Method>(new KernelPair(matrix, data, true,
                                                 name == "original_waited",
                                                 name == "graph"));
  }
  if (name == "unfused_direct" || name == "unfused_graph") {
    return std::unique_ptr<Method>(new KernelPair(matrix, data, false, false,
                                                 name == "unfused_graph"));
  }
  if (name.find("cusparse_alg") == 0) {
    bool alg2 = name.find("alg2") != std::string::npos;
    return std::unique_ptr<Method>(new VendorPair(matrix, data,
        alg2 ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG3,
        name.find("graph") != std::string::npos));
  }
#ifdef SPMM_MEGAKERNEL_IMPLEMENTATION
  return makeMegakernelMethod(name, matrix, data);
#else
  throw std::runtime_error("method is not built: " + name);
#endif
}

std::vector<std::string> selectedMethods(const std::string &filter, int order = 0) {
  auto names = methods();
  if (filter != "all") {
    if (std::find(names.begin(), names.end(), filter) == names.end()) {
      throw std::runtime_error("unknown or unbuilt method: " + filter);
    }
    return {filter};
  }
  std::rotate(names.begin(), names.begin() + order % names.size(), names.end());
  return names;
}

struct Events {
  cudaEvent_t begin = nullptr, end = nullptr;
  Events() {
    check(cudaEventCreate(&begin));
    auto status = cudaEventCreate(&end);
    if (status != cudaSuccess) {
      cudaEventDestroy(begin);
      check(status);
    }
  }
  ~Events() { cudaEventDestroy(begin); cudaEventDestroy(end); }
  void start() {
    check(cudaEventRecord(begin, 0));
    // Start must precede work even if a future method uses a nonblocking stream.
    check(cudaEventSynchronize(begin));
  }
  double stop() {
    check(cudaDeviceSynchronize());
    check(cudaEventRecord(end, 0));
    check(cudaEventSynchronize(end));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, begin, end));
    return milliseconds / 1000.0;
  }
};

int integer(const char *argument, int minimum, int maximum) {
  std::string text(argument);
  size_t consumed = 0;
  int value = std::stoi(text, &consumed);
  if (consumed != text.size() || value < minimum || value > maximum) {
    throw std::runtime_error("invalid integer: " + text);
  }
  return value;
}

struct Condition {
  int rows, structure, tileRows, order;
  std::string family, regime;
};

Condition parseCondition(char **argv) {
  Condition c;
  c.rows = integer(argv[2], 1, 1048576);
  c.family = argv[3];
  c.structure = integer(argv[4], 1, 16);
  c.tileRows = integer(argv[5], 1, 16);
  if ((c.family != "banded" && c.family != "block_diagonal") ||
      (c.family == "banded" && c.structure != 3 && c.structure != 5 && c.structure != 9) ||
      (c.family == "block_diagonal" && c.structure != 4 && c.structure != 16) ||
      (c.tileRows != 1 && c.tileRows != 4 && c.tileRows != 16)) {
    throw std::runtime_error("unsupported family, structure or tile size");
  }
  return c;
}

cudaDeviceProp deviceInfo() {
  int device = 0;
  check(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device));
  return properties;
}

int slotCount(const Condition &condition, size_t l2Bytes) {
  if (condition.regime == "repeated") return 1;
  if (condition.regime != "rotating" || condition.rows < 4096) {
    throw std::runtime_error("rotation requires rows >= 4096");
  }
  size_t xBytes = static_cast<size_t>(condition.rows) * Features * sizeof(float);
  return static_cast<int>(std::max<size_t>(2, 2 * l2Bytes / xBytes + 1));
}

int correctness(const std::string &filter, bool small) {
  std::cout << "Method,Rows,Family,Structure,Tile Rows,Variant,Stage,Correct,Max Error,Status\n";
  bool allCorrect = true;
  const std::vector<int> sizes = small ? std::vector<int>{1, 17, 65}
                                      : std::vector<int>{1, 3, 17, 65, 127, 257};
  for (int rows : sizes) {
    for (const std::string &family : {"banded", "block_diagonal", "identity",
                                      "zero", "empty_row", "duplicates"}) {
      std::vector<int> structures = family == "banded" ? std::vector<int>{3, 5, 9}
          : family == "block_diagonal" ? std::vector<int>{4, 16}
                                      : std::vector<int>{1};
      for (int structure : structures) {
        for (int tile : {1, 4, 16}) {
          for (const auto &name : selectedMethods(filter)) {
            for (int variant = 0; variant < 3; ++variant) {
              int stage = 0;
              try {
                Matrix matrix = makeMatrix(rows, family, structure);
                std::vector<std::vector<float>> inputs{makeInput(rows, variant)};
                DeviceData data(matrix, tile, 1, inputs); // Fresh buffers per variant.
                auto method = makeMethod(name, matrix, data);
                std::vector<float> readback(inputs[0].size());
                for (stage = 0; stage < 3; ++stage) {
                  if (stage == 1) {
                    inputs[0] = makeInput(rows, variant + 101);
                    check(cudaMemcpy(data.slots[0].x, inputs[0].data(),
                        inputs[0].size() * sizeof(float), cudaMemcpyHostToDevice));
                  } else if (stage == 2) {
                    replaceSparseValues(matrix, variant + 313);
                    check(cudaMemcpy(data.values, matrix.values.data(),
                        matrix.values.size() * sizeof(float), cudaMemcpyHostToDevice));
                  }
                  auto expected = reference(matrix, inputs[0]);
                  poison(data, 0);
                  method->launch(0);
                  check(cudaDeviceSynchronize());
                  double error = verify(*method, data, 0, expected, readback);
                  bool correct = std::isfinite(error) && error <= Tolerance;
                  allCorrect = allCorrect && correct;
                  std::cout << name << ',' << rows << ',' << family << ',' << structure
                            << ',' << tile << ',' << variant << ',' << stage << ','
                            << correct << ',' << error << "," << (correct ? "ok" : "failed") << '\n';
                }
              } catch (const std::exception &error) {
                allCorrect = false;
                std::cout << name << ',' << rows << ',' << family << ',' << structure
                          << ',' << tile << ',' << variant << ',' << stage
                          << ",0,inf,error\n";
                std::cerr << name << ": " << error.what() << '\n';
                cudaGetLastError();
              }
            }
          }
        }
      }
    }
  }
  return allCorrect ? 0 : 1;
}

void rawHeader() {
  std::cout << "Phase,Method,Rows,Family,Structure,Tile Rows,Regime,Process Order,Trial,"
               "Calls,Slots,Device Seconds,Wall Seconds,Correct,Max Error,CSR Bytes,"
               "Dense Bytes,Workspace Bytes,X Pool Bytes,L2 Bytes,Allocation Seconds,"
               "Preparation Seconds,Setup Seconds,Status,X Bytes,H Bytes,Y Bytes,"
               "Schedule Bytes,Event Bytes,Library Workspace Bytes,Device Malloc Seconds,"
               "Initial Copy Seconds,Inspection and Schedule Seconds,Library Preparation Seconds,"
               "Graph Preparation Seconds,Global H Write Bytes per Call,H Fully Written\n";
}

void rawRow(const std::string &phase, const std::string &name, const Condition &c,
            int trial, int calls, int slots, double gpu, double wall, double error,
            size_t csrBytes, size_t denseBytes, size_t workspaceBytes, size_t l2Bytes,
            double allocation, double preparation, const std::string &status = "ok",
            const Method *method = nullptr, const DeviceData *data = nullptr) {
  bool correct = std::isfinite(error) && error <= Tolerance && status == "ok";
  std::cout << phase << ',' << name << ',' << c.rows << ',' << c.family << ','
            << c.structure << ',' << c.tileRows << ',' << c.regime << ',' << c.order
            << ',' << trial << ',' << calls << ',' << slots << ',' << gpu << ','
            << wall << ',' << correct << ',' << error << ',' << csrBytes << ','
            << denseBytes << ',' << workspaceBytes << ','
            << static_cast<size_t>(c.rows) * Features * sizeof(float) * slots << ','
            << l2Bytes << ',' << allocation << ',' << preparation << ','
            << allocation + preparation << ',' << status << ','
            << denseBytes / 3 << ',' << denseBytes / 3 << ',' << denseBytes / 3 << ','
            << (method ? method->scheduleBytes : 0) << ','
            << (method ? method->eventBytes : 0) << ','
            << (method ? method->libraryBytes : 0) << ','
            << (data ? data->allocationSeconds : 0) << ','
            << (data ? data->copySeconds : 0) << ','
            << (method ? method->inspectionSeconds : 0) << ','
            << (method ? method->librarySeconds : 0) << ','
            << (method ? method->graphSeconds : 0) << ','
            << (method ? method->hWriteBytes : 0) << ','
            << (method ? method->writesH() : false) << '\n';
}

int benchmark(const Condition &c, const std::string &filter,
              const std::string &statsPath) {
  auto properties = deviceInfo();
  int slots = slotCount(c, properties.l2CacheSize);
  int calls = std::max(100, 2 * slots);
  Matrix matrix = makeMatrix(c.rows, c.family, c.structure);
  std::vector<std::vector<float>> inputs;
  std::vector<Reference> expected;
  for (int slot = 0; slot < slots; ++slot) {
    inputs.push_back(makeInput(c.rows, static_cast<uint64_t>(c.order) * 1000003 + slot));
    expected.push_back(reference(matrix, inputs.back()));
  }
  size_t csrBytes = matrix.rowPtr.size() * sizeof(int) +
                   matrix.columns.size() * (sizeof(int) + sizeof(float));
  size_t denseBytes = static_cast<size_t>(c.rows) * Features * sizeof(float) * slots * 3;
  std::vector<float> readback(static_cast<size_t>(c.rows) * Features);
  std::ofstream statsFile;
  if (!statsPath.empty()) {
    statsFile.open(statsPath);
    if (!statsFile) throw std::runtime_error("cannot open native Stats file");
    statsFile << std::setprecision(17);
  }
  bool allCorrect = true, header = true;
  rawHeader();
  for (const auto &name : selectedMethods(filter, c.order)) {
    try {
      check(cudaDeviceSynchronize());
      auto coldBegin = Clock::now();
      DeviceData data(matrix, c.tileRows, slots, inputs);
      double allocation = seconds(coldBegin);
      auto preparationBegin = Clock::now();
      auto method = makeMethod(name, matrix, data);
      check(cudaDeviceSynchronize());
      double preparation = seconds(preparationBegin);
      method->launch(0);
      check(cudaDeviceSynchronize());
      check(cudaMemcpy(readback.data(), data.slots[0].y,
                       readback.size() * sizeof(float), cudaMemcpyDeviceToHost));
      double cold = seconds(coldBegin);
      double error = compare(readback, expected[0].y);
      if (method->writesH()) error = std::max(error, verify(*method, data, 0, expected[0], readback));
      rawRow("cold_end_to_end", name, c, 0, 1, slots, 0, cold, error,
             csrBytes, denseBytes, method->workspaceBytes(), properties.l2CacheSize,
             allocation, preparation, "ok", method.get(), &data);
      if (!std::isfinite(error) || error > Tolerance) throw std::runtime_error("cold output failed");
      for (int slot = 0; slot < slots; ++slot) method->launch(slot);
      check(cudaDeviceSynchronize());
      NativeStats native(name, c.family + "_" + std::to_string(c.structure) + "_" + std::to_string(c.rows));
      Events events;
      for (int trial = 0; trial < Trials; ++trial) {
        check(cudaDeviceSynchronize());
        auto begin = Clock::now();
        events.start();
        for (int call = 0; call < calls; ++call) method->launch(call % slots);
        double gpu = events.stop() / calls;
        double wall = seconds(begin) / calls;
        error = 0;
        for (int slot = 0; slot < slots; ++slot) {
          error = std::max(error, verify(*method, data, slot, expected[slot], readback));
        }
        bool correct = std::isfinite(error) && error <= Tolerance;
        allCorrect = allCorrect && correct;
        rawRow("steady", name, c, trial, calls, slots, gpu, wall, error,
               csrBytes, denseBytes, method->workspaceBytes(), properties.l2CacheSize,
               allocation, preparation, correct ? "ok" : "failed", method.get(), &data);
        auto *info = native.value.ProfilingInfoTrials[trial];
        info->AnalysisTime.ElapsedTimeArray = {{allocation, "Allocation"}, {preparation, "Preparation"}};
        info->ExecutorTime.ElapsedTimeArray = {{gpu, "Device"}, {wall, "Wall"}};
        info->ErrorPerExecute = {correct, error};
      }
      for (int trial = 0; trial < Trials; ++trial) {
        int slot = trial % slots;
        check(cudaDeviceSynchronize());
        auto begin = Clock::now();
        events.start();
        check(cudaMemcpy(data.slots[slot].x, inputs[slot].data(),
                         readback.size() * sizeof(float), cudaMemcpyHostToDevice));
        method->launch(slot);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(readback.data(), data.slots[slot].y,
                         readback.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double gpu = events.stop();
        double wall = seconds(begin);
        error = compare(readback, expected[slot].y);
        if (method->writesH()) error = std::max(error, verify(*method, data, slot, expected[slot], readback));
        bool correct = std::isfinite(error) && error <= Tolerance;
        allCorrect = allCorrect && correct;
        rawRow("resident_end_to_end", name, c, trial, 1, slots, gpu, wall, error,
               csrBytes, denseBytes, method->workspaceBytes(), properties.l2CacheSize,
               allocation, preparation, correct ? "ok" : "failed", method.get(), &data);
      }
      if (statsFile) {
        if (header) {
          statsFile << native.value.printCSVHeader()
                    << "Rows,Family,Structure,Tile Rows,Regime,Process Order,Calls,Slots,L2 Bytes\n";
          header = false;
        }
        statsFile << native.value.printCSV() << c.rows << ',' << c.family << ','
                  << c.structure << ',' << c.tileRows << ',' << c.regime << ','
                  << c.order << ',' << calls << ',' << slots << ',' << properties.l2CacheSize << '\n';
      }
    } catch (const std::exception &error) {
      allCorrect = false;
      rawRow("error", name, c, -1, 0, slots, 0, 0,
             std::numeric_limits<double>::infinity(), csrBytes, denseBytes, 0,
             properties.l2CacheSize, 0, 0, "error");
      std::cerr << name << ": " << error.what() << '\n';
      cudaGetLastError();
    }
  }
  return allCorrect ? 0 : 1;
}

int trace(Condition c, const std::string &name) {
  auto properties = deviceInfo();
  int slots = slotCount(c, properties.l2CacheSize);
  auto matrix = makeMatrix(c.rows, c.family, c.structure);
  std::vector<std::vector<float>> inputs;
  for (int slot = 0; slot < slots; ++slot) inputs.push_back(makeInput(c.rows, slot));
  auto expected = reference(matrix, inputs[0]);
  DeviceData data(matrix, c.tileRows, slots, inputs);
  auto method = makeMethod(name, matrix, data);
  for (int slot = 0; slot < slots; ++slot) method->launch(slot);
  check(cudaDeviceSynchronize());
  check(cudaProfilerStart());
  method->launch(0);
  check(cudaDeviceSynchronize());
  check(cudaProfilerStop());
  std::vector<float> readback(inputs[0].size());
  double error = verify(*method, data, 0, expected, readback);
  std::cout << "Method,Rows,Family,Structure,Tile Rows,Regime,Slots,Correct,Max Error\n"
            << name << ',' << c.rows << ',' << c.family << ',' << c.structure << ','
            << c.tileRows << ',' << c.regime << ',' << slots << ','
            << (std::isfinite(error) && error <= Tolerance) << ',' << error << '\n';
  return std::isfinite(error) && error <= Tolerance ? 0 : 1;
}

void releaseData(DeviceData &data) {
  cudaDeviceSynchronize();
  for (auto &slot : data.slots) {
    cudaFree(slot.x);
    cudaFree(slot.h);
    cudaFree(slot.y);
    slot = Slot{};
  }
  cudaFree(data.rowPtr);
  cudaFree(data.columns);
  cudaFree(data.values);
  data.rowPtr = data.columns = nullptr;
  data.values = nullptr;
}

} // namespace

DeviceData::DeviceData(const Matrix &matrix, int tileRows_, int slotCount,
                       const std::vector<std::vector<float>> &inputs)
    : rows(matrix.rows), tileRows(tileRows_), slots(slotCount) {
  try {
    size_t pointerBytes = matrix.rowPtr.size() * sizeof(int);
    size_t indexBytes = matrix.columns.size() * sizeof(int);
    size_t valueBytes = matrix.values.size() * sizeof(float);
    size_t denseBytes = static_cast<size_t>(rows) * Features * sizeof(float);
    auto allocate = [&](auto **destination, size_t bytes) {
      auto begin = Clock::now();
      check(cudaMalloc(destination, bytes));
      allocationSeconds += seconds(begin);
    };
    auto copy = [&](void *destination, const void *source, size_t bytes) {
      auto begin = Clock::now();
      check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
      copySeconds += seconds(begin);
    };
    allocate(&rowPtr, pointerBytes);
    if (indexBytes) allocate(&columns, indexBytes);
    if (valueBytes) allocate(&values, valueBytes);
    copy(rowPtr, matrix.rowPtr.data(), pointerBytes);
    if (indexBytes) copy(columns, matrix.columns.data(), indexBytes);
    if (valueBytes) copy(values, matrix.values.data(), valueBytes);
    for (int slot = 0; slot < slotCount; ++slot) {
      allocate(&slots[slot].x, denseBytes);
      allocate(&slots[slot].h, denseBytes);
      allocate(&slots[slot].y, denseBytes);
      copy(slots[slot].x, inputs[slot].data(), denseBytes);
    }
    allocatedBytes = pointerBytes + indexBytes + valueBytes + denseBytes * slotCount * 3;
    check(cudaDeviceSynchronize());
  } catch (...) { releaseData(*this); throw; }
}

DeviceData::~DeviceData() { releaseData(*this); }

int main(int argc, char **argv) {
  std::cout << std::setprecision(17);
  std::cerr << std::setprecision(17);
  try {
    if (argc == 2 && std::string(argv[1]) == "--device-info") {
      auto properties = deviceInfo();
      int runtime = 0, driver = 0;
      check(cudaRuntimeGetVersion(&runtime));
      check(cudaDriverGetVersion(&driver));
      std::cout << "{\"name\":\"" << properties.name << "\",\"l2_bytes\":"
                << properties.l2CacheSize << ",\"sm_count\":" << properties.multiProcessorCount
                << ",\"compute_major\":" << properties.major << ",\"compute_minor\":"
                << properties.minor << ",\"total_memory_bytes\":" << properties.totalGlobalMem
                << ",\"runtime_version\":" << runtime << ",\"driver_version\":" << driver
                << ",\"cooperative_launch\":" << properties.cooperativeLaunch << "}\n";
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--correctness") {
      std::string filter = "all";
      bool small = false;
      for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--sanitizer-cases") small = true;
        else if (filter == "all") filter = argv[i];
        else throw std::runtime_error("unexpected correctness argument");
      }
      return correctness(filter, small);
    }
    if (argc >= 8 && std::string(argv[1]) == "--benchmark") {
      auto condition = parseCondition(argv);
      const std::vector<int> sizes{64, 512, 4096, 32768, 131072, 262144, 1048576};
      if (std::find(sizes.begin(), sizes.end(), condition.rows) == sizes.end()) {
        throw std::runtime_error("rows are outside the frozen benchmark sweep");
      }
      condition.regime = argv[6];
      condition.order = integer(argv[7], 0, 2);
      std::string filter = "all", stats;
      for (int i = 8; i < argc; ++i) {
        if (std::string(argv[i]) == "--stats" && i + 1 < argc) stats = argv[++i];
        else if (filter == "all") filter = argv[i];
        else throw std::runtime_error("unexpected benchmark argument");
      }
      return benchmark(condition, filter, stats);
    }
    if ((argc == 7 || argc == 8) && std::string(argv[1]) == "--trace") {
      auto condition = parseCondition(argv);
      condition.regime = argc == 8 ? argv[7] : "repeated";
      condition.order = 0;
      auto selection = selectedMethods(argv[6]);
      if (selection.size() != 1) throw std::runtime_error("trace requires one method");
      return trace(condition, selection[0]);
    }
    std::cerr << "Usage:\n"
              << "  --device-info\n"
              << "  --correctness [method|all] [--sanitizer-cases]\n"
              << "  --benchmark ROWS FAMILY STRUCTURE TILE REGIME ORDER [method|all] [--stats PATH]\n"
              << "  --trace ROWS FAMILY STRUCTURE TILE METHOD [REGIME=repeated]\n";
    return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

} // namespace spmm_megakernel

int main(int argc, char **argv) { return spmm_megakernel::main(argc, argv); }
