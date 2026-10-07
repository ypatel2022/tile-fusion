#include "../Cuda_SpMM_SpMM_Graph.h"
#include "../../example/SpMM_SpMM_Structured_Inputs.h"
#include "SpMM_SpMM_Structured_GPU.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr int ThreadsPerBlock = 128;

class FusedDirect : public FusedSpMMSpMMSeqReduceRowBalance {
protected:
  Timer analysis() override {
    auto t = FusedSpMMSpMMSeqReduceRowBalance::analysis();
    // Keep a valid second launch at 100% fusion, matching the graph's no-op node.
    if (UFMGridDim == 0) {
      UFMGridDim = 1;
    }
    return t;
  }

public:
  using FusedSpMMSpMMSeqReduceRowBalance::FusedSpMMSpMMSeqReduceRowBalance;
};

// This control removes the original implementation's wait between kernels.
// Both launches use the same stream, which already preserves their dependency.
class FusedSameStream : public FusedSpMMSpMMSeqReduceRowBalance {
protected:
  Timer execute() override {
    Timer t;
    dim3 blockDim(NBlockDim, MBlockDim, 1);
    dim3 fusedGrid(MGridDim, NGridDim, 1);
    dim3 deferredGrid(UFMGridDim > 0 ? UFMGridDim : 1, NGridDim, 1);
    t.startGPU();
    csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<fusedGrid, blockDim>>>(
        InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
        InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx,
        OutTensor->DXx, DFPtr, DFId);
    csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel<<<deferredGrid,
                                                       blockDim>>>(
        UFDim, InTensor->N, InTensor->K, InTensor->DACsrAp, InTensor->DACsrI,
        InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx, DUFPtr);
    cudaStreamSynchronize(0);
    t.stopGPU("FusedSpMMSpMMSameStream");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  using FusedSpMMSpMMSeqReduceRowBalance::FusedSpMMSpMMSeqReduceRowBalance;
};

template <typename Implementation>
class CheckedOutput : public Implementation {
protected:
  bool verify(double &error) override {
    bool correct = Implementation::verify(error);
    // The inherited maximum-error comparison does not reject NaNs.
    for (int i = 0; i < this->InTensor->M * this->InTensor->N; ++i) {
      if (!std::isfinite(this->OutTensor->Xx[i])) {
        error = std::numeric_limits<double>::infinity();
        return false;
      }
    }
    return correct;
  }

public:
  using Implementation::Implementation;
};

template <typename Implementation>
bool runMethod(CudaTensorInputs *input, const TestParameters &parameters,
               const char *method, int runs, int order, int fusedPercent,
               bool printHeader,
               const spmm_spmm_benchmark::StructuredCase *structured = nullptr,
               int tileRows = 4) {
  if (structured) {
    // A recoverable failure in an earlier method must not invalidate this one.
    cudaGetLastError();
  }
  auto *stats = new Stats(method, "SpMMSpMM", runs + 1, parameters._matrix_name, 1);
  auto *benchmark =
      new CheckedOutput<Implementation>(input, stats,
                                        structured ? 32 * tileRows : ThreadsPerBlock);
  benchmark->run();

  bool correct = true;
  double fusedRows = stats->OtherStats.at("Number of Fused Rows")[0];
  if (fusedPercent >= 0 && fusedRows != input->M * fusedPercent / 100) {
    std::cerr << parameters._matrix_name
              << ": unexpected fused row count: " << fusedRows << '\n';
    correct = false;
  }
  if (structured &&
      (std::string(method) == "direct_same_stream" ||
       std::string(method) == "graph") &&
      fusedRows != structured->eligibleRows(tileRows)) {
    std::cerr << parameters._matrix_name
              << ": unexpected structured fused row count: " << fusedRows << '\n';
    correct = false;
    for (int trial = 0; trial <= runs; ++trial) {
      stats->ProfilingInfoTrials[trial]->ErrorPerExecute =
          std::make_pair(false, std::numeric_limits<double>::infinity());
    }
  }
  for (int trial = 0; trial <= runs; ++trial) {
    const auto *info = stats->ProfilingInfoTrials[trial];
    correct = correct && info->ErrorPerExecute.first;
  }

  // Keep the header identical when a different method runs first.
  stats->ProfilingInfoTrials[0]->AnalysisTime.ElapsedTimeArray[0].second = "Analysis";
  auto matrixInfo = parameters.print_csv(true);
  if (printHeader) {
    std::cout << benchmark->printStatsHeader() << std::get<0>(matrixInfo)
              << "Process Order,Warmup Trials,Target Fused Percent,Fused Ratio";
    if (structured) {
      std::cout << ",Device,Matrix Family,Structure Size,Tile Eligible Ratio,GPU Tile Rows";
    }
    std::cout << '\n';
  }
  std::cout << benchmark->printStats() << std::get<1>(matrixInfo)
            << order << ",1," << fusedPercent << ',' << fusedRows / input->M;
  if (structured) {
    std::cout << ",gpu," << structured->family() << ',' << structured->structure
              << ',' << double(structured->eligibleRows(tileRows)) / input->M
              << ',' << tileRows;
  }
  std::cout << '\n';
  if (!correct) {
    std::cerr << parameters._matrix_name << ": " << method
              << " failed correctness verification\n";
  }
  delete benchmark;
  delete stats;
  return correct;
}

CSC *makeMatrix(int rows, int fusedPercent) {
  auto *matrix = sym_lib::tridiag(rows, 0.25, 0.5, 0.25);
  // tridiag supplies both triangles and values despite its default metadata.
  matrix->stype = 0;
  matrix->is_pattern = false;
  if (fusedPercent < 0) {
    return matrix;
  }

  auto *csr = sym_lib::csc_to_csr(matrix);
  delete matrix;
  // With 32 features and 128 threads, each producer tile has four rows.
  // Select the two interior rows first to keep 50% close to the tridiagonal case.
  const int priority[] = {2, 0, 1, 3};
  for (int row = 0; row < rows; ++row) {
    int tile = row / 4 * 4;
    bool fused = priority[row % 4] < fusedPercent / 25;
    bool crossesTile = false;
    for (int j = csr->p[row]; j < csr->p[row + 1]; ++j) {
      crossesTile |= csr->i[j] / 4 != row / 4;
      if (fused) {
        csr->i[j] = tile + csr->i[j] % 4;
      }
    }
    if (!fused && !crossesTile) {
      int offDiagonal = csr->p[row] + (csr->i[csr->p[row]] == row);
      csr->i[offDiagonal] = (tile + 4) % rows;
    }
  }
  matrix = sym_lib::csr_to_csc(csr);
  delete csr;
  return matrix;
}

bool parseInteger(const char *text, int minimum, int maximum, int &value) {
  char *end;
  errno = 0;
  long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < minimum ||
      parsed > maximum) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

void printUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " [features=32] [runs=100] [order=0] [fused-percent]\n"
            << "  features: 32, 64, 128, or 256 dense columns\n"
            << "  runs: 1..10000 measured runs, plus one discarded warmup\n"
            << "  order: 0..2, rotates which method runs first\n"
            << "  fused-percent: 0, 25, 50, 75, or 100; requires features=32\n"
            << "  omit fused-percent to use the original tridiagonal matrices\n"
            << "       " << program
            << " --structured [runs=100] [order=0] [rows=all] [tile-rows=4]\n"
            << "  rows: all, 64, 512, 4096, 32768, 262144, or 1048576\n"
            << "  tile-rows: 1, 4, or 16; original kernels use 32 threads per row\n"
            << "  optional profiling filters: [matrix-name=all] [method-index=all]\n"
            << "  methods: 0/1 unfused direct/graph, 2/3 ALG2 direct/graph,\n"
            << "           4/5 ALG3 direct/graph, 6/7 fused direct/graph\n";
}

bool runStructuredMethod(int method, CudaTensorInputs *input,
                         const TestParameters &parameters, int runs, int order,
                         bool printHeader,
                         const spmm_spmm_benchmark::StructuredCase &testCase,
                         int tileRows) {
  switch (method) {
  case 0:
    return runMethod<UnfusedPair<false>>(
        input, parameters, "unfused_direct", runs, order, -1, printHeader,
        &testCase, tileRows);
  case 1:
    return runMethod<UnfusedPair<true>>(
        input, parameters, "unfused_graph", runs, order, -1, printHeader,
        &testCase, tileRows);
  case 2:
    return runMethod<CuSparsePair<CUSPARSE_SPMM_CSR_ALG2, false>>(
        input, parameters, "cusparse_alg2_direct", runs, order, -1, printHeader,
        &testCase, tileRows);
  case 3:
    return runMethod<CuSparsePair<CUSPARSE_SPMM_CSR_ALG2, true>>(
        input, parameters, "cusparse_alg2_graph", runs, order, -1, printHeader,
        &testCase, tileRows);
  case 4:
    return runMethod<CuSparsePair<CUSPARSE_SPMM_CSR_ALG3, false>>(
        input, parameters, "cusparse_alg3_direct", runs, order, -1, printHeader,
        &testCase, tileRows);
  case 5:
    return runMethod<CuSparsePair<CUSPARSE_SPMM_CSR_ALG3, true>>(
        input, parameters, "cusparse_alg3_graph", runs, order, -1, printHeader,
        &testCase, tileRows);
  case 6:
    return runMethod<FusedSameStream>(
        input, parameters, "direct_same_stream", runs, order, -1, printHeader,
        &testCase, tileRows);
  default:
    return runMethod<FusedSpMMSpMMSeqReduceRowBalanceGraph>(
        input, parameters, "graph", runs, order, -1, printHeader, &testCase, tileRows);
  }
}

int runStructured(int argc, char *argv[]) {
  int runs = 100, order = 0, rowFilter = 0, tileRows = 4, methodFilter = -1;
  std::string matrixFilter = argc > 6 ? argv[6] : "all";
  if (argc > 8 ||
      (argc > 2 && !parseInteger(argv[2], 1, 10000, runs)) ||
      (argc > 3 && !parseInteger(argv[3], 0, 2, order)) ||
      (argc > 4 && std::string(argv[4]) != "all" &&
       (!parseInteger(argv[4], 64, 1048576, rowFilter) ||
        !spmm_spmm_benchmark::validRows(rowFilter))) ||
      (argc > 5 && (!parseInteger(argv[5], 1, 16, tileRows) ||
                    (tileRows != 1 && tileRows != 4 && tileRows != 16))) ||
      (argc > 7 && std::string(argv[7]) != "all" &&
       !parseInteger(argv[7], 0, 7, methodFilter))) {
    printUsage(argv[0]);
    return 1;
  }

  bool correct = true, printHeader = true;
  int caseIndex = 0;
  std::cout << std::setprecision(10);
  for (const auto &testCase : spmm_spmm_benchmark::structuredCases(rowFilter)) {
    if (matrixFilter != "all" && matrixFilter != testCase.name()) {
      continue;
    }
    auto *matrix = spmm_spmm_benchmark::makeStructuredMatrix(testCase);
    TestParameters parameters;
    parameters._matrix_name = testCase.name();
    parameters._order_method = SYM_ORDERING::NONE;
    parameters._dim1 = parameters._dim2 = testCase.rows;
    parameters._nnz = matrix->nnz;
    parameters._density = double(matrix->nnz) / testCase.rows / testCase.rows;
    parameters._b_cols = 32;
    auto *input = new CudaTensorInputs(
        testCase.rows, 32, testCase.rows, testCase.rows, matrix, matrix, 1,
        runs + 1, "SpMMSpMMStructuredBenchmark");
    size_t denseElements = static_cast<size_t>(testCase.rows) * 32;
    spmm_spmm_benchmark::initializeDense(input->Bx, denseElements);
    cudaMemcpy(input->DBx, input->Bx, denseElements * sizeof(float),
               cudaMemcpyHostToDevice);

    auto *stats = new Stats("cpu", "SpMMSpMM", 1, parameters._matrix_name, 1);
    auto *reference = new SeqSpMMSpMM(input, stats);
    reference->run();
    std::copy(reference->OutTensor->Xx,
              reference->OutTensor->Xx + denseElements, input->CorrectSol);
    input->IsSolProvided = true;
    delete reference;
    delete stats;

    for (int position = 0; position < 8; ++position) {
      int method = (caseIndex + order + position) % 8;
      if (methodFilter >= 0 && method != methodFilter) {
        continue;
      }
      bool methodCorrect = runStructuredMethod(
          method, input, parameters, runs, order, printHeader, testCase, tileRows);
      printHeader = false;
      correct = methodCorrect && correct;
    }
    delete input;
    delete matrix;
    ++caseIndex;
  }
  return correct && caseIndex > 0 ? 0 : 1;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc > 1 && std::string(argv[1]) == "--structured") {
    return runStructured(argc, argv);
  }
  int features = 32, runs = 100, order = 0, fusedPercent = -1;
  if (argc == 2 && std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return 0;
  }
  if (argc > 5 ||
      (argc > 1 && !parseInteger(argv[1], 32, 256, features)) ||
      (argc > 2 && !parseInteger(argv[2], 1, 10000, runs)) ||
      (argc > 3 && !parseInteger(argv[3], 0, 2, order)) ||
      (argc > 4 && (!parseInteger(argv[4], 0, 100, fusedPercent) ||
                    fusedPercent % 25 != 0 || features != 32)) ||
      (features != 32 && features != 64 && features != 128 && features != 256)) {
    printUsage(argv[0]);
    return 1;
  }

  const int sizes[] = {64, 512, 4096, 32768, 262144, 1048576};
  std::cout << std::setprecision(10);
  int caseIndex = 0;
  for (int rows : sizes) {
    auto *matrix = makeMatrix(rows, fusedPercent);
    std::string matrixName = fusedPercent < 0
        ? "tridiagonal_" + std::to_string(rows)
        : "tile_local_" + std::to_string(rows) + "_" + std::to_string(fusedPercent);
    TestParameters parameters;
    parameters._matrix_name = matrixName;
    parameters._order_method = SYM_ORDERING::NONE;
    parameters._dim1 = parameters._dim2 = rows;
    parameters._nnz = matrix->nnz;
    parameters._density = double(matrix->nnz) / rows / rows;
    parameters._b_cols = features;
    auto *input = new CudaTensorInputs(rows, features, rows, rows, matrix, matrix,
                                       1, runs + 1, "SpMMSpMMGraphBenchmark");
    for (int i = 0; i < rows * features; ++i) {
      input->Bx[i] = (((i % 251) * 37) % 251 - 125) / 128.0f;
    }
    cudaMemcpy(input->DBx, input->Bx,
               static_cast<size_t>(rows) * features * sizeof(float),
               cudaMemcpyHostToDevice);

    auto *stats = new Stats("cpu", "SpMMSpMM", 1, matrixName, 1);
    auto *reference = new SeqSpMMSpMM(input, stats);
    reference->run();
    std::copy(reference->OutTensor->Xx,
              reference->OutTensor->Xx + rows * features, input->CorrectSol);
    input->IsSolProvided = true;
    delete reference;
    delete stats;

    bool correct = true;
    // Graph construction is in analysis(); trial zero warms its first replay.
    for (int position = 0; position < 3; ++position) {
      int method = (caseIndex + order + position) % 3;
      bool printHeader = caseIndex == 0 && position == 0;
      bool methodCorrect;
      if (method == 0) {
        methodCorrect = runMethod<FusedDirect>(
            input, parameters, "direct", runs, order, fusedPercent, printHeader);
      } else if (method == 1) {
        methodCorrect = runMethod<FusedSameStream>(
            input, parameters, "direct_same_stream", runs, order, fusedPercent,
            printHeader);
      } else {
        methodCorrect = runMethod<FusedSpMMSpMMSeqReduceRowBalanceGraph>(
            input, parameters, "graph", runs, order, fusedPercent, printHeader);
      }
      correct = correct && methodCorrect;
    }
    delete input;
    delete matrix;
    if (!correct) {
      return 1;
    }
    ++caseIndex;
  }
  return 0;
}
