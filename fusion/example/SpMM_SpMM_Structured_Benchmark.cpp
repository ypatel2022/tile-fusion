#if !defined(MKL) || !defined(__AVX2__)
#error "The structured CPU benchmark requires MKL and AVX2."
#endif

#include "SpMM_SpMM_SP_Demo_Utils.h"
#include "SpMM_SpMM_Structured_Inputs.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using namespace sym_lib;
using namespace spmm_spmm_benchmark;

namespace {

constexpr int Features = 32;
const char *Methods[] = {"cpu_unfused", "cpu_mkl", "cpu_avx2_unfused",
                         "cpu_avx2_fused"};

template <typename Implementation>
class CheckedOutput : public Implementation {
protected:
  bool verify(double &error) override {
    bool correct = Implementation::verify(error);
    // The existing maximum-error comparison does not reject NaNs.
    for (size_t i = 0; i < size_t(this->InTensor->L) * this->InTensor->N; ++i) {
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

bool printMethodStats(SWBench *benchmark, Stats *stats,
                      const TestParameters &parameters,
                      const ScheduleParameters &schedule,
                      const StructuredCase &matrixCase, int order,
                      bool &printHeader) {
  bool correct = true;
  for (const auto *info : stats->ProfilingInfoTrials) {
    correct = info->ErrorPerExecute.first && correct;
  }
  // All methods must produce the same header regardless of rotated order.
  stats->ProfilingInfoTrials[0]->AnalysisTime.ElapsedTimeArray[0].second =
      "Analysis";
  auto matrixInfo = parameters.print_csv(true);
  auto scheduleInfo = schedule.print_csv(true);
  if (printHeader) {
    std::cout << benchmark->printStatsHeader() << std::get<0>(scheduleInfo)
              << std::get<0>(matrixInfo)
              << "Process Order,Warmup Trials,Target Fused Percent,Fused Ratio,"
                 "Device,Matrix Family,Structure Size,Tile Eligible Ratio\n";
    printHeader = false;
  }
  std::cout << benchmark->printStats() << std::get<1>(scheduleInfo)
            << std::get<1>(matrixInfo) << order << ",1,-1,-1,cpu,"
            << matrixCase.family() << ',' << matrixCase.structure << ','
            << double(matrixCase.eligibleRows()) / matrixCase.rows << '\n';
  // Preserve completed measurements if an unchanged implementation crashes
  // during its existing cleanup.
  std::cout.flush();
  if (!correct) {
    std::cerr << parameters._matrix_name << ": " << stats->Name
              << " failed correctness verification\n";
  }
  return correct;
}

bool runMethod(int method, TensorInputs<float> *input,
               const TestParameters &parameters, ScheduleParameters &schedule,
               const StructuredCase &matrixCase, int runs, int order,
               bool &printHeader) {
  auto *stats = new Stats(Methods[method], "SpMMSpMM", runs + 1,
                          parameters._matrix_name, input->NumThreads);
  stats->OtherStats["PackingType"] = {Separated};
  stats->OtherStats["TilingMethod"] = {Fixed};
  bool correct;
  switch (method) {
  case 0: {
    auto *unfused = new CheckedOutput<SpMMSpMMUnFusedParallelSP>(input, stats);
    unfused->run();
    correct = printMethodStats(unfused, stats, parameters, schedule, matrixCase,
                               order, printHeader);
    delete unfused;
    break;
  }
  case 1: {
    auto *mkl = new CheckedOutput<SpMMSpMMMKL_SP>(input, stats);
    mkl->run();
    correct = printMethodStats(mkl, stats, parameters, schedule, matrixCase,
                               order, printHeader);
    delete mkl;
    break;
  }
  case 2: {
    auto *unfusedAvx2 =
        new CheckedOutput<SpMMSpMMUnFusedParallelVectorizedAVX2SP>(
            input, stats, schedule);
    unfusedAvx2->run();
    correct = printMethodStats(unfusedAvx2, stats, parameters, schedule,
                               matrixCase, order, printHeader);
    delete unfusedAvx2;
    break;
  }
  default: {
    auto *fusedAvx2 =
        new CheckedOutput<SpMMSpMMFusedOneSparseMatInterLayerVectorizedAvx256SP>(
            input, stats, schedule);
    fusedAvx2->run();
    correct = printMethodStats(fusedAvx2, stats, parameters, schedule, matrixCase,
                               order, printHeader);
    delete fusedAvx2;
    break;
  }
  }
  delete stats;
  return correct;
}

void printUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " [threads=1] [runs=100] [order=0] [rows=all]\n"
            << "  threads: 1, 4, or 32\n"
            << "  runs: 1..10000 measured runs, plus one discarded warmup\n"
            << "  order: 0..2, rotates which method runs first\n"
            << "  rows: all, 64, 512, 4096, 32768, 262144, or 1048576\n";
}

} // namespace

int main(int argc, char *argv[]) {
  int threads = 1, runs = 100, order = 0, rowsFilter = 0;
  if (argc == 2 && std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return 0;
  }
  if (argc > 5 ||
      (argc > 1 && !parseInteger(argv[1], 1, 32, threads)) ||
      (threads != 1 && threads != 4 && threads != 32) ||
      (argc > 2 && !parseInteger(argv[2], 1, 10000, runs)) ||
      (argc > 3 && !parseInteger(argv[3], 0, 2, order)) ||
      (argc > 4 && std::string(argv[4]) != "all" &&
       (!parseInteger(argv[4], 64, 1048576, rowsFilter) ||
        !validRows(rowsFilter)))) {
    printUsage(argv[0]);
    return 1;
  }

  // Map the positional benchmark CLI onto the existing CPU demo settings.
  ScheduleParameters configuredSchedule;
  TestParameters configuredParameters;
  std::string threadArgument = std::to_string(threads);
  const char *configuration[] = {argv[0], "-nt", threadArgument.c_str(),
                                 "-bc", "32", "-ip", "128", "-tm", "-1",
                                 "-tn", "32", "-ed", "0"};
  parse_args(sizeof(configuration) / sizeof(configuration[0]), configuration,
             &configuredSchedule, &configuredParameters);

  omp_set_dynamic(0);
  omp_set_num_threads(threads);
  mkl_set_dynamic(0);
  mkl_set_num_threads(threads);
  std::cout << std::setprecision(10);
  bool allCorrect = true, printHeader = true;
  int caseIndex = 0;
  for (const auto &matrixCase : structuredCases(rowsFilter)) {
    auto *matrix = makeStructuredMatrix(matrixCase);
    TestParameters parameters = configuredParameters;
    parameters._matrix_name = matrixCase.name();
    parameters._order_method = SYM_ORDERING::NONE;
    parameters._dim1 = parameters._dim2 = matrixCase.rows;
    parameters._nnz = matrix->nnz;
    parameters._density = double(matrix->nnz) / matrixCase.rows / matrixCase.rows;
    parameters._b_cols = Features;
    parameters._embed_dim = 0;

    ScheduleParameters schedule(configuredSchedule);
    schedule._num_w_partition =
        std::max(matrixCase.rows / schedule.IterPerPartition, 2 * threads);
    auto *input = new TensorInputs<float>(
        matrixCase.rows, Features, matrixCase.rows, matrixCase.rows, matrix,
        matrix, threads, runs + 1, "SpMMSpMMStructuredBenchmark");
    initializeDense(input->Bx, size_t(matrixCase.rows) * Features);

    // The sequential implementation is independent of every timed method.
    auto *stats = new Stats("cpu_reference", "SpMMSpMM", 1,
                            parameters._matrix_name, 1);
    auto *reference = new SpMMSpMMUnFusedSP(input, stats);
    reference->run();
    std::copy(reference->OutTensor->Xx,
              reference->OutTensor->Xx + size_t(matrixCase.rows) * Features,
              input->CorrectSol);
    input->IsSolProvided = true;
    delete reference;
    delete stats;
    delete matrix;

    for (int offset = 0; offset < 4; ++offset) {
      int method = (caseIndex + order + offset) % 4;
      bool correct = runMethod(method, input, parameters, schedule, matrixCase,
                               runs, order, printHeader);
      allCorrect = correct && allCorrect;
    }
    delete input;
    ++caseIndex;
  }
  return allCorrect ? 0 : 1;
}
