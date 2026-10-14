#include "../Cuda_GeMM_SpMM_Graph.h"

#include <iostream>
#include <string>
#include <vector>

using namespace sym_lib;

// Runs one GPU method and returns whether every trial matched the CPU
// reference. Only correct methods get a CSV row, since a failed method may
// have timed nothing.
bool runGpuMethod(CudaTensorInputs *In, const float *C,
                  const TestParameters &Tp, ScheduleParameters Sp,
                  const std::string &Name, int TileRows, SpMMKernel Kernel,
                  Schedule Sched, bool UseGraph, int NumTrial,
                  int ThreadsPerBlock, bool &PrintHeader) {
  auto *stats = new swiftware::benchmark::Stats(
      Name, "GeMMSpMM", NumTrial, Tp._matrix_name, Sp._num_threads);
  auto *method = new GeMMSpMMTileFusionGraph(In, stats, C, TileRows, Kernel,
                                             Sched, UseGraph, ThreadsPerBlock);
  method->run();
  bool correct = true;
  for (int i = 0; i < NumTrial; ++i) {
    correct = correct && stats->ProfilingInfoTrials[i]->ErrorPerExecute.first;
  }
  Sp.IterPerPartition = TileRows;
  auto spCsv = Sp.print_csv(true);
  auto tpCsv = Tp.print_csv(true);
  if (PrintHeader) {
    std::cout << method->printStatsHeader() << std::get<0>(spCsv)
              << std::get<0>(tpCsv) << std::endl;
    PrintHeader = false;
  }
  if (correct) {
    std::cout << method->printStats() << std::get<1>(spCsv)
              << std::get<1>(tpCsv) << std::endl;
  } else {
    std::cerr << Tp._matrix_name << ": " << Name << " with tile rows "
              << TileRows << " failed verification" << std::endl;
  }
  delete method;
  delete stats;
  return correct;
}

int main(const int argc, const char *argv[]) {
  TestParameters tp;
  tp._order_method = SYM_ORDERING::NONE;
  ScheduleParameters sp;
  parse_args(argc, argv, &sp, &tp);
  CSC *aCSC = get_matrix_from_parameter(&tp);
  if (aCSC->m != aCSC->n) {
    return -1;
  }
  CSC *aCSCFull = nullptr;
  if (aCSC->stype == -1 || aCSC->stype == 1) {
    aCSCFull = make_full(aCSC);
  } else {
    aCSCFull = copy_sparse(aCSC);
  }
  delete aCSC;
  tp._dim1 = aCSCFull->m;
  tp._dim2 = aCSCFull->n;
  tp._nnz = aCSCFull->nnz;
  tp._density = (double)tp._nnz / (double)(tp._dim1 * tp._dim2);
  // Distinct values in A, and in B below, so a misplaced row or a value array
  // read at the wrong offset changes D. Values do not affect performance.
  for (int j = 0; j < aCSCFull->nnz; ++j) {
    aCSCFull->x[j] = 0.5 + (j * 7919 % 97) / 97.0;
  }

  int n = aCSCFull->m, k = tp._b_cols;
  tp._embed_dim = k; // C is k x k
  int numThread = sp._num_threads, numTrial = 7, threadsPerBlock = 128;
  auto *inGeMMSpMM =
      new CudaTensorInputs(n, k, n, n, aCSCFull, aCSCFull, numThread, numTrial,
                           "GeMM_SpMM_Graph_Demo");
  // C is scaled by 1/k so that D1 stays in the range of B.
  for (int i = 0; i < n * k; ++i) {
    inGeMMSpMM->Bx[i] = (int((i * 2654435761u >> 8) % 251) - 125) / 128.0f;
  }
  cudaMemcpy(inGeMMSpMM->DBx, inGeMMSpMM->Bx, size_t(n) * k * sizeof(float),
             cudaMemcpyHostToDevice);
  std::vector<float> c(size_t(k) * k);
  for (int i = 0; i < k * k; ++i) {
    c[i] = ((i % 13) * 7 % 13 - 6) / (6.0f * k);
  }
  // Float GeMM and SpMM accumulate in different orders on the CPU and GPU.
  inGeMMSpMM->Threshold = 1e-4;

  // The sequential reference runs once; GPU methods are checked against it.
  auto *stats = new swiftware::benchmark::Stats("CPU_Unfused_Seq", "GeMMSpMM",
                                                1, tp._matrix_name, numThread);
  auto *cpuGeMMSpMM = new SeqGeMMSpMM(inGeMMSpMM, stats, c.data());
  cpuGeMMSpMM->run();
  std::copy(cpuGeMMSpMM->OutTensor->Xx,
            cpuGeMMSpMM->OutTensor->Xx + size_t(n) * k, inGeMMSpMM->CorrectSol);
  inGeMMSpMM->IsSolProvided = true;
  delete cpuGeMMSpMM;
  delete stats;

  // -ip selects one tile size, as for the CPU GeMM-SpMM fixed tiles; otherwise
  // sweep tiles smaller than the matrix.
  std::vector<int> tileRows;
  if (sp.IterPerPartition > 1) {
    tileRows.push_back(sp.IterPerPartition);
  } else {
    for (int t : {1024, 4096, 16384, 65536, 262144}) {
      if (t < n) {
        tileRows.push_back(t);
      }
    }
  }
  const SpMMKernel kernels[] = {SpMMKernel::SeqReduceRowBalance,
                                SpMMKernel::CuSparseAlg2,
                                SpMMKernel::CuSparseAlg3};
  const std::string kernelNames[] = {"SeqReduceRowBalance", "CuSparse_ALG2",
                                     "CuSparse_ALG3"};

  const Schedule schedules[] = {Schedule::TileFusion,
                                Schedule::TileFusionBarrier,
                                Schedule::TiledUnfused};
  const std::string scheduleNames[] = {"Fused_Graph", "Fused_Barrier_Graph",
                                       "Unfused_Tiled_Graph"};

  bool printHeader = tp.print_header, correct = true;
  // One tile of n rows is the unfused GeMM then SpMM.
  for (int i = 0; i < 3; ++i) {
    correct &= runGpuMethod(inGeMMSpMM, c.data(), tp, sp,
                            "GPU_Unfused_Direct_" + kernelNames[i], n,
                            kernels[i], Schedule::TileFusion, false, numTrial,
                            threadsPerBlock, printHeader);
    correct &= runGpuMethod(inGeMMSpMM, c.data(), tp, sp,
                            "GPU_Unfused_Graph_" + kernelNames[i], n,
                            kernels[i], Schedule::TileFusion, true, numTrial,
                            threadsPerBlock, printHeader);
  }
  for (int t : tileRows) {
    for (int i = 0; i < 3; ++i) {
      for (int s = 0; s < 3; ++s) {
        correct &= runGpuMethod(
            inGeMMSpMM, c.data(), tp, sp,
            "GPU_" + scheduleNames[s] + "_" + kernelNames[i], t, kernels[i],
            schedules[s], true, numTrial, threadsPerBlock, printHeader);
      }
    }
  }

  delete inGeMMSpMM;
  delete aCSCFull;
  return correct ? 0 : 1;
}
