#include "../Cuda_SpMM_SpMM_Demo_Utils.h"
#include "../../example/SpMM_SpMM_Structured_Inputs.h"

#include <cublas_v2.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int Features = 32;

template <cusparseSpMMAlg_t Algorithm, bool Replay>
class CuBlasCuSparsePair : public SeqSpMMSpMM {
  const float *Weights;
  float *DeviceWeights = nullptr;
  void *Workspace = nullptr;
  cublasHandle_t Blas = nullptr;
  cusparseHandle_t Sparse = nullptr;
  cusparseSpMatDescr_t A = nullptr;
  cusparseDnMatDescr_t H = nullptr, Y = nullptr;
  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;
  float Alpha = 1.0f, Beta = 0.0f;
  bool Valid = true;

  void launchPair() {
    if (!Valid) {
      return;
    }
    // cuBLAS uses column-major storage: H^T = W^T X^T writes row-major H.
    Valid = cublasSgemm(Blas, CUBLAS_OP_N, CUBLAS_OP_N, Features,
                       InTensor->M, Features, &Alpha, DeviceWeights, Features,
                       InTensor->DBx, Features, &Beta, OutTensor->DACx,
                       Features) == CUBLAS_STATUS_SUCCESS;
    if (Valid) {
      Valid = cusparseSpMM(Sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, A, H,
                           &Beta, Y, CUDA_R_32F, Algorithm, Workspace) ==
              CUSPARSE_STATUS_SUCCESS;
    }
  }

protected:
  Timer analysis() override {
    Timer t;
    t.start();
    Valid = cudaMalloc(&DeviceWeights, Features * Features * sizeof(float)) ==
            cudaSuccess;
    if (Valid) {
      Valid = cudaMemcpy(DeviceWeights, Weights,
                          Features * Features * sizeof(float),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    }
    Valid = (cublasCreate(&Blas) == CUBLAS_STATUS_SUCCESS) && Valid;
    Valid = (cusparseCreate(&Sparse) == CUSPARSE_STATUS_SUCCESS) && Valid;
    if (Valid) {
      // Avoid reduced-precision Tensor Core math in this float32 comparison.
      Valid = cublasSetMathMode(Blas, CUBLAS_PEDANTIC_MATH) ==
              CUBLAS_STATUS_SUCCESS;
      Valid = (cusparseCreateCsr(
          &A, InTensor->M, InTensor->M, InTensor->ACsr->nnz,
          InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
          CUDA_R_32F) == CUSPARSE_STATUS_SUCCESS) && Valid;
      Valid = (cusparseCreateDnMat(&H, InTensor->M, Features, Features,
                                   OutTensor->DACx, CUDA_R_32F,
                                   CUSPARSE_ORDER_ROW) ==
               CUSPARSE_STATUS_SUCCESS) && Valid;
      Valid = (cusparseCreateDnMat(&Y, InTensor->M, Features, Features,
                                   OutTensor->DXx, CUDA_R_32F,
                                   CUSPARSE_ORDER_ROW) ==
               CUSPARSE_STATUS_SUCCESS) && Valid;
    }
    size_t workspaceBytes = 0;
    if (Valid) {
      Valid = cusparseSpMM_bufferSize(
          Sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, A, H, &Beta, Y,
          CUDA_R_32F, Algorithm, &workspaceBytes) == CUSPARSE_STATUS_SUCCESS;
    }
    if (Valid && workspaceBytes != 0) {
      Valid = cudaMalloc(&Workspace, workspaceBytes) == cudaSuccess;
    }

    cudaStream_t preparationStream = nullptr;
    if (Valid) {
      Valid = cudaStreamCreate(&preparationStream) == cudaSuccess;
    }
    if (Valid) {
      Valid = cublasSetStream(Blas, preparationStream) == CUBLAS_STATUS_SUCCESS;
      Valid = (cusparseSetStream(Sparse, preparationStream) ==
               CUSPARSE_STATUS_SUCCESS) && Valid;
      Valid = (cusparseSpMM_preprocess(
          Sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, A, H, &Beta, Y,
          CUDA_R_32F, Algorithm, Workspace) == CUSPARSE_STATUS_SUCCESS) && Valid;
      // Prime both modes before capture or timing, including library first use.
      launchPair();
      Valid = (cudaStreamSynchronize(preparationStream) == cudaSuccess) && Valid;
    }
    if (Replay && Valid) {
      Valid = cudaStreamBeginCapture(preparationStream,
                                     cudaStreamCaptureModeGlobal) == cudaSuccess;
      if (Valid) {
        launchPair();
        Valid = (cudaStreamEndCapture(preparationStream, &Graph) == cudaSuccess) &&
                Valid;
      }
      if (Valid) {
        Valid = cudaGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0) ==
                cudaSuccess;
      }
    }
    if (Blas) {
      Valid = (cublasSetStream(Blas, 0) == CUBLAS_STATUS_SUCCESS) && Valid;
    }
    if (Sparse) {
      Valid = (cusparseSetStream(Sparse, 0) == CUSPARSE_STATUS_SUCCESS) && Valid;
    }
    if (preparationStream) {
      Valid = (cudaStreamDestroy(preparationStream) == cudaSuccess) && Valid;
    }
    if (!Valid) {
      std::cerr << "GEMM-SpMM library or graph preparation failed\n";
    }
    t.stop("Analysis");
    return t;
  }

  Timer execute() override {
    Timer t;
    t.startGPU();
    if (Replay && Valid) {
      Valid = cudaGraphLaunch(GraphExec, 0) == cudaSuccess;
    } else if (!Replay) {
      launchPair();
    }
    Valid = (cudaStreamSynchronize(0) == cudaSuccess) && Valid;
    t.stopGPU("GEMMSpMM");
    if (Valid) {
      Valid = cudaMemcpy(OutTensor->Xx, OutTensor->DXx,
                          size_t(InTensor->M) * Features * sizeof(float),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    return t;
  }

  bool verify(double &error) override {
    if (!Valid) {
      error = std::numeric_limits<double>::infinity();
      return false;
    }
    bool correct = SeqSpMMSpMM::verify(error);
    for (size_t i = 0; i < size_t(InTensor->M) * Features; ++i) {
      if (!std::isfinite(OutTensor->Xx[i])) {
        error = std::numeric_limits<double>::infinity();
        return false;
      }
    }
    return correct;
  }

public:
  CuBlasCuSparsePair(CudaTensorInputs *input, Stats *stats, const float *weights)
      : SeqSpMMSpMM(input, stats), Weights(weights) {}

  ~CuBlasCuSparsePair() {
    if (GraphExec) cudaGraphExecDestroy(GraphExec);
    if (Graph) cudaGraphDestroy(Graph);
    if (A) cusparseDestroySpMat(A);
    if (H) cusparseDestroyDnMat(H);
    if (Y) cusparseDestroyDnMat(Y);
    if (Sparse) cusparseDestroy(Sparse);
    if (Blas) cublasDestroy(Blas);
    cudaFree(Workspace);
    cudaFree(DeviceWeights);
  }
};

void computeReference(CudaTensorInputs *input, const float *weights) {
  std::vector<double> intermediate(size_t(input->M) * Features, 0.0);
  for (int row = 0; row < input->M; ++row) {
    for (int k = 0; k < Features; ++k) {
      double x = input->Bx[size_t(row) * Features + k];
      for (int feature = 0; feature < Features; ++feature) {
        intermediate[size_t(row) * Features + feature] +=
            x * weights[k * Features + feature];
      }
    }
  }
  for (int row = 0; row < input->M; ++row) {
    for (int feature = 0; feature < Features; ++feature) {
      double value = 0.0;
      for (int entry = input->ACsr->p[row];
           entry < input->ACsr->p[row + 1]; ++entry) {
        value += double(input->HACsrVal[entry]) *
                 intermediate[size_t(input->ACsr->i[entry]) * Features + feature];
      }
      input->CorrectSol[size_t(row) * Features + feature] = float(value);
    }
  }
  input->IsSolProvided = true;
}

template <cusparseSpMMAlg_t Algorithm, bool Replay>
bool runMethod(CudaTensorInputs *input, const float *weights,
               const TestParameters &parameters,
               const spmm_spmm_benchmark::StructuredCase &testCase,
               const char *method, int runs, int order, bool printHeader) {
  cudaGetLastError();
  auto *stats = new Stats(method, "GEMMSpMM", runs + 1,
                          parameters._matrix_name, 1);
  auto *benchmark = new CuBlasCuSparsePair<Algorithm, Replay>(input, stats, weights);
  benchmark->run();
  bool correct = true;
  for (const auto *info : stats->ProfilingInfoTrials) {
    correct = info->ErrorPerExecute.first && correct;
  }
  auto matrixInfo = parameters.print_csv(true);
  if (printHeader) {
    std::cout << benchmark->printStatsHeader() << std::get<0>(matrixInfo)
              << "Process Order,Warmup Trials,Target Fused Percent,Fused Ratio,"
                 "Device,Matrix Family,Structure Size,Tile Eligible Ratio,"
                 "GPU Tile Rows,Input Features\n";
  }
  std::cout << benchmark->printStats() << std::get<1>(matrixInfo)
            << order << ",1,-1,0,gpu," << testCase.family() << ','
            << testCase.structure << ",-1,0," << Features << '\n';
  std::cout.flush();
  if (!correct) {
    std::cerr << parameters._matrix_name << ": " << method
              << " failed correctness verification\n";
  }
  delete benchmark;
  delete stats;
  return correct;
}

void printUsage(const char *program) {
  std::cerr << "Usage: " << program << " [runs=100] [order=0] [rows=all]\n"
            << "  H = X W; Y = A H; X has 32 columns, W is 32 by 32\n"
            << "  runs: 1..10000 measured runs, plus one discarded warmup\n"
            << "  order: 0..2, rotates which method runs first\n"
            << "  rows: all, 64, 512, 4096, 32768, 262144, or 1048576\n";
}

} // namespace

int main(int argc, char *argv[]) {
  int runs = 100, order = 0, rowFilter = 0;
  if (argc == 2 && std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return 0;
  }
  if (argc > 4 ||
      (argc > 1 && !spmm_spmm_benchmark::parseInteger(argv[1], 1, 10000, runs)) ||
      (argc > 2 && !spmm_spmm_benchmark::parseInteger(argv[2], 0, 2, order)) ||
      (argc > 3 && std::string(argv[3]) != "all" &&
       (!spmm_spmm_benchmark::parseInteger(argv[3], 64, 1048576, rowFilter) ||
        !spmm_spmm_benchmark::validRows(rowFilter)))) {
    printUsage(argv[0]);
    return 1;
  }
  float weights[Features * Features];
  spmm_spmm_benchmark::initializeDense(weights, Features * Features);
  for (float &value : weights) {
    value /= Features;
  }
  std::cout << std::setprecision(10);
  bool correct = true;
  int caseIndex = 0;
  for (const auto &testCase : spmm_spmm_benchmark::structuredCases(rowFilter)) {
    auto *matrix = spmm_spmm_benchmark::makeStructuredMatrix(testCase);
    TestParameters parameters;
    parameters._matrix_name = testCase.name();
    parameters._order_method = SYM_ORDERING::NONE;
    parameters._dim1 = parameters._dim2 = testCase.rows;
    parameters._nnz = matrix->nnz;
    parameters._density = double(matrix->nnz) / testCase.rows / testCase.rows;
    parameters._b_cols = Features;
    auto *input = new CudaTensorInputs(testCase.rows, Features, testCase.rows,
                                       testCase.rows, matrix, matrix, 1,
                                       runs + 1, "GEMMSpMMStructuredBenchmark");
    delete matrix;
    size_t denseElements = size_t(testCase.rows) * Features;
    spmm_spmm_benchmark::initializeDense(input->Bx, denseElements);
    cudaMemcpy(input->DBx, input->Bx, denseElements * sizeof(float),
               cudaMemcpyHostToDevice);
    computeReference(input, weights);
    for (int position = 0; position < 4; ++position) {
      int method = (caseIndex + order + position) % 4;
      bool printHeader = caseIndex == 0 && position == 0;
      bool methodCorrect;
      if (method == 0) {
        methodCorrect = runMethod<CUSPARSE_SPMM_CSR_ALG2, false>(
            input, weights, parameters, testCase, "cublas_cusparse_alg2_direct",
            runs, order, printHeader);
      } else if (method == 1) {
        methodCorrect = runMethod<CUSPARSE_SPMM_CSR_ALG2, true>(
            input, weights, parameters, testCase, "cublas_cusparse_alg2_graph",
            runs, order, printHeader);
      } else if (method == 2) {
        methodCorrect = runMethod<CUSPARSE_SPMM_CSR_ALG3, false>(
            input, weights, parameters, testCase, "cublas_cusparse_alg3_direct",
            runs, order, printHeader);
      } else {
        methodCorrect = runMethod<CUSPARSE_SPMM_CSR_ALG3, true>(
            input, weights, parameters, testCase, "cublas_cusparse_alg3_graph",
            runs, order, printHeader);
      }
      correct = methodCorrect && correct;
    }
    delete input;
    ++caseIndex;
  }
  return correct ? 0 : 1;
}
