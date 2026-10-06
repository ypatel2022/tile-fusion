#ifndef SPARSE_FUSION_SPMM_SPMM_STRUCTURED_GPU_H
#define SPARSE_FUSION_SPMM_SPMM_STRUCTURED_GPU_H

#include "../Cuda_SpMM_SpMM_Demo_Utils.h"

#include <limits>

// Benchmark adapters keep the original kernels and buffers, while submitting
// both dependent operations before waiting for the pair on the default stream.
template <bool Replay>
class UnfusedPair : public SpMMSpMMSeqReduceRowBalance {
  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;
  bool Valid = true;

protected:
  Timer analysis() override {
    Timer t;
    t.start();
    if (Replay) {
      Valid = (cudaGraphCreate(&Graph, 0) == cudaSuccess) && Valid;
    }
    if (Replay && Valid) {
      void *producerArgs[] = {
          &InTensor->M, &InTensor->N, &InTensor->K, &InTensor->DACsrAp,
          &InTensor->DACsrI, &InTensor->DACsrVal, &InTensor->DBx,
          &OutTensor->DACx};
      cudaKernelNodeParams params{};
      params.func = reinterpret_cast<void *>(csrspmm_seqreduce_rowbalance_kernel);
      params.gridDim = dim3(MGridDim, NGridDim, 1);
      params.blockDim = dim3(NBlockDim, MBlockDim, 1);
      params.kernelParams = producerArgs;
      cudaGraphNode_t producer = nullptr;
      Valid = (cudaGraphAddKernelNode(&producer, Graph, nullptr, 0, &params) ==
               cudaSuccess) && Valid;

      void *consumerArgs[] = {
          &InTensor->M, &InTensor->N, &InTensor->K, &InTensor->DACsrAp,
          &InTensor->DACsrI, &InTensor->DACsrVal, &OutTensor->DACx,
          &OutTensor->DXx};
      params.kernelParams = consumerArgs;
      cudaGraphNode_t consumer = nullptr;
      Valid = (cudaGraphAddKernelNode(&consumer, Graph, &producer, 1, &params) ==
               cudaSuccess) && Valid;
      if (Valid) {
        Valid = (cudaGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0) ==
                 cudaSuccess);
      }
    }
    t.stop("Analysis");
    return t;
  }

  Timer execute() override {
    Timer t;
    t.startGPU();
    if (Replay && Valid) {
      Valid = (cudaGraphLaunch(GraphExec, 0) == cudaSuccess) && Valid;
    } else if (!Replay && Valid) {
      dim3 grid(MGridDim, NGridDim, 1);
      dim3 block(NBlockDim, MBlockDim, 1);
      csrspmm_seqreduce_rowbalance_kernel<<<grid, block>>>(
          InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
          InTensor->DACsrI, InTensor->DACsrVal, InTensor->DBx, OutTensor->DACx);
      csrspmm_seqreduce_rowbalance_kernel<<<grid, block>>>(
          InTensor->M, InTensor->N, InTensor->K, InTensor->DACsrAp,
          InTensor->DACsrI, InTensor->DACsrVal, OutTensor->DACx, OutTensor->DXx);
    }
    Valid = (cudaStreamSynchronize(0) == cudaSuccess) && Valid;
    t.stopGPU("UnfusedSpMMSpMM");
    Valid = (cudaGetLastError() == cudaSuccess) && Valid;
    OutTensor->copyDeviceToHost();
    return t;
  }

  bool verify(double &error) override {
    if (!Valid) {
      error = std::numeric_limits<double>::infinity();
      return false;
    }
    return SpMMSpMMSeqReduceRowBalance::verify(error);
  }

public:
  using SpMMSpMMSeqReduceRowBalance::SpMMSpMMSeqReduceRowBalance;

  ~UnfusedPair() {
    if (GraphExec) {
      cudaGraphExecDestroy(GraphExec);
    }
    if (Graph) {
      cudaGraphDestroy(Graph);
    }
  }
};

template <cusparseSpMMAlg_t Algorithm, bool Replay>
class CuSparsePair : public SpMMSpMMCuSparse {
  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;
  cusparseSpMatDescr_t SecondA = nullptr;
  bool Valid = true;

  void launchPair() {
    if (!Valid) {
      return;
    }
    Valid = (cusparseSpMM(cusparse_handle, transA, transB, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, Alg, WorkspaceMul1) ==
             CUSPARSE_STATUS_SUCCESS) && Valid;
    Valid = (cusparseSpMM(cusparse_handle, transA, transB, &alpha, SecondA, matC,
                         &beta, matD, CUDA_R_32F, Alg, WorkspaceMul2) ==
             CUSPARSE_STATUS_SUCCESS) && Valid;
  }

protected:
  Timer analysis() override {
    Timer t;
    t.start();
    // Both variants use identical preparation. Separate sparse descriptors keep
    // each operation's preprocessing buffer active throughout graph capture.
    SpMMSpMMCuSparse::analysis();
    auto descriptorStatus = cusparseCreateCsr(
        &SecondA, InTensor->M, InTensor->K, InTensor->ACsr->nnz,
        InTensor->DACsrAp, InTensor->DACsrI, InTensor->DACsrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F);
    Valid = (descriptorStatus == CUSPARSE_STATUS_SUCCESS) && Valid;
    cudaStream_t preparationStream = nullptr;
    Valid = (cudaStreamCreate(&preparationStream) == cudaSuccess) && Valid;
    Valid = (cusparseSetStream(cusparse_handle, preparationStream) ==
             CUSPARSE_STATUS_SUCCESS) && Valid;
    if (Valid) {
      auto firstStatus = cusparseSpMM_preprocess(
          cusparse_handle, transA, transB, &alpha, matA, matB, &beta, matC,
          CUDA_R_32F, Alg, WorkspaceMul1);
      auto secondStatus = cusparseSpMM_preprocess(
          cusparse_handle, transA, transB, &alpha, SecondA, matC, &beta, matD,
          CUDA_R_32F, Alg, WorkspaceMul2);
      Valid = firstStatus == CUSPARSE_STATUS_SUCCESS &&
              secondStatus == CUSPARSE_STATUS_SUCCESS;
      if (!Valid) {
        std::cerr << "cuSPARSE preprocessing failed: " << firstStatus << ", "
                  << secondStatus << '\n';
      }
    }
    // Prime both direct and graph variants identically before measured runs.
    launchPair();
    Valid = (cudaStreamSynchronize(preparationStream) == cudaSuccess) && Valid;
    if (Replay && Valid) {
      auto captureStatus = cudaStreamBeginCapture(preparationStream,
                                                  cudaStreamCaptureModeGlobal);
      Valid = captureStatus == cudaSuccess;
      launchPair();
      auto endStatus = cudaStreamEndCapture(preparationStream, &Graph);
      Valid = (endStatus == cudaSuccess) && Valid;
      if (!Valid) {
        std::cerr << "cuSPARSE graph capture failed: "
                  << cudaGetErrorString(captureStatus) << ", "
                  << cudaGetErrorString(endStatus) << '\n';
      }
      if (Valid) {
        Valid = (cudaGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0) ==
                 cudaSuccess);
      }
    }
    Valid = (cusparseSetStream(cusparse_handle, 0) == CUSPARSE_STATUS_SUCCESS) &&
            Valid;
    Valid = (cudaStreamDestroy(preparationStream) == cudaSuccess) && Valid;
    t.stop("Analysis");
    return t;
  }

  Timer execute() override {
    Timer t;
    t.startGPU();
    if (Replay && Valid) {
      Valid = (cudaGraphLaunch(GraphExec, 0) == cudaSuccess) && Valid;
    } else if (!Replay) {
      launchPair();
    }
    Valid = (cudaStreamSynchronize(0) == cudaSuccess) && Valid;
    t.stopGPU("CuSparseSpMMSpMM");
    OutTensor->copyDeviceToHost();
    return t;
  }

  bool verify(double &error) override {
    // Priming has already produced a correct output: a failed replay must not
    // pass verification merely because that output was left untouched.
    if (!Valid) {
      error = std::numeric_limits<double>::infinity();
      return false;
    }
    return SpMMSpMMCuSparse::verify(error);
  }

public:
  CuSparsePair(CudaTensorInputs *input, Stats *stats, int)
      : SpMMSpMMCuSparse(input, stats, Algorithm) {}

  ~CuSparsePair() {
    if (GraphExec) {
      cudaGraphExecDestroy(GraphExec);
    }
    if (Graph) {
      cudaGraphDestroy(Graph);
    }
    if (SecondA) {
      cusparseDestroySpMat(SecondA);
    }
  }
};

#endif // SPARSE_FUSION_SPMM_SPMM_STRUCTURED_GPU_H
