#ifndef SPARSE_FUSION_CUDA_SPMM_SPMM_GRAPH_H
#define SPARSE_FUSION_CUDA_SPMM_SPMM_GRAPH_H

#include "Cuda_SpMM_SpMM_Demo_Utils.h"

#include <stdexcept>
#include <string>

// Replays the existing fused/deferred kernel pair with fixed buffers and schedule.
class FusedSpMMSpMMSeqReduceRowBalanceGraph
    : public FusedSpMMSpMMSeqReduceRowBalance {
  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;

  static void checkCuda(cudaError_t Status, const char *Operation) {
    if (Status != cudaSuccess) {
      throw std::runtime_error(std::string(Operation) + ": " +
                               cudaGetErrorString(Status));
    }
  }

protected:
  Timer analysis() override {
    Timer t;
    t.start();
    FusedSpMMSpMMSeqReduceRowBalance::analysis();

    checkCuda(cudaGraphCreate(&Graph, 0), "cudaGraphCreate");
    dim3 blockDim(NBlockDim, MBlockDim, 1);

    void *fusedArgs[] = {
        &InTensor->M,       &InTensor->N,        &InTensor->K,
        &InTensor->DACsrAp, &InTensor->DACsrI,   &InTensor->DACsrVal,
        &InTensor->DBx,     &OutTensor->DACx,    &OutTensor->DXx,
        &DFPtr,            &DFId};
    cudaKernelNodeParams fusedParams{};
    fusedParams.func = reinterpret_cast<void *>(
        csr_fusedTile_spmmspmm_seqreduce_rowbalance_kernel);
    fusedParams.gridDim = dim3(MGridDim, NGridDim, 1);
    fusedParams.blockDim = blockDim;
    fusedParams.kernelParams = fusedArgs;
    cudaGraphNode_t fusedNode;
    checkCuda(cudaGraphAddKernelNode(&fusedNode, Graph, nullptr, 0, &fusedParams),
              "cudaGraphAddKernelNode (fused)");

    void *deferredArgs[] = {
        &UFDim,            &InTensor->N,        &InTensor->K,
        &InTensor->DACsrAp, &InTensor->DACsrI,   &InTensor->DACsrVal,
        &OutTensor->DACx,   &OutTensor->DXx,     &DUFPtr};
    cudaKernelNodeParams deferredParams{};
    deferredParams.func = reinterpret_cast<void *>(
        csr_unfusedTile_spmmspmm_seqreduce_rowbalance_kernel);
    // Keep a valid second node when every row is fused; UFDim == 0 makes it a no-op.
    deferredParams.gridDim = dim3(UFMGridDim > 0 ? UFMGridDim : 1, NGridDim, 1);
    deferredParams.blockDim = blockDim;
    deferredParams.kernelParams = deferredArgs;
    cudaGraphNode_t deferredNode;
    // All intermediate rows must be ready before the deferred consumers run.
    checkCuda(cudaGraphAddKernelNode(&deferredNode, Graph, &fusedNode, 1,
                                    &deferredParams),
              "cudaGraphAddKernelNode (deferred)");
    checkCuda(cudaGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0),
              "cudaGraphInstantiate");
    t.stop("FusedSpMMSpMMGraphAnalysis");
    return t;
  }

  Timer execute() override {
    Timer t;
    t.startGPU();
    checkCuda(cudaGraphLaunch(GraphExec, 0), "cudaGraphLaunch");
    checkCuda(cudaStreamSynchronize(0), "cudaStreamSynchronize");
    t.stopGPU("FusedSpMMSpMMGraph");
    OutTensor->copyDeviceToHost();
    return t;
  }

public:
  FusedSpMMSpMMSeqReduceRowBalanceGraph(CudaTensorInputs *In1, Stats *Stat1,
                                       int ThreadPerBlock = 256)
      : FusedSpMMSpMMSeqReduceRowBalance(In1, Stat1, ThreadPerBlock) {
    HFPtr = HFId = HUFPtr = nullptr;
    DFPtr = DFId = DUFPtr = nullptr;
  }

  FusedSpMMSpMMSeqReduceRowBalanceGraph(
      const FusedSpMMSpMMSeqReduceRowBalanceGraph &) = delete;
  FusedSpMMSpMMSeqReduceRowBalanceGraph &operator=(
      const FusedSpMMSpMMSeqReduceRowBalanceGraph &) = delete;

  ~FusedSpMMSpMMSeqReduceRowBalanceGraph() {
    if (GraphExec) {
      cudaGraphExecDestroy(GraphExec);
    }
    if (Graph) {
      cudaGraphDestroy(Graph);
    }
  }
};

#endif // SPARSE_FUSION_CUDA_SPMM_SPMM_GRAPH_H
