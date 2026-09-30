#ifndef SPARSE_FUSION_CUDA_SPMM_SPMM_GRAPH_H
#define SPARSE_FUSION_CUDA_SPMM_SPMM_GRAPH_H

#include "Cuda_SpMM_SpMM_Demo_Utils.h"

// Replays the existing fused/deferred kernel pair with fixed buffers and schedule.
class FusedSpMMSpMMSeqReduceRowBalanceGraph
    : public FusedSpMMSpMMSeqReduceRowBalance {
  cudaGraph_t Graph = nullptr;
  cudaGraphExec_t GraphExec = nullptr;

protected:
  Timer analysis() override {
    Timer t;
    t.start();
    FusedSpMMSpMMSeqReduceRowBalance::analysis();

    cudaGraphCreate(&Graph, 0);
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
    cudaGraphAddKernelNode(&fusedNode, Graph, nullptr, 0, &fusedParams);

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
    cudaGraphAddKernelNode(&deferredNode, Graph, &fusedNode, 1, &deferredParams);
    cudaGraphInstantiate(&GraphExec, Graph, nullptr, nullptr, 0);
    t.stop("FusedSpMMSpMMGraphAnalysis");
    return t;
  }

  Timer execute() override {
    Timer t;
    t.startGPU();
    cudaGraphLaunch(GraphExec, 0);
    cudaStreamSynchronize(0);
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
