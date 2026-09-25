#ifndef SPARSE_FUSION_CUDA_SPMM_SPMM_H
#define SPARSE_FUSION_CUDA_SPMM_SPMM_H

#include <cuda_runtime_api.h>
#include <memory>

namespace swiftware {
namespace cuda {

// Borrowed host CSR storage. Indices are zero-based; values are float32.
struct CsrView {
  int rows;
  int columns;
  int nonzeros;
  const int *row_offsets;
  const int *column_indices;
  const float *values;
};

enum class ExecutionMode { Unfused, UnfusedGraph, Fused, FusedGraph };

struct FusionInfo {
  int producer_tiles;
  int fused_rows;
  int deferred_rows;
};

// Reusable execution of Y = A * (B * X), for dense row-major X and Y.
// Construction inspects and copies A/B (including values) and instantiates both
// CUDA Graphs. Changing either sparse matrix requires a new object. tile_rows
// is in [1, 128]; matrix dimensions and dense_columns must be positive.
// X/Y are borrowed device buffers with fixed addresses; X values may change
// between launches. They must not overlap and must outlive all submitted work.
//
// Launch is asynchronous. Order all uses of this object's workspace through
// one stream or explicit events; concurrent uses are not supported. Synchronize
// before destroying this object or its X/Y buffers. Create, launch, and destroy
// on the same CUDA device. Invalid arguments throw std::invalid_argument; CUDA
// API failures throw std::runtime_error. Check asynchronous errors when the
// caller synchronizes the stream.
class SpMMSpMMGraph {
public:
  SpMMSpMMGraph(CsrView a, CsrView b, int dense_columns,
                const float *x, float *y, int tile_rows = 32);
  ~SpMMSpMMGraph();

  SpMMSpMMGraph(const SpMMSpMMGraph &) = delete;
  SpMMSpMMGraph &operator=(const SpMMSpMMGraph &) = delete;

  void launch(cudaStream_t stream,
              ExecutionMode mode = ExecutionMode::FusedGraph) const;
  FusionInfo info() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cuda
} // namespace swiftware

#endif
