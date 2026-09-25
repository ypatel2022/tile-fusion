#include "sparse-fusion/Cuda_SpMM_SpMM.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace swiftware {
namespace cuda {
namespace {

constexpr int feature_tile = 32;
constexpr int warps_per_block = 4;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T> class DeviceBuffer {
public:
  T *data = nullptr;
  DeviceBuffer() = default;
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  void allocate(size_t count) {
    if (count) check(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)), "cudaMalloc");
  }

  void copy(const T *source, size_t count) {
    allocate(count);
    if (count) check(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
  }
};

struct CsrData {
  const int *offsets;
  const int *indices;
  const float *values;
};

struct DeviceCsr {
  DeviceBuffer<int> offsets, indices;
  DeviceBuffer<float> values;

  void copy(CsrView matrix) {
    offsets.copy(matrix.row_offsets, size_t(matrix.rows) + 1);
    indices.copy(matrix.column_indices, matrix.nonzeros);
    values.copy(matrix.values, matrix.nonzeros);
  }

  CsrData view() const { return {offsets.data, indices.data, values.data}; }
};

int ceil_div(int size, int tile) { return size / tile + (size % tile != 0); }

void validate(CsrView matrix) {
  if (matrix.rows <= 0 || matrix.columns <= 0 || matrix.nonzeros < 0 ||
      !matrix.row_offsets || (matrix.nonzeros && (!matrix.column_indices || !matrix.values)))
    throw std::invalid_argument("CSR dimensions and storage are invalid");
  if (matrix.row_offsets[0] != 0 || matrix.row_offsets[matrix.rows] != matrix.nonzeros)
    throw std::invalid_argument("CSR row offsets must span [0, nonzeros]");
  for (int row = 0; row < matrix.rows; ++row) {
    if (matrix.row_offsets[row] < 0 || matrix.row_offsets[row] > matrix.row_offsets[row + 1])
      throw std::invalid_argument("CSR row offsets must be nondecreasing");
  }
  for (int j = 0; j < matrix.nonzeros; ++j) {
    if (matrix.column_indices[j] < 0 || matrix.column_indices[j] >= matrix.columns)
      throw std::invalid_argument("CSR column index is out of bounds");
  }
}

// One warp per sparse row; adjacent lanes access adjacent dense columns.
// A row list selects deferred consumers; nullptr selects consecutive rows.
__global__ void spmm(CsrData matrix, const float *input, float *output,
                     int columns, int row_count, const int *rows) {
  int index = int(blockIdx.x) * warps_per_block + int(threadIdx.y);
  int column = int(blockIdx.y) * feature_tile + int(threadIdx.x);
  if (index >= row_count || column >= columns) return;
  int row = rows ? rows[index] : index;
  float sum = 0.f;
  for (int j = matrix.offsets[row]; j < matrix.offsets[row + 1]; ++j)
    sum = fmaf(matrix.values[j], input[size_t(matrix.indices[j]) * columns + column], sum);
  output[size_t(row) * columns + column] = sum;
}

__global__ void fused_spmm(CsrData a, CsrData b, const float *input,
                           float *intermediate, float *output, int columns,
                           int producer_rows, int tile_rows,
                           const int *tile_offsets, const int *local_rows) {
  extern __shared__ float tile[];
  int first_row = int(blockIdx.x) * tile_rows;
  int column = int(blockIdx.y) * feature_tile + int(threadIdx.x);
  int lane = int(threadIdx.x);

  for (int local = int(threadIdx.y); local < tile_rows; local += warps_per_block) {
    float sum = 0.f;
    if (local < producer_rows - first_row && column < columns) {
      int row = first_row + local;
      for (int j = b.offsets[row]; j < b.offsets[row + 1]; ++j)
        sum = fmaf(b.values[j], input[size_t(b.indices[j]) * columns + column], sum);
      // Deferred consumers need this value after this kernel completes.
      intermediate[size_t(row) * columns + column] = sum;
    }
    tile[local * feature_tile + lane] = sum;
  }

  // All threads participate, including tail rows and tail feature lanes.
  __syncthreads();

  for (size_t index = size_t(tile_offsets[blockIdx.x]) + threadIdx.y;
       index < size_t(tile_offsets[blockIdx.x + 1]); index += warps_per_block) {
    int row = local_rows[index];
    if (column < columns) {
      float sum = 0.f;
      for (int j = a.offsets[row]; j < a.offsets[row + 1]; ++j)
        sum = fmaf(a.values[j], tile[(a.indices[j] - first_row) * feature_tile + lane], sum);
      output[size_t(row) * columns + column] = sum;
    }
  }
}

cudaGraphNode_t add_kernel(cudaGraph_t graph, void *function, dim3 grid,
                           size_t shared_bytes, void **arguments,
                           cudaGraphNode_t dependency = nullptr) {
  cudaKernelNodeParams parameters{};
  parameters.func = function;
  parameters.gridDim = grid;
  parameters.blockDim = dim3(feature_tile, warps_per_block);
  parameters.sharedMemBytes = shared_bytes;
  parameters.kernelParams = arguments;
  cudaGraphNode_t node;
  check(cudaGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                               dependency ? 1 : 0, &parameters), "cudaGraphAddKernelNode");
  return node;
}

} // namespace

struct SpMMSpMMGraph::Impl {
  DeviceCsr a, b;
  DeviceBuffer<float> intermediate;
  DeviceBuffer<int> tile_offsets, local_rows, deferred_rows;
  const float *input = nullptr;
  float *output = nullptr;
  int columns = 0, producer_rows = 0, output_rows = 0, tile_rows = 0;
  FusionInfo stats{};
  cudaGraph_t fused_graph = nullptr, unfused_graph = nullptr;
  cudaGraphExec_t fused_exec = nullptr, unfused_exec = nullptr;

  ~Impl() {
    if (fused_exec) cudaGraphExecDestroy(fused_exec);
    if (unfused_exec) cudaGraphExecDestroy(unfused_exec);
    if (fused_graph) cudaGraphDestroy(fused_graph);
    if (unfused_graph) cudaGraphDestroy(unfused_graph);
  }

  dim3 row_grid(int rows) const {
    return dim3(ceil_div(rows, warps_per_block), ceil_div(columns, feature_tile));
  }
  dim3 tile_grid() const { return dim3(stats.producer_tiles, ceil_div(columns, feature_tile)); }
  size_t shared_bytes() const { return size_t(tile_rows) * feature_tile * sizeof(float); }

  void initialize(CsrView host_a, CsrView host_b, int width,
                  const float *x, float *y, int rows_per_tile) {
    validate(host_a);
    validate(host_b);
    if (host_a.columns != host_b.rows || width <= 0 || !x || !y ||
        rows_per_tile <= 0 || rows_per_tile > 128)
      throw std::invalid_argument("Incompatible matrices, buffers, or tile size (1..128)");
    size_t input_bytes = size_t(host_b.columns) * width * sizeof(float);
    size_t output_bytes = size_t(host_a.rows) * width * sizeof(float);
    auto x_address = reinterpret_cast<std::uintptr_t>(x);
    auto y_address = reinterpret_cast<std::uintptr_t>(y);
    if ((x_address <= y_address && y_address - x_address < input_bytes) ||
        (y_address < x_address && x_address - y_address < output_bytes))
      throw std::invalid_argument("Dense input and output must not overlap");

    columns = width;
    producer_rows = host_b.rows;
    output_rows = host_a.rows;
    tile_rows = rows_per_tile;
    input = x;
    output = y;
    stats.producer_tiles = ceil_div(producer_rows, tile_rows);

    int device;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (ceil_div(columns, feature_tile) > properties.maxGridSize[1] ||
        std::max({stats.producer_tiles, ceil_div(producer_rows, warps_per_block),
                  ceil_div(output_rows, warps_per_block)}) > properties.maxGridSize[0] ||
        shared_bytes() > properties.sharedMemPerBlock)
      throw std::invalid_argument("Workload exceeds device launch limits");

    // A consumer is local only if every predecessor belongs to the same tile.
    // Unsorted indices and duplicate entries are both valid CSR input.
    std::vector<int> owner(output_rows, -1), offsets(size_t(stats.producer_tiles) + 1, 0);
    std::vector<int> deferred;
    for (int row = 0; row < output_rows; ++row) {
      int begin = host_a.row_offsets[row], end = host_a.row_offsets[row + 1];
      int tile = begin == end ? 0 : host_a.column_indices[begin] / tile_rows;
      for (int j = begin; j < end; ++j) {
        if (host_a.column_indices[j] / tile_rows != tile) { tile = -1; break; }
      }
      if (tile >= 0) {
        owner[row] = tile;
        ++offsets[tile + 1];
      } else {
        deferred.push_back(row);
      }
    }
    for (int tile = 0; tile < stats.producer_tiles; ++tile)
      offsets[tile + 1] += offsets[tile];
    std::vector<int> local(offsets.back()), cursor(offsets);
    for (int row = 0; row < output_rows; ++row)
      if (owner[row] >= 0) local[cursor[owner[row]]++] = row;
    stats.fused_rows = int(local.size());
    stats.deferred_rows = int(deferred.size());

    a.copy(host_a);
    b.copy(host_b);
    intermediate.allocate(size_t(producer_rows) * columns);
    tile_offsets.copy(offsets.data(), offsets.size());
    local_rows.copy(local.data(), local.size());
    deferred_rows.copy(deferred.data(), deferred.size());
    build_graphs();
  }

  void build_graphs() {
    auto av = a.view(), bv = b.view();
    const int *all_rows = nullptr;
    void *producer_args[] = {&bv, &input, &intermediate.data, &columns, &producer_rows, &all_rows};
    void *consumer_args[] = {&av, &intermediate.data, &output, &columns, &output_rows, &all_rows};
    void *fused_args[] = {&av, &bv, &input, &intermediate.data, &output, &columns,
                         &producer_rows, &tile_rows, &tile_offsets.data, &local_rows.data};
    void *deferred_args[] = {&av, &intermediate.data, &output, &columns,
                            &stats.deferred_rows, &deferred_rows.data};

    check(cudaGraphCreate(&unfused_graph, 0), "cudaGraphCreate");
    auto producer = add_kernel(unfused_graph, reinterpret_cast<void *>(spmm),
                               row_grid(producer_rows), 0, producer_args);
    add_kernel(unfused_graph, reinterpret_cast<void *>(spmm), row_grid(output_rows),
               0, consumer_args, producer);
    check(cudaGraphInstantiate(&unfused_exec, unfused_graph, nullptr, nullptr, 0), "cudaGraphInstantiate");

    check(cudaGraphCreate(&fused_graph, 0), "cudaGraphCreate");
    auto fused = add_kernel(fused_graph, reinterpret_cast<void *>(fused_spmm),
                            tile_grid(), shared_bytes(), fused_args);
    if (stats.deferred_rows)
      add_kernel(fused_graph, reinterpret_cast<void *>(spmm), row_grid(stats.deferred_rows),
                 0, deferred_args, fused);
    check(cudaGraphInstantiate(&fused_exec, fused_graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
  }

  void launch(cudaStream_t stream, ExecutionMode mode) const {
    dim3 block(feature_tile, warps_per_block);
    switch (mode) {
    case ExecutionMode::Unfused:
      spmm<<<row_grid(producer_rows), block, 0, stream>>>(
          b.view(), input, intermediate.data, columns, producer_rows, nullptr);
      check(cudaGetLastError(), "producer SpMM launch");
      spmm<<<row_grid(output_rows), block, 0, stream>>>(
          a.view(), intermediate.data, output, columns, output_rows, nullptr);
      check(cudaGetLastError(), "consumer SpMM launch");
      break;
    case ExecutionMode::UnfusedGraph:
      check(cudaGraphLaunch(unfused_exec, stream), "cudaGraphLaunch");
      break;
    case ExecutionMode::Fused:
      fused_spmm<<<tile_grid(), block, shared_bytes(), stream>>>(
          a.view(), b.view(), input, intermediate.data, output, columns, producer_rows,
          tile_rows, tile_offsets.data, local_rows.data);
      check(cudaGetLastError(), "fused SpMM launch");
      if (stats.deferred_rows) {
        spmm<<<row_grid(stats.deferred_rows), block, 0, stream>>>(
            a.view(), intermediate.data, output, columns, stats.deferred_rows, deferred_rows.data);
        check(cudaGetLastError(), "deferred SpMM launch");
      }
      break;
    case ExecutionMode::FusedGraph:
      check(cudaGraphLaunch(fused_exec, stream), "cudaGraphLaunch");
      break;
    default:
      throw std::invalid_argument("Unknown execution mode");
    }
  }
};

SpMMSpMMGraph::SpMMSpMMGraph(CsrView a, CsrView b, int dense_columns,
                            const float *x, float *y, int tile_rows)
    : impl_(new Impl) {
  impl_->initialize(a, b, dense_columns, x, y, tile_rows);
}

SpMMSpMMGraph::~SpMMSpMMGraph() = default;

void SpMMSpMMGraph::launch(cudaStream_t stream, ExecutionMode mode) const {
  impl_->launch(stream, mode);
}

FusionInfo SpMMSpMMGraph::info() const noexcept { return impl_->stats; }

} // namespace cuda
} // namespace swiftware
