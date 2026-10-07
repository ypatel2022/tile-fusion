#include "evaluation.h"

#include <cooperative_groups.h>
#include <cuda/atomic>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace spmm_megakernel {
namespace {

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

struct KernelData {
  int rows;
  const int *rowPtr, *columns;
  const float *values, *x;
  float *h, *y;
};

struct TileSchedule {
  int *localPtr = nullptr, *localId = nullptr;
  int *deferredPtr = nullptr, *deferredId = nullptr;
  int *reversePtr = nullptr, *reverseId = nullptr, *dependencies = nullptr;
  unsigned long long *events = nullptr;
  unsigned char *hStoreMask = nullptr;
};

// One lane owns one feature; preserve the original CSR accumulation order.
template <bool SharedInput>
__device__ __forceinline__ float rowSum(
    int row, int feature, const KernelData &data, const float *input,
    const float *sharedH = nullptr, int first = 0, int end = 0) {
  float sum = 0.0f;
  for (int p = data.rowPtr[row]; p < data.rowPtr[row + 1]; ++p) {
    int column = data.columns[p];
    float value = SharedInput && column >= first && column < end
        ? sharedH[(column - first) * Features + feature]
        : input[column * Features + feature];
    sum += data.values[p] * value;
  }
  return sum;
}

__global__ void barrierKernel(KernelData data) {
  int tiles = (data.rows + blockDim.y - 1) / blockDim.y;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int row = tile * blockDim.y + threadIdx.y;
    if (row < data.rows) {
      data.h[row * Features + threadIdx.x] =
          rowSum<false>(row, threadIdx.x, data, data.x);
    }
  }
  cooperative_groups::this_grid().sync();
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    int row = tile * blockDim.y + threadIdx.y;
    if (row < data.rows) {
      data.y[row * Features + threadIdx.x] =
          rowSum<false>(row, threadIdx.x, data, data.h);
    }
  }
}

// Each producer notifies once. The last notification owns the consumer, so
// occupied blocks never wait for an unscheduled producer block.
__device__ bool notify(unsigned long long &word, unsigned int epoch, int count) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> event(word);
  unsigned long long observed = event.load(cuda::memory_order_acquire);
  for (;;) {
    unsigned int remaining = static_cast<unsigned int>(observed >> 32) == epoch
        ? static_cast<unsigned int>(observed) - 1
        : static_cast<unsigned int>(count - 1);
    unsigned long long replacement =
        (static_cast<unsigned long long>(epoch) << 32) | remaining;
    if (event.compare_exchange_strong(observed, replacement,
                                      cuda::memory_order_acq_rel,
                                      cuda::memory_order_acquire)) {
      return remaining == 0;
    }
  }
}

template <bool SharedH>
__global__ void eventKernel(KernelData data, TileSchedule schedule,
                            unsigned int epoch) {
  extern __shared__ float sharedH[];
  __shared__ int readyConsumer;
  int tile = blockIdx.x;
  int first = tile * blockDim.y;
  int end = min(first + static_cast<int>(blockDim.y), data.rows);
  int row = first + threadIdx.y;
  int feature = threadIdx.x;
  if (row < data.rows) {
    float sum = rowSum<false>(row, feature, data, data.x);
    if (!SharedH || !schedule.hStoreMask || schedule.hStoreMask[row]) {
      data.h[row * Features + feature] = sum;
    }
    if (SharedH) sharedH[threadIdx.y * Features + feature] = sum;
  }
  __syncthreads();

  int local = schedule.localPtr[tile] + threadIdx.y;
  if (local < schedule.localPtr[tile + 1]) {
    row = schedule.localId[local];
    data.y[row * Features + feature] =
        rowSum<SharedH>(row, feature, data, data.h, sharedH, first, end);
  }
  // The leader's release publishes H stores from every thread in this block.
  __syncthreads();
  for (int edge = schedule.reversePtr[tile];
       edge < schedule.reversePtr[tile + 1]; ++edge) {
    int consumer = schedule.reverseId[edge];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      readyConsumer = notify(schedule.events[consumer], epoch,
                             schedule.dependencies[consumer]) ? consumer : -1;
    }
    // The final acquire sees every producer; pass that visibility to all lanes.
    __syncthreads();
    if (readyConsumer >= 0) {
      int index = schedule.deferredPtr[readyConsumer] + threadIdx.y;
      if (index < schedule.deferredPtr[readyConsumer + 1]) {
        row = schedule.deferredId[index];
        data.y[row * Features + feature] =
            rowSum<SharedH>(row, feature, data, data.h, sharedH, first, end);
      }
    }
    // Keep the trigger and the producer's shared H alive for the entire callback.
    __syncthreads();
  }
}

template <typename T>
T *upload(const std::vector<T> &values) {
  if (values.empty()) return nullptr;
  T *device = nullptr;
  check(cudaMalloc(&device, values.size() * sizeof(T)));
  try {
    check(cudaMemcpy(device, values.data(), values.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
  } catch (...) {
    cudaFree(device);
    throw;
  }
  return device;
}

class Megakernel : public Method {
  DeviceData &Data;
  bool Barrier, Shared;
  TileSchedule Schedule;
  unsigned int Epoch = 0;
  int Tiles, Grid;
  size_t SharedBytes;

  void cleanup() {
    cudaDeviceSynchronize();
    cudaFree(Schedule.localPtr);
    cudaFree(Schedule.localId);
    cudaFree(Schedule.deferredPtr);
    cudaFree(Schedule.deferredId);
    cudaFree(Schedule.reversePtr);
    cudaFree(Schedule.reverseId);
    cudaFree(Schedule.dependencies);
    cudaFree(Schedule.events);
    cudaFree(Schedule.hStoreMask);
  }

  void inspect(const Matrix &matrix) {
    std::vector<int> localPtr(Tiles + 1), localId;
    std::vector<int> deferredPtr(Tiles + 1), deferredId;
    std::vector<int> counts(Tiles), reversePtr(Tiles + 1), reverseId;
    std::vector<std::vector<int>> reverse(Tiles);
    std::vector<unsigned char> storedH(Shared ? matrix.rows : 0, 0);
    for (int tile = 0; tile < Tiles; ++tile) {
      int first = tile * Data.tileRows;
      int end = std::min(first + Data.tileRows, matrix.rows);
      std::vector<int> producers;
      for (int row = first; row < end; ++row) {
        // Same topology-only predicate as the original GPU inspector. Its
        // class header also defines kernels and cannot enter a second CUDA TU.
        bool local = true;
        for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
          int column = matrix.columns[p];
          if (column < first || column >= end) local = false;
        }
        if (local) {
          localId.push_back(row);
        } else {
          deferredId.push_back(row);
          for (int p = matrix.rowPtr[row]; p < matrix.rowPtr[row + 1]; ++p) {
            producers.push_back(matrix.columns[p] / Data.tileRows);
            // A deferred callback can run in another producer CTA, including
            // for home-tile H. Cover every possible global H read.
            if (Shared) storedH[matrix.columns[p]] = 1;
          }
        }
      }
      localPtr[tile + 1] = localId.size();
      deferredPtr[tile + 1] = deferredId.size();
      std::sort(producers.begin(), producers.end());
      producers.erase(std::unique(producers.begin(), producers.end()),
                      producers.end());
      counts[tile] = producers.size();
      for (int producer : producers) reverse[producer].push_back(tile);
    }
    for (int tile = 0; tile < Tiles; ++tile) {
      reverseId.insert(reverseId.end(), reverse[tile].begin(), reverse[tile].end());
      reversePtr[tile + 1] = reverseId.size();
    }
    Schedule.localPtr = upload(localPtr);
    Schedule.localId = upload(localId);
    Schedule.deferredPtr = upload(deferredPtr);
    Schedule.deferredId = upload(deferredId);
    Schedule.reversePtr = upload(reversePtr);
    Schedule.reverseId = upload(reverseId);
    Schedule.dependencies = upload(counts);
    scheduleBytes = (localPtr.size() + localId.size() + deferredPtr.size() +
                     deferredId.size() + reversePtr.size() + reverseId.size() +
                     counts.size()) * sizeof(int);
    if (Shared) {
      size_t storedRows = std::count(storedH.begin(), storedH.end(), 1);
      hWriteBytes = storedRows * Features * sizeof(float);
      // nullptr means store all H; avoid an all-ones mask when nothing is saved.
      if (storedRows < static_cast<size_t>(matrix.rows)) {
        Schedule.hStoreMask = upload(storedH);
        scheduleBytes += storedH.size() * sizeof(unsigned char);
      }
    }
    if (!reverseId.empty()) {
      eventBytes = static_cast<size_t>(Tiles) * sizeof(unsigned long long);
      check(cudaMalloc(&Schedule.events, eventBytes));
      check(cudaMemset(Schedule.events, 0, eventBytes));
    }
  }

public:
  Megakernel(const std::string &name, const Matrix &matrix, DeviceData &data)
      : Data(data), Barrier(name == "megakernel_barrier"),
        Shared(name == "megakernel_events_shared"),
        Tiles((data.rows + data.tileRows - 1) / data.tileRows), Grid(Tiles),
        SharedBytes(Shared ? data.tileRows * Features * sizeof(float) : 0) {
    auto begin = std::chrono::steady_clock::now();
    try {
      hWriteBytes = static_cast<size_t>(Data.rows) * Features * sizeof(float);
      if (!Barrier) inspect(matrix);
      if (Barrier) {
        int device, multiprocessors, active, supported;
        check(cudaGetDevice(&device));
        check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                                     device));
        check(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, barrierKernel, Features * Data.tileRows, 0));
        if (!supported || active == 0) {
          throw std::runtime_error("cooperative megakernel launch is unsupported");
        }
        Grid = std::min(Tiles, active * multiprocessors);
      }
      check(cudaDeviceSynchronize());
      inspectionSeconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - begin).count();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  void launch(int slot) override {
    const auto &buffers = Data.slots[slot];
    KernelData parameters{Data.rows, Data.rowPtr, Data.columns, Data.values,
                           buffers.x, buffers.h, buffers.y};
    dim3 block(Features, Data.tileRows, 1);
    if (Barrier) {
      void *arguments[] = {&parameters};
      check(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(barrierKernel),
                                        dim3(Grid), block, arguments, 0, 0));
    } else {
      // All calls use the ordered default stream. Never reuse a generation.
      if (Epoch == std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("event epoch exhausted; create a new plan");
      }
      ++Epoch;
      if (Shared) {
        eventKernel<true><<<Grid, block, SharedBytes>>>(parameters, Schedule, Epoch);
      } else {
        eventKernel<false><<<Grid, block>>>(parameters, Schedule, Epoch);
      }
    }
    check(cudaGetLastError());
  }

  size_t workspaceBytes() const override { return scheduleBytes + eventBytes; }
  bool writesH() const override {
    return hWriteBytes == static_cast<size_t>(Data.rows) * Features * sizeof(float);
  }
  ~Megakernel() { cleanup(); }
};

} // namespace

std::unique_ptr<Method> makeMegakernelMethod(const std::string &name,
                                            const Matrix &matrix,
                                            DeviceData &data) {
  if (name != "megakernel_barrier" && name != "megakernel_events_global" &&
      name != "megakernel_events_shared") {
    throw std::runtime_error("unknown megakernel method: " + name);
  }
  return std::unique_ptr<Method>(new Megakernel(name, matrix, data));
}

} // namespace spmm_megakernel
