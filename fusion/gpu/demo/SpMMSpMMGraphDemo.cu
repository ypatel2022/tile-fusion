#include "sparse-fusion/Cuda_SpMM_SpMM.h"

#include <cuda_runtime.h>

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using swiftware::cuda::CsrView;
using swiftware::cuda::ExecutionMode;
using swiftware::cuda::SpMMSpMMGraph;

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

struct Csr {
  int rows;
  int columns;
  std::vector<int> offsets{0};
  std::vector<int> indices;
  std::vector<float> values;

  Csr(int rows, int columns) : rows(rows), columns(columns) {}

  void append_row(std::initializer_list<int> columns) {
    for (int column : columns) {
      indices.push_back(column);
      const int entry = static_cast<int>(values.size());
      values.push_back((entry % 2 ? -1.0f : 1.0f) *
                       (1 + entry % 5) * 0.125f);
    }
    offsets.push_back(static_cast<int>(indices.size()));
  }

  CsrView view() const {
    return {rows, columns, static_cast<int>(indices.size()), offsets.data(),
            indices.data(), values.data()};
  }
};

// Empty producer rows are deliberate: their intermediate output must be zero.
Csr producer(int rows, int columns) {
  Csr b(rows, columns);
  for (int row = 0; row < rows; ++row) {
    if (row % 11 == 0)
      b.append_row({});
    else
      b.append_row({row % columns, (row + 1) % columns});
  }
  return b;
}

struct DeviceData {
  float *x = nullptr;
  float *y = nullptr;
  cudaStream_t stream = nullptr;

  DeviceData() = default;
  DeviceData(const DeviceData &) = delete;
  DeviceData &operator=(const DeviceData &) = delete;

  ~DeviceData() {
    if (stream) cudaStreamSynchronize(stream);
    cudaFree(y);
    cudaFree(x);
    if (stream) cudaStreamDestroy(stream);
  }
};

std::vector<double> reference(const Csr &a, const Csr &b,
                              const std::vector<float> &x, int width) {
  std::vector<double> intermediate(static_cast<size_t>(b.rows) * width, 0.0);
  std::vector<double> y(static_cast<size_t>(a.rows) * width, 0.0);
  for (int row = 0; row < b.rows; ++row)
    for (int p = b.offsets[row]; p < b.offsets[row + 1]; ++p)
      for (int column = 0; column < width; ++column)
        intermediate[static_cast<size_t>(row) * width + column] +=
            static_cast<double>(b.values[p]) *
            x[static_cast<size_t>(b.indices[p]) * width + column];
  for (int row = 0; row < a.rows; ++row)
    for (int p = a.offsets[row]; p < a.offsets[row + 1]; ++p)
      for (int column = 0; column < width; ++column)
        y[static_cast<size_t>(row) * width + column] +=
            static_cast<double>(a.values[p]) *
            intermediate[static_cast<size_t>(a.indices[p]) * width + column];
  return y;
}

const char *mode_name(ExecutionMode mode) {
  switch (mode) {
  case ExecutionMode::Unfused: return "unfused";
  case ExecutionMode::UnfusedGraph: return "unfused graph";
  case ExecutionMode::Fused: return "fused";
  case ExecutionMode::FusedGraph: return "fused graph";
  }
  return "unknown";
}

void reject_invalid_csr() {
  DeviceData device;
  check(cudaMalloc(&device.x, sizeof(float)));
  check(cudaMalloc(&device.y, sizeof(float)));
  const int bad_offsets[] = {0, 1}; // Terminal offset disagrees with nnz=0.
  const int empty_offsets[] = {0, 0};
  const CsrView bad{1, 1, 0, bad_offsets, nullptr, nullptr};
  const CsrView empty{1, 1, 0, empty_offsets, nullptr, nullptr};
  try {
    SpMMSpMMGraph invalid(bad, empty, 1, device.x, device.y);
  } catch (const std::invalid_argument &) {
    std::cout << "PASS invalid CSR rejected\n";
    return;
  }
  throw std::runtime_error("Malformed CSR was accepted");
}

void run_case(const char *name, const Csr &a, const Csr &b, int width,
              int tile_rows, int expected_deferred) {
  std::vector<float> x(static_cast<size_t>(b.columns) * width);
  std::vector<float> y(static_cast<size_t>(a.rows) * width);
  const std::vector<float> poison(y.size(),
                                  std::numeric_limits<float>::quiet_NaN());
  DeviceData device;
  check(cudaMalloc(&device.x, x.size() * sizeof(float)));
  check(cudaMalloc(&device.y, y.size() * sizeof(float)));
  check(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking));
  SpMMSpMMGraph plan(a.view(), b.view(), width, device.x, device.y, tile_rows);
  const auto info = plan.info();
  if (info.producer_tiles != (b.rows + tile_rows - 1) / tile_rows ||
      info.fused_rows + info.deferred_rows != a.rows ||
      info.deferred_rows != expected_deferred)
    throw std::runtime_error(std::string(name) + ": incorrect row partition");

  const ExecutionMode modes[] = {ExecutionMode::Unfused,
                                 ExecutionMode::UnfusedGraph,
                                 ExecutionMode::Fused,
                                 ExecutionMode::FusedGraph};
  for (ExecutionMode mode : modes) {
    for (int replay = 0; replay < 2; ++replay) {
      for (size_t i = 0; i < x.size(); ++i)
        x[i] = (static_cast<int>((i * 7 + replay * 13) % 31) - 15) * 0.0625f;
      const auto expected = reference(a, b, x, width);
      check(cudaMemcpyAsync(device.x, x.data(), x.size() * sizeof(float),
                            cudaMemcpyHostToDevice, device.stream));
      // Every launch must overwrite Y, including outputs of empty CSR rows.
      check(cudaMemcpyAsync(device.y, poison.data(), y.size() * sizeof(float),
                            cudaMemcpyHostToDevice, device.stream));
      if (mode == ExecutionMode::FusedGraph)
        plan.launch(device.stream); // Also exercise the default mode.
      else
        plan.launch(device.stream, mode);
      check(cudaMemcpyAsync(y.data(), device.y, y.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, device.stream));
      check(cudaStreamSynchronize(device.stream));
      for (size_t i = 0; i < y.size(); ++i) {
        const double tolerance = 1e-5 + 1e-5 * std::abs(expected[i]);
        if (!std::isfinite(y[i]) || std::abs(y[i] - expected[i]) > tolerance)
          throw std::runtime_error(std::string(name) + ", " + mode_name(mode) +
                                   ", replay " + std::to_string(replay) +
                                   ": mismatch at element " + std::to_string(i));
      }
    }
  }
  std::cout << "PASS " << name << " (" << info.producer_tiles << " tiles, "
            << info.fused_rows << " fused rows, " << info.deferred_rows
            << " deferred rows; all four modes, two input versions)\n";
}

} // namespace

int main() {
  try {
    Csr local(64, 64);
    for (int row = 0; row < local.rows; ++row) {
      const int first = (row / 16) * 16;
      local.append_row({row, first + (row + 1) % 16});
    }
    run_case("all-local", local, producer(64, 41), 64, 16, 0);

    Csr mixed(37, 65);
    int deferred = 0;
    for (int row = 0; row < mixed.rows; ++row) {
      switch (row % 5) {
      case 0: mixed.append_row({}); break;
      case 1: mixed.append_row({row % 32, (row + 7) % 32}); break;
      case 2: mixed.append_row({32 + row % 32, 32 + (row + 5) % 32}); break;
      case 3: mixed.append_row({0, 64}); ++deferred; break;
      case 4: mixed.append_row({64}); break;
      }
    }
    run_case("mixed rectangular", mixed, producer(65, 41), 35, 32, deferred);

    Csr remote(17, 65);
    for (int row = 0; row < remote.rows; ++row)
      remote.append_row({row, 64});
    run_case("no-local rectangular", remote, producer(65, 23), 7, 32, 17);

    Csr empty_a(9, 33), empty_b(33, 11);
    for (int row = 0; row < empty_a.rows; ++row) empty_a.append_row({});
    for (int row = 0; row < empty_b.rows; ++row) empty_b.append_row({});
    run_case("zero-nnz", empty_a, empty_b, 1, 32, 0);
    reject_invalid_csr();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
