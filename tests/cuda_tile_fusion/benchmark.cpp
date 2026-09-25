#include "SWBench.h"
#include "sparse-fusion/Cuda_SpMM_SpMM.h"
#include "sparse-fusion/SpMM_SpMM.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using swiftware::benchmark::Stats;
using swiftware::benchmark::SWBench;
using swiftware::benchmark::Timer;
using swiftware::cuda::CsrView;
using swiftware::cuda::ExecutionMode;
using swiftware::cuda::SpMMSpMMGraph;
using Clock = std::chrono::steady_clock;
constexpr int TileRows = 32;
constexpr int Degree = 8;
constexpr ExecutionMode Modes[] = {ExecutionMode::Unfused,
                                    ExecutionMode::UnfusedGraph,
                                    ExecutionMode::Fused,
                                    ExecutionMode::FusedGraph};
constexpr const char *ModeNames[] = {"unfused", "unfused_graph", "fused",
                                    "fused_graph"};

void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

int argument(const char *text, int minimum, int maximum) {
  size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (text[consumed] != '\0' || value < minimum || value > maximum)
    throw std::invalid_argument("Argument outside supported range: " +
                                std::string(text));
  return value;
}

struct Csr {
  int rows;
  std::vector<int> offsets;
  std::vector<int> indices;
  std::vector<float> values;

  Csr(int rows, int locality, bool producer) : rows(rows), offsets(rows + 1) {
    indices.reserve(static_cast<size_t>(rows) * Degree);
    values.reserve(indices.capacity());
    for (int row = 0; row < rows; ++row) {
      const bool local = locality == 100 || (locality == 50 && row % 2 == 0);
      const int tile = row / TileRows;
      for (int entry = 0; entry < Degree; ++entry) {
        int column = (row + entry) % rows;
        if (!producer) {
          // Remote rows read two producer tiles; all eight indices are unique.
          const int source_tile = !local && entry >= Degree / 2
                                      ? (tile + 1) % (rows / TileRows) : tile;
          column = source_tile * TileRows + (row + entry * 3) % TileRows;
        }
        indices.push_back(column);
        const float sign = (row + entry + (producer ? 1 : 0)) % 2 ? -1.f : 1.f;
        // Each row has an absolute value sum of one, with mixed signs.
        values.push_back(sign * (entry + 1) / 36.f);
      }
      offsets[row + 1] = static_cast<int>(indices.size());
    }
  }

  CsrView view() const {
    return {rows, rows, static_cast<int>(indices.size()), offsets.data(),
            indices.data(), values.data()};
  }
};

std::vector<double> reference(const Csr &a, const Csr &b,
                              const std::vector<float> &x, int width) {
  const std::vector<double> av(a.values.begin(), a.values.end());
  const std::vector<double> bv(b.values.begin(), b.values.end());
  const std::vector<double> input(x.begin(), x.end());
  std::vector<double> intermediate(x.size(), 0.0), y(x.size(), 0.0);
  swiftware::sparse::spmmCsrSequential(b.rows, width, b.rows, b.offsets.data(),
      b.indices.data(), bv.data(), input.data(), intermediate.data());
  swiftware::sparse::spmmCsrSequential(a.rows, width, a.rows, a.offsets.data(),
      a.indices.data(), av.data(), intermediate.data(), y.data());
  return y;
}

struct Workload {
  float *x = nullptr;
  float *y = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  std::unique_ptr<SpMMSpMMGraph> plan;
  std::vector<float> output;
  double setup_seconds = 0;

  explicit Workload(size_t elements) : output(elements) {}
  Workload(const Workload &) = delete;
  Workload &operator=(const Workload &) = delete;

  ~Workload() {
    if (stream) cudaStreamSynchronize(stream);
    plan.reset();
    if (stop) cudaEventDestroy(stop);
    if (start) cudaEventDestroy(start);
    cudaFree(y);
    cudaFree(x);
    if (stream) cudaStreamDestroy(stream);
  }

  void initialize(const Csr &a, const Csr &b,
                  const std::vector<float> &input, int width) {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaMalloc(&x, input.size() * sizeof(float)));
    check(cudaMalloc(&y, output.size() * sizeof(float)));
    check(cudaMemcpyAsync(x, input.data(), input.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    check(cudaStreamSynchronize(stream));
    const auto begin = Clock::now();
    plan.reset(new SpMMSpMMGraph(a.view(), b.view(), width, x, y, TileRows));
    setup_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
  }
};

class Benchmark : public SWBench {
  Workload &workload_;
  const std::vector<double> &expected_;
  ExecutionMode mode_;
  int replays_;
  int warmup_;

  Timer analysis() override {
    Timer timer;
    timer.ElapsedTimeArray.emplace_back(workload_.setup_seconds, "PlanSetup");
    return timer;
  }

  void preExecute() override {
    for (int replay = 0; replay < warmup_; ++replay)
      workload_.plan->launch(workload_.stream, mode_);
    // Exclude graph upload, warmup, and output poisoning from both timers.
    check(cudaMemsetAsync(workload_.y, 0xff,
                          workload_.output.size() * sizeof(float), workload_.stream));
    check(cudaStreamSynchronize(workload_.stream));
  }

  Timer execute() override {
    const auto begin = Clock::now();
    check(cudaEventRecord(workload_.start, workload_.stream));
    for (int replay = 0; replay < replays_; ++replay)
      workload_.plan->launch(workload_.stream, mode_);
    const auto submitted = Clock::now();
    check(cudaEventRecord(workload_.stop, workload_.stream));
    check(cudaEventSynchronize(workload_.stop));
    const auto complete = Clock::now();
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, workload_.start, workload_.stop));
    Timer timer;
    timer.ElapsedTimeArray.emplace_back(milliseconds * 1e-3 / replays_, "Gpu");
    St->OtherStats["wall_seconds"] = {
        std::chrono::duration<double>(complete - begin).count() / replays_};
    St->OtherStats["submit_seconds"] = {
        std::chrono::duration<double>(submitted - begin).count() / replays_};
    return timer;
  }

  bool verify(double &error) override {
    check(cudaMemcpyAsync(workload_.output.data(), workload_.y,
                          workload_.output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, workload_.stream));
    check(cudaStreamSynchronize(workload_.stream));
    error = 0;
    for (size_t i = 0; i < expected_.size(); ++i) {
      const double difference = std::abs(workload_.output[i] - expected_[i]);
      if (!std::isfinite(workload_.output[i]) ||
          difference > 1e-5 + 1e-5 * std::abs(expected_[i]))
        throw std::runtime_error("Output mismatch at element " + std::to_string(i));
      error = std::max(error, difference);
    }
    // Stats' standard error field rounds to six decimal places.
    St->OtherStats["max_abs_error"] = {error};
    return true;
  }

public:
  Benchmark(Stats &stats, Workload &workload, const std::vector<double> &expected,
            ExecutionMode mode, int replays, int warmup)
      : SWBench(&stats), workload_(workload), expected_(expected), mode_(mode),
        replays_(replays), warmup_(warmup) {}
};

} // namespace

int main(int argc, char **argv) {
  if ((argc == 2 && std::string(argv[1]) == "--help") || argc < 4 || argc > 8) {
    std::cerr << "Usage: " << argv[0]
              << " rows width locality_percent [replays=200] [trials=7]"
                 " [warmup=20] [order_offset=0]\n"
                 "rows: multiples of 32 in [64, 262144]; width: [1, 256]; "
                 "locality_percent: 0, 50, or 100; order_offset: [0, 3].\n";
    return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
  }
  try {
    const int rows = argument(argv[1], 64, 262144);
    const int width = argument(argv[2], 1, 256);
    const int locality = argument(argv[3], 0, 100);
    const int replays = argc > 4 ? argument(argv[4], 1, 100000) : 200;
    const int trials = argc > 5 ? argument(argv[5], 1, 1000) : 7;
    const int warmup = argc > 6 ? argument(argv[6], 1, 100000) : 20;
    const int order_offset = argc > 7 ? argument(argv[7], 0, 3) : 0;
    if (rows % TileRows || (locality != 0 && locality != 50 && locality != 100))
      throw std::invalid_argument("Rows must be divisible by 32; locality must be 0, 50, or 100");

    int device = 0, runtime = 0, driver = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaRuntimeGetVersion(&runtime));
    check(cudaDriverGetVersion(&driver));
    std::cerr << "GPU=" << properties.name << " runtime=" << runtime
              << " driver=" << driver << '\n';

    const Csr a(rows, locality, false), b(rows, 100, true);
    std::vector<float> input(static_cast<size_t>(rows) * width);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = (static_cast<int>((i * 7 + i / width * 13) % 127) - 63) / 64.f;
    const auto expected = reference(a, b, input, width);
    Workload workload(input.size());
    workload.initialize(a, b, input, width);
    const auto info = workload.plan->info();
    if (info.fused_rows != rows * locality / 100 ||
        info.deferred_rows != rows - info.fused_rows)
      throw std::runtime_error("Unexpected inspector row partition");

    bool header = true;
    for (int trial = 0; trial < trials; ++trial) {
      for (int order = 0; order < 4; ++order) {
        const int mode = (trial + order + order_offset) % 4;
        Stats stats(ModeNames[mode], "SpMM_SpMM", 1, "synthetic", 1);
        stats.ProfilingInfoTrials[0]->resizeValueArray(1, 1, 0);
        stats.OtherStats = {
            {"rows", {double(rows)}}, {"width", {double(width)}},
            {"locality_percent", {double(locality)}}, {"degree", {Degree}},
            {"tile_rows", {TileRows}}, {"trial", {double(trial)}},
            {"trials", {double(trials)}}, {"order", {double(order)}},
            {"order_offset", {double(order_offset)}},
            {"replays", {double(replays)}}, {"warmup", {double(warmup)}},
            {"fused_rows", {double(info.fused_rows)}},
            {"deferred_rows", {double(info.deferred_rows)}},
            {"producer_tiles", {double(info.producer_tiles)}}};
        Benchmark benchmark(stats, workload, expected, Modes[mode], replays, warmup);
        benchmark.run();
        if (header) {
          std::cout << stats.printCSVHeader() << '\n';
          header = false;
        }
        std::cout << stats.printCSV() << '\n';
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
