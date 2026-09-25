// One GNN layer's linear operations: H = X W; Y = A H.
// Reuse the repository's CPU kernels and cache-based tile inspector.
#include "GCN_Layer_MKL_Forward_Utils.h"
#include "aggregation/sparse_io.h"
#include "aggregation/sparse_utilities.h"
#include "sparse-fusion/Fusion_Inspector.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace swiftware::benchmark;

static void check(sparse_status_t status) {
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("MKL sparse operation failed");
}

static int positive(const char *text) {
  std::string value(text);
  size_t end = 0;
  int n = std::stoi(value, &end);
  if (end != value.size() || n <= 0)
    throw std::runtime_error("Arguments must be positive integers");
  return n;
}

struct Problem {
  std::unique_ptr<sym_lib::CSR> a;
  int n, features, hidden, threads;
  std::vector<float> values, x, w, reference;
  sparse_matrix_t mklA = nullptr;

  Problem(const std::string &path, int f, int h, int t)
      : features(f), hidden(h), threads(t) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open matrix: " + path);
    sym_lib::CSC *loaded = nullptr;
    // This reader sorts coordinates; the older read_mtx assumes column order.
    sym_lib::read_mtx_csc_real(file, loaded, false);
    std::unique_ptr<sym_lib::CSC> csc(loaded);
    if (!csc || csc->m != csc->n || csc->m == 0 ||
        csc->m > size_t(std::numeric_limits<int>::max() / std::max(f, h)))
      throw std::runtime_error("Expected a nonempty square matrix of supported size");
    if (csc->stype != 0)
      csc.reset(sym_lib::make_full(csc.get()));
    a.reset(sym_lib::csc_to_csr(csc.get()));
    n = int(a->m);
    // Symmetric degree normalization, as in the repository's GCN example.
    values.resize(a->nnz);
    for (int i = 0; i < n; ++i) {
      int degree = a->p[i + 1] - a->p[i];
      if (!degree) throw std::runtime_error("Add self-loops to isolated nodes");
      for (int j = a->p[i]; j < a->p[i + 1]; ++j) {
        int neighborDegree = a->p[a->i[j] + 1] - a->p[a->i[j]];
        values[j] = float(a->x[j] / std::sqrt(double(degree) * neighborDegree));
        if (!std::isfinite(values[j])) throw std::runtime_error("Invalid adjacency value");
      }
    }
    x.resize(size_t(n) * f);
    w.resize(size_t(f) * h);
    reference.resize(size_t(n) * h);
    // Mixed signs and nonconstant features make indexing mistakes observable.
    std::mt19937 rng(42);
    auto randomValue = [&]() { return float(int(rng() % 2001) - 1000) / 1000.f; };
    for (auto &v : x) v = randomValue();
    for (auto &v : w) v = randomValue() / std::sqrt(float(f));
    check(mkl_sparse_s_create_csr(&mklA, SPARSE_INDEX_BASE_ZERO, n, n,
                                a->p, a->p + 1, a->i, values.data()));
  }

  ~Problem() { if (mklA) mkl_sparse_destroy(mklA); }

  void mkl(float *intermediate, float *output) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, hidden, features, 1, x.data(), features, w.data(), hidden,
                0, intermediate, hidden);
    matrix_descr description{};
    description.type = SPARSE_MATRIX_TYPE_GENERAL;
    check(mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklA, description,
                         SPARSE_LAYOUT_ROW_MAJOR, intermediate, hidden, hidden,
                         0, output, hidden));
  }

  void makeReference() {
    mkl_set_num_threads(threads);
    std::vector<float> intermediate(reference.size());
    mkl(intermediate.data(), reference.data());
    for (float v : reference)
      if (!std::isfinite(v)) throw std::runtime_error("Nonfinite MKL reference");
    // Independently check up to 16 entire output rows with scalar double math.
    for (int sample = 0; sample < std::min(n, 16); ++sample) {
      int row = int(size_t(sample) * (n - 1) / std::max(1, std::min(n, 16) - 1));
      for (int col = 0; col < hidden; ++col) {
        double expected = 0;
        for (int j = a->p[row]; j < a->p[row + 1]; ++j) {
          double dot = 0;
          for (int k = 0; k < features; ++k)
            dot += double(x[size_t(a->i[j]) * features + k]) * w[size_t(k) * hidden + col];
          expected += double(values[j]) * dot;
        }
        if (std::abs(expected - reference[size_t(row) * hidden + col]) >
            1e-4 + 1e-4 * std::abs(expected))
          throw std::runtime_error("Scalar reference check failed");
      }
    }
  }
};

enum Mode { Library, Unfused, Fused };

class LayerBench : public SWBench {
  Problem &p;
  Mode mode;
  int cacheBytes, warmups;
  bool warmed = false;
  std::vector<float> intermediate, output;
  std::unique_ptr<sym_lib::MultiDimensionalSet> schedule;

  void callKernel() {
    if (mode == Library) {
      p.mkl(intermediate.data(), output.data());
    } else if (mode == Unfused) {
      forwardForOneLayerWithMKLGeMMAndSpMMSPVectorized(
          p.n, p.a->p, p.a->i, p.values.data(), p.x.data(), p.features,
          p.w.data(), p.hidden, output.data(), intermediate.data(), p.threads);
    } else {
      forwardForOneLayerFusedParallelSeparatedVectorizedSP(
          p.n, p.a->p, p.a->i, p.values.data(), p.features, p.hidden,
          p.x.data(), p.w.data(), output.data(), intermediate.data(), p.threads,
          schedule->n1_, schedule->ptr1_, schedule->ptr2_,
          schedule->ker_begin_, schedule->id_);
    }
  }

  Timer analysis() override {
    Timer timer;
    timer.start();
    if (mode == Fused) {
      sym_lib::ScheduleParameters parameters;
      parameters._num_threads = p.threads;
      parameters.TileM = cacheBytes;
      parameters.TileN = 32;
      InspectorForTileFusedCSRVariableTileSize inspector(parameters, St);
      schedule.reset(inspector.generateVariableTileSizeScheduleGeMMSpMM(
          p.a.get(), p.features, p.hidden, sizeof(float)));
    }
    timer.stop("Inspection");
    St->OtherStats["FusedRows"] = {mode == Fused ? double(schedule->getNumberOfFusedNodes()) : 0.};
    return timer;
  }

  void preExecute() override {
    // Fused tiles use outer OpenMP parallelism and single-thread MKL calls.
    mkl_set_num_threads(mode == Fused ? 1 : p.threads);
    if (!warmed) {
      for (int i = 0; i < warmups; ++i) {
        std::fill(output.begin(), output.end(), 0.f);
        callKernel();
        double error = 0;
        verify(error);
      }
      warmed = true;
    }
    // The vectorized SpMM accumulates into output. Reset equally for all modes,
    // outside timing, as in the repository. Keep buffers allocated across trials.
    std::fill(output.begin(), output.end(), 0.f);
  }

  Timer execute() override {
    Timer timer;
    timer.start();
    callKernel();
    timer.stop();
    return timer;
  }

  bool verify(double &error) override {
    error = 0;
    for (size_t i = 0; i < output.size(); ++i) {
      double difference = std::abs(double(output[i]) - p.reference[i]);
      error = std::max(error, difference);
      if (!std::isfinite(output[i]) || difference > 1e-4 + 1e-4 * std::abs(p.reference[i]))
        throw std::runtime_error("Output verification failed at element " + std::to_string(i));
    }
    return true;
  }

public:
  LayerBench(Problem &problem, Mode m, int cache, int warm, Stats &stats)
      : SWBench(&stats), p(problem), mode(m), cacheBytes(cache), warmups(warm),
        intermediate(p.reference.size()), output(p.reference.size()) {
    // Predeclare inspector statistics so all three CSV rows share one schema.
    for (const auto &key : {"FusedIterations", "Tile Size Mean", "Tile Size STD", "NumPartitions"})
      stats.OtherStats[key] = {0.};
    stats.OtherStats["Nodes"] = {double(p.n)};
    stats.OtherStats["Nonzeros"] = {double(p.a->nnz)};
    stats.OtherStats["Features"] = {double(p.features)};
    stats.OtherStats["Hidden"] = {double(p.hidden)};
    stats.OtherStats["CacheBytes"] = {double(cache)};
    stats.OtherStats["Warmups"] = {double(warm)};
  }
};

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Usage: gnn_fusion_benchmark MATRIX FEATURES HIDDEN THREADS CACHE_BYTES [TRIALS=7] [WARMUPS=3] [ORDER=1]\n"
                   "ORDER 1/2/3 rotates implementation order. HIDDEN must be a multiple of 32.\n";
      return 0;
    }
    if (argc < 6 || argc > 9) throw std::runtime_error("See --help for arguments");
    int features = positive(argv[2]), hidden = positive(argv[3]);
    int threads = positive(argv[4]), cache = positive(argv[5]);
    int trials = argc > 6 ? positive(argv[6]) : 7;
    int warmups = argc > 7 ? positive(argv[7]) : 3;
    int order = argc > 8 ? positive(argv[8]) : 1;
    if (hidden % 32 || order > 3 || features > 4096 || hidden > 4096)
      throw std::runtime_error("Unsupported feature dimensions or implementation order");
    // Inspector needs room for at least one initial row tile.
    if (cache < 4LL * (features + hidden + 2LL * features * hidden))
      throw std::runtime_error("Cache budget too small for these feature dimensions");
    omp_set_dynamic(0);
    mkl_set_dynamic(0);
    Problem problem(argv[1], features, hidden, threads);
    problem.makeReference();
    const char *names[] = {"unfused_mkl", "unfused_avx2", "tile_fused_avx2"};
    std::string header;
    for (int i = 0; i < 3; ++i) {
      int mode = (i + order - 1) % 3;
      Stats stats(names[mode], "GEMM_SpMM", trials,
                  std::filesystem::path(argv[1]).stem().string(), threads);
      LayerBench bench(problem, Mode(mode), cache, warmups, stats);
      bench.run();
      if (i == 0) {
        header = stats.printCSVHeader();
        std::cout << header << '\n';
      } else if (header != stats.printCSVHeader()) {
        throw std::runtime_error("Inconsistent CSV schema");
      }
      std::cout << stats.printCSV() << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
