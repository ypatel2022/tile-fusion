#include "gemm_dag.cuh"
#include "SWBench.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>

using swiftware::benchmark::Stats;
using swiftware::benchmark::SWBench;
using swiftware::benchmark::Timer;
using Clock = std::chrono::steady_clock;

enum class Mode { Sequential, SerialGraph, DagGraph };

// All modes use the same inputs, output buffers, CPU reference, and GEMM kernel.
struct Workload {
    int n;
    size_t bytes;
    float *d_A, *d_B;
    float* d_C[num_gemms];
    std::vector<float> reference[num_gemms];

    explicit Workload(int size) : n(size), bytes(size_t(n) * n * sizeof(float)) {
        std::vector<float> A(size_t(n) * n), B(size_t(n) * n);
        initialize_inputs(A, B, n);
        std::fprintf(stderr, "Computing CPU reference for %d GEMMs, n=%d...\n", num_gemms, n);
        reference_dag(A, B, reference, n);
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        for (auto& output : d_C) CUDA_CHECK(cudaMalloc(&output, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));
    }

    ~Workload() {
        for (auto output : d_C) CUDA_CHECK(cudaFree(output));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
    }
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

// Follow the repository's setup -> analysis -> trials -> verify -> teardown flow.
class GemmDagBench : public SWBench {
    Workload& work;
    Mode mode;
    int iterations, warmup;
    cudaStream_t stream{};
    cudaEvent_t start{}, stop{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};

    void launch() {
        if (mode == Mode::Sequential) {
            launch_sequential(work.d_A, work.d_B, work.d_C, work.n, stream);
        } else {
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        }
    }

    void setup() override {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    Timer analysis() override {
        Timer time;
        if (mode != Mode::Sequential) {
            auto begin = Clock::now();
            if (mode == Mode::SerialGraph) {
                // Capture the exact same single-stream sequence as the baseline.
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                launch_sequential(work.d_A, work.d_B, work.d_C, work.n, stream);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            } else {
                graph = build_graph(work.d_A, work.d_B, work.d_C, work.n);
            }
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
            CUDA_CHECK(cudaGraphUpload(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            preparation_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        }
        // Match SWBench's seconds convention; keep one-time preparation separate.
        time.ElapsedTimeArray.emplace_back(preparation_seconds, "Graph preparation seconds");
        return time;
    }

    void preExecute() override {
        // Poison outputs outside timing so a missing write cannot reuse another
        // mode's correct result. Each replay overwrites all outputs from fixed A/B.
        for (auto output : work.d_C) {
            CUDA_CHECK(cudaMemsetAsync(output, 0xff, work.bytes, stream));
        }
        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    Timer execute() override {
        // One event pair surrounds a batch of complete DAGs. There is no sync
        // between GEMMs or replays; the single stream orders successive DAGs.
        auto begin = Clock::now();
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        double wall_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        CUDA_CHECK(cudaGetLastError());
        float gpu_ms;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

        // Report seconds per complete 30-GEMM evaluation, not per kernel.
        double gpu_seconds = (gpu_ms / 1000.0) / iterations;
        wall_seconds /= iterations;
        if (!(gpu_seconds > 0.0) || !(wall_seconds > 0.0)) {
            std::fprintf(stderr, "Invalid timing measurement.\n");
            std::exit(EXIT_FAILURE);
        }
        gpu_samples.push_back(gpu_seconds);
        wall_samples.push_back(wall_seconds);
        Timer time;
        // Timer::startGPU() records on the default stream; our events explicitly
        // use the workload stream. Reuse Timer for SWBench's CSV storage only.
        time.ElapsedTimeArray.emplace_back(gpu_seconds, "GPU seconds per DAG");
        time.ElapsedTimeArray.emplace_back(wall_seconds, "Wall seconds per DAG");
        return time;
    }

    bool verify(double& error) override {
        // Copies and CPU comparisons are outside both timed regions.
        bool ok = verify_outputs(work.d_C, work.reference, work.n, error);
        passed = passed && ok;
        return ok;
    }

    void teardown() override {
        if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
        if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

public:
    bool passed = true;
    double preparation_seconds = 0.0;
    std::vector<double> gpu_samples, wall_samples;

    GemmDagBench(Stats* stats, Workload& workload, Mode selected, int repeats, int warmups)
        : SWBench(stats), work(workload), mode(selected), iterations(repeats), warmup(warmups) {}
};

int parse_integer(const char* value, const char* name, int minimum, int maximum) {
    errno = 0;
    char* end;
    long parsed = std::strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "%s must be an integer in [%d, %d].\n", name, minimum, maximum);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(parsed);
}

int main(int argc, char** argv) {
    if (argc > 5 || (argc > 1 && std::strcmp(argv[1], "--help") == 0)) {
        std::fprintf(stderr, "Usage: %s [n=64 [replays=100 [trials=7 [warmup=10]]]]\n"
                             "n: 1..1024; replays: 1..1000000; trials: 1..1000; warmup: 1..1000000\n"
                             "CSV goes to stdout; summaries and errors go to stderr.\n", argv[0]);
        return argc > 5 ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    int n = argc > 1 ? parse_integer(argv[1], "n", 1, 1024) : 64;
    int iterations = argc > 2 ? parse_integer(argv[2], "replays", 1, 1000000) : 100;
    int trials = argc > 3 ? parse_integer(argv[3], "trials", 1, 1000) : 7;
    int warmup = argc > 4 ? parse_integer(argv[4], "warmup", 1, 1000000) : 10;

    cudaDeviceProp device;
    int device_id, runtime_version, driver_version;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&device, device_id));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    std::fprintf(stderr, "GPU: %s; CUDA runtime: %d; driver: %d\n",
                 device.name, runtime_version, driver_version);
    Workload work(n);

    // Initialize the kernel before measuring any mode's preparation or replay.
    launch_sequential(work.d_A, work.d_B, work.d_C, work.n, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const Mode modes[] = {Mode::Sequential, Mode::SerialGraph, Mode::DagGraph};
    const char* names[] = {"GPU_Sequential_Launches", "GPU_Serial_Graph", "GPU_DAG_Graph"};
    double gpu_medians[3], wall_medians[3];
    for (int i = 0; i < 3; ++i) {
        Stats stats(names[i], "GEMM_DAG", trials, "dense_" + std::to_string(n), 1);
        for (auto* trial : stats.ProfilingInfoTrials) {
            trial->resizeValueArray(1, 2, 0);  // Two timed regions, no PAPI counters.
        }
        GemmDagBench benchmark(&stats, work, modes[i], iterations, warmup);
        benchmark.run();
        if (!benchmark.passed) {
            std::fprintf(stderr, "%s failed CPU verification; stopping benchmark.\n", names[i]);
            return EXIT_FAILURE;
        }
        gpu_medians[i] = median(benchmark.gpu_samples);
        wall_medians[i] = median(benchmark.wall_samples);
        stats.OtherStats["Median GPU Seconds Per DAG"] = {gpu_medians[i]};
        stats.OtherStats["Median Wall Seconds Per DAG"] = {wall_medians[i]};
        stats.OtherStats["GPU Speedup vs Sequential"] = {gpu_medians[0] / gpu_medians[i]};
        stats.OtherStats["Wall Speedup vs Sequential"] = {wall_medians[0] / wall_medians[i]};

        if (i == 0) {
            std::cout << "GPU,CUDA Runtime,CUDA Driver,N,GEMMs,Replays Per Trial,Warmup Replays,"
                      << stats.printCSVHeader() << '\n';
        }
        std::cout << '"' << device.name << "\"," << runtime_version << ',' << driver_version
                  << ',' << n << ',' << num_gemms << ',' << iterations << ',' << warmup << ','
                  << stats.printCSV() << '\n';
        std::fprintf(stderr, "%s: median GPU %.3f us/DAG, wall %.3f us/DAG; preparation %.3f ms; PASS\n",
                     names[i], gpu_medians[i] * 1e6, wall_medians[i] * 1e6,
                     benchmark.preparation_seconds * 1e3);
    }
    std::fprintf(stderr, "GPU speedups: serial graph / ordinary launches = %.3fx; "
                         "DAG graph / ordinary launches = %.3fx; DAG / serial graph = %.3fx\n",
                 gpu_medians[0] / gpu_medians[1], gpu_medians[0] / gpu_medians[2],
                 gpu_medians[1] / gpu_medians[2]);
    return EXIT_SUCCESS;
}
