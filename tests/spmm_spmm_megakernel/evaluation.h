#ifndef SPMM_SPMM_MEGAKERNEL_EVALUATION_H
#define SPMM_SPMM_MEGAKERNEL_EVALUATION_H

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace spmm_megakernel {

constexpr int Features = 32;

struct Matrix {
  int rows;
  std::vector<int> rowPtr, columns;
  std::vector<float> values;
};

struct Slot {
  float *x = nullptr, *h = nullptr, *y = nullptr;
};

struct DeviceData {
  int rows, tileRows;
  int *rowPtr = nullptr, *columns = nullptr;
  float *values = nullptr;
  std::vector<Slot> slots;
  size_t allocatedBytes = 0;
  double allocationSeconds = 0, copySeconds = 0;

  DeviceData(const Matrix &matrix, int tileRows, int slotCount,
             const std::vector<std::vector<float>> &inputs);
  ~DeviceData();
  DeviceData(const DeviceData &) = delete;
  DeviceData &operator=(const DeviceData &) = delete;
};

// Constructors prepare reusable schedules. launch() submits exactly the work
// required for the current values; correctness and timing use this same path.
struct Method {
  size_t scheduleBytes = 0, eventBytes = 0, libraryBytes = 0;
  size_t hWriteBytes = 0;
  double inspectionSeconds = 0, librarySeconds = 0, graphSeconds = 0;
  virtual ~Method() = default;
  virtual void launch(int slot) = 0;
  virtual size_t workspaceBytes() const = 0;
  virtual bool writesH() const { return true; }
};

#ifdef SPMM_MEGAKERNEL_IMPLEMENTATION
std::unique_ptr<Method> makeMegakernelMethod(const std::string &name,
                                            const Matrix &matrix,
                                            DeviceData &data);
#endif

} // namespace spmm_megakernel

#endif
