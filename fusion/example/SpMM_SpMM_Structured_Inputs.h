#ifndef SPARSE_FUSION_SPMM_SPMM_STRUCTURED_INPUTS_H
#define SPARSE_FUSION_SPMM_SPMM_STRUCTURED_INPUTS_H

#include "aggregation/def.h"
#include "aggregation/sparse_utilities.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

namespace spmm_spmm_benchmark {

struct StructuredCase {
  int rows;
  int structure; // Total diagonals for a band, or rows in each dense block.
  bool blockDiagonal;

  const char *family() const { return blockDiagonal ? "block_diagonal" : "banded"; }

  std::string name() const {
    return std::string(family()) + "_" + std::to_string(structure) + "_" +
           std::to_string(rows);
  }

  // Input property for four-row GPU producer tiles, independent of CPU scheduling.
  int eligibleRows() const {
    if (blockDiagonal) {
      return structure == 4 ? rows : 0;
    }
    return structure == 3 ? rows / 2 + 2 : (structure == 5 ? 4 : 0);
  }
};

inline bool validRows(int rows) {
  const int sizes[] = {64, 512, 4096, 32768, 262144, 1048576};
  return std::find(std::begin(sizes), std::end(sizes), rows) != std::end(sizes);
}

inline std::vector<StructuredCase> structuredCases(int rowFilter = 0) {
  std::vector<StructuredCase> cases;
  const int sizes[] = {64, 512, 4096, 32768, 262144, 1048576};
  for (int rows : sizes) {
    if (rowFilter != 0 && rows != rowFilter) {
      continue;
    }
    for (int diagonals : {3, 5, 9}) {
      cases.push_back({rows, diagonals, false});
    }
    for (int blockRows : {4, 16}) {
      cases.push_back({rows, blockRows, true});
    }
  }
  return cases;
}

inline sym_lib::CSC *makeStructuredMatrix(const StructuredCase &test) {
  int halfBand = test.structure / 2;
  int nnz = test.blockDiagonal
      ? test.rows * test.structure
      : test.rows * test.structure - halfBand * (halfBand + 1);
  auto *csr = new sym_lib::CSR(test.rows, test.rows, nnz);
  int entry = 0;
  for (int row = 0; row < test.rows; ++row) {
    int first = test.blockDiagonal ? row / test.structure * test.structure
                                   : std::max(0, row - halfBand);
    int end = test.blockDiagonal ? first + test.structure
                                 : std::min(test.rows, row + halfBand + 1);
    csr->p[row] = entry;
    for (int column = first; column < end; ++column) {
      csr->i[entry] = column;
      // These positive weights keep repeated products well scaled. Dense blocks
      // retain a diagonal contribution rather than making H and Y identical.
      csr->x[entry] = row == column ? 0.5
          : (test.blockDiagonal ? 0.5 / (test.structure - 1) : 0.25 / halfBand);
      ++entry;
    }
  }
  csr->p[test.rows] = entry;
  auto *matrix = sym_lib::csr_to_csc(csr);
  delete csr;
  return matrix;
}

inline void initializeDense(float *values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    int index = static_cast<int>(i % 251);
    values[i] = ((index * 37) % 251 - 125) / 128.0f;
  }
}

inline bool parseInteger(const char *text, int minimum, int maximum, int &value) {
  char *end;
  errno = 0;
  long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < minimum ||
      parsed > maximum) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

} // namespace spmm_spmm_benchmark

#endif // SPARSE_FUSION_SPMM_SPMM_STRUCTURED_INPUTS_H
