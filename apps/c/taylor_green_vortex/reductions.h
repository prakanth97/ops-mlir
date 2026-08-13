#ifndef REDUCTIONS_H
#define REDUCTIONS_H
#include <stdlib.h>
#include "ops/OPSWrapper.h"

double readDatValueAt(ops_dat dat, int global_i, int global_j, int global_k) {
  int ndim = dat->block ? dat->block->dims : 1;
  if (ndim != 3) {
    throw std::runtime_error("readDatValueAt: expected a 3D dat");
  }

  int local_i = global_i - dat->d_m[0];
  int local_j = global_j - dat->d_m[1];
  int local_k = global_k - dat->d_m[2];

  int size0 = dat->size[0];
  int size1 = dat->size[1];
  int size2 = dat->size[2];

  if (local_i < 0 || local_i >= size0 ||
      local_j < 0 || local_j >= size1 ||
      local_k < 0 || local_k >= size2) {
    throw std::runtime_error("readDatValueAt: coordinate out of bounds");
  }

  std::size_t flat =
    static_cast<std::size_t>(local_i) * (static_cast<std::size_t>(size1) * size2) +
    static_cast<std::size_t>(local_j) * static_cast<std::size_t>(size2) +
    static_cast<std::size_t>(local_k);

  return reinterpret_cast<double *>(dat->data)[flat];
}

#endif
