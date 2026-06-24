
#ifndef OPS_MLIR_RUNTIME_CORE_H
#define OPS_MLIR_RUNTIME_CORE_H

#include "ops_lib_core.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ops_mlir {

enum class ArgKind { Dat, Gbl, Idx, Reduce, Unknown };

struct DatDesc {
  std::uintptr_t handle;
  int index;
  std::uintptr_t block;

  int dim;
  int type_size;
  int elem_size;

  std::vector<int64_t> size;
  std::vector<int64_t> base;
  std::vector<int64_t> d_m;
  std::vector<int64_t> d_p;
  std::vector<int64_t> stride;

  std::string name;
  std::string type;

  std::uintptr_t data;
  std::uintptr_t data_d;
};

struct StencilDesc {
  std::uintptr_t handle;
  int index;
  int dims;
  int points;
  std::string name;

  std::uintptr_t stencil;
  std::uintptr_t stride;
  std::uintptr_t mgrid_stride;
  int type;
};

struct ArgDesc {
  DatDesc dat;
  StencilDesc stencil;
  int dim;
  int elem_size;

  std::uintptr_t data;
  std::uintptr_t data_d;
  int acc;
  int argtype;
  int opt;
};

struct LoopDesc {
  std::string kernel_name;
  std::uintptr_t kernel_token;

  std::uintptr_t block;
  int dims;

  std::vector<int64_t> range;
  std::vector<ArgDesc> args;
};

}

#endif // OPS_MLIR_RUNTIME_CORE_H
