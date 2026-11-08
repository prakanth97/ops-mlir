//===- Manifest.h - kernel summary and manifest shared by the importers -*- C++ -*-===//

#ifndef OPS_MLIR_FRONTEND_COMMON_MANIFEST_H
#define OPS_MLIR_FRONTEND_COMMON_MANIFEST_H

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ops_mlir::frontend {

struct ParamSummary {
  enum Kind { Acc, Gbl, Red, Idx } kind = Acc;
  mlir::Type elem;
  bool isConst = false;
  bool legacyMacro = false;
  int64_t ncomp = 1;
  int64_t gblDim = 1;
  std::set<std::vector<int64_t>> reads, writes;
  bool dynamicRead = false;
  std::string style = "accumulate";
  std::string kindHint;
};

struct ConstUse {
  std::string name;
  mlir::Type elem;
  int64_t offset = 0, stride = 0;
  bool dynamicIndex = false;
  bool operator<(const ConstUse &o) const {
    return std::tie(name, offset, stride, dynamicIndex) < std::tie(o.name, o.offset, o.stride, o.dynamicIndex);
  }
};

struct KernelSummary {
  std::string kernel;
  int ndim = 0;
  std::vector<ParamSummary> params;
  std::set<ConstUse> consts;
  unsigned idxAxesRead = 0;
  unsigned droppedPrintf = 0;
  unsigned prints = 0;
};

std::string typeName(mlir::Type t);

std::set<std::string> boundConstants(ops::KernelOp kernel);

std::string writeManifest(const KernelSummary &summary, ops::KernelOp kernel, mlir::ModuleOp module,
                          const char *importer);

std::string writeFailedManifest(const std::string &kernel, int ndim, const std::string &reason, const char *importer);

} // namespace ops_mlir::frontend

#endif // OPS_MLIR_FRONTEND_COMMON_MANIFEST_H
