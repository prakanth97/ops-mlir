//===- Passes.h - conversion passes of the ops dialect ----------*- C++ -*-===//

#ifndef OPS_MLIR_CONVERSION_PASSES_H
#define OPS_MLIR_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace arith { class ArithDialect; }
namespace LLVM { class LLVMDialect; }
namespace func { class FuncDialect; }
namespace memref { class MemRefDialect; }
namespace scf { class SCFDialect; }
} // namespace mlir

namespace ops_mlir::ops {

class OPSDialect;

#define GEN_PASS_DECL
#include "ops-mlir/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ops-mlir/Conversion/Passes.h.inc"

} // namespace ops_mlir::ops

#endif // OPS_MLIR_CONVERSION_PASSES_H
