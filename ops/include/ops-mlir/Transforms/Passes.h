//===- Passes.h - transformation passes of the ops dialect -------*- C++ -*-===//

#ifndef OPS_MLIR_TRANSFORMS_PASSES_H
#define OPS_MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;
namespace arith { class ArithDialect; }
namespace scf { class SCFDialect; }
} // namespace mlir

namespace ops_mlir::ops {

class OPSDialect;

#define GEN_PASS_DECL
#include "ops-mlir/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ops-mlir/Transforms/Passes.h.inc"

} // namespace ops_mlir::ops

#endif // OPS_MLIR_TRANSFORMS_PASSES_H
