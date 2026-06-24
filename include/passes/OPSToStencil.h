#ifndef OPS_MLIR_PASSES_OPS_TO_STENCIL_H
#define OPS_MLIR_PASSES_OPS_TO_STENCIL_H

#include "mlir/IR/BuiltinOps.h"

namespace ops_mlir {

/// Lower every ops.par_loop in `module` into its own func.func with
/// memref<...> parameters and a stencil.apply body, in place.
void convertOPSParLoopsToStencil(mlir::ModuleOp module);

} // namespace ops_mlir

#endif // OPS_MLIR_PASSES_OPS_TO_STENCIL_H
