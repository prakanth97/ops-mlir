#ifndef STENCIL_MLIR_DIALECT_STENCIL_STENCIL_OPS_H
#define STENCIL_MLIR_DIALECT_STENCIL_STENCIL_OPS_H

#include "Dialect/stencil/StencilTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/stencil/StencilOps.h.inc"

#endif // STENCIL_MLIR_DIALECT_STENCIL_STENCIL_OPS_H
