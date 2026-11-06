//===- OPSAttrs.h - attributes of the `ops` dialect ---------------*- C++ -*-===//

#ifndef OPS_MLIR_DIALECT_OPS_IR_OPSATTRS_H
#define OPS_MLIR_DIALECT_OPS_IR_OPSATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"

#include "ops-mlir/Dialect/OPS/IR/OPSEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.h.inc"

#endif // OPS_MLIR_DIALECT_OPS_IR_OPSATTRS_H
