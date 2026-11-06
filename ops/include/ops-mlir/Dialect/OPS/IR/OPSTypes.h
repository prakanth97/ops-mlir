//===- OPSTypes.h - types of the `ops` dialect --------------------*- C++ -*-===//

#ifndef OPS_MLIR_DIALECT_OPS_IR_OPSTYPES_H
#define OPS_MLIR_DIALECT_OPS_IR_OPSTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"

#define GET_TYPEDEF_CLASSES
#include "ops-mlir/Dialect/OPS/IR/OPSTypes.h.inc"

namespace ops_mlir::ops {
/// Highest block rank supported by the dialect (OPS itself allows 5).
constexpr int64_t kMaxRank = 3;
/// True for the element types of dats, accessors, globals and reductions: f16, f32, f64, i32, i64.
bool isSupportedElementType(mlir::Type type);
} // namespace ops_mlir::ops

#endif // OPS_MLIR_DIALECT_OPS_IR_OPSTYPES_H
