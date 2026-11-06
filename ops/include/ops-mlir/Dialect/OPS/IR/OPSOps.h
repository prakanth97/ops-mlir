//===- OPSOps.h - operations of the `ops` dialect -----------------*- C++ -*-===//

#ifndef OPS_MLIR_DIALECT_OPS_IR_OPSOPS_H
#define OPS_MLIR_DIALECT_OPS_IR_OPSOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.h"
#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"
#include "ops-mlir/Dialect/OPS/IR/OPSTypes.h"

#define GET_OP_CLASSES
#include "ops-mlir/Dialect/OPS/IR/OPSOps.h.inc"

#endif // OPS_MLIR_DIALECT_OPS_IR_OPSOPS_H
