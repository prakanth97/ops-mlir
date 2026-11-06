#include "ops-mlir/Dialect/OPS/IR/OPSDialect.h"

#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.h"
#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Dialect/OPS/IR/OPSTypes.h"

using namespace mlir;
using namespace ops_mlir::ops;

#include "ops-mlir/Dialect/OPS/IR/OPSOpsDialect.cpp.inc"

void OPSDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "ops-mlir/Dialect/OPS/IR/OPSOps.cpp.inc"
      >();
}
