#include "Dialect/OPS/OPSDialect.h"
#include "Dialect/OPS/OPSOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ops_mlir::ops;

#include "Dialect/OPS/OPSDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/OPS/OPSOpsAttrs.cpp.inc"

namespace ops_mlir::ops {

void OPSDialect::initialize() {
  addOperations<ParLoopOp>();
  addAttributes<ArgAttr>();
  addAttributes<DatAttr>();
  addAttributes<StencilAttr>();
}

mlir::Type OPSDialect::parseType(mlir::DialectAsmParser &parser) const {
  return nullptr;
}

void OPSDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &os) const {
  os << type;
}

} // namespace ops_mlir::ops
