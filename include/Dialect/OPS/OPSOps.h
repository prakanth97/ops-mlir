#ifndef OPS_MLIR_DIALECT_OPS_OPSOPS_H
#define OPS_MLIR_DIALECT_OPS_OPSOPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/OPS/OPSOpsAttrs.h.inc"

namespace ops_mlir::ops {

class ParLoopOp : public mlir::Op<ParLoopOp> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "ops.par_loop"; }

  static llvm::ArrayRef<llvm::StringRef> getOperandNames() { return {}; }

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() {
    static llvm::StringRef attrNames[] = {
        "kernel_ptr", "kernel_name", "dims", "range", "args", "block_dims"};
    return llvm::ArrayRef(attrNames);
  }

  mlir::StringAttr getKernelNameAttr() {
    return (*this)->getAttrOfType<mlir::StringAttr>("kernel_name");
  }

  void print(mlir::OpAsmPrinter &p);
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
};

} // namespace ops_mlir::ops

#endif // OPS_MLIR_DIALECT_OPS_OPSOPS_H
