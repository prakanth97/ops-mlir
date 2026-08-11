//===- KernelIRBuilder.h - C++ kernel body -> MLIR --------------*- C++ -*-===//
//
// Translates a single OPS kernel function's C++ source into a real MLIR
// function body, for backends (e.g. CUDA).
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H
#define OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

namespace ops_mlir {

class KernelIRBuilder {
public:
  explicit KernelIRBuilder(mlir::MLIRContext &context) : context_(context) {}

  mlir::func::FuncOp generate(const std::string &sourceFile,
                              const std::string &kernelName, int indexRank,
                              const std::map<std::string, const void *> &constants,
                              const std::vector<int> &constArgDims,
                              llvm::raw_ostream &errs);

private:
  mlir::MLIRContext &context_;
};

} // namespace ops_mlir

#endif // OPS_MLIR_RUNTIME_KERNEL_IR_BUILDER_H
