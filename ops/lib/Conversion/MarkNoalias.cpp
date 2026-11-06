#include "ops-mlir/Conversion/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSMARKNOALIAS
#include "ops-mlir/Conversion/Passes.h.inc"

namespace {

using namespace mlir;

struct OpsMarkNoaliasPass : public impl::OpsMarkNoaliasBase<OpsMarkNoaliasPass> {
  void runOnOperation() override {
    getOperation().walk([&](LLVM::LLVMFuncOp func) {
      // The C interface wrapper takes pointers to descriptors, which convert-func-to-llvm already marks.
      if (func.getName().starts_with("_mlir_ciface_")) return;
      auto abi = func->getAttrOfType<DictionaryAttr>("ops.abi");
      if (!abi) return;
      auto args = abi.getAs<DenseI64ArrayAttr>("noalias_args");
      if (!args) return;
      for (int64_t index : args.asArrayRef())
        if (index >= 0 && static_cast<unsigned>(index) < func.getNumArguments() && isa<LLVM::LLVMPointerType>(func.getArgument(index).getType()))
          func.setArgAttr(index, LLVM::LLVMDialect::getNoAliasAttrName(), UnitAttr::get(func.getContext()));
    });
  }
};

} // namespace

} // namespace ops_mlir::ops
