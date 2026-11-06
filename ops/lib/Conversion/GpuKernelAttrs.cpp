#include "ops-mlir/Conversion/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSGPUKERNELATTRS
#include "ops-mlir/Conversion/Passes.h.inc"

namespace {

using namespace mlir;

struct OpsGpuKernelAttrsPass : public impl::OpsGpuKernelAttrsBase<OpsGpuKernelAttrsPass> {
  void runOnOperation() override {
    getOperation().walk([&](LLVM::LLVMFuncOp func) {
      if (!func->hasAttr("gpu.kernel") && !func->hasAttr("nvvm.kernel")) return;
      // Every pointer argument of a kernel of ours belongs to a dat (or a reduction buffer): the pointers come in pairs
      // (allocated, aligned) at the head of an expanded memref descriptor.
      unsigned n = func.getNumArguments();
      for (unsigned i = 0; i + 1 < n; ++i) {
        if (!isa<LLVM::LLVMPointerType>(func.getArgument(i).getType()) || !isa<LLVM::LLVMPointerType>(func.getArgument(i + 1).getType())) continue;
        func.setArgAttr(i + 1, LLVM::LLVMDialect::getNoAliasAttrName(), UnitAttr::get(func.getContext()));
        ++i; // the next descriptor starts after this pair
      }
    });
  }
};

} // namespace

} // namespace ops_mlir::ops
