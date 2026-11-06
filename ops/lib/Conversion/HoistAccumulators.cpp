#include "ops-mlir/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSHOISTACCUMULATORS
#include "ops-mlir/Conversion/Passes.h.inc"

namespace {

using namespace mlir;

struct OpsHoistAccumulatorsPass : public impl::OpsHoistAccumulatorsBase<OpsHoistAccumulatorsPass> {
  void runOnOperation() override {
    SmallVector<memref::AllocaOp> accumulators;
    getOperation().walk([&](memref::AllocaOp alloca) {
      if (alloca->hasAttr("ops.accumulator")) accumulators.push_back(alloca);
    });
    for (memref::AllocaOp alloca : accumulators) {
      // The nearest enclosing parallel region (a thread of its own), else the function.
      Block *target = nullptr;
      for (Operation *op = alloca->getParentOp(); op; op = op->getParentOp()) {
        if (auto parallel = dyn_cast<omp::ParallelOp>(op)) {
          target = &parallel.getRegion().front();
          break;
        }
        if (auto func = dyn_cast<func::FuncOp>(op)) {
          target = &func.getBody().front();
          break;
        }
      }
      if (!target || alloca->getBlock() == target) continue;
      alloca->moveBefore(&target->front());
      alloca->removeAttr("ops.accumulator");
    }
  }
};

} // namespace

} // namespace ops_mlir::ops
