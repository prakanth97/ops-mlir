#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Transforms/Passes.h"
#include "ops-mlir/Transforms/StoreForwarding.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_FORWARDSTORES
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

struct ForwardStoresPass : public impl::ForwardStoresBase<ForwardStoresPass> {
  void runOnOperation() override {
    getOperation().walk([&](ParLoopOp loop) {
      forwardStoresInLoop(loop);
      narrowAccessModes(loop);
    });
  }
};

} // namespace

} // namespace ops_mlir::ops
