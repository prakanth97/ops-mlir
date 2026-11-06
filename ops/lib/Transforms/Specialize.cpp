#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseMap.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSSPECIALIZE
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

struct OpsSpecializePass : public impl::OpsSpecializeBase<OpsSpecializePass> {
  using OpsSpecializeBase::OpsSpecializeBase;

  void runOnOperation() override {
    if (mode == "none") return;
    if (mode != "ranges" && mode != "shapes") {
      getOperation().emitError("ops-specialize: unknown mode '") << mode << "' (none, ranges or shapes)";
      return signalPassFailure();
    }
    for (ChainOp chain : getOperation().getOps<ChainOp>()) {
      // mode=shapes: the sizes of the dats are given by `ops.shape_values` (one entry per chain argument, empty for the
      // arguments that are not dats); convert-ops-to-scf gives the memrefs static shapes and strides.
      if (mode == "shapes" && !chain->getAttrOfType<ArrayAttr>("ops.shape_values")) {
        chain.emitError("ops-specialize mode=shapes needs an `ops.shape_values` entry for each argument of the chain");
        return signalPassFailure();
      }
      if (failed(specializeRanges(chain))) return signalPassFailure();
    }
  }

  /// The range operands of each loop become the constants of `ops.range_values`. The chain keeps its arguments:
  /// the ABI does not change, the values are simply no longer used.
  LogicalResult specializeRanges(ChainOp chain) {
    auto ranges = chain->getAttrOfType<ArrayAttr>("ops.range_values");
    SmallVector<ParLoopOp> loops;
    for (Operation &op : chain.getBody().front())
      if (auto loop = dyn_cast<ParLoopOp>(&op)) loops.push_back(loop);
    if (loops.empty()) return success();
    if (!ranges || ranges.size() != loops.size())
      return chain.emitError("ops-specialize mode=ranges needs an `ops.range_values` entry for each loop");

    OpBuilder b(chain.getContext());
    b.setInsertionPointToStart(&chain.getBody().front());
    DenseMap<int64_t, Value> constants;
    auto constant = [&](int64_t v) {
      auto [it, inserted] = constants.try_emplace(v);
      if (inserted) it->second = arith::ConstantIndexOp::create(b, chain.getLoc(), v);
      return it->second;
    };
    for (auto [i, loop] : llvm::enumerate(loops)) {
      auto values = dyn_cast<DenseI64ArrayAttr>(ranges[i]);
      if (!values || values.size() != loop.getRange().size())
        return loop.emitError("`ops.range_values` entry ") << i << " does not match the range of the loop";
      for (auto [k, operand] : llvm::enumerate(loop.getRangeMutable()))
        operand.set(constant(values[k]));
    }
    return success();
  }
};

} // namespace

} // namespace ops_mlir::ops
