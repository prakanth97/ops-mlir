#include "ops-mlir/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OPSFPCONTRACT
#include "ops-mlir/Conversion/Passes.h.inc"

namespace {

using namespace mlir;

struct OpsFpContractPass : public impl::OpsFpContractBase<OpsFpContractPass> {
  void runOnOperation() override {
    getOperation().walk([&](arith::ArithFastMathInterface op) {
      // Only the operations that are fused (an addition or subtraction of a product): setting the flag on the others
      // changes nothing and clutters the IR.
      if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op.getOperation())) return;
      arith::FastMathFlags flags = arith::FastMathFlags::contract;
      if (auto current = op.getFastMathFlagsAttr()) flags = flags | current.getValue();
      op->setAttr(op.getFastMathAttrName(), arith::FastMathFlagsAttr::get(op->getContext(), flags));
    });
  }
};

} // namespace

} // namespace ops_mlir::ops
