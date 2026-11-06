#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"

namespace ops_mlir::ops {

#define GEN_PASS_DEF_OUTLINEGROUPS
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

struct OutlineGroupsPass : public impl::OutlineGroupsBase<OutlineGroupsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    SmallVector<ChainOp> chains(module.getOps<ChainOp>());
    for (ChainOp chain : chains)
      if (failed(outline(chain, symbols))) return signalPassFailure();
  }

  LogicalResult outline(ChainOp chain, SymbolTable &symbols) {
    auto ranges = chain->getAttrOfType<ArrayAttr>("ops.range_values");
    auto shapes = chain->getAttrOfType<ArrayAttr>("ops.shape_values");
    SmallVector<ParLoopOp> loops;
    for (Operation &op : chain.getBody().front())
      if (auto loop = dyn_cast<ParLoopOp>(&op)) loops.push_back(loop);
    if (loops.empty()) return success();

    MLIRContext *ctx = chain.getContext();
    OpBuilder moduleBuilder(chain);
    for (auto [n, loop] : llvm::enumerate(loops)) {
      // The group's arguments: the distinct values of the loop's operands, in order of appearance.
      SmallVector<Value> operands;
      DenseMap<Value, unsigned> argOf;
      for (Value v : loop->getOperands())
        if (argOf.try_emplace(v, operands.size()).second) operands.push_back(v);

      // Values the body uses from the chain: only constants can be copied along.
      SetVector<Value> captured;
      getUsedValuesDefinedAbove(loop.getBody(), captured);
      for (Value v : captured) {
        if (argOf.count(v)) continue;
        Operation *def = v.getDefiningOp();
        if (!def || !matchPattern(v, m_Constant()))
          return loop.emitError("the loop body uses a value defined in the chain that is neither an operand nor a "
                                "constant: pass it as a scalar operand");
      }

      SmallVector<Type> inputTypes;
      for (Value v : operands) inputTypes.push_back(v.getType());
      SmallVector<Type> resultTypes(loop->getResultTypes());
      std::string name = ("group_" + Twine(n)).str();
      auto group = ChainOp::create(moduleBuilder, loop.getLoc(), name, FunctionType::get(ctx, inputTypes, resultTypes));
      symbols.insert(group);
      Block *entry = group.addEntryBlock();

      IRMapping mapper;
      for (auto [i, v] : llvm::enumerate(operands)) mapper.map(v, entry->getArgument(i));
      OpBuilder b = OpBuilder::atBlockEnd(entry);
      for (Value v : captured)
        if (!argOf.count(v)) b.clone(*v.getDefiningOp(), mapper);
      Operation *clone = b.clone(*loop, mapper);
      ReturnOp::create(b, loop.getLoc(), clone->getResults());

      if (ranges && n < ranges.size()) group->setAttr("ops.range_values", ArrayAttr::get(ctx, {ranges[n]}));
      if (shapes) {
        // The sizes of the group's own arguments (the chain's argument of the same value).
        SmallVector<Attribute> own;
        for (Value v : operands) {
          auto arg = dyn_cast<BlockArgument>(v);
          if (arg && arg.getOwner() == &chain.getBody().front() && arg.getArgNumber() < shapes.size())
            own.push_back(shapes[arg.getArgNumber()]);
          else
            own.push_back(DenseI64ArrayAttr::get(ctx, {}));
        }
        group->setAttr("ops.shape_values", ArrayAttr::get(ctx, own));
      }

      // In the chain: replace the loop by a call.
      OpBuilder cb(loop);
      auto call = CallOp::create(cb, loop.getLoc(), resultTypes, name, operands);
      loop->replaceAllUsesWith(call->getResults());
    }
    for (ParLoopOp loop : loops) loop.erase();

    // Constants only the loop bodies used are dead now.
    for (Operation &op : llvm::make_early_inc_range(chain.getBody().front().without_terminator()))
      if (isa<arith::ConstantOp>(op) && op.use_empty()) op.erase();
    return success();
  }
};

} // namespace

} // namespace ops_mlir::ops
