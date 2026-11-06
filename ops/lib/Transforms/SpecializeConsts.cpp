#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringSet.h"

#include <algorithm>

namespace ops_mlir::ops {

#define GEN_PASS_DEF_SPECIALIZECONSTS
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/// The value of type `type` stored little-endian at `offset` in `bytes`, or null.
TypedAttr decode(Type type, ArrayRef<int8_t> bytes, int64_t offset) {
  if (!type.isIntOrFloat()) return nullptr;
  unsigned bits = type.getIntOrFloatBitWidth();
  if (bits > 64 || bits % 8) return nullptr;
  int64_t size = bits / 8;
  if (offset < 0 || offset + size > static_cast<int64_t>(bytes.size())) return nullptr;
  uint64_t raw = 0;
  for (int64_t i = 0; i < size; ++i) raw |= static_cast<uint64_t>(static_cast<uint8_t>(bytes[offset + i])) << (8 * i);
  if (auto ft = dyn_cast<FloatType>(type))
    return FloatAttr::get(type, APFloat(ft.getFloatSemantics(), APInt(bits, raw)));
  return IntegerAttr::get(type, APInt(bits, raw));
}

/// Replaces an `ops.const_ref` of a constant with known value, with no index or a constant one.
struct FoldConstRef : public OpRewritePattern<ConstRefOp> {
  FoldConstRef(MLIRContext *ctx, const llvm::StringSet<> &names, DictionaryAttr values, bool &hadError)
      : OpRewritePattern(ctx), names(names), values(values), hadError(hadError) {}

  LogicalResult matchAndRewrite(ConstRefOp ref, PatternRewriter &rewriter) const override {
    if (!names.contains(ref.getName())) return failure();
    int64_t index = 0;
    if (ref.getIndex()) {
      APInt value;
      if (!matchPattern(ref.getIndex(), m_ConstantInt(&value))) return failure(); // not (yet) constant
      index = value.getSExtValue();
    }
    auto bytes = values ? values.getAs<DenseI8ArrayAttr>(ref.getName()) : DenseI8ArrayAttr();
    if (!bytes) {
      hadError = true;
      ref.emitError("no value for constant @") << ref.getName() << " in ops.const_values";
      return failure();
    }
    int64_t offset = static_cast<int64_t>(ref.getOffset()) + index * static_cast<int64_t>(ref.getArrayStride());
    TypedAttr attr = decode(ref.getType(), bytes.asArrayRef(), offset);
    if (!attr) {
      hadError = true;
      ref.emitError("constant @") << ref.getName() << " has " << bytes.size() << " bytes: cannot read "
                                 << ref.getType() << " at byte offset " << offset;
      return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(ref, attr);
    return success();
  }

  const llvm::StringSet<> &names;
  DictionaryAttr values;
  bool &hadError;
};

struct SpecializeConstsPass : public impl::SpecializeConstsBase<SpecializeConstsPass> {
  using SpecializeConstsBase::SpecializeConstsBase;

  void runOnOperation() override {
    for (ChainOp chain : getOperation().getOps<ChainOp>())
      if (failed(specialize(chain))) return signalPassFailure();
  }

  LogicalResult specialize(ChainOp chain) {
    MLIRContext *ctx = chain.getContext();
    auto values = chain->getAttrOfType<DictionaryAttr>("ops.const_values");
    auto roles = chain->getAttrOfType<DictionaryAttr>("ops.const_roles");

    llvm::StringSet<> fold;
    if (!names.empty()) {
      for (const std::string &n : names) fold.insert(n);
    } else if (roles) {
      for (NamedAttribute roleEntry : roles)
        if (auto role = dyn_cast<StringAttr>(roleEntry.getValue()); role && role.getValue() == "bound")
          fold.insert(roleEntry.getName().getValue());
    }

    if (!fold.empty()) {
      if (failed(foldBindings(chain, fold, values))) return failure();

      bool hadError = false;
      RewritePatternSet patterns(ctx);
      patterns.add<FoldConstRef>(ctx, fold, values, hadError);
      for (RegisteredOperationName op : ctx->getRegisteredOperations())
        if (op.getDialectNamespace() == "arith" || op.getDialectNamespace() == "scf")
          op.getCanonicalizationPatterns(patterns, ctx);
      FrozenRewritePatternSet frozen(std::move(patterns));

      (void)applyPatternsGreedily(chain, frozen);
      if (hadError) return failure();

      // Loops that index a folded constant dynamically: unroll them, so that the index becomes constant.
      SmallVector<scf::ForOp> loops;
      chain.walk([&](scf::ForOp forOp) {
        bool indexesConst = false;
        forOp.walk([&](ConstRefOp ref) { indexesConst |= ref.getIndex() && fold.contains(ref.getName()); });
        if (!indexesConst) return;
        std::optional<APInt> trips = forOp.getStaticTripCount();
        if (trips && trips->ule(static_cast<uint64_t>(maxTripCount))) loops.push_back(forOp);
      });
      for (scf::ForOp forOp : loops) // inner loops come first
        if (failed(loopUnrollFull(forOp))) return forOp.emitError("cannot unroll the loop");
      if (!loops.empty()) {
        (void)applyPatternsGreedily(chain, frozen);
        if (hadError) return failure();
      }
    }

    // What remains must be compilable.
    WalkResult walk = chain.walk([&](ConstRefOp ref) {
      if (!ref.getIndex()) return WalkResult::advance();
      ref.emitError("ops.const_ref @") << ref.getName() << " has a dynamic index that cannot be resolved: the loop "
                                       << "must be unrollable (constant trip count <= " << maxTripCount
                                       << ") and the constant must be folded (role bound)";
      return WalkResult::interrupt();
    });
    return failure(walk.wasInterrupted());
  }

  /// Replaces the chain arguments recorded in `ops.binding_consts` for folded constants by their values.
  LogicalResult foldBindings(ChainOp chain, const llvm::StringSet<> &fold, DictionaryAttr values) {
    auto bindings = chain->getAttrOfType<ArrayAttr>("ops.binding_consts");
    if (!bindings) return success();

    struct Entry {
      DictionaryAttr attr;
      StringRef name;
      int64_t offset, arg;
    };
    SmallVector<Entry> entries;
    for (Attribute a : bindings) {
      auto d = cast<DictionaryAttr>(a);
      entries.push_back({d, cast<StringAttr>(d.get("name")).getValue(),
                         cast<IntegerAttr>(d.get("offset")).getInt(), cast<IntegerAttr>(d.get("arg")).getInt()});
    }

    // Highest argument first, so that erasing one does not move the others still to do.
    SmallVector<size_t> order(entries.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return entries[a].arg > entries[b].arg; });

    SmallVector<bool> folded(entries.size(), false);
    SmallVector<int64_t> erasedArgs;
    for (size_t i : order) {
      const Entry &e = entries[i];
      if (!fold.contains(e.name)) continue;
      BlockArgument arg = chain.getArgument(e.arg);
      auto bytes = values ? values.getAs<DenseI8ArrayAttr>(e.name) : DenseI8ArrayAttr();
      if (!bytes) return chain.emitError("no value for constant @") << e.name << " in ops.const_values";
      TypedAttr attr = decode(arg.getType(), bytes.asArrayRef(), e.offset);
      if (!attr)
        return chain.emitError("constant @") << e.name << " has " << bytes.size() << " bytes: cannot read "
                                              << arg.getType() << " at byte offset " << e.offset;
      for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
        auto loop = dyn_cast<ParLoopOp>(use.getOwner());
        if (!loop) return use.getOwner()->emitError("a bound constant is used outside a loop's scalar operands");
        unsigned scalarIndex = use.getOperandNumber() - loop.getScalars().getBeginOperandIndex();
        if (scalarIndex >= loop.getScalars().size())
          return loop.emitError("a bound constant is used outside a loop's scalar operands");
        Block &body = loop.getBodyBlock();
        OpBuilder b(&body, body.begin());
        Value value = arith::ConstantOp::create(b, loop.getLoc(), attr);
        BlockArgument blockArg = loop.getScalarArg(scalarIndex);
        blockArg.replaceAllUsesWith(value);
        body.eraseArgument(loop.getNumDats() + scalarIndex);
        loop.getScalarsMutable().erase(scalarIndex);
      }
      if (failed(chain.eraseArgument(e.arg))) return chain.emitError("cannot remove a chain argument");
      folded[i] = true;
      erasedArgs.push_back(e.arg);
    }

    // The remaining entries point at moved arguments.
    Builder b(chain.getContext());
    SmallVector<Attribute> kept;
    for (size_t i = 0; i < entries.size(); ++i) {
      if (folded[i]) continue;
      int64_t arg = entries[i].arg;
      for (int64_t erased : erasedArgs)
        if (erased < entries[i].arg) --arg;
      SmallVector<NamedAttribute> attrs;
      for (NamedAttribute na : entries[i].attr)
        attrs.push_back(na.getName() == "arg" ? b.getNamedAttr("arg", b.getI64IntegerAttr(arg)) : na);
      kept.push_back(b.getDictionaryAttr(attrs));
    }
    if (kept.empty()) chain->removeAttr("ops.binding_consts");
    else chain->setAttr("ops.binding_consts", b.getArrayAttr(kept));
    return success();
  }
};

} // namespace

} // namespace ops_mlir::ops
