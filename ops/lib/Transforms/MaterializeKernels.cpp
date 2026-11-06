#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"
#include "ops-mlir/Transforms/Passes.h"
#include "ops-mlir/Transforms/StoreForwarding.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"

#include <tuple>

namespace ops_mlir::ops {

#define GEN_PASS_DEF_MATERIALIZEKERNELS
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;


struct ChainState {
  ChainOp chain;
  std::map<std::pair<std::string, std::pair<int64_t, const void *>>, unsigned> argOf; // key -> chain arg index
  SmallVector<Attribute> bindings;
};

struct MaterializeKernelsPass : public impl::MaterializeKernelsBase<MaterializeKernelsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    for (ChainOp chain : module.getOps<ChainOp>())
      if (failed(materializeChain(chain, symbols))) return signalPassFailure();
  }

  LogicalResult materializeChain(ChainOp chain, SymbolTable &symbols) {
    ChainState state;
    state.chain = chain;
    if (auto existing = chain->getAttrOfType<ArrayAttr>("ops.binding_consts"))
      state.bindings.assign(existing.begin(), existing.end());

    SmallVector<ParLoopOp> loops;
    for (Operation &op : chain.getBody().front())
      if (auto loop = dyn_cast<ParLoopOp>(&op)) loops.push_back(loop);

    for (ParLoopOp loop : loops) {
      if (!loop.getKernel() || !loop.isBodyEmpty()) continue;
      auto kernel = symbols.lookup<KernelOp>(*loop.getKernel());
      if (!kernel) return loop.emitError("no ops.kernel named @") << *loop.getKernel();
      if (failed(materializeLoop(loop, kernel, state))) return failure();
    }
    if (!state.bindings.empty()) chain->setAttr("ops.binding_consts", ArrayAttr::get(chain.getContext(), state.bindings));
    return success();
  }

  /// The chain argument that carries constant (`name`, `offset`, `type`), added on first use.
  unsigned constArgument(ChainState &state, StringRef name, int64_t offset, Type type, Location loc) {
    auto key = std::make_pair(name.str(), std::make_pair(offset, type.getAsOpaquePointer()));
    auto it = state.argOf.find(key);
    if (it != state.argOf.end()) return it->second;
    ChainOp chain = state.chain;
    unsigned index = chain.getNumArguments();
    (void)chain.insertArgument(index, type, DictionaryAttr(), loc);
    MLIRContext *ctx = chain.getContext();
    Builder b(ctx);
    state.bindings.push_back(b.getDictionaryAttr({b.getNamedAttr("name", b.getStringAttr(name)),
                                                  b.getNamedAttr("offset", b.getI64IntegerAttr(offset)),
                                                  b.getNamedAttr("arg", b.getI64IntegerAttr(index))}));
    state.argOf[key] = index;
    return index;
  }

  LogicalResult materializeLoop(ParLoopOp loop, KernelOp kernel, ChainState &state) {
    if (!kernel.getBody().hasOneBlock())
      return kernel.emitError("kernel @") << kernel.getSymName() << " must have a single block";
    Block &kernelBlock = kernel.getBody().front();
    Block &body = loop.getBodyBlock();

    // Kernel parameters by kind, in order.
    struct Gbl {
      unsigned base, size;
    };
    DenseMap<Value, Value> paramToBlockArg; // accessor / reduction parameters
    DenseMap<Value, Gbl> gbls;
    unsigned acc = 0, red = 0, gblBase = 0;
    for (BlockArgument p : kernelBlock.getArguments()) {
      if (isa<AccType>(p.getType())) {
        if (acc >= loop.getNumDats())
          return loop.emitError("kernel @") << kernel.getSymName() << " has more accessor parameters than the loop has dats";
        BlockArgument target = loop.getAccessorArg(acc++);
        // The kernel only knows the components it uses: its accessor may have fewer than the dat.
        auto kernelAcc = cast<AccType>(p.getType());
        auto loopAcc = cast<AccType>(target.getType());
        if (kernelAcc.getElementType() != loopAcc.getElementType() || kernelAcc.getRank() != loopAcc.getRank() ||
            kernelAcc.getNcomp() > loopAcc.getNcomp())
          return loop.emitError("kernel @") << kernel.getSymName() << " parameter " << p.getArgNumber() << " has type "
                                            << p.getType() << " but the dat operand has accessor type " << target.getType();
        paramToBlockArg[p] = target;
      } else if (auto gbl = dyn_cast<GblType>(p.getType())) {
        gbls[p] = {gblBase, static_cast<unsigned>(gbl.getSize())};
        gblBase += gbl.getSize();
      } else if (isa<RedType>(p.getType())) {
        if (red >= loop->getNumResults())
          return loop.emitError("kernel @") << kernel.getSymName() << " has more reduction parameters than the loop has results";
        BlockArgument target = loop.getReductionArg(red++);
        if (target.getType() != p.getType())
          return loop.emitError("kernel @") << kernel.getSymName() << " parameter " << p.getArgNumber() << " has type "
                                            << p.getType() << " but the loop's reduction has type " << target.getType();
        paramToBlockArg[p] = target;
      }
    }
    if (acc != loop.getNumDats())
      return loop.emitError("kernel @") << kernel.getSymName() << " has " << acc << " accessor parameters but the loop has "
                                        << loop.getNumDats() << " dats";
    if (red != loop->getNumResults())
      return loop.emitError("kernel @") << kernel.getSymName() << " has " << red << " reduction parameters but the loop has "
                                        << loop->getNumResults() << " results";
    if (gblBase != loop.getScalars().size())
      return loop.emitError("kernel @") << kernel.getSymName() << " takes " << gblBase
                                        << " global components but the loop passes " << loop.getScalars().size() << " scalars";

    IRMapping mapper;
    for (auto &[param, target] : paramToBlockArg) mapper.map(param, target);

    OpBuilder b(loop.getContext());
    b.setInsertionPoint(body.getTerminator());
    DenseMap<std::pair<StringRef, std::pair<int64_t, const void *>>, Value> constInLoop;

    // The loop operand (and block argument) of a constant at a fixed offset, one per distinct constant.
    auto constValue = [&](ConstRefOp ref) -> Value {
      auto key = std::make_pair(ref.getName(), std::make_pair(static_cast<int64_t>(ref.getOffset()),
                                                            ref.getType().getAsOpaquePointer()));
      auto found = constInLoop.find(key);
      if (found == constInLoop.end()) {
        unsigned chainArg = constArgument(state, ref.getName(), static_cast<int64_t>(ref.getOffset()), ref.getType(), ref.getLoc());
        // New scalar operand, and its block argument after the existing scalars.
        unsigned pos = loop.getNumDats() + loop.getScalars().size();
        loop.getScalarsMutable().append(state.chain.getArgument(chainArg));
        BlockArgument blockArg = body.insertArgument(pos, ref.getType(), ref.getLoc());
        found = constInLoop.try_emplace(key, blockArg).first;
      }
      return found->second;
    };
    auto globalValue = [&](GblReadOp read, Value &out) -> LogicalResult {
      auto it = gbls.find(read.getGbl());
      if (it == gbls.end()) return read.emitError("global read of a value that is not a kernel parameter");
      BlockArgument scalar = loop.getScalarArg(it->second.base + read.getIndex());
      if (scalar.getType() != read.getType())
        return read.emitError("global component has type ") << read.getType() << " but the loop passes " << scalar.getType();
      out = scalar;
      return success();
    };

    for (Operation &op : kernelBlock.without_terminator()) {
      if (auto read = dyn_cast<GblReadOp>(&op)) {
        Value scalar;
        if (failed(globalValue(read, scalar))) return failure();
        mapper.map(read.getResult(), scalar);
        continue;
      }
      if (auto ref = dyn_cast<ConstRefOp>(&op); ref && !ref.getIndex()) {
        mapper.map(ref.getResult(), constValue(ref));
        continue;
      }
      Operation *cloned = b.clone(op, mapper);
      // Constants and globals read inside the regions of the operation (under an `if`, in a loop).
      SmallVector<Operation *> nested;
      cloned->walk([&](Operation *n) {
        if (n != cloned && isa<ConstRefOp, GblReadOp>(n)) nested.push_back(n);
      });
      for (Operation *n : nested) {
        if (auto read = dyn_cast<GblReadOp>(n)) {
          Value scalar;
          if (failed(globalValue(read, scalar))) return failure();
          read.getResult().replaceAllUsesWith(scalar);
          read->erase();
        } else if (auto ref = cast<ConstRefOp>(n); !ref.getIndex()) {
          ref.getResult().replaceAllUsesWith(constValue(ref));
          ref->erase();
        }
      }
    }
    // The loop is now valid or not: report what is wrong with the kernel against this loop's stencils.
    if (failed(verify(loop))) return failure();
    // A kernel may read back what it wrote to an output: use the stored values, and diagnose the reads of write-only
    // operands that no store feeds.
    forwardStoresInLoop(loop);
    for (unsigned i = 0; i < loop.getNumDats(); ++i) {
      if (loop.getAccess(i) != AccessMode::Write) continue;
      for (Operation *user : loop.getAccessorArg(i).getUsers())
        if (isa<ReadOp, ReadDynOp>(user))
          return user->emitError("kernel @") << kernel.getSymName() << " reads the write-only dat operand #" << i
                                             << " before it has written that point";
    }
    return success();
  }
};

} // namespace


} // namespace ops_mlir::ops
