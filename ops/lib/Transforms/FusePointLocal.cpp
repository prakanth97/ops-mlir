#include "ops-mlir/Analysis/FusionPlanner.h"
#include "ops-mlir/Transforms/Passes.h"

#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <map>

namespace ops_mlir::ops {

#define GEN_PASS_DEF_FUSEPOINTLOCAL
#include "ops-mlir/Transforms/Passes.h.inc"

namespace {

using namespace mlir;

/// What a fused loop does to one dat: the join of what its members do.
struct FusedOperand {
  Value dat;
  bool reads = false, writes = false, inc = true; // inc: every member access is inc
  int64_t rank = 0;
  std::vector<int64_t> strides;
  std::vector<std::vector<int64_t>> points;

  AccessMode mode() const {
    if (inc && writes) return AccessMode::Inc;
    if (reads && writes) return AccessMode::RW;
    return writes ? AccessMode::Write : AccessMode::Read;
  }
};

struct FusePointLocalPass : public impl::FusePointLocalBase<FusePointLocalPass> {
  using FusePointLocalBase::FusePointLocalBase;

  void runOnOperation() override {
    if (guarded) {
      getOperation().emitError("ops-fuse-point-local: guarded fusion is not implemented yet");
      return signalPassFailure();
    }
    if (placement != "earliest" && placement != "latest") {
      getOperation().emitError("ops-fuse-point-local: unknown placement '") << placement << "' (earliest or latest)";
      return signalPassFailure();
    }
    for (ChainOp chain : getOperation().getOps<ChainOp>())
      if (failed(fuseChain(chain))) return signalPassFailure();
  }

  LogicalResult fuseChain(ChainOp chain) {
    for (Operation &op : chain.getBody().front())
      if (auto loop = dyn_cast<ParLoopOp>(&op))
        if (loop.getKernel() && loop.isBodyEmpty())
          return loop.emitError("the kernel of this loop is not materialised: run ops-materialize-kernels first");

    auto extracted = extractLoops(chain);
    if (failed(extracted)) return failure();
    std::vector<LoopInfo> &loops = *extracted;

    FusionOptions opts;
    opts.maxGroupSize = std::max<int64_t>(1, maxLoops);
    opts.reorder = reorder;
    opts.allowGuarded = false;
    opts.boxRatio = boxRatio;
    opts.placement = placement == "latest" ? FusionOptions::Placement::Latest : FusionOptions::Placement::Earliest;
    FusionPlan plan = planFusion(loops, opts);

    bool identity = true;
    for (size_t g = 0; g < plan.groups.size(); ++g)
      identity &= plan.groups[g].loops.size() == 1 && plan.groups[g].loops[0] == g;
    if (identity) return success();

    // Segments: the loops between two reductions. A segment is rebuilt right before the operation that follows
    // its last loop; reductions stay where they are, so nothing moves across them.
    std::vector<size_t> segmentOf(loops.size());
    size_t segment = 0;
    for (size_t i = 0; i < loops.size(); ++i) {
      if (loops[i].hasReduction) {
        if (i > 0 && !loops[i - 1].hasReduction) ++segment;
        segmentOf[i] = segment++;
      } else {
        segmentOf[i] = segment;
      }
    }
    std::map<size_t, Operation *> sentinel; // segment -> operation after its last loop
    for (size_t i = 0; i < loops.size(); ++i)
      if (!loops[i].hasReduction) sentinel[segmentOf[i]] = loops[i].op->getNextNode();

    DenseMap<Operation *, Range> finalRange;
    SmallVector<ParLoopOp> toErase;
    for (const FusedGroup &g : plan.groups) {
      const LoopInfo &first = loops[g.loops.front()];
      if (first.hasReduction) {
        finalRange[first.op] = first.range;
        continue;
      }
      Operation *before = sentinel[segmentOf[g.loops.front()]];
      if (g.loops.size() == 1) {
        first.op->moveBefore(before);
        finalRange[first.op] = first.range;
        continue;
      }
      OpBuilder b(before);
      ParLoopOp fused = buildFused(b, loops, g);
      if (!fused) return failure();
      finalRange[fused] = g.range;
      for (size_t l : g.loops) toErase.push_back(loops[l].op);
    }
    for (ParLoopOp old : toErase) old.erase();

    SmallVector<Attribute> ranges;
    Builder b(chain.getContext());
    for (Operation &op : chain.getBody().front())
      if (auto loop = dyn_cast<ParLoopOp>(&op)) ranges.push_back(b.getDenseI64ArrayAttr(finalRange.lookup(loop)));
    chain->setAttr("ops.range_values", b.getArrayAttr(ranges));
    return success();
  }

  ParLoopOp buildFused(OpBuilder &b, std::vector<LoopInfo> &loops, const FusedGroup &group) {
    SmallVector<ParLoopOp> members;
    for (size_t l : group.loops) members.push_back(loops[l].op);
    MLIRContext *ctx = b.getContext();

    // Union of the dat operands.
    SmallVector<FusedOperand> operands;
    DenseMap<Value, unsigned> operandOf;
    for (ParLoopOp m : members) {
      for (unsigned i = 0; i < m.getNumDats(); ++i) {
        Value dat = m.getDats()[i];
        auto [it, inserted] = operandOf.try_emplace(dat, operands.size());
        if (inserted) {
          FusedOperand o;
          o.dat = dat;
          o.rank = m.getStencil(i).getRank();
          o.strides.assign(m.getStencil(i).getStrides().begin(), m.getStencil(i).getStrides().end());
          operands.push_back(std::move(o));
        }
        FusedOperand &o = operands[it->second];
        AccessMode mode = m.getAccess(i);
        o.reads |= mode == AccessMode::Read || mode == AccessMode::RW || mode == AccessMode::Inc ||
                   mode == AccessMode::Min || mode == AccessMode::Max;
        o.writes |= mode != AccessMode::Read;
        o.inc &= mode == AccessMode::Inc;
        StencilAttr stencil = m.getStencil(i);
        if (!std::equal(o.strides.begin(), o.strides.end(), stencil.getStrides().begin(), stencil.getStrides().end())) {
          m.emitError("the same dat is accessed with different strides in loops that would be fused");
          return nullptr;
        }
        for (int64_t p = 0; p < stencil.getNumPoints(); ++p) {
          std::vector<int64_t> point(stencil.getPoint(p).begin(), stencil.getPoint(p).end());
          if (std::find(o.points.begin(), o.points.end(), point) == o.points.end()) o.points.push_back(point);
        }
      }
    }
    // A write-only mode is only right if no member reads (checked above through `reads`); the join of read and
    // write is rw.

    SmallVector<Value> datValues, scalarValues;
    SmallVector<Attribute> accesses, stencils;
    SmallVector<Type> blockTypes;
    for (FusedOperand &o : operands) {
      datValues.push_back(o.dat);
      accesses.push_back(AccessModeAttr::get(ctx, o.mode()));
      SmallVector<int64_t> flat;
      for (auto &p : o.points) flat.append(p.begin(), p.end());
      stencils.push_back(StencilAttr::get(ctx, o.rank, flat, o.strides));
      auto dat = cast<DatType>(o.dat.getType());
      blockTypes.push_back(AccType::get(ctx, dat.getElementType(), dat.getRank(), dat.getNcomp()));
    }
    DenseMap<Value, unsigned> scalarOf;
    for (ParLoopOp m : members)
      for (Value s : m.getScalars())
        if (scalarOf.try_emplace(s, scalarValues.size()).second) {
          scalarValues.push_back(s);
          blockTypes.push_back(s.getType());
        }

    ParLoopOp first = members.front();
    SmallVector<Location> memberLocs;
    for (ParLoopOp m : members) memberLocs.push_back(m.getLoc());
    Location loc = FusedLoc::get(ctx, memberLocs);
    SmallVector<Value> range(first.getRange().begin(), first.getRange().end());
    auto fused = ParLoopOp::create(b, loc, TypeRange{}, range, datValues, scalarValues, FlatSymbolRefAttr(),
                                   b.getI64IntegerAttr(-1), b.getI64IntegerAttr(first.getBlock()),
                                   b.getArrayAttr(accesses), b.getArrayAttr(stencils), ArrayAttr());
    SmallVector<Attribute> memberIds;
    for (ParLoopOp m : members) memberIds.push_back(b.getI64IntegerAttr(static_cast<int64_t>(m.getLoopId())));
    fused->setAttr("ops.fused_loops", b.getArrayAttr(memberIds));

    SmallVector<Location> argLocs(blockTypes.size(), loc);
    OpBuilder::InsertionGuard guard(b);
    Block *body = b.createBlock(&fused.getBody(), {}, blockTypes, argLocs);
    for (ParLoopOp m : members) {
      IRMapping mapper;
      for (unsigned i = 0; i < m.getNumDats(); ++i)
        mapper.map(m.getAccessorArg(i), body->getArgument(operandOf.lookup(m.getDats()[i])));
      for (unsigned i = 0; i < m.getScalars().size(); ++i)
        mapper.map(m.getScalarArg(i), body->getArgument(operands.size() + scalarOf.lookup(m.getScalars()[i])));
      for (Operation &op : m.getBodyBlock().without_terminator()) b.clone(op, mapper);
    }
    YieldOp::create(b, loc);
    return fused;
  }
};

} // namespace

} // namespace ops_mlir::ops
