#include "ops-mlir/Conversion/Passes.h"
#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/DenseMap.h"

#include <string>

namespace ops_mlir::ops {

#define GEN_PASS_DEF_CONVERTOPSTOSCF
#include "ops-mlir/Conversion/Passes.h.inc"

namespace {

using namespace mlir;


/// A dat accessor of the loop being lowered.
struct Accessor {
  unsigned slot;                   // distinct dat (index into the function's memref arguments)
  Value memref;                    // the function argument
  SmallVector<int64_t> strides;    // stencil strides, OPS axis order
  int64_t ncomp;
};

/// Per reduction of the loop: the per-point accumulator and how to combine.
struct Accumulator {
  Value memref;
  ReductionKind kind;
  Type type;
};

Value neutralElement(OpBuilder &b, Location loc, ReductionKind kind, Type type) {
  if (auto ft = dyn_cast<FloatType>(type)) {
    const llvm::fltSemantics &sem = ft.getFloatSemantics();
    APFloat v(sem);
    switch (kind) {
    case ReductionKind::Add: v = APFloat::getZero(sem); break;
    case ReductionKind::Min: v = APFloat::getInf(sem); break;
    case ReductionKind::Max: v = APFloat::getInf(sem, /*Negative=*/true); break;
    }
    return arith::ConstantOp::create(b, loc, FloatAttr::get(type, v));
  }
  unsigned w = type.getIntOrFloatBitWidth();
  APInt v(w, 0);
  if (kind == ReductionKind::Min) v = APInt::getSignedMaxValue(w);
  if (kind == ReductionKind::Max) v = APInt::getSignedMinValue(w);
  return arith::ConstantOp::create(b, loc, IntegerAttr::get(type, v));
}

/// acc (op) v, with the C semantics of OPS_MIN / OPS_MAX (`a < b ? a : b`).
Value combine(OpBuilder &b, Location loc, ReductionKind kind, Value acc, Value v) {
  bool isFloat = isa<FloatType>(acc.getType());
  switch (kind) {
  case ReductionKind::Add:
    return isFloat ? Value(arith::AddFOp::create(b, loc, acc, v)) : Value(arith::AddIOp::create(b, loc, acc, v));
  case ReductionKind::Min:
  case ReductionKind::Max: {
    bool min = kind == ReductionKind::Min;
    Value cond = isFloat ? Value(arith::CmpFOp::create(b, loc, min ? arith::CmpFPredicate::OLT : arith::CmpFPredicate::OGT, acc, v))
                         : Value(arith::CmpIOp::create(b, loc, min ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt, acc, v));
    return arith::SelectOp::create(b, loc, cond, acc, v);
  }
  }
  llvm_unreachable("unknown reduction kind");
}

/// The memref type of a dat argument: dynamic sizes and strides, or (mode=shapes) the static ones of `sizes` (OPS order, x
/// first); the offset (the base shift of the dat) stays dynamic.
MemRefType datMemRefType(DatType dat, bool soa, ArrayRef<int64_t> sizes = {}) {
  MLIRContext *ctx = dat.getContext();
  int64_t rank = dat.getRank(), ncomp = dat.getNcomp();
  const bool known = static_cast<int64_t>(sizes.size()) == rank;
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic), strides(rank, ShapedType::kDynamic);
  // The last memref dimension of the block is x, the fastest: unit stride, or ncomp when the components of a point
  // are adjacent.
  int64_t stride = (ncomp > 1 && !soa) ? ncomp : 1, points = 1;
  strides[rank - 1] = stride;
  if (known) {
    for (int64_t d = 0; d < rank; ++d) {
      shape[rank - 1 - d] = sizes[d];
      strides[rank - 1 - d] = stride;
      stride *= sizes[d];
      points *= sizes[d];
    }
  }
  if (ncomp > 1) {
    shape.push_back(ncomp);
    strides.push_back(soa ? (known ? points : ShapedType::kDynamic) : 1);
  }
  return MemRefType::get(shape, dat.getElementType(), StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides));
}

/// The lowering of one group (an ops.chain with a single ops.par_loop) to a func.func. The phases are the methods
/// `run` calls, in that order; the state they share is the members.
class GroupLowering {
public:
  struct Config {
    bool soa;
    std::string target; // cpu or gpu
    std::string block;  // gpu: threads per dimension, x,y,z
    bool affine;        // cpu: affine.parallel / affine.for and affine accesses (loops without reductions)
  };

  GroupLowering(ChainOp chain, ModuleOp module, ParLoopOp loop, Config config, unsigned &formatCounter)
      : chain(chain), module(module), loop(loop), config(std::move(config)), formatCounter(formatCounter),
        ctx(chain.getContext()), loc(chain.getLoc()), rank(loop.getRank()), chainBlock(chain.getBody().front()),
        gpu(this->config.target == "gpu"), b(chain) {}

  LogicalResult run() {
    if (failed(collectArguments())) return failure();
    if (failed(createFunction())) return failure();
    cloneChainOps();
    if (failed(buildLoopShell())) return failure();
    bindAccessors();
    createAccumulators();
    openInnerLoop();
    cloneLoopBody();
    if (failed(rewriteBody())) return failure();
    finishReductions();
    if (failed(emitReturn())) return failure();
    chain.erase();
    return success();
  }

private:
  ChainOp chain;
  ModuleOp module;
  ParLoopOp loop;
  Config config;
  unsigned &formatCounter;
  MLIRContext *ctx;
  Location loc;
  unsigned rank;
  Block &chainBlock;
  const bool gpu;
  OpBuilder b;

  // The ABI (see the pass description).
  SmallVector<BlockArgument> datArgs, scalarArgs;
  DenseMap<Value, unsigned> datSlot, scalarSlot;
  unsigned numDats = 0;
  unsigned shiftBase = 0, rangeBase = 0, dispBase = 0, scalarBase = 0, bufferBase = 0;
  func::FuncOp func;
  Block *entry = nullptr;
  IRMapping mapper;
  Value one;

  // The shell around the body of one point: where the body goes (`body`), the coordinate of the point in each
  // dimension (`iv`), and what to look through to find the ops to rewrite (`walkRoot`).
  Block *body = nullptr;
  Operation *walkRoot = nullptr;
  scf::ParallelOp parallel;
  affine::AffineParallelOp affineParallel;
  bool useAffine = false;
  Value innerIv;
  SmallVector<Value, 3> gpuIvs;
  Operation *gpuActive = nullptr; // the scf.if that masks the threads outside the range
  gpu::LaunchOp gpuLaunch;
  SmallVector<int64_t, 3> gpuBlock;
  bool innerLoop = false;

  // What the loop body works on.
  SmallVector<ReductionKind> kinds;
  SmallVector<Accessor> accessors;
  SmallVector<Accumulator> accumulators;

  // --- ABI --------------------------------------------------------------------------------------------------------

  /// The dats and scalars of the loop must be arguments of the chain.
  LogicalResult collectArguments() {
    auto chainArg = [&](Value v, StringRef what) -> BlockArgument {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg || arg.getOwner() != &chainBlock) {
        loop.emitError("a ") << what << " operand of the loop is not an argument of the chain";
        return nullptr;
      }
      return arg;
    };
    for (Value dat : loop.getDats()) {
      BlockArgument arg = chainArg(dat, "dat");
      if (!arg) return failure();
      if (datSlot.try_emplace(arg, datArgs.size()).second) datArgs.push_back(arg);
    }
    for (Value s : loop.getScalars()) {
      BlockArgument arg = chainArg(s, "scalar");
      if (!arg) return failure();
      if (scalarSlot.try_emplace(arg, scalarArgs.size()).second) scalarArgs.push_back(arg);
    }
    numDats = datArgs.size();
    return success();
  }

  /// The function: its signature, the attributes the runtime reads, and an entry block to build into.
  LogicalResult createFunction() {
    SmallVector<Type> inputs;
    auto shapes = chain->getAttrOfType<ArrayAttr>("ops.shape_values"); // mode=shapes: static memref shapes
    for (BlockArgument d : datArgs) {
      ArrayRef<int64_t> sizes;
      if (shapes && d.getArgNumber() < shapes.size())
        if (auto a = dyn_cast<DenseI64ArrayAttr>(shapes[d.getArgNumber()])) sizes = a.asArrayRef();
      inputs.push_back(datMemRefType(cast<DatType>(d.getType()), config.soa, sizes));
    }
    Type indexType = IndexType::get(ctx);
    for (unsigned i = 0; i < numDats * rank + 2 * rank + rank; ++i) inputs.push_back(indexType);
    for (BlockArgument s : scalarArgs) inputs.push_back(s.getType());
    SmallVector<Type> results(loop->getResultTypes());
    if (config.target != "cpu" && !gpu)
      return chain.emitError("option 'target' must be cpu or gpu, got '") << config.target << "'";
    if (gpu) {
      // The results come back through device buffers: one rank-0 memref per reduction, after the scalars.
      for (Type t : results) inputs.push_back(MemRefType::get({ShapedType::kDynamic}, t));
      results.clear();
    }

    shiftBase = numDats;
    rangeBase = shiftBase + numDats * rank;
    dispBase = rangeBase + 2 * rank;
    scalarBase = dispBase + rank;
    bufferBase = scalarBase + scalarArgs.size();

    // The function is named by `ops.func_name` (the runtime picks group_<hash>) or the chain; a declaration of that
    // name, as a caller in the same module needs, is replaced by the definition.
    StringRef name = chain.getSymName();
    if (auto given = chain->getAttrOfType<StringAttr>("ops.func_name")) name = given.getValue();
    if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
      if (!existing.isExternal()) return chain.emitError("a function named @") << name << " already exists";
      existing.erase();
    }
    b.setInsertionPoint(chain);
    func = func::FuncOp::create(b, loc, name, FunctionType::get(ctx, inputs, results));
    func.setPublic();
    func->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
    for (unsigned i = 0; i < numDats; ++i) func.setArgAttr(i, "llvm.noalias", UnitAttr::get(ctx));
    Builder bb(ctx);
    // Position of the aligned pointer of each dat once the memref arguments are expanded to descriptors (allocated
    // pointer, aligned pointer, offset, sizes, strides).
    SmallVector<int64_t> noaliasArgs;
    for (unsigned pos = 0; Type t : inputs) {
      auto memref = dyn_cast<MemRefType>(t);
      if (!memref) break;
      noaliasArgs.push_back(pos + 1);
      pos += 3 + 2 * memref.getRank();
    }
    func->setAttr("ops.abi", bb.getDictionaryAttr({bb.getNamedAttr("noalias_args", bb.getDenseI64ArrayAttr(noaliasArgs)),
                                                   bb.getNamedAttr("rank", bb.getI64IntegerAttr(rank)),
                                                   bb.getNamedAttr("num_dats", bb.getI64IntegerAttr(numDats)),
                                                   bb.getNamedAttr("num_scalars", bb.getI64IntegerAttr(scalarArgs.size())),
                                                   bb.getNamedAttr("num_reductions", bb.getI64IntegerAttr(loop->getNumResults())),
                                                   bb.getNamedAttr("gpu", bb.getBoolAttr(gpu)),
                                                   bb.getNamedAttr("soa", bb.getBoolAttr(config.soa))}));
    entry = func.addEntryBlock();
    b.setInsertionPointToEnd(entry);
    return success();
  }

  /// The ops of the chain other than the loop (constants...), and the loop's scalars, mapped to the arguments.
  void cloneChainOps() {
    for (auto [i, s] : llvm::enumerate(scalarArgs)) mapper.map(s, entry->getArgument(scalarBase + i));
    for (Operation &op : chainBlock) {
      if (isa<ParLoopOp, ReturnOp>(&op)) continue;
      b.clone(op, mapper);
    }
    for (unsigned r = 0; r < loop->getNumResults(); ++r) kinds.push_back(loop.getReductionKind(r));
    one = arith::ConstantIndexOp::create(b, loc, 1);
  }

  // --- the loop shell ---------------------------------------------------------------------------------------------

  /// A range operand that is a constant (ops-specialize) is used as such; otherwise it is the function argument.
  Value rangeValue(unsigned slot) {
    Value operand = loop.getRange()[slot];
    if (operand.getDefiningOp() && matchPattern(operand, m_Constant())) return mapper.lookup(operand);
    return entry->getArgument(rangeBase + slot);
  }

  /// The coordinate of the point in dimension `d` (OPS axis order).
  Value iv(unsigned d) {
    if (gpu) return gpuIvs[d];
    if (innerLoop && d == 0) return innerIv;
    return body->getArgument(rank - 1 - d);
  }

  LogicalResult buildLoopShell() {
    innerLoop = !gpu && rank > 1;
    if (gpu) return buildGpuShell();
    // A reduction is a different construct in affine.parallel (reductions with results) and the accumulators of
    // ops-hoist-accumulators belong to scf.parallel / omp.parallel: loops with reductions keep the scf form.
    useAffine = config.affine && loop->getNumResults() == 0;
    if (config.affine) func->setAttr("ops.loops", StringAttr::get(ctx, useAffine ? "affine" : "scf"));
    if (useAffine) buildAffineCpuShell();
    else buildCpuShell();
    return success();
  }

  /// The bound of an affine loop that is a single value: a function argument or a constant, both symbols.
  AffineMap symbolBound() { return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1, getAffineSymbolExpr(0, ctx)); }

  /// The shell of buildCpuShell with affine.parallel (z, y) and, in openInnerLoop, an affine.for (x). The bounds are single values
  /// (range arguments or constants: symbols). affine.parallel takes one bound map per dimension and ONE list of operands that every
  /// map sees: map i is the symbol i of the list.
  void buildAffineCpuShell() {
    SmallVector<Value> lbOperands, ubOperands;
    SmallVector<int64_t> steps;
    for (int d = rank - 1; d >= (innerLoop ? 1 : 0); --d) { // outermost (z) first
      lbOperands.push_back(rangeValue(2 * d));
      ubOperands.push_back(rangeValue(2 * d + 1));
      steps.push_back(1);
    }
    const unsigned n = steps.size();
    SmallVector<AffineMap> lbMaps, ubMaps;
    for (unsigned i = 0; i < n; ++i) {
      lbMaps.push_back(AffineMap::get(/*dimCount=*/0, /*symbolCount=*/n, getAffineSymbolExpr(i, ctx)));
      ubMaps.push_back(AffineMap::get(/*dimCount=*/0, /*symbolCount=*/n, getAffineSymbolExpr(i, ctx)));
    }
    affineParallel = affine::AffineParallelOp::create(b, loc, TypeRange{}, ArrayRef<arith::AtomicRMWKind>{}, lbMaps, lbOperands, ubMaps,
                                                      ubOperands, steps);
    body = affineParallel.getBody();
    walkRoot = affineParallel;
    b.setInsertionPointToStart(body);
  }

  /// The outer dimensions (z, y) run in parallel; x, the contiguous one, is a sequential loop inside, which the
  /// compiler vectorises. (With one dimension only, that one is the parallel loop.)
  void buildCpuShell() {
    SmallVector<Value> lbs, ubs, steps, inits;
    for (int d = rank - 1; d >= (innerLoop ? 1 : 0); --d) { // outermost (z) first
      lbs.push_back(rangeValue(2 * d));
      ubs.push_back(rangeValue(2 * d + 1));
      steps.push_back(one);
    }
    for (unsigned r = 0; r < loop->getNumResults(); ++r)
      inits.push_back(neutralElement(b, loc, kinds[r], loop->getResult(r).getType()));
    parallel = scf::ParallelOp::create(b, loc, lbs, ubs, steps, inits);
    body = parallel.getBody();
    walkRoot = parallel;
    b.setInsertionPointToStart(body);
  }

  /// Thread block and grid: x is the fastest dimension of memory and of the block.
  LogicalResult buildGpuShell() {
    SmallVector<int64_t, 3> bs;
    for (StringRef part : llvm::split(config.block, ',')) {
      int64_t v;
      if (part.trim().getAsInteger(10, v) || v < 1)
        return chain.emitError("option 'block' must be positive integers x,y,z, got '") << config.block << "'";
      bs.push_back(v);
    }
    bs.resize(3, 1);
    // The dimensions the block does not have are folded into x (a 1-D block gets all the threads of the tile).
    for (unsigned d = rank; d < 3; ++d) {
      bs[0] *= bs[d];
      bs[d] = 1;
    }
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    SmallVector<Value, 3> extents, grid(3, one), blockSize;
    Value nonEmpty;
    for (unsigned d = 0; d < 3; ++d) blockSize.push_back(arith::ConstantIndexOp::create(b, loc, bs[d]));
    for (unsigned d = 0; d < rank; ++d) {
      Value extent = arith::SubIOp::create(b, loc, rangeValue(2 * d + 1), rangeValue(2 * d));
      Value positive = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, extent, zero);
      nonEmpty = nonEmpty ? Value(arith::AndIOp::create(b, loc, nonEmpty, positive)) : positive;
      grid[d] = arith::CeilDivSIOp::create(b, loc, arith::MaxSIOp::create(b, loc, extent, zero), blockSize[d]);
    }
    auto nonEmptyIf = scf::IfOp::create(b, loc, nonEmpty, /*withElseRegion=*/false);
    b.setInsertionPointToStart(&nonEmptyIf.getThenRegion().front());
    auto launch = gpu::LaunchOp::create(b, loc, grid[0], grid[1], grid[2], blockSize[0], blockSize[1], blockSize[2]);
    b.setInsertionPointToStart(&launch.getBody().front());
    gpu::TerminatorOp::create(b, loc);
    b.setInsertionPoint(launch.getBody().front().getTerminator());
    Value inRange;
    auto blockIds = launch.getBlockIds();
    auto threadIds = launch.getThreadIds();
    Value ids[3][2] = {{blockIds.x, threadIds.x}, {blockIds.y, threadIds.y}, {blockIds.z, threadIds.z}};
    for (unsigned d = 0; d < rank; ++d) {
      Value base = arith::MulIOp::create(b, loc, ids[d][0], blockSize[d]);
      Value local = arith::AddIOp::create(b, loc, base, ids[d][1]);
      Value index = arith::AddIOp::create(b, loc, local, rangeValue(2 * d));
      gpuIvs.push_back(index);
      Value inside = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, index, rangeValue(2 * d + 1));
      inRange = inRange ? Value(arith::AndIOp::create(b, loc, inRange, inside)) : inside;
    }
    auto active = scf::IfOp::create(b, loc, inRange, /*withElseRegion=*/false);
    body = &active.getThenRegion().front();
    walkRoot = launch;
    gpuActive = active;
    gpuLaunch = launch;
    gpuBlock.assign(bs.begin(), bs.end());
    b.setInsertionPointToStart(body);
    return success();
  }

  // --- the body of one point --------------------------------------------------------------------------------------

  /// The accessors of the loop as values the cloned body can use until it is rewritten (placeholders: casts of the memref).
  void bindAccessors() {
    Builder bb(ctx);
    for (unsigned i = 0; i < loop.getNumDats(); ++i) {
      unsigned slot = datSlot.lookup(loop.getDats()[i]);
      StencilAttr stencil = loop.getStencil(i);
      Accessor a;
      a.slot = slot;
      a.memref = entry->getArgument(slot);
      a.strides.assign(stencil.getStrides().begin(), stencil.getStrides().end());
      a.ncomp = cast<DatType>(loop.getDats()[i].getType()).getNcomp();
      accessors.push_back(a);
      auto cast = UnrealizedConversionCastOp::create(b, loc, loop.getAccessorArg(i).getType(), a.memref);
      cast->setAttr("ops.accessor", bb.getI64IntegerAttr(i));
      mapper.map(loop.getAccessorArg(i), cast.getResult(0));
    }
    for (unsigned i = 0; i < loop.getScalars().size(); ++i)
      mapper.map(loop.getScalarArg(i), entry->getArgument(scalarBase + scalarSlot.lookup(loop.getScalars()[i])));
  }

  /// The per-point accumulator of each reduction (a rank-0 memref the body combines into).
  void createAccumulators() {
    Builder bb(ctx);
    for (unsigned r = 0; r < loop->getNumResults(); ++r) {
      Type t = loop->getResult(r).getType();
      // On the GPU every thread of a block takes part in the reduction after the body: the accumulators live outside the
      // mask of the threads that are in the range.
      OpBuilder::InsertionGuard accumulatorGuard(b);
      if (gpu) b.setInsertionPoint(gpuActive);
      auto allocaOp = memref::AllocaOp::create(b, loc, MemRefType::get({}, t));
      // Re-initialised per iteration of the parallel loop: ops-hoist-accumulators moves it out of the loop.
      allocaOp->setAttr("ops.accumulator", UnitAttr::get(ctx));
      Value alloca = allocaOp;
      memref::StoreOp::create(b, loc, neutralElement(b, loc, kinds[r], t), alloca, ValueRange{});
      accumulators.push_back({alloca, kinds[r], t});
      auto cast = UnrealizedConversionCastOp::create(b, loc, loop.getReductionArg(r).getType(), alloca);
      cast->setAttr("ops.reduction", bb.getI64IntegerAttr(r));
      mapper.map(loop.getReductionArg(r), cast.getResult(0));
    }
  }

  void openInnerLoop() {
    if (!innerLoop) return;
    if (useAffine) {
      auto inner = affine::AffineForOp::create(b, loc, ValueRange{rangeValue(0)}, symbolBound(), ValueRange{rangeValue(1)}, symbolBound());
      innerIv = inner.getInductionVar();
      b.setInsertionPointToStart(inner.getBody());
      return;
    }
    auto inner = scf::ForOp::create(b, loc, rangeValue(0), rangeValue(1), one);
    innerIv = inner.getInductionVar();
    b.setInsertionPointToStart(inner.getBody());
  }

  void cloneLoopBody() {
    for (Operation &op : loop.getBodyBlock().without_terminator()) b.clone(op, mapper);
  }

  const Accessor *accessorOf(Value acc) {
    auto cast = acc.getDefiningOp<UnrealizedConversionCastOp>();
    if (!cast) return nullptr;
    auto idx = cast->getAttrOfType<IntegerAttr>("ops.accessor");
    return idx ? &accessors[idx.getInt()] : nullptr;
  }

  /// Memref indices of an access at constant offsets (dynamic offsets: `dyn`), memory order.
  SmallVector<Value> memrefIndices(OpBuilder &rb, Location l, const Accessor &a, ArrayRef<int64_t> offsets,
                                   ValueRange dyn, int64_t comp) {
    SmallVector<Value> idx(rank + (a.ncomp > 1 ? 1 : 0));
    for (unsigned d = 0; d < rank; ++d) {
      Value v = entry->getArgument(shiftBase + a.slot * rank + d);
      if (a.strides[d] != 0) v = arith::AddIOp::create(rb, l, v, iv(d));
      if (!offsets.empty() && offsets[d] != 0) v = arith::AddIOp::create(rb, l, v, arith::ConstantIndexOp::create(rb, l, offsets[d]));
      if (!dyn.empty()) {
        Value o = dyn[d];
        if (!o.getType().isIndex()) o = arith::IndexCastOp::create(rb, l, rb.getIndexType(), o);
        v = arith::AddIOp::create(rb, l, v, o);
      }
      idx[rank - 1 - d] = v;
    }
    if (a.ncomp > 1) idx[rank] = arith::ConstantIndexOp::create(rb, l, comp);
    return idx;
  }

  /// The access at constant offsets as an affine map of (the coordinates; the shifts): per memory dimension
  /// shift_d + i_d (unless the stencil stride along d is 0) + offset_d, the component last.
  std::pair<AffineMap, SmallVector<Value>> affineAccess(const Accessor &a, ArrayRef<int64_t> offsets, int64_t comp) {
    SmallVector<Value> operands;
    for (unsigned d = 0; d < rank; ++d) operands.push_back(iv(d));
    for (unsigned d = 0; d < rank; ++d) operands.push_back(entry->getArgument(shiftBase + a.slot * rank + d));
    SmallVector<AffineExpr> exprs(rank + (a.ncomp > 1 ? 1 : 0));
    for (unsigned d = 0; d < rank; ++d) {
      AffineExpr e = getAffineSymbolExpr(d, ctx);
      if (a.strides[d] != 0) e = e + getAffineDimExpr(d, ctx);
      if (!offsets.empty() && offsets[d] != 0) e = e + offsets[d];
      exprs[rank - 1 - d] = e;
    }
    if (a.ncomp > 1) exprs[rank] = getAffineConstantExpr(comp, ctx);
    return {AffineMap::get(rank, rank, exprs, ctx), operands};
  }

  LogicalResult lowerRead(OpBuilder &rb, ReadOp read) {
    const Accessor *a = accessorOf(read.getAcc());
    if (!a) return read.emitError("access through a value that is not an accessor of the loop");
    if (useAffine) {
      auto [map, operands] = affineAccess(*a, read.getOffsets(), read.getComp());
      auto load = affine::AffineLoadOp::create(rb, read.getLoc(), a->memref, map, operands);
      read.getResult().replaceAllUsesWith(load);
      return success();
    }
    auto load = memref::LoadOp::create(rb, read.getLoc(), a->memref, memrefIndices(rb, read.getLoc(), *a, read.getOffsets(), {}, read.getComp()));
    read.getResult().replaceAllUsesWith(load);
    return success();
  }

  LogicalResult lowerReadDyn(OpBuilder &rb, ReadDynOp dyn) {
    const Accessor *a = accessorOf(dyn.getAcc());
    if (!a) return dyn.emitError("access through a value that is not an accessor of the loop");
    auto load = memref::LoadOp::create(rb, dyn.getLoc(), a->memref, memrefIndices(rb, dyn.getLoc(), *a, {}, dyn.getOffsets(), dyn.getComp()));
    dyn.getResult().replaceAllUsesWith(load);
    return success();
  }

  LogicalResult lowerWrite(OpBuilder &rb, WriteOp write) {
    const Accessor *a = accessorOf(write.getAcc());
    if (!a) return write.emitError("access through a value that is not an accessor of the loop");
    if (useAffine) {
      auto [map, operands] = affineAccess(*a, write.getOffsets(), write.getComp());
      affine::AffineStoreOp::create(rb, write.getLoc(), write.getValue(), a->memref, map, operands);
      return success();
    }
    memref::StoreOp::create(rb, write.getLoc(), write.getValue(), a->memref, memrefIndices(rb, write.getLoc(), *a, write.getOffsets(), {}, write.getComp()));
    return success();
  }

  void lowerIndex(OpBuilder &rb, IndexOp index) {
    Location l = index.getLoc();
    unsigned d = index.getDim();
    Value v = arith::AddIOp::create(rb, l, iv(d), entry->getArgument(dispBase + d));
    index.getResult().replaceAllUsesWith(arith::IndexCastOp::create(rb, l, rb.getI32Type(), v));
  }

  void lowerPrint(OpBuilder &rb, PrintOp print) {
    Location l = print.getLoc();
    if (gpu) {
      gpu::PrintfOp::create(rb, l, print.getFormat(), print.getValues());
      return;
    }
    // printf through the LLVM dialect: a global with the format string, a variadic call.
    auto *mctx = rb.getContext();
    auto i32 = rb.getI32Type();
    auto ptrType = LLVM::LLVMPointerType::get(mctx);
    OpBuilder mb(module.getBody(), module.getBody()->begin());
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
      auto printfType = LLVM::LLVMFunctionType::get(i32, {ptrType}, /*isVarArg=*/true);
      LLVM::LLVMFuncOp::create(mb, l, "printf", printfType);
    }
    std::string text = print.getFormat().str();
    text.push_back('\0');
    auto arrayType = LLVM::LLVMArrayType::get(rb.getI8Type(), text.size());
    std::string symbol = "ops_fmt_" + std::to_string(formatCounter++) + "_" + chain.getSymName().str();
    LLVM::GlobalOp::create(mb, l, arrayType, /*isConstant=*/true, LLVM::Linkage::Internal, symbol, rb.getStringAttr(text));
    Value address = LLVM::AddressOfOp::create(rb, l, ptrType, symbol);
    SmallVector<Value> callArgs{address};
    callArgs.append(print.getValues().begin(), print.getValues().end());
    auto printfType = LLVM::LLVMFunctionType::get(i32, {ptrType}, /*isVarArg=*/true);
    LLVM::CallOp::create(rb, l, printfType, "printf", callArgs);
  }

  LogicalResult lowerReduce(OpBuilder &rb, ReduceOp reduce) {
    Location l = reduce.getLoc();
    auto cast = reduce.getRed().getDefiningOp<UnrealizedConversionCastOp>();
    auto r = cast ? cast->getAttrOfType<IntegerAttr>("ops.reduction") : IntegerAttr();
    if (!r) return reduce.emitError("reduction through a value that is not a reduction of the loop");
    const Accumulator &acc = accumulators[r.getInt()];
    Value cur = memref::LoadOp::create(rb, l, acc.memref, ValueRange{});
    memref::StoreOp::create(rb, l, combine(rb, l, acc.kind, cur, reduce.getValue()), acc.memref, ValueRange{});
    return success();
  }

  /// Rewrite the ops of the dialect that the cloned body contains, nested ones included.
  LogicalResult rewriteBody() {
    SmallVector<Operation *> toRewrite;
    walkRoot->walk([&](Operation *op) {
      if (isa<ReadOp, ReadDynOp, WriteOp, IndexOp, ReduceOp, GblReadOp, ConstRefOp, PrintOp>(op)) toRewrite.push_back(op);
    });
    for (Operation *op : toRewrite) {
      OpBuilder rb(op);
      LogicalResult status = success();
      if (auto read = dyn_cast<ReadOp>(op)) status = lowerRead(rb, read);
      else if (auto dyn = dyn_cast<ReadDynOp>(op)) status = lowerReadDyn(rb, dyn);
      else if (auto write = dyn_cast<WriteOp>(op)) status = lowerWrite(rb, write);
      else if (auto index = dyn_cast<IndexOp>(op)) lowerIndex(rb, index);
      else if (auto print = dyn_cast<PrintOp>(op)) lowerPrint(rb, print);
      else if (auto reduce = dyn_cast<ReduceOp>(op)) status = lowerReduce(rb, reduce);
      else
        return op->emitError("'") << op->getName() << "' must not remain in a loop body when lowering a group (materialise and specialise first)";
      if (failed(status)) return failure();
      op->erase();
    }
    // The placeholders have no users left.
    SmallVector<Operation *> dead;
    walkRoot->walk([&](UnrealizedConversionCastOp cast) {
      if (cast.use_empty() && (cast->hasAttr("ops.accessor") || cast->hasAttr("ops.reduction"))) dead.push_back(cast);
    });
    for (Operation *op : dead) op->erase();
    return success();
  }

  // --- the partial reductions of the points -----------------------------------------------------------------------

  void finishReductions() {
    if (gpu)
      emitGpuReduction();
    else
      b.setInsertionPointToEnd(body);
    if (!gpu && !accumulators.empty()) emitCpuReduction();
  }

  /// The threads of a warp combine their partial values with shuffles and lane 0 adds the result to the buffer of the
  /// reduction with an atomic (one atomic per warp). A block that is not a whole number of warps has every thread do its own.
  void emitGpuReduction() {
    b.setInsertionPoint(gpuLaunch.getBody().front().getTerminator());
    const int64_t threads = gpuBlock[0] * gpuBlock[1] * gpuBlock[2];
    const bool warps = threads % 32 == 0;
    Value lane0, slot;
    {
      auto tid0 = gpuLaunch.getThreadIds();
      auto bid = gpuLaunch.getBlockIds();
      auto grid = gpuLaunch.getGridSize();
      Value linThread = arith::AddIOp::create(b, loc, tid0.x,
                  arith::MulIOp::create(b, loc, arith::ConstantIndexOp::create(b, loc, gpuBlock[0]),
                  arith::AddIOp::create(b, loc, tid0.y, arith::MulIOp::create(b, loc, arith::ConstantIndexOp::create(b, loc, gpuBlock[1]), tid0.z))));
      Value linBlock = arith::AddIOp::create(b, loc, bid.x, arith::MulIOp::create(b, loc, grid.x, arith::AddIOp::create(b, loc, bid.y, arith::MulIOp::create(b, loc, grid.y, bid.z))));
      // warp (or thread, when the block is not whole warps) number over the whole grid
      const int64_t unit = warps ? 32 : 1;
      Value unitsPerBlock = arith::ConstantIndexOp::create(b, loc, threads / unit);
      Value unitInBlock = arith::DivUIOp::create(b, loc, linThread, arith::ConstantIndexOp::create(b, loc, unit));
      Value unitId = arith::AddIOp::create(b, loc, arith::MulIOp::create(b, loc, linBlock, unitsPerBlock), unitInBlock);
      slot = unitId;
    }
    if (warps) {
      auto tid = gpuLaunch.getThreadIds();
      Value c32 = arith::ConstantIndexOp::create(b, loc, 32);
      Value lin = arith::AddIOp::create(b, loc, tid.x,
                  arith::MulIOp::create(b, loc, arith::ConstantIndexOp::create(b, loc, gpuBlock[0]),
                  arith::AddIOp::create(b, loc, tid.y, arith::MulIOp::create(b, loc, arith::ConstantIndexOp::create(b, loc, gpuBlock[1]), tid.z))));
      Value lane = arith::RemUIOp::create(b, loc, lin, c32);
      lane0 = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, arith::ConstantIndexOp::create(b, loc, 0));
    }
    for (auto [i, acc] : llvm::enumerate(accumulators)) {
      Value partial = memref::LoadOp::create(b, loc, acc.memref, ValueRange{});
      if (warps) {
        Value width = arith::ConstantIntOp::create(b, loc, 32, 32);
        for (int offset = 16; offset >= 1; offset /= 2) {
          Value delta = arith::ConstantIntOp::create(b, loc, offset, 32);
          auto shuffled = gpu::ShuffleOp::create(b, loc, partial, delta, width, gpu::ShuffleMode::XOR);
          partial = combine(b, loc, acc.kind, partial, shuffled.getShuffleResult());
        }
      }
      OpBuilder::InsertionGuard guard(b);
      if (warps) {
        auto onlyLane0 = scf::IfOp::create(b, loc, lane0, /*withElseRegion=*/false);
        b.setInsertionPointToStart(&onlyLane0.getThenRegion().front());
      }
      // The partial value of the warp (or thread) goes to its own element of the buffer; the caller combines them.
      memref::StoreOp::create(b, loc, partial, entry->getArgument(bufferBase + i), ValueRange{slot});
    }
  }

  /// The partial values of the points are combined by the scf.parallel (scf.reduce).
  void emitCpuReduction() {
    SmallVector<Value> partials;
    for (const Accumulator &acc : accumulators) partials.push_back(memref::LoadOp::create(b, loc, acc.memref, ValueRange{}));
    auto reduce = scf::ReduceOp::create(b, loc, partials);
    for (auto [i, acc] : llvm::enumerate(accumulators)) {
      Block &block = reduce.getReductions()[i].front();
      b.setInsertionPointToEnd(&block);
      Value v = combine(b, loc, acc.kind, block.getArgument(0), block.getArgument(1));
      scf::ReduceReturnOp::create(b, loc, v);
    }
  }

  // --- return -----------------------------------------------------------------------------------------------------

  LogicalResult emitReturn() {
    b.setInsertionPointToEnd(entry);
    SmallVector<Value> returned;
    auto ret = cast<ReturnOp>(chainBlock.getTerminator());
    for (Value v : ret.getOperands()) {
      auto res = dyn_cast<OpResult>(v);
      if (!res || res.getOwner() != loop.getOperation())
        return ret.emitError("a returned value is not a reduction of the loop");
      if (!gpu) returned.push_back(parallel.getResult(res.getResultNumber()));
    }
    func::ReturnOp::create(b, loc, returned);
    return success();
  }
};

struct ConvertOpsToSCFPass : public impl::ConvertOpsToSCFBase<ConvertOpsToSCFPass> {
  using ConvertOpsToSCFBase::ConvertOpsToSCFBase;
  unsigned formatCounter = 0;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ChainOp> chains(module.getOps<ChainOp>());
    for (ChainOp chain : chains)
      if (failed(lowerChain(chain, module))) return signalPassFailure();
  }

  LogicalResult lowerChain(ChainOp chain, ModuleOp module) {
    SmallVector<ParLoopOp> loops;
    bool hasCalls = false;
    for (Operation &op : chain.getBody().front()) {
      if (auto loop = dyn_cast<ParLoopOp>(&op)) loops.push_back(loop);
      if (isa<CallOp>(&op)) hasCalls = true;
    }
    if (loops.empty() || hasCalls) return success();
    if (loops.size() > 1)
      return chain.emitError("a group must have a single ops.par_loop (found ") << loops.size()
                                                                              << "): run ops-outline-groups first";
    bool called = false;
    module.walk([&](CallOp call) {
      if (call.getCallee() == chain.getSymName()) called = true;
    });
    if (called)
      return chain.emitError("the group is called from another chain of this module and its signature changes: split "
                             "the groups into modules of their own first");
    if (loopStyle != "scf" && loopStyle != "affine")
      return chain.emitError("option 'loops' must be scf or affine, got '") << loopStyle << "'";
    GroupLowering::Config config{soa, target, block, loopStyle == "affine"};
    return GroupLowering(chain, module, loops.front(), std::move(config), formatCounter).run();
  }
};

} // namespace

} // namespace ops_mlir::ops
