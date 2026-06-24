//===- OPSToStencil.cpp - Lower ops.par_loop to func.func + stencil.* ---===//
//
// Each ops.par_loop becomes a callable func.func with one memref<...>
// parameter per captured Dat argument (not a pointer literal -- the
// pointer is a call-time concern, supplied by the caller, e.g. via
// mlir::ExecutionEngine, not baked into the IR), containing a
// stencil.apply body: stencil.load the memref parameters that are
// read/read-write, one stencil.access per dat (defaulting to a single
// (0, ..., 0) point -- see below), a placeholder func.call to the kernel
// (the captured ops.par_loop only carries the kernel's name/pointer, not
// its source), then stencil.store the apply's results back into the
// write/read-write/inc memref parameters.
//
// `args` is OPS_ArgAttr (struct, mirroring `struct ops_arg`): dat
// (OPS_DatAttr), stencil (OPS_StencilAttr), dim, elem_size, data, data_d,
// acc, argtype, opt. See OPSOps.td.
//
// ArgType (ops_arg_type):   0=GBL 1=DAT 2=IDX
// Access (ops_access):      0=READ 1=WRITE 2=RW 3=INC 4=MIN 5=MAX
//
// Stencil access pattern: OPS_StencilAttr only carries a *pointer* to the
// point-offset array (mirroring ops_stencil_core::stencil exactly), not
// the offsets themselves, so this pass cannot recover them from the IR
// alone and defaults every dat to a single (0, ..., 0) access point.
//
// Idx arguments (ops_arg_idx(), e.g. left_bndcon/right_bndcon's
// `const int *idx`) are modeled via stencil.index: one per dimension,
// appended to the kernel call after the dat-derived accesses, in the same
// order they appear among the loop's arguments.
//
// Known simplification (first cut): Gbl/Reduce arguments are still
// skipped.
//===----------------------------------------------------------------------===//

#include "passes/OPSToStencil.h"

#include "Dialect/OPS/OPSOps.h"
#include "Dialect/stencil/StencilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace ops_mlir::ops;

namespace {

enum ArgTypeCode : int32_t { kGbl = 0, kDat = 1, kIdx = 2 };
enum AccessCode : int32_t {
  kRead = 0,
  kWrite = 1,
  kRw = 2,
  kInc = 3,
  kMin = 4,
  kMax = 5
};

SmallVector<int64_t> datSize(DatAttr dat, int64_t ndim) {
  auto size = dat.getSize().asArrayRef();
  return SmallVector<int64_t>(size.begin(), size.begin() + ndim);
}

stencil::StencilBoundsAttr datBounds(OpBuilder &b, DatAttr dat, int64_t ndim) {
  auto size = dat.getSize().asArrayRef();
  auto d_m = dat.getDM().asArrayRef();
  auto d_p = dat.getDP().asArrayRef();
  SmallVector<int64_t> lb, ub;
  for (int64_t i = 0; i < ndim; ++i) {
    lb.push_back(d_m[i]);
    ub.push_back(size[i] + d_p[i]);
  }
  return stencil::StencilBoundsAttr::get(
      b.getContext(), DenseI64ArrayAttr::get(b.getContext(), lb),
      DenseI64ArrayAttr::get(b.getContext(), ub));
}

stencil::StencilBoundsAttr rangeBounds(OpBuilder &b, ArrayRef<int64_t> range,
                                        int64_t ndim) {
  SmallVector<int64_t> lb, ub;
  for (int64_t i = 0; i < ndim; ++i) {
    lb.push_back(range[2 * i]);
    ub.push_back(range[2 * i + 1]);
  }
  return stencil::StencilBoundsAttr::get(
      b.getContext(), DenseI64ArrayAttr::get(b.getContext(), lb),
      DenseI64ArrayAttr::get(b.getContext(), ub));
}

/// Lower a single ops.par_loop into a standalone func.func, inserted right
/// before it in the module. Returns the new function.
func::FuncOp convertParLoop(ParLoopOp loop, unsigned index, ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(loop);
  Location loc = loop.getLoc();

  int64_t ndim = loop->getAttrOfType<IntegerAttr>("dims").getInt();
  auto argsAttr = loop->getAttrOfType<ArrayAttr>("args");

  SmallVector<ArgAttr> datArgs;
  unsigned numIdxArgs = 0;
  for (Attribute a : argsAttr)
    if (auto arg = dyn_cast<ArgAttr>(a)) {
      if (arg.getArgtype() == kDat)
        datArgs.push_back(arg);
      else if (arg.getArgtype() == kIdx)
        ++numIdxArgs;
    }

  // Build the function signature: one memref<...> param per dat arg.
  SmallVector<Type> paramTypes;
  for (ArgAttr arg : datArgs)
    paramTypes.push_back(
        MemRefType::get(datSize(arg.getDat(), ndim), builder.getF64Type()));

  StringRef kernelName = loop.getKernelNameAttr().getValue();

  auto funcType = builder.getFunctionType(paramTypes, {});
  std::string fnName =
      ("ops_par_loop_" + kernelName + "_" + Twine(index)).str();
  auto func = builder.create<func::FuncOp>(loc, fnName, funcType);
  func.setPrivate();
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto rangeAttr = loop->getAttrOfType<DenseI64ArrayAttr>("range");
  auto applyBounds = rangeBounds(builder, rangeAttr.asArrayRef(), ndim);

  SmallVector<std::pair<ArgAttr, Value>> loads; // (arg, stencil.temp)
  SmallVector<std::pair<ArgAttr, Value>> writeFields; // (arg, memref param)

  for (auto [i, arg] : llvm::enumerate(datArgs)) {
    Value field = entry->getArgument(i);
    int32_t acc = arg.getAcc();
    if (acc == kRead || acc == kRw) {
      auto tempTy = stencil::TempType::get(
          ctx, datBounds(builder, arg.getDat(), ndim), builder.getF64Type());
      auto load = builder.create<stencil::LoadOp>(loc, tempTy, field);
      loads.emplace_back(arg, load.getRes());
    }
    if (acc == kWrite || acc == kRw || acc == kInc)
      writeFields.emplace_back(arg, field);
  }

  SmallVector<Value> applyArgs;
  for (auto &[arg, temp] : loads)
    applyArgs.push_back(temp);

  SmallVector<Type> resultTypes;
  for (auto &[arg, field] : writeFields)
    resultTypes.push_back(stencil::TempType::get(
        ctx, datBounds(builder, arg.getDat(), ndim), builder.getF64Type()));
  if (resultTypes.empty())
    resultTypes.push_back(
        stencil::TempType::get(ctx, applyBounds, builder.getF64Type()));

  auto apply = builder.create<stencil::ApplyOp>(loc, resultTypes, applyArgs,
                                                 applyBounds);
  Block *applyBody = builder.createBlock(&apply.getRegion());
  for (Value v : applyArgs)
    applyBody->addArgument(v.getType(), loc);

  OpBuilder bodyBuilder(applyBody, applyBody->end());

  // stencil.access defaults to a single (0,...,0) point per dat -- see
  // file header (OPS_StencilAttr only carries a raw offsets pointer).
  SmallVector<int64_t> zeroOffset(ndim, 0);
  auto offsetAttr =
      stencil::IndexAttr::get(ctx, DenseI64ArrayAttr::get(ctx, zeroOffset));

  SmallVector<Value> accessResults;
  for (BlockArgument blockArg : applyBody->getArguments())
    accessResults.push_back(bodyBuilder.create<stencil::AccessOp>(
        loc, builder.getF64Type(), blockArg, offsetAttr));

  // ops_arg_idx(): one stencil.index per dimension, per idx argument,
  // appended after the dat-derived accesses. No static shift, so offset
  // is all-zero (matching the (0,...,0) default used for stencil.access).
  SmallVector<int64_t> zeroIdxOffset(ndim, 0);
  auto idxOffsetAttr = stencil::IndexAttr::get(
      ctx, DenseI64ArrayAttr::get(ctx, zeroIdxOffset));
  SmallVector<Value> idxResults;
  for (unsigned i = 0; i < numIdxArgs; ++i)
    for (int64_t d = 0; d < ndim; ++d)
      idxResults.push_back(bodyBuilder.create<stencil::IndexOp>(
          loc, bodyBuilder.getIndexType(), bodyBuilder.getI64IntegerAttr(d),
          idxOffsetAttr));

  SmallVector<Value> callArgs = accessResults;
  callArgs.append(idxResults.begin(), idxResults.end());

  // Declare (if not already present) an external func for the kernel body
  // -- the captured ops.par_loop only carries the kernel's name/pointer,
  // not its source, so this is a placeholder for whatever eventually
  // supplies the real kernel (e.g. linking in separately-compiled LLVM IR
  // for the real kernel function).
  if (!module.lookupSymbol<func::FuncOp>(kernelName)) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    SmallVector<Type> kernelParamTypes(accessResults.size(),
                                        builder.getF64Type());
    kernelParamTypes.append(idxResults.size(), builder.getIndexType());
    auto kernelType = builder.getFunctionType(
        kernelParamTypes,
        SmallVector<Type>(std::max<size_t>(1, writeFields.size()),
                           builder.getF64Type()));
    auto decl =
        moduleBuilder.create<func::FuncOp>(loc, kernelName, kernelType);
    decl.setPrivate();
  }

  size_t numResults = std::max<size_t>(1, writeFields.size());
  auto call = bodyBuilder.create<func::CallOp>(
      loc, SmallVector<Type>(numResults, builder.getF64Type()), kernelName,
      callArgs);
  bodyBuilder.create<stencil::ReturnOp>(loc, call.getResults());

  // createBlock() above left `builder`'s insertion point inside the new
  // apply body block; move it back to right after the apply op (i.e. back
  // in the function's entry block) before emitting the store/return.
  builder.setInsertionPointAfter(apply);

  for (auto [i, fieldPair] : llvm::enumerate(writeFields)) {
    auto &[arg, field] = fieldPair;
    builder.create<stencil::StoreOp>(loc, apply.getResult(i), field,
                                      datBounds(builder, arg.getDat(), ndim));
  }

  builder.create<func::ReturnOp>(loc);
  return func;
}

} // namespace

namespace ops_mlir {

/// Lower every ops.par_loop in `module` into its own func.func, in place.
void convertOPSParLoopsToStencil(ModuleOp module) {
  SmallVector<ParLoopOp> loops;
  module.walk([&](ParLoopOp op) { loops.push_back(op); });

  for (auto [i, loop] : llvm::enumerate(loops)) {
    convertParLoop(loop, i, module);
    loop.erase();
  }
}

} // namespace ops_mlir
