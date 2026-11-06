//===- ChainBuilder.h - build an ops.chain from a compact loop description -*- C++ -*-===//
//
// Test helper. A chain is described by one LoopSpec per loop: dat arguments (dat id, access mode, stencil
// points), a range, a block. The dat ids are the chain's argument numbers, so a plan or an analysis result
// can be compared against the description directly.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_UNITTESTS_CHAINBUILDER_H
#define OPS_MLIR_UNITTESTS_CHAINBUILDER_H

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace ops_mlir::ops::testing {

using Points = std::vector<std::vector<int64_t>>;

/// Pseudo dat ids of ArgSpec for the arguments that are not dats.
constexpr int kReduction = -1; ///< a reduction (max) the loop contributes to
constexpr int kGlobal = -2;    ///< a read-only global (a scalar operand)

struct ArgSpec {
  int dat = 0;
  AccessMode access = AccessMode::Read;
  Points points = {};                          ///< stencil points; empty = the zero offset
  std::vector<int64_t> strides = {};           ///< empty = unit strides
  std::optional<Points> writes = std::nullopt; ///< offsets the body writes at (materialised loops only)
};

struct LoopSpec {
  std::vector<ArgSpec> args;
  std::vector<int64_t> range;
  int64_t block = 1;
  bool materialize = false; ///< give the loop a body with the `ops.write`s of its write operands
};

struct BuiltChain {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ChainOp chain;
};

inline mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    auto *c = new mlir::MLIRContext();
    c->loadDialect<OPSDialect, mlir::arith::ArithDialect>();
    return c;
  }();
  return *ctx;
}

/// Builds `ops.chain @chain` over dats 0..maxDat with the loops of `loops`. All dats are f64 with one
/// component on a block of `rank` dimensions. The chain carries `ops.range_values`.
inline BuiltChain buildChain(llvm::ArrayRef<LoopSpec> loops, unsigned rank) {
  using namespace mlir;
  MLIRContext &ctx = context();
  Location loc = UnknownLoc::get(&ctx);
  OpBuilder b(&ctx);
  Type f64 = b.getF64Type();

  int maxDat = -1;
  for (const LoopSpec &l : loops)
    for (const ArgSpec &a : l.args) maxDat = std::max(maxDat, a.dat);

  // Chain signature: dats, then per loop its range bounds and its global scalars.
  SmallVector<Type> inputs, results;
  for (int d = 0; d <= maxDat; ++d) inputs.push_back(DatType::get(&ctx, f64, rank, 1));
  for (const LoopSpec &l : loops) {
    for (unsigned i = 0; i < 2 * rank; ++i) inputs.push_back(b.getIndexType());
    for (const ArgSpec &a : l.args) {
      if (a.dat == kGlobal) inputs.push_back(f64);
      if (a.dat == kReduction) results.push_back(f64);
    }
  }

  BuiltChain out;
  out.module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(out.module->getBody());
  out.chain = ChainOp::create(b, loc, "chain", b.getFunctionType(inputs, results));
  Block *entry = out.chain.addEntryBlock();
  b.setInsertionPointToEnd(entry);

  SmallVector<Attribute> rangeValues;
  SmallVector<Value> returned;
  unsigned argPos = maxDat + 1;
  for (const LoopSpec &l : loops) {
    SmallVector<Value> range, dats, scalars;
    for (unsigned i = 0; i < 2 * rank; ++i) range.push_back(entry->getArgument(argPos++));
    SmallVector<Attribute> accesses, stencils, kinds;
    SmallVector<Type> accTypes, scalarTypes, redTypes;
    for (const ArgSpec &a : l.args) {
      if (a.dat == kGlobal) {
        scalars.push_back(entry->getArgument(argPos++));
        scalarTypes.push_back(f64);
        continue;
      }
      if (a.dat == kReduction) {
        kinds.push_back(ReductionKindAttr::get(&ctx, ReductionKind::Max));
        redTypes.push_back(RedType::get(&ctx, f64));
        continue;
      }
      dats.push_back(entry->getArgument(a.dat));
      accesses.push_back(AccessModeAttr::get(&ctx, a.access));
      Points points = a.points.empty() ? Points{std::vector<int64_t>(rank, 0)} : a.points;
      SmallVector<int64_t> flat;
      for (const auto &p : points) flat.append(p.begin(), p.end());
      SmallVector<int64_t> strides(rank, 1);
      if (!a.strides.empty()) strides.assign(a.strides.begin(), a.strides.end());
      stencils.push_back(StencilAttr::get(&ctx, rank, flat, strides));
      accTypes.push_back(AccType::get(&ctx, f64, rank, 1));
    }

    ArrayAttr kindsAttr = kinds.empty() ? ArrayAttr() : b.getArrayAttr(kinds);
    SmallVector<Type> resultTypes(kinds.size(), f64);
    auto loop = ParLoopOp::create(b, loc, resultTypes, range, dats, scalars, FlatSymbolRefAttr(),
                                    b.getI64IntegerAttr(-1), b.getI64IntegerAttr(l.block),
                                    b.getArrayAttr(accesses), b.getArrayAttr(stencils), kindsAttr);
    returned.append(loop.getResults().begin(), loop.getResults().end());

    SmallVector<Type> blockTypes(accTypes);
    blockTypes.append(scalarTypes);
    blockTypes.append(redTypes);
    SmallVector<Location> locs(blockTypes.size(), loc);
    OpBuilder::InsertionGuard guard(b);
    Block *body = b.createBlock(&loop.getBody(), {}, blockTypes, locs);
    if (l.materialize) {
      Value zero = arith::ConstantOp::create(b, loc, b.getF64FloatAttr(0.0));
      unsigned datIdx = 0;
      for (const ArgSpec &a : l.args) {
        if (a.dat < 0) continue;
        Value acc = body->getArgument(datIdx++);
        if (a.access == AccessMode::Read) continue;
        Points writes = a.writes ? *a.writes : (a.points.empty() ? Points{std::vector<int64_t>(rank, 0)} : a.points);
        for (const auto &w : writes) WriteOp::create(b, loc, acc, w, zero, 0);
      }
      for (unsigned r = 0; r < redTypes.size(); ++r)
        ReduceOp::create(b, loc, body->getArgument(accTypes.size() + scalarTypes.size() + r), zero);
    }
    YieldOp::create(b, loc);

    rangeValues.push_back(b.getDenseI64ArrayAttr(l.range));
  }
  b.setInsertionPointToEnd(entry);
  ReturnOp::create(b, loc, returned);
  out.chain->setAttr("ops.range_values", b.getArrayAttr(rangeValues));
  return out;
}

} // namespace ops_mlir::ops::testing

#endif // OPS_MLIR_UNITTESTS_CHAINBUILDER_H
