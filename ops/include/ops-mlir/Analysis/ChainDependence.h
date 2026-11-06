//===- ChainDependence.h - what the loops of a chain do to which dats -*- C++ -*-===//
//
// The loops of an `ops.chain` reduced to what dependence analysis needs (LoopInfo), and the dependence
// edges between them. Ranges are concrete: they come from the `ops.range_values` attribute of the chain, an
// array with one dense i64 array [lo0, hi0, lo1, hi1, ...] per `ops.par_loop`, in program order.
//
// An edge i -> j (i before j) exists when both touch a dat and their footprints conflict:
//   RAW  i writes where j reads;  WAR  i reads where j writes;  WAW  both write.
// The footprint of a read is the range expanded by the stencil. The footprint of a write is the range
// expanded by the offsets the loop writes at: the whole stencil of the operand when the kernel body is not
// materialised, otherwise the offsets of its `ops.write`s (boundary kernels write halo points, at non-zero
// offsets).
//
// An edge is *fusable* when running i and j point by point, in one kernel, gives the same result as running
// i over its whole range and then j: no value may flow between different points. It is not when
//   - j reads at a non-zero offset what i writes (RAW across points);
//   - i reads at a non-zero offset what j writes (WAR across points);
//   - either of them writes at a non-zero offset.
// A non-fusable edge still orders the loops; loops joined by no path may run in either order.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_ANALYSIS_CHAINDEPENDENCE_H
#define OPS_MLIR_ANALYSIS_CHAINDEPENDENCE_H

#include "ops-mlir/Analysis/Footprint.h"
#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ops_mlir::ops {

/// One dat operand of a loop.
struct ArgInfo {
  int dat = 0;                 ///< identity of the dat within the chain
  AccessMode access = AccessMode::Read;
  int64_t elemBytes = 0;       ///< bytes per point of the dat (all components)
  bool reads = false;          ///< read or rw
  bool writes = false;         ///< write, rw or inc
  bool readNonLocal = false;   ///< reads at a non-zero offset (or with a stride other than 1)
  bool writeNonLocal = false;  ///< writes at a non-zero offset (or with a stride other than 1)
  Range readBox, writeBox;     ///< footprints; empty when the operand does not read / write
};

/// What one loop does to one dat, all its operands on that dat merged.
struct DatAccess {
  bool reads = false, writes = false;
  bool readNonLocal = false, writeNonLocal = false;
  Range readBox, writeBox;
};
using DatAccesses = std::map<int, DatAccess>;

/// A loop of a chain.
struct LoopInfo {
  int64_t block = 0;
  unsigned rank = 0;
  Range range;                 ///< [lo0, hi0, lo1, hi1, ...]
  unsigned axes = 0;           ///< bit d set when some dat of the loop is indexed along dimension d
  bool hasReduction = false;   ///< the loop produces reductions: never fused, a barrier for reordering
  llvm::SmallVector<ArgInfo, 6> args;
  ParLoopOp op;                ///< the loop this was extracted from (null for hand-made infos)

  /// The accesses merged per dat.
  DatAccesses accesses() const;
  bool hasNonLocalRead() const;
  bool hasNonLocalAccess() const; ///< non-local read or write
};

/// Extracts the loops of `chain`, in program order. Fails (with a diagnostic on the chain) if a loop has no
/// concrete range.
llvm::FailureOr<std::vector<LoopInfo>> extractLoops(ChainOp chain);

/// How two loops conflict.
enum ConflictKind : unsigned { RAW = 1, WAR = 2, WAW = 4 };

struct DepEdge {
  size_t from = 0, to = 0;
  unsigned kinds = 0;                 ///< bitmask of ConflictKind
  bool fusable = true;                ///< all conflicts are within a point
  llvm::SmallVector<int, 2> dats;     ///< dats the conflicts are on, ascending
};

/// The dependence edges of a sequence of loops.
struct ChainDependence {
  std::vector<LoopInfo> loops;
  std::vector<DepEdge> edges;                     ///< sorted by (to, from)
  std::vector<std::vector<size_t>> predecessors;  ///< per loop: indices into `edges` of its incoming edges

  static ChainDependence build(std::vector<LoopInfo> loops);
  /// Convenience: extract then build.
  static llvm::FailureOr<ChainDependence> build(ChainOp chain);

  /// Text form, used by `-ops-print-chain-dependence` and by tests.
  void print(llvm::raw_ostream &os) const;
};

} // namespace ops_mlir::ops

#endif // OPS_MLIR_ANALYSIS_CHAINDEPENDENCE_H
