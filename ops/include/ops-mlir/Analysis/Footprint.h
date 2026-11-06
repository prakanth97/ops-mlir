//===- Footprint.h - iteration ranges and stencil footprints -----*- C++ -*-===//
//
// Geometry of the dependence analysis: boxes in OPS index space. A Range is
// [lo0, hi0, lo1, hi1, ...] with half-open intervals, x first (OPS axis order).
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_ANALYSIS_FOOTPRINT_H
#define OPS_MLIR_ANALYSIS_FOOTPRINT_H

#include "ops-mlir/Dialect/OPS/IR/OPSAttrs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace ops_mlir::ops {

using Range = llvm::SmallVector<int64_t, 6>;

/// Stands for "touches everything" in a dimension (multigrid / broadcast accesses).
constexpr int64_t kUnbounded = int64_t(1) << 40;

/// True if some dimension has no points.
bool isEmpty(llvm::ArrayRef<int64_t> range);

/// True if the two boxes share a point. Empty boxes overlap nothing.
bool overlaps(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b);

/// Number of points of the box.
double volume(llvm::ArrayRef<int64_t> range);

/// Smallest box containing both. An empty `a` yields `b`.
Range boundingBox(llvm::ArrayRef<int64_t> a, llvm::ArrayRef<int64_t> b);

/// How far a set of offsets reaches from the iteration point: per dimension the smallest and largest offset,
/// and whether the dimension is "unbounded" (a stride other than 1: multigrid, or a dat that lacks the
/// dimension, so every point of the loop touches the same elements along it).
struct Reach {
  llvm::SmallVector<int64_t, 3> lo, hi;
  llvm::SmallVector<bool, 3> unbounded;
  bool any = false;       ///< at least one offset was added
  bool pointLocal = true; ///< every offset is zero and every stride is 1

  explicit Reach(unsigned rank = 0) : lo(rank, 0), hi(rank, 0), unbounded(rank, false) {}

  /// The reach of all points of a stencil, strides included.
  static Reach of(StencilAttr stencil);
  void addOffset(llvm::ArrayRef<int64_t> offset);
  void setStrides(llvm::ArrayRef<int64_t> strides);
};

/// Points a loop over `range` may touch with offsets of `reach`: the range expanded by the extent of the
/// reach, along each dimension. An unbounded dimension is the whole line.
Range footprint(llvm::ArrayRef<int64_t> range, const Reach &reach);

/// Convenience: footprint over all points of a stencil.
Range footprint(llvm::ArrayRef<int64_t> range, StencilAttr stencil);

} // namespace ops_mlir::ops

#endif // OPS_MLIR_ANALYSIS_FOOTPRINT_H
