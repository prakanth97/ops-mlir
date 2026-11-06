//===- FusionPlanner.h - dependence-aware grouping of the loops of a chain -*- C++ -*-===//
//
// Decides which loops of a chain run together as one kernel, and in what order the kernels run. Pure
// analysis: it reads LoopInfos and returns a plan. `ops-fuse-point-local` applies the plan as IR rewriting.
//
// Two planners share the legality rules:
//   * consecutive (reorder = false): only adjacent loops fuse;
//   * DAG (default): the dependence edges of ChainDependence let independent loops be hoisted next to the
//     loops they can fuse with, so a loop that cannot join its neighbours may still join an earlier kernel.
//
// Loops are placed, in program order, into an existing group when every edge to a member is fusable and the
// groups stay acyclic (no loop moves across one it depends on); kernels are emitted in a topological order of
// the groups. Fusing loop j into a group that holds loop i (i before j) executes both at each point, in
// order, instead of running i over its whole range and then j: ChainDependence's fusable edges are exactly
// the pairs for which that is equivalent. A point outside a member's range skips that member (a "guarded"
// group), and reads of a dat that member would have written see the old value, as in sequential execution.
// A loop with a reduction is a barrier: nothing moves across it and it is never fused.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_ANALYSIS_FUSIONPLANNER_H
#define OPS_MLIR_ANALYSIS_FUSIONPLANNER_H

#include "ops-mlir/Analysis/ChainDependence.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace ops_mlir::ops {

struct FusionOptions {
  bool enabled = true;
  /// Upper bound on loops per group (register pressure / kernel size).
  size_t maxGroupSize = 8;
  /// Fuse loops with different ranges into one guarded launch over the bounding box.
  bool allowGuarded = true;
  /// Build the dependence DAG and allow loops to move past independent ones.
  bool reorder = true;
  /// Which legal group a loop joins when the DAG planner has a choice: the earliest (shortest dependence
  /// chains first, most fusion) or the latest (keeps loops near their program-order neighbours).
  enum class Placement { Earliest, Latest };
  Placement placement = Placement::Earliest;
  /// A guarded group is only formed if volume(bounding box) <= boxRatio * sum(member volumes), so a thin
  /// boundary loop does not drag a full-grid launch along.
  double boxRatio = 1.0;
};

struct FusedGroup {
  std::vector<size_t> loops; ///< indices into the chain's loops, ascending (also a valid order to run them in)
  bool guarded = false;      ///< members have differing ranges
  Range range;               ///< bounding box [lo0, hi0, lo1, hi1, ...]
};

struct FusionPlan {
  std::vector<FusedGroup> groups; ///< in execution order

  /// DAG planner diagnostics: for every loop that had to start a new kernel, why each candidate kernel was
  /// rejected (a loop can count under several reasons). Keys: "first" (nothing to join), "dependence" (a
  /// blocking edge to a member), "order" (joining would put it before a kernel it depends on), "range"
  /// (bounding-box / guard rule), "size" (group size cap), "block" (different block, rank or axes).
  std::map<std::string, size_t> blockedBy;

  /// Number of loops that execute in a different position relative to the others than program order (0 for
  /// the consecutive planner).
  size_t numReordered() const;

  /// Compact text form ("0,1,2|3|4,5g"; g marks a guarded group), part of the compiled-module cache key.
  std::string digest() const;
};

/// Memory traffic model in the style of OPS (extent x element size, read and write counted separately,
/// stencil neighbours assumed cached).
struct TrafficEstimate {
  double unfusedBytes = 0; ///< every loop on its own
  double fusedBytes = 0;   ///< per kernel: each distinct dat loaded at most once (not at all if a member writes
                           ///< it first) and stored at most once
};
TrafficEstimate estimateTraffic(llvm::ArrayRef<LoopInfo> loops, const FusionPlan &plan);

/// Groups the loops. Every loop lands in exactly one group. With reordering off the groups are runs of
/// adjacent loops; with it on they may interleave but every dependence is respected.
FusionPlan planFusion(llvm::ArrayRef<LoopInfo> loops, const FusionOptions &options);

} // namespace ops_mlir::ops

#endif // OPS_MLIR_ANALYSIS_FUSIONPLANNER_H
