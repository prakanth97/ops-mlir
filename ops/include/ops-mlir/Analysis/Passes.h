//===- Passes.h - analysis printing passes ------------------------*- C++ -*-===//
//
//   -ops-print-chain-dependence   loops of every ops.chain with their dat accesses, and the dependence edges
//   -ops-print-fusion-plan        the fusion plan of every chain: max-loops, reorder, placement, guarded,
//                                 box-ratio (printed to stderr, for FileCheck)
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_ANALYSIS_PASSES_H
#define OPS_MLIR_ANALYSIS_PASSES_H

namespace ops_mlir::ops {

/// Registers the analysis passes with the global pass registry.
void registerAnalysisPasses();

} // namespace ops_mlir::ops

#endif // OPS_MLIR_ANALYSIS_PASSES_H
