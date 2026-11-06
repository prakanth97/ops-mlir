//===- Pipelines.h - the named pipelines of the OPS JIT ----------*- C++ -*-===//
//
//   ops-jit-frontend            ops-materialize-kernels, ops-specialize-consts, canonicalize, cse
//   ops-jit-plan                ops-fuse-point-local, ops-forward-stores, canonicalize, cse, ops-outline-groups
//                               options: max-loops reorder placement guarded box-ratio
//   ops-jit-cpu                 ops-specialize, convert-ops-to-scf, OpenMP, down to the LLVM dialect
//                               options: omp specialize soa
//
// They are registered with the process-wide opdsl::PipelineRegistry (the runtime builds them by name and option
// string) and, for ops-opt, as named pipelines on the command line (-ops-jit-cpu="omp=false").
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_PIPELINES_PIPELINES_H
#define OPS_MLIR_PIPELINES_PIPELINES_H

#include "opdsl/Pipeline/PipelineRegistry.h"

#include <array>
#include <cstdint>
#include <string>

namespace ops_mlir::ops {

/// Adds the pipelines of this library to `registry`.
void registerOPSPipelines(opdsl::PipelineRegistry &registry = opdsl::PipelineRegistry::global());

/// What the user asked for, as the pipelines take it. The runtime fills it from its options (and, for a device, from the device and
/// the toolkit); the option strings that the registry parses are written here, next to the pipelines they belong to.
struct PipelineSettings {
  // planning (ops-jit-plan)
  bool fusion = true;
  int64_t fusionMax = 8;
  bool fusionReorder = true;
  std::string fusionPlacement = "earliest";
  bool fusionGuarded = false;
  // lowering, both targets
  std::string specialize = "none"; ///< none, ranges or shapes
  bool soa = false;                ///< layout of multi-component dats (a property of the OPS instance)
  bool fma = false;
  // CPU (ops-jit-cpu)
  bool omp = true;
  std::string loops = "scf";       ///< scf or affine
  std::string affinePasses;        ///< a pass pipeline run on the affine form (implies loops=affine)
  // GPU (ops-jit-cuda)
  std::string gpuBlock = "32,4,1"; ///< thread block x,y,z
  std::string gpuArch;             ///< NVVM chip, e.g. sm_80
  std::string gpuFeatures;         ///< NVVM target features, e.g. +ptx83
  std::string ptxasOptions;
};

/// The options of ops-jit-plan.
std::string planPipelineOptions(const PipelineSettings &settings);

/// The pipeline that lowers one group of a block of `rank` dimensions for the host or a device, with its options. The options are part
/// of the cache key of the group: other options compile again.
struct GroupPipeline {
  std::string name, options;
};
GroupPipeline groupPipeline(bool device, int rank, const PipelineSettings &settings);

/// The thread block for a block of `rank` dimensions: the option x,y,z, with the dimensions the block does not have folded into x.
std::array<int64_t, 3> gpuBlockFor(const std::string &option, int rank);

/// Makes the pipelines of the global registry available as named pipelines of mlir-opt style tools. Call after
/// registerOPSPipelines().
void registerOPSPipelinesWithMLIR();

} // namespace ops_mlir::ops

#endif // OPS_MLIR_PIPELINES_PIPELINES_H
