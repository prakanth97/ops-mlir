//===- Options.h - run-time options of libopsjit (OPS_JIT_* variables) -----*- C++ -*-===//
//
// Plan section 4.8. Read from the environment by ops_jit_init(); ops_jit_set_option(name, value) overrides one (before
// the first loop is queued: a later change only affects plans built afterwards).
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_OPTIONS_H
#define OPS_MLIR_RUNTIME_OPTIONS_H

#include <string>

namespace ops_mlir::runtime {

struct Options {
  bool enabled = true;           ///< OPS_JIT
  std::string claim = "all";     ///< OPS_JIT_CLAIM: all | fused | none
  bool omp = true;               ///< OPS_JIT_CPU: omp (true) | seq
  bool fusion = true;            ///< OPS_JIT_FUSION
  int fusionMax = 8;             ///< OPS_JIT_FUSION_MAX
  bool fusionReorder = true;     ///< OPS_JIT_FUSION_REORDER
  std::string fusionPlacement = "earliest"; ///< OPS_JIT_FUSION_PLACEMENT
  bool fusionGuarded = false;    ///< OPS_JIT_FUSION_GUARDED
  std::string specialize = "none"; ///< OPS_JIT_SPECIALIZE: none | ranges | shapes
  bool cache = true;             ///< OPS_JIT_CACHE
  std::string cacheDir;          ///< OPS_JIT_CACHE_DIR (empty: the default location)
  bool verify = false;           ///< OPS_JIT_VERIFY
  bool explain = false;          ///< OPS_JIT_EXPLAIN: print why a loop is not claimed
  bool plan = false;             ///< OPS_JIT_PLAN: print the groups
  bool trace = false;            ///< OPS_JIT_TRACE: print every group before it runs (to find the one that crashes)
  bool stats = false;            ///< OPS_JIT_STATS: counters and phase times at exit
  std::string dumpDir;           ///< OPS_JIT_DUMP_DIR
  bool dumpData = false;         ///< OPS_JIT_DUMP_DATA: also record the first call of every group (inputs, checksums) for ops-jit-replay
  int runMax = 512;              ///< OPS_JIT_RUN_MAX: most loops compiled and planned together
  bool sync = false;             ///< OPS_JIT_SYNC (device builds)
  double verifyFloatTol = 1e-5;  ///< OPS_JIT_VERIFY_FLOAT_TOL: relative tolerance of verify mode for single precision dats
  double verifyHalfTol = 1e-2;   ///< OPS_JIT_VERIFY_HALF_TOL: relative tolerance of verify mode for half precision dats
  std::string loops = "scf";     ///< OPS_JIT_LOOPS: scf | affine (the CPU lowering goes through the affine dialect)
  std::string affinePasses;      ///< OPS_JIT_AFFINE_PASSES: a pass pipeline run on the affine form (implies loops=affine; + is a comma, ; a blank)
  bool fma = false;              ///< OPS_JIT_FMA: let the CPU code generator fuse multiply-adds (results change in the last bits: verify mode then reports differences)
  std::string printfPolicy = "drop"; ///< OPS_JIT_PRINTF: keep | drop | decline (what to do with printf in kernels)
  std::string gpuBlock = "32,4,1"; ///< OPS_JIT_GPU_BLOCK: thread block (x,y,z) of the generated kernels
  std::string gpuArch;           ///< OPS_JIT_GPU_ARCH: e.g. sm_80 (empty: the device in use)
  std::string ptxasOpts;         ///< OPS_JIT_PTXAS_OPTS: extra ptxas options (e.g. -v)

  /// Reads the OPS_JIT_* environment variables.
  static Options fromEnvironment();

  /// Sets one option by its environment name (e.g. "OPS_JIT_FUSION"). Returns false for an unknown name or value.
  bool set(const std::string &name, const std::string &value);

  /// The options that change generated code or plans, as text: part of the cache keys.
  std::string codegenDigest() const;
};

/// The process-wide options.
Options &options();

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_OPTIONS_H
