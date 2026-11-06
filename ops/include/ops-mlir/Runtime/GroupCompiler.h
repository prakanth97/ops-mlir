//===- GroupCompiler.h - planned group -> compiled function (memory and disk caches) -*- C++ -*-===//
//
// Plan section 4.5 (buildPlan step 3) and 4.7 (GroupKey). Takes the canonical module of one group (an `ops.chain`
// after ops-jit-plan), looks its key up in the memory and disk caches, and on a miss lowers it with a named pipeline,
// compiles it with opdsl::JITSession and stores the object. No dependency on OPS: libopsjit, the replay tool and the
// unit tests share it.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_GROUPCOMPILER_H
#define OPS_MLIR_RUNTIME_GROUPCOMPILER_H

#include "opdsl/Cache/PersistentCache.h"
#include "opdsl/JIT/JITSession.h"
#include "opdsl/Pipeline/PipelineRegistry.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace ops_mlir::runtime {

/// Bump when the ABI, the packed wrapper or the meaning of a compiled group changes: it is part of every key.
constexpr int kCodeFormat = 12;

struct GroupCode {
  opdsl::JITSession::PackedFn fn = nullptr;
  std::string entry;
  std::string key;
  std::string pipeline, pipelineOptions; ///< what it was lowered with (for dumps)
};

struct CompileStats {
  uint64_t compiles = 0, memoryHits = 0, diskHits = 0;
  double tLowering = 0, tLlvm = 0, tCacheIO = 0;
};

struct CompilerOptions {
  bool useCache = true;
  std::string cacheDir; ///< empty: the default location
  std::string dumpDir;  ///< non-empty: group and lowered IR are written there
  opdsl::JITOptions jit;
};

class GroupCompiler {
public:
  GroupCompiler(mlir::MLIRContext &ctx, const opdsl::PipelineRegistry &registry, CompilerOptions options);

  /// The compiled group for `group`. `pipeline` and `pipelineOptions` name the lowering (e.g. "ops-jit-cpu",
  /// "omp=true specialize=none soa=false"); they are part of the key. Null with `error` set on failure.
  std::shared_ptr<GroupCode> compile(mlir::ModuleOp group, const std::string &pipeline, const std::string &pipelineOptions,
                                     std::string &error);

  const CompileStats &stats() const { return stats_; }
  opdsl::JITSession &session();

private:
  mlir::MLIRContext &ctx;
  const opdsl::PipelineRegistry &registry;
  CompilerOptions options;
  std::unique_ptr<opdsl::JITSession> session_;
  std::unique_ptr<opdsl::PersistentCache> cache_;
  std::unordered_map<std::string, std::shared_ptr<GroupCode>> code_;
  CompileStats stats_;
};

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_GROUPCOMPILER_H
