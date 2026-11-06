//===- Artifact.h - the per-kernel artifacts embedded by the translator --------*- C++ -*-===//
//
// `ops_jit_artifact` (OPS, ops_jit_artifact.h) holds the bytecode of an `ops.kernel` and a manifest (plan section 4.2).
// The runtime parses each once, keyed by the digest, and keeps the module and the decoded manifest.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_RUNTIME_ARTIFACT_H
#define OPS_MLIR_RUNTIME_ARTIFACT_H

#include "ops-mlir/Dialect/OPS/IR/OPSOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct ops_jit_artifact;

namespace ops_mlir::runtime {

struct ParamInfo {
  enum Kind { Acc, Gbl, Reduce, Idx } kind = Acc;
  std::string elem;      ///< f64, f32, i32 ... (not for idx)
  int64_t ncomp = 1;     ///< accessor: components the kernel uses
  int64_t dim = 1;       ///< gbl: elements the kernel reads; reduce: 1
  bool isConst = false;  ///< accessor: read-only
  std::string style;     ///< reduce: "accumulate" | "assign"
};

struct ConstInfo {
  std::string name, elem;
  int64_t offset = 0;
  bool dynamicIndex = false;
  int64_t arrayStride = 0;
  bool bound = false; ///< role "bound": its value is folded into the code
};

using Offsets = std::vector<std::vector<int64_t>>;

struct Manifest {
  std::string kernel, digest, importer;
  int ndim = 0;
  std::vector<ParamInfo> params;
  std::vector<ConstInfo> consts;
  std::map<int, Offsets> reads, writes; ///< constant offsets per accessor parameter
  std::set<int> dynamicReads;           ///< accessor parameters read at data-dependent offsets
  std::vector<int> idxAxesRead;
  int droppedPrintf = 0;
  int prints = 0; ///< printf calls the kernel keeps as ops.print
};

/// A kernel ready for the lifter: decoded manifest and the parsed `ops.kernel`.
struct KernelArtifact {
  Manifest manifest;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ops::KernelOp kernel;
  std::string error; ///< non-empty: unusable (reason)
};

/// Decodes the manifest text; on failure returns false and sets `error`.
bool parseManifest(const std::string &json, Manifest &out, std::string &error);

/// OPS element type name ("double", "float", "int", "long", ...) -> MLIR element type name ("f64", ...); empty if
/// unsupported by the JIT.
std::string elemName(const std::string &opsType);
unsigned elemBytes(const std::string &name);

/// Caches the parsed artifacts of an `mlir::MLIRContext`.
class ArtifactCache {
public:
  explicit ArtifactCache(mlir::MLIRContext &ctx) : ctx(ctx) {}

  /// The decoded artifact, or null with `reason` when it cannot be used (status failed, unknown version, bytecode that
  /// does not parse).
  const KernelArtifact *get(const ops_jit_artifact *artifact, std::string &reason);

private:
  mlir::MLIRContext &ctx;
  std::unordered_map<const ops_jit_artifact *, std::unique_ptr<KernelArtifact>> byPointer;
  std::unordered_map<std::string, KernelArtifact *> byDigest;
};

} // namespace ops_mlir::runtime

#endif // OPS_MLIR_RUNTIME_ARTIFACT_H
