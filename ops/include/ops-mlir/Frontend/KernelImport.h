//===- KernelImport.h - OPS kernel (C++) -> ops.kernel + manifest ------*- C++ -*-===//
//
// Provider A of the plan: a build-time frontend that reads one OPS user kernel with clang (in the translation unit
// of the application, with its include paths and definitions) and emits
//
//   * an MLIR module holding `ops.kernel @<name>`, and
//   * a manifest (JSON, plan section 4.2): what the kernel's parameters are, which constants it reads, which
//     offsets it accesses, whether it reduces, ...
//
// Anything outside the supported subset makes the import fail with a reason (status "failed" in the manifest): the
// loops that call the kernel then run through the stock OPS code.
//
//===----------------------------------------------------------------------===//

#ifndef OPS_MLIR_FRONTEND_KERNELIMPORT_H
#define OPS_MLIR_FRONTEND_KERNELIMPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>
#include <vector>

namespace ops_mlir::frontend {

struct ImportOptions {
  std::string kernelName;
  /// The translation unit to parse: the application source file that defines (or includes the header that
  /// defines) the kernel. Preferred, because it brings the application's globals, structs and macros.
  std::string sourceFile;
  /// Alternatively a header that defines the kernel: parsed after `#include <ops_lib_core.h>`.
  std::string headerFile;
  /// The dimension of the block the kernel runs on (1-3).
  int ndim = 0;
  /// Optional parameter kinds in order (acc, gbl, red, idx): the translator knows them from the call sites. Without
  /// them they are inferred from the parameter types and the use of the parameter.
  std::vector<std::string> kinds;
  /// The application's compiler flags (-I, -D, ...).
  std::vector<std::string> clangArgs;
  /// Clang's resource directory (builtin headers). Empty: the one this build was configured with.
  std::string resourceDir;
};

struct ImportResult {
  bool ok = false;
  std::string reason; ///< why the import failed
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string manifest;               ///< JSON text, with the digest of the module and the manifest
  std::vector<std::string> warnings;  ///< things that were dropped or approximated (printf)
  std::vector<std::string> dependencies; ///< files the translation unit read (for build dependency files)
};

/// "A/1.0": the provider and its version, recorded in manifests and part of their digests.
const char *importerId();

/// Imports the kernel. `ctx` needs the ops, arith, math and scf dialects.
ImportResult importKernel(mlir::MLIRContext &ctx, const ImportOptions &options);

/// A manifest for a kernel that could not be imported.
std::string failedManifest(const ImportOptions &options, const std::string &reason);

} // namespace ops_mlir::frontend

#endif // OPS_MLIR_FRONTEND_KERNELIMPORT_H
