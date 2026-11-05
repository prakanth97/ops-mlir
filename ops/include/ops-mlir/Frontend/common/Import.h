//===- Import.h - options and result shared by the kernel importers ----*- C++ -*-===//

#ifndef OPS_MLIR_FRONTEND_COMMON_IMPORT_H
#define OPS_MLIR_FRONTEND_COMMON_IMPORT_H

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
  /// Fortran: module sources the kernel uses, in dependency order.
  std::vector<std::string> modules;
};

struct ImportResult {
  bool ok = false;
  std::string reason; ///< why the import failed
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string manifest;               ///< JSON text, with the digest of the module and the manifest
  std::vector<std::string> warnings;  ///< things that were dropped or approximated (printf)
  std::vector<std::string> dependencies; ///< files the import read (for build dependency files)
};

} // namespace ops_mlir::frontend

#endif // OPS_MLIR_FRONTEND_COMMON_IMPORT_H
